"""
SpecVLM Pipeline - CLIP-Based Composition + Aspect Scoring

Architecture:
    Bulk Encoder (SigLIP-2 ViT-g/14) → 1536-d Embeddings
        ↓
    SpecVLMPipeline (pure CLIP math, no LLM)
        ├─ Overall calibrated score (pos vs neg aesthetic prompts)
        └─ Per-aspect scores (Technical / Composition / Lighting / Narrative / Human)
        ↓
    grade_pipeline_v2 Step 4d: score fusion
        (tech*0.40 + composition*0.30 + semantic*0.30)

VRAM Protocol:
    1. SigLIP2Encoder.encode_images() → all embeddings computed
    2. SpecVLMPipeline.grade_images() → pure numpy, no GPU load
    3. TechnicalHead (TOPIQ NR + MANIQA) → IQA scoring
    4. grade_pipeline_v2 → score fusion, PersonalHead, LanceDB
"""

from __future__ import annotations

import os
import json
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

# NOTE: torch is deliberately NOT imported here. This module's runtime path is
# pure-numpy cosine scoring over embeddings the caller already holds — it never
# ran a torch op (the import was dead: one reference, the import line itself).
# But grade_pipeline_v2 imports `_cd_brief_implies_empty` from here during the
# early-exit gate, so that dead import dragged 347 MB of torch into the
# CUDA-free grade worker at exactly the wrong moment: measured peak_wset jumped
# 0.72 -> 1.39 GB in that window, while the 2.7 GB SigLIP subprocess was about
# to load. If a torch op is ever genuinely needed here, import it inside the
# function that needs it — never at module scope.
import numpy as np

# Model paths
MODEL_DIR = Path("models/specvlm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Priority gate threshold
DRAFT_CONFIDENCE_THRESHOLD = 0.85

# Batch inference settings
BATCH_SIZE  = 24   # images per single VLM forward pass (was 16)
MAX_WAIT_MS = 500  # partial-batch flush threshold (ms)

# Score thresholds outside which 7B verify is skipped regardless of confidence.
# Only images in the grey zone [VERIFY_SCORE_LO, VERIFY_SCORE_HI] are worth
# the 7B latency — clearly Weak or clearly Strong shots don't benefit.
VERIFY_SCORE_LO = 0.32
VERIFY_SCORE_HI = 0.75

# FlashAttention-2: enabled when flash_attn package is installed
try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_KWARGS = {"attn_implementation": "flash_attention_2"}
    print("[specvlm] FlashAttention-2 available")
except ImportError:
    _FLASH_ATTN_KWARGS = {}

# ── Creative Direction brief context (Task 2: Subject Intrusion constraint) ────
# Set via set_cd_brief() before grading in a creative direction context.
_CD_BRIEF: str = ""
_CD_EMPTY_KEYWORDS = {"empty", "liminal", "desert", "void", "abandoned", "desolate"}

_SUBJECT_INTRUSION_CONSTRAINT = (
    "\nCREATIVE DIRECTION CONSTRAINT: The current brief implies an absence of people. "
    "The presence of even a single human subject is a BINARY FAILURE for this brief. "
    "Do NOT score based on lighting or composition if a person is present. "
    "If any person is visible: categorize as 'Subject Intrusion', set score to ≤ 0.40, "
    "and begin reasoning_log with 'disqualification: Subject Intrusion — person detected'.\n"
)


def set_cd_brief(brief: str) -> None:
    """Register the Creative Direction style brief so Subject Intrusion logic activates."""
    global _CD_BRIEF
    _CD_BRIEF = brief or ""


def _cd_brief_implies_empty() -> bool:
    text = _CD_BRIEF.lower()
    return any(kw in text for kw in _CD_EMPTY_KEYWORDS)

# ── Aesthetic text prompts for CLIP-based scoring ──────────────────────────────
# Used by SpecVLMPipeline when pre-computed SigLIP-2 embeddings are available.
# Encoded once by SigLIP2Encoder in grade_pipeline_v2 before unloading.

_POS_PROMPTS: List[str] = [
    # Street / documentary / human-centric
    "a stunning street photograph with decisive moment and perfect composition",
    "award-winning documentary photography with authentic emotion and visual impact",
    "compelling candid photography with strong story and human connection",
    "masterful street photography with dynamic layering and visual hierarchy",
    # Architecture / geometric / structural
    "fine art architectural photography with bold geometry, strong lines and spatial depth",
    "graphic urban photography with precise geometric abstraction and tonal balance",
    "architectural composition with beautiful symmetry, shadow play and structural elegance",
    # Liminal / atmospheric / mood
    "powerful liminal space photograph — empty, atmospheric, quietly unsettling",
    "evocative environmental photography with mood, solitude and visual tension",
    "minimalist street scene with striking emptiness, fog, or reflective surfaces",
    # Low-light / fine art / intentional grain and softness
    "cinematic low-light street photography with intentional grain and moody atmosphere",
    "fine art photography with intentional soft focus, dreamlike quality, and artistic vision",
    "night street photography with available light, film grain, and atmospheric depth",
    "high contrast low-key photography with dramatic shadows and expressive tonal range",
]

_NEG_PROMPTS: List[str] = [
    "a snapshot with poor composition and no visual intent",
    "overexposed photo with bad framing and no clear subject",
    "cluttered confusing image with no focal point and distracting background",
    "low quality photo with blown exposure and flat uninteresting scene",
    "boring snapshot with no visual interest, no intention and no reason to look twice",
]

# Aspect-specific prompts for the verify pass (richer per-dimension breakdown)
_ASPECT_PROMPTS: Dict[str, List[str]] = {
    # Short, concrete noun-phrase prompts work best with SigLIP-2's image-caption training.

    # Technical: intentional softness, grain, and vintage lens rendering are valid fine-art
    # choices — the negative only targets genuine equipment/shooting failures.
    "Technical":     ["photograph with purposeful visual execution — whether crisp and clean "
                      "or rendered through vintage glass with organic grain and intentional softness",
                      "technically ruined photo — severe chromatic aberration, dead pixels, "
                      "extreme accidental camera shake destroying all detail"],

    "Composition":   ["well composed, leading lines, strong framing, clear subject",
                      "cluttered frame, no clear subject, bad cropping, random composition"],

    # Lighting: moody low-key and available light are positive — 'underlit' removed from
    # negative because it matches intentional low-light fine-art photography incorrectly.
    "Lighting":      ["evocative light with atmosphere — moody low-key available light, "
                      "dramatic shadows, cinematic darkness, golden hour warmth, or intentional "
                      "shadow play that adds dimension and mood",
                      "flat uninspired light with no mood, harshly overexposed blown highlights, "
                      "or fluorescent flatness that strips all atmosphere and tonal dimension"],

    "Narrative":     ["decisive moment, emotion, storytelling, atmosphere, solitude, tension, mood, quiet drama",
                      "accidental snapshot, no intent, boring frame, nothing to look at"],

    # Human/Culture: short concrete phrase so SigLIP-2 can match visual content.
    # Low score is expected for architectural/liminal; Step 4c weights penalise this ~0×.
    "Human/Culture": ["people, human figures, faces, crowd, street life",
                      "empty scene, no people, deserted, nobody present"],
}


# ── VisualMetadata dataclass ───────────────────────────────────────────────────


@dataclass
class VisualMetadata:
    path:          str
    filename:      str
    clip_score:    float           # calibrated 0–1 from CLIP math
    aspect_ratio:  str             # "3:2", "16:9", "1:1", etc.
    clip_tier:     str             # "Strong" / "Mid" / "Weak"
    aspect_scores: Dict[str, float]  # Technical / Composition / Lighting / Narrative / Human/Culture
    photo_genre:   str             # "Street" | "Architectural" | "Liminal"

    @property
    def is_architectural(self) -> bool:
        return self.photo_genre == "Architectural"

    @property
    def is_liminal(self) -> bool:
        return self.photo_genre == "Liminal"

    @property
    def is_fine_art(self) -> bool:
        return self.photo_genre == "FineArt"


def _detect_aspect_ratio(path: str) -> str:
    """Return nearest common ratio string from image dimensions."""
    try:
        from PIL import Image as _PIL
        with _PIL.open(path) as _img:
            w, h = _img.size
        ratio = w / h if h else 1.0
        # Map to nearest standard ratio
        _RATIOS = [(1/1,"1:1"),(4/3,"4:3"),(3/2,"3:2"),(16/9,"16:9"),(2/3,"2:3"),(3/4,"3:4"),(9/16,"9:16")]
        return min(_RATIOS, key=lambda x: abs(x[0] - ratio))[1]
    except Exception:
        return "unknown"


def _detect_genre(aspect_scores: Dict[str, float]) -> str:
    """
    Classify the photo genre from CLIP aspect scores.

    Priority order: Architectural → Liminal → FineArt → Street.

    Architectural: structural/geometric — buildings, symmetry, urban geometry.
        Human/Culture < 0.38 AND Composition > 0.52
    Liminal: empty/atmospheric — fog, corridors, underpasses, reflections.
        Human/Culture < 0.35 AND (Lighting > 0.50 OR Narrative > 0.48)
    FineArt: moody/atmospheric shots — low-light, vintage lens, available light,
        intentional soft focus. Can include people. Defined by evocative lighting
        and strong narrative intent rather than technical precision.
        Lighting > 0.50 AND Narrative > 0.45
    Street: everything else — candid, human-centric, decisive moment.
    """
    human = aspect_scores.get("Human/Culture", 0.5)
    comp  = aspect_scores.get("Composition",   0.5)
    light = aspect_scores.get("Lighting",      0.5)
    narr  = aspect_scores.get("Narrative",     0.5)

    # Architectural: strong geometry overrides atmospheric mood
    if human < 0.38 and comp > 0.52:
        return "Architectural"
    # Liminal: empty/atmospheric with no people
    if human < 0.35 and (light > 0.50 or narr > 0.48):
        return "Liminal"
    # FineArt: moody lighting + strong narrative, regardless of people presence.
    # Captures low-light, available-light, vintage-lens, atmospheric shots.
    if light > 0.50 and narr > 0.45:
        return "FineArt"
    # Catch-all for strongly peopleless frames that just miss the thresholds above
    # (e.g. comp 0.50, light 0.48). They must NOT be graded with the human-centric
    # Street weights (Narrative 0.44 / Human 0.05), which crush empty architectural
    # and liminal shots for lacking a subject. Route to a structure/atmosphere genre
    # whose Human/Culture weight is 0.
    if human < 0.40:
        return "Architectural" if comp >= light else "Liminal"
    return "Street"


# User-selected niche → SpecVLM genre. When the user explicitly picks a peopleless
# niche we honour it instead of per-image auto-detection, so a subject-less
# architectural/liminal frame gets genre weights with Human/Culture≈0 and is not
# penalised for lacking a human subject (auto-detect can mis-file it as "Street").
_PRESET_GENRE: Dict[str, str] = {
    "architectural": "Architectural",
    "urban_city":    "Architectural",
    "liminal":       "Liminal",
    "minimalist":    "Liminal",
    "abstract":      "Liminal",
    "macro":         "Liminal",
    "fine_art":      "FineArt",
    "night":         "FineArt",
    "landscape":     "FineArt",
}


def _preset_genre(preset: str) -> Optional[str]:
    """Map a user-selected niche (key/label/legacy) to a SpecVLM genre, or None."""
    if not preset:
        return None
    try:
        from niche_registry import resolve_key as _rk
        key = _rk(preset) or preset.lower().strip()
    except Exception:
        key = preset.lower().strip()
    return _PRESET_GENRE.get(key)


def build_visual_metadata(
    path: str,
    clip_score: float,
    aspect_scores: Dict[str, float],
) -> VisualMetadata:
    if clip_score >= 0.60:
        tier = "Strong"
    elif clip_score >= 0.41:
        tier = "Mid"
    else:
        tier = "Weak"
    return VisualMetadata(
        path         = path,
        filename     = Path(path).name,
        clip_score   = round(clip_score, 3),
        aspect_ratio = _detect_aspect_ratio(path),
        clip_tier    = tier,
        aspect_scores= {k: round(v, 3) for k, v in aspect_scores.items()},
        photo_genre  = _detect_genre(aspect_scores),
    )


# ── SpecVLM Result ─────────────────────────────────────────────────────────────


class SpecVLMResult:
    """Result from SpecVLM grading with reasoning log."""

    def __init__(
        self,
        path: str,
        score: float,
        confidence: float,
        reasoning_log: str,
        is_verified: bool = False,
        draft_score: Optional[float] = None,
        verify_score: Optional[float] = None,
        breakdown: Optional[Dict[str, float]] = None,
    ):
        self.path          = path
        self.score         = score
        self.confidence    = confidence
        self.reasoning_log = reasoning_log
        self.is_verified   = is_verified
        self.draft_score   = draft_score
        self.verify_score  = verify_score
        self.breakdown     = breakdown or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "score": self.score,
            "confidence": self.confidence,
            "reasoning_log": self.reasoning_log,
            "is_verified": self.is_verified,
            "draft_score": self.draft_score,
            "verify_score": self.verify_score,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── SpecVLM Pipeline Orchestrator ──────────────────────────────────────────────


def _raw_discriminant(
    img_emb: np.ndarray,
    pos_embs: np.ndarray,
    neg_embs: np.ndarray,
) -> float:
    """
    Raw discriminant = best-positive-similarity minus best-negative-similarity.

    In 1536-d space this typically spans only ±0.05, so it must be calibrated
    across the batch before mapping to a [0, 1] score.
    """
    return float(np.max(img_emb @ pos_embs.T)) - float(np.max(img_emb @ neg_embs.T))


_SCORE_FLOOR = 0.10
_SCORE_CEIL  = 0.95
_ANCHORS_PATH = Path(__file__).resolve().parent.parent / "cache" / "calibration_anchors.json"


def probe_fingerprint(pos_embs: np.ndarray, neg_embs: np.ndarray) -> str:
    """Identity of the scale the anchors were derived against.

    Hashing the probe EMBEDDINGS rather than the prompt text is deliberate: the
    embeddings are already a function of all three things the discriminant
    distribution depends on — the encoder tier (1536/1024/768-d are different
    spaces), the encoder checkpoint, and the probe set itself. One hash covers
    all three and cannot drift out of sync with them.

    This matters more than it looks: the positive probes are RAG-augmented, so
    uploading a reference PDF changes the probe set and therefore the scale.
    """
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(pos_embs, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(neg_embs, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def aspect_fingerprint(aspect_pos: np.ndarray, aspect_neg: np.ndarray) -> str:
    """Identity of the per-aspect scale. Separate from the overall fingerprint
    so changing the aspect prompts does not needlessly invalidate the overall
    anchors, and vice versa."""
    import hashlib
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(aspect_pos, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(aspect_neg, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def load_aspect_anchors(aspect_pos: np.ndarray, aspect_neg: np.ndarray,
                        aspect_names: "list"):
    """[(lo, hi)] per aspect in `aspect_names` order, or None.

    Each aspect gets its OWN pair: Composition and Technical have different
    discriminant distributions, so one shared scale would systematically flatter
    whichever aspect happens to sit higher.
    """
    try:
        import json
        if not _ANCHORS_PATH.exists():
            return None
        d = json.loads(_ANCHORS_PATH.read_text(encoding="utf-8"))
        stored = d.get("aspects") or {}
        want = aspect_fingerprint(aspect_pos, aspect_neg)
        if d.get("aspect_fingerprint") != want:
            print(f"[specvlm] aspect anchors are STALE — re-run "
                  f"scripts/derive_calibration_anchors.py")
            return None
        missing = [a for a in aspect_names if a not in stored]
        if missing:
            print(f"[specvlm] aspect anchors missing for {missing} — "
                  f"per-aspect scores stay batch-relative")
            return None
        return [(float(stored[a][0]), float(stored[a][1])) for a in aspect_names]
    except Exception as err:
        print(f"[specvlm] could not read aspect anchors ({err})")
        return None


def load_anchors(pos_embs: np.ndarray, neg_embs: np.ndarray):
    """(lo, hi) for this probe set, or None if absent or stale.

    Returns None rather than guessing. A stale anchor grades every photo against
    the wrong scale while looking completely healthy, which is a worse failure
    than the batch-relative bug this replaces.
    """
    try:
        import json
        if not _ANCHORS_PATH.exists():
            # Say WHERE. A silent miss here is indistinguishable from a stale
            # fingerprint at the call site, and both look like "anchors just
            # don't work" in the log.
            print(f"[specvlm] no anchors file at {_ANCHORS_PATH}")
            return None
        d = json.loads(_ANCHORS_PATH.read_text(encoding="utf-8"))
        want = probe_fingerprint(pos_embs, neg_embs)
        if d.get("fingerprint") != want:
            print(f"[specvlm] calibration anchors are STALE "
                  f"(probes/encoder/tier changed: {d.get('fingerprint')} != {want}) "
                  f"— re-run scripts/derive_calibration_anchors.py")
            return None
        return float(d["lo"]), float(d["hi"])
    except Exception as err:
        print(f"[specvlm] could not read calibration anchors ({err})")
        return None


def _calibrate(raw: np.ndarray, anchors: "Optional[tuple]" = None) -> np.ndarray:
    """
    Map raw discriminants onto [0.10, 0.95] against a FIXED scale.

    Why a fixed scale
    -----------------
    This used to min-max stretch each batch: batch min → 0.10, batch max → 0.95.
    That is grading on a curve. The same photograph scored 0.10 beside strong
    work and 0.95 beside weak work — Weak or Strong from identical pixels — and
    every batch was guaranteed to contain one of each, so a folder of uniformly
    excellent frames always manufactured rejects. It also made grades
    incomparable between culls, and reduced the absolute 0.60/0.41 thresholds to
    decoration, since their input had already been batch-normalised.

    CLAUDE.md:79 forbids exactly this. It survived because the 2026-06
    absolute-grading work removed quantile calibration from grade_pipeline_v2
    and stopped there; this min-max lived in the CLIP scorer.

    Why not simply drop the stretch
    -------------------------------
    The raw discriminant spans only about ±0.05 in 1536-d space, so unscaled it
    puts every photo in one bucket. An earlier IQR attempt compressed everything
    into [0.33, 0.67] — all Mid, TOPIQ irrelevant. The expansion has to stay;
    only its reference changes, from "this batch" to a fixed corpus.

    `anchors` is (lo, hi): the p1 and p99 of the raw discriminant over a
    reference corpus, derived by scripts/derive_calibration_anchors.py and
    fingerprinted on tier + encoder + probe set. Percentiles rather than
    min/max so one outlier frame cannot move the scale for everything graded
    afterwards; values outside clamp.

    With no anchors this degrades to the old batch-relative behaviour and SAYS
    SO. Grading silently against an absent or stale scale would be worse than
    the bug being fixed here.
    """
    if anchors is None:
        print("[specvlm] WARNING: no calibration anchors — falling back to "
              "batch-relative scoring. Grades are NOT comparable across culls. "
              "Run scripts/derive_calibration_anchors.py")
        if len(raw) == 1:
            return np.clip(0.52 + raw * 1.40, _SCORE_FLOOR, _SCORE_CEIL)
        lo   = float(np.min(raw))
        hi   = float(np.max(raw))
        span = max(hi - lo, 1e-4)
        return np.clip((raw - lo) / span * 0.85 + _SCORE_FLOOR,
                       _SCORE_FLOOR, _SCORE_CEIL)

    lo, hi = float(anchors[0]), float(anchors[1])
    span = max(hi - lo, 1e-6)
    span_out = _SCORE_CEIL - _SCORE_FLOOR
    # No len(raw)==1 special case: a fixed scale is defined for one photo
    # exactly as it is for a thousand, which is the point.
    return np.clip((raw - lo) / span * span_out + _SCORE_FLOOR,
                   _SCORE_FLOOR, _SCORE_CEIL)


def _raw_aspect_discriminants(
    img_emb: np.ndarray,
    aspect_pos: np.ndarray,   # (A, D)
    aspect_neg: np.ndarray,   # (A, D)
) -> np.ndarray:
    """Raw per-aspect discriminants (A,) — calibrated per-aspect across the batch."""
    return (img_emb @ aspect_pos.T) - (img_emb @ aspect_neg.T)


_ASPECT_LABEL = {
    "Technical":    "Technical",
    "Composition":  "Composition",
    "Lighting":     "Lighting",
    "Narrative":    "Moment",
    "Human/Culture":"Human",
}


def _tier(score: float) -> str:
    if score >= 0.60:
        return "strong"
    if score >= 0.41:
        return "mid"
    return "weak"


# Per-aspect photographer observations keyed by score band.
# Written in the voice of a photo editor marking contact sheets — direct, specific.
_ASPECT_NOTES: Dict[str, list] = {
    "Composition": [
        (0.78, "Frame is airtight — every element earns its place."),
        (0.62, "Geometry works; the eye moves without fighting the edges."),
        (0.45, "Framing is serviceable but the edges carry dead weight."),
        (0.00, "Frame is loose — crop it or reshoot it."),
    ],
    "Lighting": [
        (0.78, "Light has direction and authority — shadow play is doing the work."),
        (0.62, "Light is readable; contrast holds."),
        (0.45, "Flat light. No drama, no depth — nothing to push the subject forward."),
        (0.00, "Light is fighting the image. Blown highlights or dead-flat exposure."),
    ],
    # Alternative Lighting notes for intentional low-key / chiaroscuro / available-light work.
    # Used when genre is FineArt/Liminal or Narrative score suggests deliberate mood.
    "Lighting_moody": [
        (0.78, "Light has direction and authority — shadow play is doing the work."),
        (0.62, "Light is readable; contrast holds."),
        (0.45, "Low-key rendering — shadow weight reads as intentional mood."),
        (0.00, "Deep shadow dominance. Chiaroscuro or available-light approach — darkness as intent."),
    ],
    "Narrative": [
        (0.78, "The moment is decisive — gesture or tension frozen at exactly the right frame."),
        (0.62, "A moment caught, not staged — feels authentic."),
        (0.45, "Something is happening but nothing is at stake."),
        (0.00, "No moment. The scene is static and the camera just witnessed it."),
    ],
    "Human/Culture": [
        (0.78, "The human subject commands the frame — presence is undeniable."),
        (0.62, "Human element adds weight; the figure belongs here."),
        (0.45, "Figures are present but incidental — they don't anchor anything."),
        (0.00, "No human element. Architectural or environmental — works only if intentional."),
    ],
    "Technical": [
        (0.78, "Technical execution disappears into the image — as it should."),
        (0.62, "Technically clean. No distraction."),
        (0.45, "Some softness or exposure drift. Manageable, not invisible."),
        (0.00, "Technical failure is visible — motion blur, clipping, or heavy noise."),
    ],
    # Alternative Technical notes for FineArt/moody images where soft focus,
    # grain, and low-light rendering are deliberate aesthetic choices.
    "Technical_moody": [
        (0.78, "Technical execution disappears into the image — as it should."),
        (0.62, "Technically clean. No distraction."),
        (0.45, "Soft rendering or organic grain — intentional aesthetic signature, not a failure."),
        (0.00, "Technical compromise is visible — but in fine-art work, intentional grain and glow are valid."),
    ],
}

# Overall verdict by dominant aspect + tier
_VERDICT: Dict[str, Dict[str, str]] = {
    "strong": {
        "Narrative":     "Street photographer's instinct — right place, right frame, right moment.",
        "Composition":   "Geometric authority. The structure carries the image.",
        "Lighting":      "Light as subject. Everything else serves the atmosphere.",
        "Human/Culture": "The figure is the photograph. Everything else is context.",
        "Technical":     "Technically confident — the craft is invisible.",
    },
    "mid": {
        "Narrative":     "The moment is there but the frame doesn't fully commit to it.",
        "Composition":   "Decent bones. The structure works but doesn't surprise.",
        "Lighting":      "Light is present but not working hard enough.",
        "Human/Culture": "The human element is in the frame but not in control of it.",
        "Technical":     "Technically adequate. Won't lose the shot but won't win it either.",
    },
    "weak": {
        "Narrative":     "No decisive moment — the shutter fired but nothing was caught.",
        "Composition":   "The frame is not resolved. Too much, too little, or in the wrong place.",
        "Lighting":      "Light is the problem here, not the solution.",
        "Human/Culture": "The subject is lost. Distance, angle, or timing killed it.",
        "Technical":     "Technical compromise dominates. The image can't recover from it.",
    },
}


def _aspect_note(key: str, value: float, moody: bool = False) -> str:
    # Use moody-aware notes for Lighting and Technical when the image is
    # intentionally dark/low-key/soft so CLIP score dips don't read as failures.
    lookup_key = f"{key}_moody" if moody and key in ("Lighting", "Technical") else key
    for threshold, note in _ASPECT_NOTES.get(lookup_key, _ASPECT_NOTES.get(key, [])):
        if value >= threshold:
            return note
    return ""


def _build_reasoning(
    score: float,
    aspect_scores: Dict[str, float],
    is_verified: bool,
    grade: str = "",
    genre: str = "",
) -> str:
    if grade:
        tier = "strong" if "Strong" in grade else "weak" if "Weak" in grade else "mid"
    else:
        tier = _tier(score)
    pct  = int(round(score * 100))

    sorted_aspects = sorted(aspect_scores.items(), key=lambda x: -x[1]) if aspect_scores else []
    top_key    = sorted_aspects[0][0]  if sorted_aspects else "Narrative"
    bottom_key = sorted_aspects[-1][0] if sorted_aspects else "Technical"

    # Intentional low-light / chiaroscuro / soft-focus detection:
    # FineArt/Liminal genre implies darkness/mood is deliberate.
    # For other genres, strong Narrative intent alongside weaker Lighting
    # also signals a deliberate low-key or available-light choice.
    narrative_score = aspect_scores.get("Narrative", 0.0)
    lighting_score  = aspect_scores.get("Lighting",  1.0)
    is_moody = (
        genre in ("FineArt", "Liminal")
        or (narrative_score >= 0.38 and lighting_score < 0.55)
    )

    verdict = _VERDICT.get(tier, {}).get(top_key, "")
    # For weak Lighting verdict on moody images, replace the penalizing
    # "Light is the problem" verdict with atmospheric-intent language.
    if tier == "weak" and top_key == "Lighting" and is_moody:
        verdict = "Atmospheric depth through shadow — low-key is the visual language here."
    elif tier == "mid" and top_key == "Lighting" and is_moody:
        verdict = "Shadow and atmosphere doing most of the work — mood over exposure."

    top_label    = _ASPECT_LABEL.get(top_key,    top_key)
    bottom_label = _ASPECT_LABEL.get(bottom_key, bottom_key)

    lines = [f"{tier.capitalize()}  {pct}%"]
    if verdict:
        lines.append(verdict)
    lines.append("")

    for k, v in sorted_aspects:
        note = _aspect_note(k, v, moody=is_moody)
        if note:
            label = _ASPECT_LABEL.get(k, k)
            lines.append(f"{label}: {note}")

    lines.append(f"\nBest: {top_label}   ·   Weakest: {bottom_label}")
    return "\n".join(lines)


class SpecVLMPipeline:
    """
    CLIP-based aesthetic grading pipeline using SigLIP-2 embeddings.

    When pre-computed embeddings + text reference embeddings are provided
    (passed from grade_pipeline_v2 before SigLIP-2 is unloaded), grading is
    purely CPU / numpy — no extra GPU models loaded.

    Draft pass  : overall positive vs negative aesthetic similarity.
    Verify pass : per-aspect scoring (Technical / Composition / Lighting /
                  Narrative / Human) for images whose draft confidence is low.
    """

    def __init__(self) -> None:
        pass

    def grade_images(
        self,
        image_paths: List[str],
        progress=None,
        scan_mode: bool = False,
        preset: str = "",
        embeddings: Optional[np.ndarray] = None,
        pos_text_embs: Optional[np.ndarray] = None,
        neg_text_embs: Optional[np.ndarray] = None,
        aspect_pos_embs: Optional[np.ndarray] = None,
        aspect_neg_embs: Optional[np.ndarray] = None,
        aspect_names: Optional[List[str]] = None,
    ) -> List[SpecVLMResult]:
        """
        Grade a batch of images.

        When `embeddings` and `pos_text_embs` / `neg_text_embs` are supplied the
        pipeline runs entirely on pre-computed SigLIP-2 vectors (fast, no GPU).
        """
        if not image_paths:
            return []

        _p = progress or (lambda f, d: None)
        n  = len(image_paths)

        if embeddings is None or pos_text_embs is None or neg_text_embs is None:
            raise RuntimeError(
                "SpecVLMPipeline requires pre-computed SigLIP-2 embeddings and "
                "text reference embeddings. Make sure grade_pipeline_v2 passes "
                "embeddings=, pos_text_embs=, neg_text_embs= to grade_images()."
            )

        have_aspects = (
            aspect_pos_embs is not None
            and aspect_neg_embs is not None
            and aspect_names is not None
        )

        # ── Pass 1: raw discriminants for all images ──────────────────────────
        _p(0.51, "Scoring images…")
        raw_overall = np.array([
            _raw_discriminant(embeddings[i], pos_text_embs, neg_text_embs)
            for i in range(n)
        ])

        # Per-aspect raw matrix (n, A) — only computed if aspects are available
        raw_aspects: Optional[np.ndarray] = None
        if have_aspects:
            raw_aspects = np.stack([
                _raw_aspect_discriminants(embeddings[i], aspect_pos_embs, aspect_neg_embs)
                for i in range(n)
            ])  # (n, A)

        # ── Calibrate: stretch the batch distribution to fill the grade range ─
        # Overall score: percentile-stretch to [0.05, 0.95]
        cal_overall = _calibrate(raw_overall,
                                 anchors=load_anchors(pos_text_embs, neg_text_embs))

        # Aspect scores: calibrate each dimension independently across the batch
        cal_aspects: Optional[np.ndarray] = None
        if raw_aspects is not None:
            # Per-aspect anchors, not one shared pair: the five dimensions have
            # different discriminant distributions. Calibrating them against a
            # single scale would systematically flatter whichever sits highest.
            _asp_anchors = load_aspect_anchors(
                aspect_pos_embs, aspect_neg_embs, aspect_names or [])
            cal_aspects = np.stack([
                _calibrate(raw_aspects[:, j],
                           anchors=_asp_anchors[j] if _asp_anchors else None)
                for j in range(raw_aspects.shape[1])
            ], axis=1)  # (n, A)

        # ── Pass 2: build results, run verify pass for uncertain images ───────
        results: List[SpecVLMResult] = []

        # Genre-aware aspect weights (sum to 1.0 per genre).
        # Human/Culture = 0.0 for Liminal/Architectural so empty-scene photos
        # are not penalised for lacking human subjects.
        _GENRE_W: Dict[str, Dict[str, float]] = {
            # Technical weight kept very low for Street/Liminal — intentional grain,
            # soft lens, and low-light are valid artistic choices, not failures.
            "Street":       {"Narrative": 0.44, "Composition": 0.30, "Lighting": 0.18, "Technical": 0.03, "Human/Culture": 0.05},
            "Liminal":      {"Narrative": 0.36, "Lighting": 0.30,   "Composition": 0.29, "Technical": 0.05, "Human/Culture": 0.00},
            # Architectural keeps higher Technical — buildings should be sharp.
            "Architectural":{"Composition": 0.44, "Lighting": 0.26, "Technical": 0.18, "Narrative": 0.12, "Human/Culture": 0.00},
            # FineArt: atmospheric mood and narrative intent are everything.
            # Technical near-zero — vintage glass, soft focus, and high-ISO grain
            # are intentional aesthetic signatures, not technical failures.
            # Lighting is the dominant dimension (low-key, available light, cinematic dark).
            "FineArt":      {"Lighting": 0.42, "Narrative": 0.33, "Composition": 0.18, "Technical": 0.04, "Human/Culture": 0.03},
        }

        _forced_genre = _preset_genre(preset)
        if _forced_genre:
            print(f"[specvlm] niche '{preset}' → forcing genre '{_forced_genre}' (Human/Culture not penalised)")

        for i, path in enumerate(image_paths):
            is_verified = False

            # Always populate aspect scores — every photo gets all 5 bars
            aspect_scores: Dict[str, float] = {}
            if have_aspects and cal_aspects is not None:
                aspect_scores = {
                    name: round(float(cal_aspects[i, j]), 3)
                    for j, name in enumerate(aspect_names)
                }

            # Genre-aware weighted score: weights aspects by what matters for each genre.
            # Blended 60/40 with the overall CLIP discriminant so holistic aesthetic
            # quality (pos vs neg prompts) still contributes alongside genre logic.
            # Honour the user-selected niche when it maps to a peopleless genre;
            # otherwise fall back to per-image auto-detection.
            genre = _forced_genre or (_detect_genre(aspect_scores) if aspect_scores else "Street")
            w = _GENRE_W.get(genre, _GENRE_W["Street"])
            genre_score = sum(aspect_scores.get(k, 0.5) * v for k, v in w.items())
            overall_clip = float(cal_overall[i])
            # Peopleless genres are unfairly dragged by the holistic CLIP discriminant
            # whose NEG prompts ("no clear subject", "flat uninteresting", "boring …
            # no reason to look twice") match empty/atmospheric scenes BY DESIGN. For
            # those genres lean on the genre-weighted score, halve the holistic weight,
            # and lift the floor so a strong empty frame isn't dumped into deep-Weak
            # for simply lacking a conventional subject.
            if genre in ("Liminal", "Architectural", "FineArt"):
                draft_score = float(np.clip(0.80 * genre_score + 0.20 * overall_clip, 0.30, 0.88))
            else:
                draft_score = float(np.clip(0.60 * genre_score + 0.40 * overall_clip, 0.15, 0.85))

            confidence  = min(1.0, abs(draft_score - 0.5) * 2.0)

            reasoning = _build_reasoning(draft_score, aspect_scores, is_verified, genre=genre)

            results.append(SpecVLMResult(
                path          = path,
                score         = draft_score,
                confidence    = confidence,
                reasoning_log = reasoning,
                is_verified   = is_verified,
                draft_score   = draft_score,
                breakdown     = aspect_scores,
            ))

            _p(
                0.51 + 0.35 * (i + 1) / n,
                f"Graded {i + 1}/{n} — {draft_score:.2f}",
            )

        return results

    def unload(self) -> None:
        pass
