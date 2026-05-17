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

import torch
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
    return "Street"


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


def _calibrate(raw: np.ndarray) -> np.ndarray:
    """
    Min-Max stretch: batch min → 0.10, batch max → 0.95.

    Using full Min-Max (not IQR) guarantees the worst photo in the batch scores
    near 0.10 and the best near 0.95 regardless of how similar the batch is.
    IQR compressed scores into [0.33, 0.67] for homogeneous batches, causing
    every photo to land in Mid and making TOPIQ's contribution irrelevant.
    """
    lo   = float(np.min(raw))
    hi   = float(np.max(raw))
    span = max(hi - lo, 1e-4)
    return np.clip((raw - lo) / span * 0.85 + 0.10, 0.10, 0.95)


def _raw_aspect_discriminants(
    img_emb: np.ndarray,
    aspect_pos: np.ndarray,   # (A, D)
    aspect_neg: np.ndarray,   # (A, D)
) -> np.ndarray:
    """Raw per-aspect discriminants (A,) — calibrated per-aspect across the batch."""
    return (img_emb @ aspect_pos.T) - (img_emb @ aspect_neg.T)


_TIER_DESC = {
    "strong": "Strong visual intent — decisive moment, bold geometry, or atmospheric power.",
    "mid":    "Some strong elements but inconsistent execution or missing visual tension.",
    "weak":   "Blurry, poorly framed, flat light, or no clear visual subject or intent.",
}

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


def _build_reasoning(
    score: float,
    aspect_scores: Dict[str, float],
    is_verified: bool,
) -> str:
    tier     = _tier(score)
    pct      = int(round(score * 100))
    lines    = [f"{tier.capitalize()}  {pct}%", _TIER_DESC[tier]]

    if aspect_scores:
        lines.append("")
        for k, v in sorted(aspect_scores.items(), key=lambda x: -x[1]):
            label = _ASPECT_LABEL.get(k, k)
            bar   = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"{label:<12} {bar}  {int(v*100)}%")
        top    = _ASPECT_LABEL.get(max(aspect_scores, key=aspect_scores.get),
                                   max(aspect_scores, key=aspect_scores.get))
        bottom = _ASPECT_LABEL.get(min(aspect_scores, key=aspect_scores.get),
                                   min(aspect_scores, key=aspect_scores.get))
        lines.append(f"\nBest: {top}   ·   Weakest: {bottom}")

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
        cal_overall = _calibrate(raw_overall)

        # Aspect scores: calibrate each dimension independently across the batch
        cal_aspects: Optional[np.ndarray] = None
        if raw_aspects is not None:
            cal_aspects = np.stack([
                _calibrate(raw_aspects[:, j])
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
            "Liminal":      {"Narrative": 0.38, "Lighting": 0.42,   "Composition": 0.14, "Technical": 0.03, "Human/Culture": 0.00},
            # Architectural keeps higher Technical — buildings should be sharp.
            "Architectural":{"Composition": 0.44, "Lighting": 0.26, "Technical": 0.18, "Narrative": 0.12, "Human/Culture": 0.00},
            # FineArt: atmospheric mood and narrative intent are everything.
            # Technical near-zero — vintage glass, soft focus, and high-ISO grain
            # are intentional aesthetic signatures, not technical failures.
            # Lighting is the dominant dimension (low-key, available light, cinematic dark).
            "FineArt":      {"Lighting": 0.42, "Narrative": 0.33, "Composition": 0.18, "Technical": 0.04, "Human/Culture": 0.03},
        }

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
            genre = _detect_genre(aspect_scores) if aspect_scores else "Street"
            w = _GENRE_W.get(genre, _GENRE_W["Street"])
            genre_score = sum(aspect_scores.get(k, 0.5) * v for k, v in w.items())
            overall_clip = float(cal_overall[i])
            draft_score  = float(np.clip(0.60 * genre_score + 0.40 * overall_clip, 0.15, 0.85))

            confidence  = min(1.0, abs(draft_score - 0.5) * 2.0)

            reasoning = _build_reasoning(draft_score, aspect_scores, is_verified)

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
