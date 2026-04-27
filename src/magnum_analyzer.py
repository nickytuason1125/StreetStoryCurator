import os, json, logging
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

_log = logging.getLogger(__name__)
from typing import Dict, List, Optional, Tuple
import torch
import clip
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine

# Relative import for src/ package; fall back to absolute when run as a script
try:
    from .exif_handler import sort_by_timeline
except ImportError:
    from exif_handler import sort_by_timeline  # type: ignore

try:
    from human_grader import get_human_aesthetic_score as _get_human_score
    _HUMAN_OK = True
except ImportError:
    _HUMAN_OK = False
    def _get_human_score(img_path, **kw): return 0.5  # type: ignore

# ---------------------------------------------------------------------------
# Competition presets  (display name → weight dict)
# ---------------------------------------------------------------------------

COMPETITION_PRESETS: Dict[str, Optional[Dict[str, float]]] = {
    "Magnum Editor":        {"decisive_moment": 0.25, "layering_depth": 0.20, "juxtaposition": 0.15, "light_atmosphere": 0.15, "authenticity": 0.15, "composition_geometry": 0.10},
    "LSPF (London Street)": {"decisive_moment": 0.30, "layering_depth": 0.15, "juxtaposition": 0.20, "light_atmosphere": 0.15, "authenticity": 0.10, "composition_geometry": 0.10},
    "SPI (International)":  {"decisive_moment": 0.20, "layering_depth": 0.15, "juxtaposition": 0.30, "light_atmosphere": 0.15, "authenticity": 0.10, "composition_geometry": 0.10},
    "Custom":               None,
}

# Backward-compatible aliases used by the rest of the module
PROFILES = COMPETITION_PRESETS

# Reference weight table — human_perception is always applied as a fixed
# +0.2 coefficient on top of the 6 competition criteria, so it is excluded
# from _CRITERIA_KEYS to avoid double-counting with preset weight dicts.
JUDGING_CRITERIA: Dict[str, float] = {
    "decisive_moment":      0.18,
    "layering_depth":       0.14,
    "juxtaposition":        0.14,
    "light_atmosphere":     0.14,
    "authenticity":         0.12,
    "composition_geometry": 0.08,
    "human_perception":     0.20,   # fixed coefficient, not in preset weight dicts
}
_CRITERIA_KEYS = [k for k in JUDGING_CRITERIA if k != "human_perception"]

# Maps preset display name → short key used inside _generate_critique
_PRESET_TO_PROFILE: Dict[str, str] = {
    "Magnum Editor":        "magnum",
    "LSPF (London Street)": "lspf",
    "SPI (International)":  "spi",
    "Custom":               "custom",
}

def _resolve_weights(preset: str, custom_weights: Optional[Dict] = None) -> Dict[str, float]:
    """Return the weight dict for a preset, falling back to Magnum Editor."""
    if custom_weights:
        return custom_weights
    w = COMPETITION_PRESETS.get(preset)
    return w if w is not None else COMPETITION_PRESETS["Magnum Editor"]  # type: ignore

# First 8 prompt → positive, last 3 → negative / cliché
COMPETITION_PROMPTS: List[str] = [
    "decisive moment in street photography",        # +
    "candid human interaction",                     # +
    "layered urban environment",                    # +
    "juxtaposition of old and new",                 # +
    "natural light and long shadows",               # +
    "authentic street scene",                       # +
    "open narrative ambiguity",                     # +
    "geometric framing leading lines",              # +
    "touristy cliché",                              # −
    "over saturated HDR",                           # −
    "posed or staged",                              # −
]
_N_POSITIVE = 8
_N_NEGATIVE = 3

# Sequence rationale labels (one per slot)
RATIONALE: List[str] = [
    "1. Establishing  — wide/multi-layered context, sets tone & geometry.",
    "2. Subject/Moment — peak gesture or interaction, narrative anchor.",
    "3. Detail/Geometry — texture, line, or micro-contrast, visual rhythm.",
    "4. Contrast/Shift  — light, mood, or perspective break, keeps pace.",
    "5. Atmosphere      — negative space, ambiguity, lingering mood, resolution.",
]


# ---------------------------------------------------------------------------
# MagnumStreetAnalyzer
# ---------------------------------------------------------------------------

class MagnumStreetAnalyzer:

    def __init__(self, model_root="./models", cache_path="cache/magnum_scores.json", preset="Magnum Editor", custom_weights=None):
        self.model_root = Path(model_root)
        self.model_root.mkdir(exist_ok=True)
        self.device = torch.device("cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, download_root=str(self.model_root))
        self.model.eval()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = self._load_cache()
        self.weights = _resolve_weights(preset, custom_weights)
        # Internal state used by the rest of the class
        self.profile      = _PRESET_TO_PROFILE.get(preset, "magnum")
        self._text_embs: Optional[np.ndarray] = None

    def apply_preset(self, preset_name, custom_weights=None):
        """Switch to a named competition preset and clear the score cache."""
        self.weights = _resolve_weights(preset_name, custom_weights)
        self.profile = _PRESET_TO_PROFILE.get(preset_name, self.profile)
        self._clear_cache()  # force re-scoring with new weights

    def _clear_cache(self):
        self.cache.clear()
        self._text_embs = None          # text embeddings are weight-independent but cheap to rebuild
        if self.cache_path.exists():
            self.cache_path.unlink()

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2), encoding="utf-8"
        )

    # ---------------------------------------------- cached text embeddings

    def _get_text_embs(self) -> np.ndarray:
        """Encode COMPETITION_PROMPTS once and cache in memory. Shape: (11, 512)."""
        if self._text_embs is None:
            tokens = clip.tokenize(COMPETITION_PROMPTS).to(self.device)
            with torch.no_grad():
                embs = self.model.encode_text(tokens)
                embs = embs / embs.norm(dim=-1, keepdim=True)
            self._text_embs = embs.cpu().numpy()
        return self._text_embs

    # ----------------------------------------------------------------- public

    def analyze_folder(self, folder_path: str) -> List[Tuple[str, Dict]]:
        """
        Grade all images in folder_path under the active profile.

        Pixel/CLIP signals are loaded from disk cache when available;
        only new images trigger inference.  Switching the profile and
        calling this again re-weights cached signals instantly.

        Returns list of (absolute_path, result_dict) tuples where
        result_dict contains per-criterion scores, final weighted score,
        grade, critique, and embedding.
        """
        results: List[Tuple[str, Dict]] = []
        for f in sorted(Path(folder_path).iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                path = str(f.resolve())
                if path not in self.cache:
                    self.cache[path] = self._analyze_signals(path)
                    self._save_cache()
                signals = self.cache[path]
                if "error" in signals:
                    results.append((path, signals))
                else:
                    results.append((path, self._apply_profile(signals)))
        return results

    def switch_profile(self, preset: str, custom_weights=None) -> None:
        """Change the active preset without clearing the cache (use apply_preset to also clear)."""
        self.weights = _resolve_weights(preset, custom_weights)
        self.profile = _PRESET_TO_PROFILE.get(preset, self.profile)

    # --------------------------------------------------------------- analysis

    def _apply_profile(self, signals: Dict) -> Dict:
        """
        Apply self.weights to a pre-computed signals dict.
        Returns a full result dict (score, grade, critique included).
        This is O(1) — no image I/O or inference.
        """
        criterion_scores = {k: signals[k] for k in _CRITERIA_KEYS}
        semantic_align   = signals["semantic_align"]
        human_score      = signals.get("human_perception", 0.5)

        weighted_sum = sum(self.weights.get(k, 0) * v for k, v in criterion_scores.items())
        final        = float(np.clip(
            weighted_sum + 0.2 * human_score, 0.0, 1.0
        ))

        grade = (
            "Strong \u2705" if final > 0.68
            else "Mid \u26a0\ufe0f" if final > 0.48
            else "Weak \u274c"
        )
        scores_for_critique = {**criterion_scores, "human_perception": human_score}
        return {
            **{k: round(v, 3) for k, v in criterion_scores.items()},
            "semantic_align":   round(semantic_align, 3),
            "human_perception": round(human_score,    3),
            "score":            round(final, 3),
            "grade":            grade,
            "critique":         self._generate_critique(scores_for_critique, grade),
            "profile":          self.profile,
            "subject_count":    signals["subject_count"],
            "embedding":        signals["embedding"],
            "dims":             signals["dims"],
        }

    def _analyze_signals(self, img_path: str) -> Dict:
        """
        Compute all profile-agnostic pixel and CLIP signals for one image.
        Result is cached to disk; does NOT include score/grade/critique.
        """
        try:
            return self._analyze_signals_inner(img_path)
        except Exception as exc:
            _log.warning("Signal analysis failed for %s: %s", img_path, exc)
            return {"error": str(exc)}

    def _analyze_signals_inner(self, img_path: str) -> Dict:
        """Inner implementation — called exclusively by _analyze_signals."""
        img = cv2.imread(img_path)
        if img is None:
            return {"error": "Load failed"}

        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w    = img.shape[:2]
        pil_img = Image.open(img_path).convert("RGB")

        # ── 1. DECISIVE MOMENT ─────────────────────────────────────────────
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml"
        )
        raw_faces  = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        raw_bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

        # detectMultiScale returns an empty tuple () when nothing found, not []
        face_count    = len(raw_faces)  if not isinstance(raw_faces,  tuple) else 0
        body_count    = len(raw_bodies) if not isinstance(raw_bodies, tuple) else 0
        subject_count = face_count + body_count

        moment_density = min((face_count * 1.5 + body_count) / 6.0, 1.0)

        # Motion-freeze proxy: high gradient std → frozen peak action
        gx       = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy       = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        gradient = np.hypot(gx, gy)
        freeze_score = min(float(np.std(gradient)) / 80.0, 1.0)

        decisive_moment = 0.5 * moment_density + 0.5 * freeze_score

        # ── 2. LAYERING & DEPTH ────────────────────────────────────────────
        # Split into horizontal row-thirds: foreground (bottom) / mid / background (top)
        zones = np.array_split(gray, 3, axis=0)
        zone_density = [float(np.mean(cv2.Canny(z, 50, 150) > 0)) for z in zones]
        spread        = max(zone_density) - min(zone_density)
        layering_depth = float(np.clip(0.3 * spread + 0.7 * np.mean(zone_density), 0.0, 1.0))

        # ── 3. JUXTAPOSITION / TENSION ─────────────────────────────────────
        # FIX: np.vsplit → top & bottom rows;  np.hsplit → left & right columns
        top,  bottom = np.array_split(gray, 2, axis=0)   # row-wise: handles odd heights
        left, right  = np.array_split(gray, 2, axis=1)   # col-wise: handles odd widths

        mean_top, mean_bottom = float(np.mean(top)), float(np.mean(bottom))
        # Ratio of darker-half to brighter-half: 1.0 = identical, 0.0 = max contrast
        tb_ratio    = min(mean_top, mean_bottom) / max(max(mean_top, mean_bottom), 1.0)
        tb_contrast = 1.0 - tb_ratio                                   # ∈ [0, 1]

        lr_contrast = float(np.abs(np.mean(left) - np.mean(right))) / 255.0
        juxtaposition = float(np.clip(0.5 * tb_contrast + 0.5 * lr_contrast, 0.0, 1.0))

        # ── 4. LIGHT & ATMOSPHERE ──────────────────────────────────────────
        hist     = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_sum = float(hist.sum())
        shadow    = float(np.sum(hist[:40]))  / hist_sum
        highlight = float(np.sum(hist[215:])) / hist_sum
        contrast  = float(np.sqrt(np.var(gray.astype(np.float32))))

        light_atmosphere = float(np.clip(
            0.4 * min(contrast / 70.0, 1.0)
            + 0.3 * (1.0 - abs(shadow - highlight))
            + 0.3 * min(shadow + highlight, 1.0),
            0.0, 1.0,
        ))

        # ── 5. AUTHENTICITY ────────────────────────────────────────────────
        # Penalise: over-saturated clipping  +  excessive horizontal symmetry
        hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_clip = float(np.sum(hsv[:, :, 1] > 220)) / (h * w)

        # TM_CCOEFF_NORMED ∈ [-1, 1]; near +1 → highly symmetric (likely posed/touristic)
        # Clip to [0, 1] so anti-correlated images aren't rewarded beyond max authenticity
        raw_sym   = float(cv2.matchTemplate(
            gray, gray[:, ::-1].copy(), cv2.TM_CCOEFF_NORMED
        )[0, 0])
        sym_score = float(np.clip(raw_sym, 0.0, 1.0))    # 1.0 = symmetric, 0.0 = random

        authenticity = float(np.clip(1.0 - 0.4 * sat_clip - 0.6 * sym_score, 0.0, 1.0))

        # ── 6. COMPOSITION & GEOMETRY ──────────────────────────────────────
        edges_comp = cv2.Canny(gray, 50, 150)
        lines      = cv2.HoughLinesP(
            edges_comp, 1, np.pi / 180,
            threshold=30, minLineLength=w // 5, maxLineGap=15,
        )
        line_density = min(len(lines) / 10.0, 1.0) if lines is not None else 0.0

        # Energy at rule-of-thirds intersection points (4 hotspots)
        hotspots = [
            (int(w * col), int(h * row))
            for row in (0.33, 0.66)
            for col in (0.33, 0.66)
        ]
        thirds_energy = float(np.mean([
            float(np.mean(
                gradient[
                    max(0, y - 15):min(h, y + 15),
                    max(0, x - 15):min(w, x + 15),
                ]
            )) / 100.0
            for x, y in hotspots
        ]))
        composition_geometry = float(np.clip(
            0.5 * line_density + 0.5 * min(thirds_energy, 1.0), 0.0, 1.0
        ))

        # ── 7. CLIP SEMANTIC ALIGNMENT ─────────────────────────────────────
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb_t = self.model.encode_image(img_tensor)
            img_emb_t = img_emb_t / img_emb_t.norm(dim=-1, keepdim=True)

        img_emb   = img_emb_t.cpu().numpy()            # (1, 512)
        txt_embs  = self._get_text_embs()              # (11, 512) — cached after first call
        sims      = (img_emb @ txt_embs.T).flatten()   # (11,)

        positive       = float(np.mean(sims[:_N_POSITIVE]))
        negative_mean  = float(np.mean(sims[_N_POSITIVE:]))
        semantic_align = float(np.clip((positive - negative_mean + 1.0) / 2.0, 0.0, 1.0))

        # ── 8. HUMAN AESTHETIC PERCEPTION (LAION v2.5+) ────────────────────
        human_score = _get_human_score(img_path)

        # Return raw signals only — score/grade/critique are applied per-profile
        # by _apply_profile() so the cache remains profile-agnostic.
        return {
            "decisive_moment":      round(decisive_moment,      3),
            "layering_depth":       round(layering_depth,       3),
            "juxtaposition":        round(juxtaposition,        3),
            "light_atmosphere":     round(light_atmosphere,     3),
            "authenticity":         round(authenticity,         3),
            "composition_geometry": round(composition_geometry, 3),
            "semantic_align":       round(semantic_align,       3),
            "human_perception":     round(human_score,          3),
            "subject_count":        subject_count,
            "embedding":            img_emb.flatten().tolist(),
            "dims":                 (w, h),
        }

    # ------------------------------------------------------------ critique

    def _generate_critique(self, scores: Dict[str, float], grade: str) -> str:
        """
        Generate a one-sentence critique tailored to the active profile's
        primary criterion, so feedback aligns with what that jury values most.
        """
        p = self.profile

        def _human_suffix() -> str:
            hp = scores.get("human_perception", 0)
            if hp > 0.65:
                return " Strong aesthetic harmony & professional visual flow."
            if hp < 0.40:
                return " Lacks curated polish; leans snapshot or over-processed."
            return ""

        if grade.startswith("Strong"):
            if p == "lspf":
                if scores["decisive_moment"] > 0.70:
                    critique = "Unmistakable peak moment — exactly the kind of original, unguarded gesture LSPF rewards."
                elif scores["authenticity"] > 0.65:
                    critique = "Raw street authenticity with strong timing. Hard to fake, harder to plan."
                else:
                    critique = "Cohesive street image with confident framing and emotional resonance."
            elif p == "spi":
                if scores["juxtaposition"] > 0.65:
                    critique = "Conceptual tension is sharp and unforced — the kind of irony SPI juries remember."
                elif scores["decisive_moment"] > 0.65:
                    critique = "Decisive moment reinforces the juxtaposition rather than standing alone. SPI-strong."
                else:
                    critique = "Cohesive street image with confident framing and emotional resonance."
            else:
                if scores["decisive_moment"] > 0.70:
                    critique = "Peak timing & gesture. Clear narrative tension without feeling staged."
                elif scores["layering_depth"] > 0.65:
                    critique = "Exceptional depth separation. Foreground/mid/background work together to tell a layered story."
                elif scores["juxtaposition"] > 0.60:
                    critique = "Strong visual contrast/irony. Juxtaposition adds narrative weight without being forced."
                else:
                    critique = "Cohesive street image with confident framing and emotional resonance."
            return critique + _human_suffix()

        if grade.startswith("Mid"):
            if p == "lspf":
                if scores["decisive_moment"] < 0.45:
                    critique = "Moment is present but not peak. LSPF wants the fraction of a second before or after this."
                elif scores["authenticity"] < 0.40:
                    critique = "Feels slightly constructed. LSPF juries are unforgiving of any hint of staging."
                else:
                    critique = "Solid execution, but lacks the decisive moment or layered tension that elevates it."
            elif p == "spi":
                if scores["juxtaposition"] < 0.45:
                    critique = "Lacks the conceptual contradiction SPI prizes. Look for visual irony within the same frame."
                elif scores["light_atmosphere"] < 0.40:
                    critique = "Flat light weakens the narrative tension. SPI rewards images where light amplifies the concept."
                else:
                    critique = "Solid execution, but lacks the decisive moment or layered tension that elevates it."
            else:
                if scores["authenticity"] < 0.40:
                    critique = "Feels slightly posed or processed. Street purity could be stronger."
                elif scores["light_atmosphere"] < 0.40:
                    critique = "Flat or harsh lighting. Consider waiting for directional light or shadow play."
                else:
                    critique = "Solid execution, but lacks the decisive moment or layered tension that elevates it."
            return critique + _human_suffix()

        # Weak — profile-specific lowest-scoring dimension
        if p == "lspf":
            critique = "Neither the moment nor the authenticity cuts through. Street photography lives in the unrepeatable."
        elif p == "spi":
            critique = "No clear conceptual tension or juxtaposition. SPI wants images that argue with themselves."
        else:
            critique = "Competent technically, but leans cliché, flat, or static. Revisit framing, timing, or light."

        if scores.get("human_perception", 0) > 0.65:
            return critique + " Strong aesthetic harmony & professional visual flow."
        elif scores.get("human_perception", 0) < 0.40:
            return critique + " Lacks curated polish; leans snapshot or over-processed."
        return critique

    # --------------------------------------------------------- sequencing

    def sequence_story(
        self,
        results: List[Tuple[str, Dict]],
        target: int = 5,
    ) -> Tuple[List[str], List[str]]:
        """
        Select and order `target` images into a Magnum-style contact-sheet sequence.

        Role slots (pacing order):
          0  Establishing  — wide, layered, landscape-oriented
          1  Subject/Moment — peak gesture, human presence
          2  Detail/Geometry — texture, lines, micro-contrast
          3  Contrast/Shift  — tonal break, flat light, mood shift
          4  Atmosphere      — no subjects, negative space, resolution

        Greedy lookahead: tries every image as the opener, keeps the sequence
        with the best combined flow + pacing + tension score.

        Returns (ordered_path_list, rationale_list).
        """
        valid = [(p, r) for p, r in results if "error" not in r]
        if len(valid) < target:
            paths = [p for p, _ in valid]
            return paths, RATIONALE[: len(paths)]

        ranked = sorted(valid, key=lambda x: x[1]["score"], reverse=True)[:20]
        embeddings = np.array([r[1]["embedding"] for r in ranked])   # (N, 512)

        # Role predicates — each takes an integer index into `ranked`
        def _establishing(i: int) -> bool:
            r = ranked[i][1]
            w, h = r["dims"]
            return r["layering_depth"] > 0.50 and w > h * 1.20

        def _subject_moment(i: int) -> bool:
            r = ranked[i][1]
            return r["decisive_moment"] > 0.45 and r["subject_count"] > 0

        def _detail_geometry(i: int) -> bool:
            r = ranked[i][1]
            return r["juxtaposition"] > 0.55 or r["composition_geometry"] > 0.60

        def _contrast_shift(i: int) -> bool:
            r = ranked[i][1]
            return r["light_atmosphere"] < 0.40 or r["authenticity"] < 0.50

        def _atmosphere(i: int) -> bool:
            r = ranked[i][1]
            return r["semantic_align"] > 0.50 and r["subject_count"] == 0

        role_predicates = [
            _establishing,
            _subject_moment,
            _detail_geometry,
            _contrast_shift,
            _atmosphere,
        ]

        best_seq:   List[int] = []
        best_score: float     = -1.0

        for start in range(len(ranked)):
            seq:  List[int] = [start]
            used: set       = {start}

            for slot in range(1, target):
                predicate  = role_predicates[slot]
                candidates = [i for i in range(len(ranked)) if i not in used and predicate(i)]
                if not candidates:                          # role fallback: any unused image
                    candidates = [i for i in range(len(ranked)) if i not in used]
                if not candidates:
                    break

                # Pick the candidate most visually similar to the previous frame
                prev_emb  = embeddings[seq[-1]].reshape(1, -1)
                cand_sims = _sk_cosine(prev_emb, embeddings[candidates]).flatten()
                seq.append(candidates[int(np.argmax(cand_sims))])
                used.add(seq[-1])

            if len(seq) < target:
                continue

            # Score this candidate sequence on three axes:
            # flow (visual continuity), pacing (score diversity), tension arc
            pair_sims = [
                float(_sk_cosine(
                    embeddings[seq[i]].reshape(1, -1),
                    embeddings[seq[i + 1]].reshape(1, -1),
                )[0, 0])
                for i in range(target - 1)
            ]
            seq_scores = [ranked[x][1]["score"] for x in seq]
            tension    = float(np.mean([
                ranked[x][1]["juxtaposition"] + ranked[x][1]["decisive_moment"]
                for x in seq
            ]))
            score_std  = float(np.std(seq_scores))
            flow       = float(np.mean(pair_sims))
            pacing     = 1.0 - score_std / max(score_std, 0.35)
            total      = 0.45 * flow + 0.35 * pacing + 0.20 * tension

            if total > best_score:
                best_score = total
                best_seq   = seq[:]

        if not best_seq:
            best_seq = list(range(min(target, len(ranked))))

        paths = [ranked[x][0] for x in best_seq]
        return paths, RATIONALE[: len(paths)]


# ---------------------------------------------------------------------------
# Module-level utility: rescore without re-running inference
# ---------------------------------------------------------------------------

def rescore(
    results:  List[Tuple[str, Dict]],
    preset:   str,
    custom_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, Dict]]:
    """
    Re-rank already-analyzed results under a different competition preset.
    No image I/O or model inference is performed — weights are applied to
    the criterion scores already present in each result dict.

    Parameters
    ----------
    results        : output of MagnumStreetAnalyzer.analyze_folder()
    preset         : target preset name, e.g. "LSPF (London Street)"
    custom_weights : optional fully custom weight dict (overrides preset)

    Returns
    -------
    New list of (path, result_dict) tuples scored under the target preset.
    """
    weights = _resolve_weights(preset, custom_weights)
    profile = _PRESET_TO_PROFILE.get(preset, "magnum")
    rescored: List[Tuple[str, Dict]] = []

    for path, r in results:
        if "error" in r:
            rescored.append((path, r))
            continue

        # Verify all criteria are present (guard against old cache formats)
        missing = [k for k in _CRITERIA_KEYS if k not in r]
        if missing:
            rescored.append((path, {"error": f"Missing criteria keys: {missing}"}))
            continue

        criterion_scores = {k: r[k] for k in _CRITERIA_KEYS}
        semantic_align   = r.get("semantic_align", 0.5)
        human_score      = r.get("human_perception", 0.5)
        weighted_sum     = sum(weights.get(k, 0) * v for k, v in criterion_scores.items())
        final            = float(np.clip(weighted_sum + 0.2 * human_score, 0.0, 1.0))

        grade = (
            "Strong \u2705" if final > 0.68
            else "Mid \u26a0\ufe0f" if final > 0.48
            else "Weak \u274c"
        )

        # Minimal shell just for _generate_critique — no CLIP load occurs
        _shell         = object.__new__(MagnumStreetAnalyzer)
        _shell.profile = profile
        _shell.weights = weights

        scores_for_critique = {**criterion_scores, "human_perception": human_score}
        rescored.append((path, {
            **{k: round(v, 3) for k, v in criterion_scores.items()},
            "semantic_align":   round(semantic_align, 3),
            "human_perception": round(human_score,    3),
            "score":            round(final, 3),
            "grade":            grade,
            "critique":         _shell._generate_critique(scores_for_critique, grade),
            "profile":          profile,
            "preset":           preset,
            "subject_count":    r.get("subject_count", 0),
            "embedding":        r.get("embedding", []),
            "dims":             r.get("dims", (0, 0)),
        }))

    return rescored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="MagnumStreetAnalyzer — competition-grade street photo grader"
    )
    parser.add_argument("folder", nargs="?", default=".",
                        help="Folder of images to analyze (default: .)")
    parser.add_argument(
        "--preset", "-p", default="Magnum Editor",
        choices=[k for k in COMPETITION_PRESETS if k != "Custom"],
        help="Competition preset (default: 'Magnum Editor')",
    )
    parser.add_argument(
        "--compare", "-c", action="store_true",
        help="Show scores under all three presets side-by-side",
    )
    args = parser.parse_args()

    analyzer = MagnumStreetAnalyzer(preset=args.preset)
    print(f"\nPreset: {args.preset}\n")

    results = analyzer.analyze_folder(args.folder)
    if not results:
        print("No images found.")
        sys.exit(0)

    print(f"Analyzed {len(results)} image(s):\n")

    _compare_presets = [k for k in COMPETITION_PRESETS if k != "Custom"]

    if args.compare:
        # Rescore from cached signals — no re-inference
        all_scored = {
            p: (results if p == args.preset else rescore(results, p))
            for p in _compare_presets
        }
        col_w = 20
        header = f"  {'Filename':<42}" + "".join(f"  {p:>{col_w}}" for p in _compare_presets)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for path, _ in results:
            fname = Path(path).name[:40]
            row = []
            for p in _compare_presets:
                match = next((r for fp, r in all_scored[p] if fp == path), {})
                sym   = "\u2705" if "Strong" in match.get("grade","") else ("\u26a0\ufe0f" if "Mid" in match.get("grade","") else "\u274c")
                row.append(f"{match.get('score',0):.3f} {sym}")
            print(f"  {fname:<42}" + "".join(f"  {s:>{col_w}}" for s in row))
    else:
        for path, r in results:
            if "error" in r:
                print(f"  ERROR  {Path(path).name}: {r['error']}")
                continue
            print(
                f"  {r['grade']}  {Path(path).name:<44}"
                f"  score={r['score']:.3f}"
                f"  dm={r['decisive_moment']:.2f}"
                f"  jux={r['juxtaposition']:.2f}"
                f"  auth={r['authenticity']:.2f}"
                f"  layer={r['layering_depth']:.2f}"
                f"  light={r['light_atmosphere']:.2f}"
                f"  comp={r['composition_geometry']:.2f}"
                f"  clip={r['semantic_align']:.2f}"
            )
            print(f"         {r['critique']}\n")

    print("\nBuilding story sequence ...\n")
    paths, rationale = analyzer.sequence_story(results)
    for label, p in zip(rationale, paths):
        print(f"  {label}")
        print(f"     → {Path(p).name}\n")
