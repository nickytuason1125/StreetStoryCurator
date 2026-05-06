"""
Step 2 — Q-Align aesthetic scorer (4-bit NF4 quantized).

Q-Align (q-future/one-align) is a vision-language model fine-tuned to output
quality-level tokens: bad / poor / fair / good / excellent.
The aesthetic score is a soft weighted sum over those five token logits.

Sequential VRAM protocol
────────────────────────
1.  SigLIPEncoder.encode_images()   → all embeddings computed
2.  SigLIPEncoder.unload()          → GPU cleared
3.  QAlignGrader()                  → 4-bit model loads into freed VRAM
4.  QAlignGrader.score()            → aesthetic scores for all photos
5.  QAlignGrader.unload()           → GPU cleared for next step

Falls back to a lightweight NIMA-based proxy when the model is unavailable
so the pipeline degrades gracefully.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Optional

MODEL_ID = "q-future/one-align"

# Ordered quality levels; weights map to 0.0 → 1.0 linearly.
_QUALITY_WORDS = ["bad", "poor", "fair", "good", "excellent"]
_QUALITY_WEIGHTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)

# Aesthetic scoring prompt (used with the model's chat template).
_PROMPT = (
    "<|im_start|>system\nYou are an expert photo editor.<|im_end|>\n"
    "<|im_start|>user\n<image>\nRate the overall aesthetic quality "
    "of this photo.<|im_end|>\n<|im_start|>assistant\nThe quality of "
    "the photo is"
)


class QAlignGrader:

    def __init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tok = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=False
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._model.eval()

        # Pre-compute quality-word token ids once.
        self._quality_ids = [
            self._tok(w, add_special_tokens=False)["input_ids"][0]
            for w in _QUALITY_WORDS
        ]

    # ── scoring ───────────────────────────────────────────────────────────────

    def score(self, paths: list[str], progress=None) -> list[float]:
        """
        Return aesthetic scores in [0, 1] for each image path.
        1.0 = excellent, 0.0 = bad.
        """
        from PIL import Image as _PIL
        scores: list[float] = []

        for i, path in enumerate(paths):
            try:
                img  = _PIL.open(path).convert("RGB")
                inp  = self._tok(_PROMPT, return_tensors="pt").to(self._model.device)
                with torch.no_grad():
                    out    = self._model(**inp, images=[img])
                    logits = out.logits[0, -1, self._quality_ids]   # (5,)
                    probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                scores.append(float(np.dot(probs, _QUALITY_WEIGHTS)))
            except Exception:
                scores.append(0.5)   # neutral on failure

            if progress:
                progress(
                    0.5 + (i + 1) / len(paths) * 0.35,
                    desc=f"Q-Align: {i+1}/{len(paths)}",
                )

        return scores

    # ── VRAM release ──────────────────────────────────────────────────────────

    def unload(self) -> None:
        del self._model
        torch.cuda.empty_cache()


# ── Fallback: NIMA-based proxy ────────────────────────────────────────────────

class _NIMAFallback:
    """
    Lightweight aesthetic proxy using the existing NIMA ONNX model.
    Used when Q-Align / bitsandbytes are not installed.
    """

    def __init__(self) -> None:
        import onnxruntime as ort
        from pathlib import Path
        model_path = Path("models/nima.onnx")
        if not model_path.exists():
            raise FileNotFoundError("models/nima.onnx not found")
        self._sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._in  = self._sess.get_inputs()[0].name

    def score(self, paths: list[str], progress=None) -> list[float]:
        import cv2
        scores: list[float] = []
        for i, path in enumerate(paths):
            try:
                img = cv2.imread(path)
                img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
                img = img[..., ::-1].transpose(2, 0, 1)[np.newaxis]
                out = self._sess.run(None, {self._in: img})[0]
                # NIMA outputs a 10-class distribution; expected score mapped to [0,1]
                weights = np.arange(1, 11, dtype=np.float32) / 10.0
                scores.append(float(np.dot(out.flatten(), weights)))
            except Exception:
                scores.append(0.5)
            if progress:
                progress(
                    0.5 + (i + 1) / len(paths) * 0.35,
                    desc=f"Scoring: {i+1}/{len(paths)}",
                )
        return scores

    def unload(self) -> None:
        pass   # ONNX session holds no GPU memory


# ── Fallback: V1 LightweightStreetScorer ─────────────────────────────────────

class _V1LightweightFallback:
    """
    Final-tier fallback using the existing V1 OpenCV + ONNX scorer.
    Produces real differentiated scores without any GPU or HuggingFace models.
    Used when both Q-Align and NIMA are unavailable.
    """

    def __init__(self) -> None:
        from lightweight_analyzer import LightweightStreetScorer
        self._scorer = LightweightStreetScorer()

    @staticmethod
    def _remap(raw: float) -> float:
        """Map V1 best-preset score (threshold >0.59) to V2 bucket ranges (≥0.70/0.40/0.00)."""
        if raw > 0.59:
            t = (raw - 0.59) / 0.41
            return float(np.clip(0.70 + t * 0.30, 0.70, 1.0))
        elif raw >= 0.40:
            t = (raw - 0.40) / 0.19
            return float(np.clip(0.40 + t * 0.29, 0.40, 0.69))
        else:
            return float(np.clip(raw, 0.0, 0.39))

    def _analyze_one(self, path: str) -> dict:
        result = self._scorer._analyze(path, preset="Classic Street")
        bd = result.get("breakdown", {})

        # Use the actual final score from _analyze(), not Best_Score.
        # Best_Score omits the 0.90 deflation applied inside _analyze(), making
        # it 10-15% higher than the real score and causing Strong inflation.
        actual_score = float(result.get("score", 0.5))
        score = self._remap(actual_score)
        pre_gate = score

        # ── Culling gates ─────────────────────────────────────────────────────
        tech_raw         = float(bd.get("_tech_raw",         bd.get("Technical", 0.5)))
        intentional      = bool( bd.get("_intentional_soft", False))
        blown            = float(bd.get("_blown",            0.0))
        best_sharp       = float(bd.get("_best_sharp",       999.0))
        blur_cv          = float(bd.get("_blur_cv",          1.0))
        high_key         = bool( bd.get("_high_key",         False))
        backlit          = bool( bd.get("_backlit",          False))
        directional_blur = bool( bd.get("_directional_blur", False))
        chiaroscuro      = bool( bd.get("_chiaroscuro",      False))
        comp             = float(bd.get("Composition",       0.5))
        niche            = str(  bd.get("Detected_Niche",    ""))

        gate_applied = "none"

        # ── Context classification ────────────────────────────────────────────
        fine_art_niche = any(x in niche for x in
                             ("Fine Art", "High Key", "Macro", "Motion"))

        # Selective focus: subject sharp, background soft (bokeh/telephoto)
        selective_focus = blur_cv > 0.45 and best_sharp > 80

        # Artistic motion: directional blur + compositional intent = panning/long exp
        artistic_motion = directional_blur and comp >= 0.52

        # Artistic uniform soft: dreamy look, clean sensor (no grain), high comp
        artistic_soft = (
            not intentional and blur_cv < 0.35
            and best_sharp > 30 and comp >= 0.58
        )

        # Fine art broad exempt: fine art niche OR chiaroscuro — these styles
        # intentionally violate "normal" technical rules
        fine_art_exempt = fine_art_niche or chiaroscuro

        blur_exempt = intentional or selective_focus or artistic_motion or fine_art_exempt

        # ── Blur / sharpness gates (skip for intentional / fine art) ─────────
        if best_sharp < 55 and not blur_exempt and not artistic_soft:
            score = min(score, 0.32)
            gate_applied = f"blur_hard(sharp={best_sharp:.0f})"

        elif best_sharp < 110 and tech_raw < 0.32 and not blur_exempt and not artistic_soft:
            factor = 0.80 + 0.20 * (tech_raw / 0.32)
            score  = score * factor
            gate_applied = f"blur_soft(sharp={best_sharp:.0f},f={factor:.2f})"

        elif artistic_soft and best_sharp < 90 and not intentional:
            score = score * 0.93     # ≤7% nudge only
            gate_applied = "artistic_soft_nudge"

        # ── Technical quality gate (exposure + noise + sharpness combined) ───
        # Skip for fine art / tonal extreme styles that intentionally break
        # conventional exposure rules.
        if not fine_art_exempt:
            if tech_raw < 0.22:
                score = min(score, 0.34)
                gate_applied = (gate_applied if gate_applied != "none" else "") + "+tech_hard"
            elif tech_raw < 0.38 and not selective_focus and not artistic_motion:
                factor = 0.82 + 0.18 * (tech_raw - 0.22) / 0.16
                score  = score * factor
                gate_applied = (gate_applied if gate_applied != "none" else "none") \
                               + f"+tech_soft(f={factor:.2f})"

        # ── Blown highlights gate ─────────────────────────────────────────────
        # Skip for high-key, backlit/silhouette, chiaroscuro, and fine art niches —
        # all of which legitimately have >28% bright pixels.
        if blown > 0.28 and not (high_key or backlit or chiaroscuro or fine_art_niche):
            score = min(score, 0.58)
            gate_applied += f"+blown({blown:.2f})"

        score = float(np.clip(score, 0.0, 1.0))

        import os as _os
        ctx = ("fine_art"       if fine_art_exempt   else
               "artistic_motion"if artistic_motion   else
               "selective_focus"if selective_focus   else
               "intentional"    if intentional       else
               "artistic_soft"  if artistic_soft     else
               "high_key"       if high_key          else
               "backlit"        if backlit           else "standard")
        print(
            f"[grade] {_os.path.basename(path)}: "
            f"raw={actual_score:.3f}  remapped={pre_gate:.3f}  final={score:.3f}  "
            f"niche={niche}  ctx={ctx}  "
            f"sharp={best_sharp:.0f}  blur_cv={blur_cv:.2f}  blown={blown:.2f}  "
            f"gate={gate_applied}"
        )

        # Normalise dimension labels regardless of which preset was applied.
        def _pick(*keys: str) -> float:
            for k in keys:
                v = bd.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return round(float(v), 3)
            return 0.0

        breakdown = {
            "Technical":     _pick("Technical", "News Sharpness", "Cleanliness",
                                   "Execution", "Sharpness & Detail", "Exposure"),
            "Composition":   _pick("Composition", "Framing", "Context",
                                   "Geometry & Balance", "Negative Space", "Layered Depth"),
            "Lighting":      _pick("Lighting", "Atmosphere", "Natural Light",
                                   "Mood & Tone", "Available Light", "Natural Light Quality"),
            "Authenticity":  _pick("Decisive Moment", "Authenticity", "Narrative",
                                   "Immediacy", "Narrative Suggestion", "Conceptual Weight"),
            "Human/Culture": _pick("Human/Culture", "Subject Isolation", "Sense of Place",
                                   "Presence", "Character Presence", "Human Impact"),
        }

        return {
            "score":     score,
            "breakdown": breakdown,
            "critique":  result.get("critique", ""),
        }

    def score(self, paths: list[str], progress=None) -> list[float]:
        scores: list[float] = []
        n = len(paths)
        for i, path in enumerate(paths):
            try:
                scores.append(self._analyze_one(path)["score"])
            except Exception:
                scores.append(0.5)
            if progress:
                progress(0.5 + (i + 1) / n * 0.35, desc=f"Scoring {i + 1}/{n}")
        return scores

    def score_detailed(self, paths: list[str], progress=None) -> list[dict]:
        """Return list of {score, breakdown, critique} dicts — one per image."""
        results: list[dict] = []
        n = len(paths)
        for i, path in enumerate(paths):
            try:
                results.append(self._analyze_one(path))
            except Exception:
                results.append({"score": 0.5, "breakdown": {}, "critique": ""})
            if progress:
                progress(0.5 + (i + 1) / n * 0.35, desc=f"Scoring {i + 1}/{n}")
        return results

    def unload(self) -> None:
        pass


def get_grader() -> "QAlignGrader | _NIMAFallback | _V1LightweightFallback":
    """
    Grader selection with three tiers:
      1. QAlignGrader          — 4-bit NF4 VLM, GPU recommended
      2. _NIMAFallback         — NIMA ONNX, CPU-only
      3. _V1LightweightFallback — V1 OpenCV+ONNX scorer, always available
    """
    try:
        import bitsandbytes   # noqa: F401
        import transformers   # noqa: F401
        return QAlignGrader()
    except Exception:
        pass

    try:
        return _NIMAFallback()
    except Exception:
        pass

    return _V1LightweightFallback()
