"""
Qwen2.5-VL-3B-Instruct Vision-Language Reasoning Engine

Speed target: 100 photos in ~2 minutes.

Architecture (hybrid):
    CLIP score  — computed instantly from SigLIP-2 embeddings already in hand.
    VLM reason  — Qwen2.5-VL-3B-Instruct looks at each image and writes the
                  critique; the score comes from CLIP so generation can be short
                  (~40 tokens vs 200), cutting per-photo time to ~0.5–1.5 s.

Typical throughput on 6 GB GPU (RTX 3060):
    ~0.8–2 s/photo  →  80–200 s for 100 photos

VRAM: ~2.2 GB INT4 — loads after SigLIP-2 unloads, frees before PersonalHead.
"""
from __future__ import annotations

import gc
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

MODEL_ID        = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_CACHE_DIR = Path("models/qwen_vlm")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Short prompt — fewer output tokens = faster generation
# Scores come from CLIP; we only need a brief critique from the VLM.
_REASON_PROMPT = """\
Look at this photograph — it may be street, architectural, liminal, urban geometry, \
or environmental — and write ONE sentence naming its single strongest visual element \
and its single biggest weakness. Be specific and genre-aware \
(e.g. "The symmetrical geometry and fog create an exceptional liminal tension, \
but the horizon is slightly tilted"; or "The harsh noon light kills shadow detail, \
but the layered depth with three planes is exceptional").\
"""

# Max image dimension fed to the VLM.  480 px gives ~400-700 vision tokens —
# enough to read composition / moment / light without the cost of larger sizes.
_MAX_VLM_PX = 480
_MAX_NEW_TOKENS = 60   # one sentence is plenty


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class VLMResult:
    path:      str
    score:     float                          # comes from CLIP caller
    aspects:   Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    critique:  str = ""


# ── Reasoning formatter ────────────────────────────────────────────────────────

_ASPECT_DETAIL = {
    "Technical":   "sharpness, exposure, and noise",
    "Composition": "framing, geometry, and visual flow",
    "Lighting":    "directional light and tonal contrast",
    "Moment":      "decisive moment and story tension",
    "Human":       "emotional presence and candid energy",
}


_ASPECT_SHORT = {
    "Technical":   "Technical",
    "Composition": "Composition",
    "Lighting":    "Lighting",
    "Moment":      "Moment",
    "Human":       "Human",
}


def build_vlm_reasoning(
    score: float,
    aspects: Dict[str, float],
    critique: str,
) -> str:
    tier = "Strong" if score >= 0.60 else ("Mid" if score >= 0.41 else "Weak")
    pct  = int(round(score * 100))
    lines = [f"{tier}  {pct}%"]
    if critique:
        lines.append(critique)
    if aspects:
        lines.append("")
        for k, v in sorted(aspects.items(), key=lambda x: -x[1]):
            label = _ASPECT_SHORT.get(k, k)
            bar   = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"{label:<12} {bar}  {int(v*100)}%")
        top    = _ASPECT_SHORT.get(max(aspects, key=aspects.get), max(aspects, key=aspects.get))
        bottom = _ASPECT_SHORT.get(min(aspects, key=aspects.get), min(aspects, key=aspects.get))
        lines.append(f"\nBest: {top}   ·   Weakest: {bottom}")
    return "\n".join(lines)


# ── Grader class ───────────────────────────────────────────────────────────────

class QwenVLMGrader:
    """
    Loads Qwen2.5-VL-3B-Instruct as INT4 (~2.2 GB VRAM).

    grade_images() accepts pre-computed CLIP scores (from SigLIP-2) and uses
    the VLM only to generate per-photo reasoning text — keeping generation
    length short for maximum throughput.
    """

    _INT4_VRAM_GB = 2.2

    def __init__(self, device: str = "auto", progress=None):
        _p = progress or (lambda f, d: None)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            free_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved(0)
            ) / 1e9
            if free_gb < self._INT4_VRAM_GB:
                print(
                    f"[qwen_vlm] Only {free_gb:.1f} GB free "
                    f"(need ~{self._INT4_VRAM_GB:.1f}) — falling back to CPU"
                )
                self.device = "cpu"

        _p(0.52, "Loading Qwen2.5-VL-3B (first run downloads ~6 GB)…")
        print(f"[qwen_vlm] Loading {MODEL_ID} on {self.device}…")

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLCls
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration as _VLCls
            except ImportError:
                raise ImportError(
                    "transformers >= 4.45 with Qwen2-VL support required.\n"
                    "Run:  pip install --upgrade transformers"
                )

        from transformers import AutoProcessor

        base_kw = dict(
            pretrained_model_name_or_path=MODEL_ID,
            cache_dir=str(MODEL_CACHE_DIR),
            trust_remote_code=True,
        )

        if self.device == "cuda":
            _p(0.53, "Quantising Qwen2.5-VL to INT4…")
            self._model = self._load_int4(_VLCls, base_kw)
        else:
            _p(0.53, "Loading on CPU (slow, no GPU)…")
            self._model = _VLCls.from_pretrained(
                **base_kw, torch_dtype=torch.float32, device_map="cpu"
            )

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir=str(MODEL_CACHE_DIR), trust_remote_code=True
        )

        _p(0.56, "Qwen2.5-VL ready — generating reasoning…")
        print("[qwen_vlm] Model ready.")

    # ------------------------------------------------------------------
    def _load_int4(self, cls, base_kw: dict):
        from transformers import BitsAndBytesConfig

        extra: dict = {}
        try:
            # Flash Attention 2 cuts prefill cost ~30 % on Ampere+ GPUs
            extra["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

        try:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = cls.from_pretrained(
                **base_kw, quantization_config=bnb, device_map="auto", **extra
            )
            print("[qwen_vlm] INT4 via BitsAndBytes + flash_attn2 (~2.2 GB VRAM)")
            return model
        except Exception as e_bnb:
            print(f"[qwen_vlm] INT4 failed ({e_bnb}) — trying FP16 (no flash_attn2)")
            return cls.from_pretrained(
                **base_kw, torch_dtype=torch.float16, device_map="auto"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _resize(path: str):
        """Load and resize image to _MAX_VLM_PX on the long edge."""
        from PIL import Image as _PIL
        img = _PIL.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_VLM_PX:
            scale = _MAX_VLM_PX / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)), _PIL.Resampling.LANCZOS
            )
        return img

    # ------------------------------------------------------------------
    def _reason_one(self, path: str) -> str:
        """Ask the VLM for a one-sentence critique. Returns '' on failure."""
        try:
            img = self._resize(path)
        except Exception as e:
            print(f"[qwen_vlm] Image load failed {path}: {e}")
            return ""

        messages = [{
            "role": "user",
            "content": [
                {"type": "image",  "image": img},
                {"type": "text",   "text":  _REASON_PROMPT},
            ],
        }]

        try:
            text   = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[img], return_tensors="pt", padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    do_sample=False,
                )

            n_in   = inputs["input_ids"].shape[1]
            result = self._processor.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()
            print(f"[qwen_vlm] {Path(path).name}: {result[:80]!r}")
            return result

        except Exception as e:
            print(f"[qwen_vlm] Inference failed {path}: {e}")
            return ""

    # ------------------------------------------------------------------
    def grade_images(
        self,
        paths: List[str],
        clip_scores: np.ndarray,          # (N,) float32 from SigLIP-2 CLIP
        clip_aspects: Optional[List[Dict[str, float]]] = None,
        progress=None,
    ) -> List[VLMResult]:
        """
        Generate per-photo reasoning.  Scores come from CLIP (already computed).

        Args:
            paths:        image file paths
            clip_scores:  calibrated CLIP scores (0-1) from SpecVLMPipeline
            clip_aspects: per-photo aspect dicts from SpecVLMPipeline (optional)
            progress:     (frac, msg) callback

        Returns:
            List[VLMResult] with score=clip_score and VLM critique in reasoning.
        """
        _p = progress or (lambda f, d: None)
        n  = len(paths)
        if clip_aspects is None:
            clip_aspects = [{}] * n

        results: List[VLMResult] = []
        t0 = time.time()

        # Pre-load next image in background while GPU runs on current one
        with ThreadPoolExecutor(max_workers=2) as pool:
            next_fut = pool.submit(self._resize, paths[0]) if n > 0 else None

            for i, path in enumerate(paths):
                # Retrieve pre-loaded image
                try:
                    img = next_fut.result() if next_fut else self._resize(path)
                except Exception as e:
                    print(f"[qwen_vlm] Preload failed {path}: {e}")
                    img = None

                # Kick off loading for next image
                next_fut = (
                    pool.submit(self._resize, paths[i + 1]) if i + 1 < n else None
                )

                # VLM critique
                if img is not None:
                    critique = self._run_one(img, path)
                else:
                    critique = ""

                score   = float(clip_scores[i])
                aspects = clip_aspects[i]
                results.append(VLMResult(
                    path      = path,
                    score     = score,
                    aspects   = aspects,
                    reasoning = build_vlm_reasoning(score, aspects, critique),
                    critique  = critique,
                ))

                done    = i + 1
                elapsed = time.time() - t0
                if done < n:
                    eta_s   = int(elapsed / done * (n - done))
                    eta_str = f" — ~{eta_s // 60}m{eta_s % 60:02d}s left"
                else:
                    eta_str = ""
                frac = 0.56 + (done / n) * 0.29   # spans 0.56 → 0.85
                _p(frac, f"VLM reasoning: {done}/{n}{eta_str}")

        return results

    def _run_one(self, img, path: str) -> str:
        """Shared inference logic for a pre-loaded PIL image."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  _REASON_PROMPT},
            ],
        }]
        try:
            text   = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[img], return_tensors="pt", padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False
                )
            n_in = inputs["input_ids"].shape[1]
            result = self._processor.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()
            print(f"[qwen_vlm] {Path(path).name}: {result[:80]!r}")
            return result
        except Exception as e:
            print(f"[qwen_vlm] Inference failed {path}: {e}")
            return ""

    # ------------------------------------------------------------------
    def unload(self) -> None:
        for attr in ("_model", "_processor"):
            obj = getattr(self, attr, None)
            if obj is not None:
                if hasattr(obj, "cpu"):
                    try:
                        obj.cpu()
                    except Exception:
                        pass
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("[qwen_vlm] Unloaded.")
