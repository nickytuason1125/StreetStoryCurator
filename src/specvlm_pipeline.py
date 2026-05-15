"""
SpecVLM Pipeline - Speculative Decoding for High-Speed Grading

Architecture:
    Bulk Encoder (SigLIP-2 ViT-g/14) → Embeddings
        ↓
    Priority-Gate Controller
        ├─ Draft Model (DeepSeek-R1-Distill-Qwen-1.5B INT4)
        │   └─ → Confidence > 0.88? → SKIP 7B
        └─ Verify Model (DeepSeek-R1-Distill-Qwen-7B INT4)
            └─ → Only triggered when confidence ≤ 0.88
        ↓
    Reasoning Log Aggregation
        ↓
    Score Calculation (weighted ensemble)

VRAM Protocol:
    1. SigLIP2Encoder.encode_images() → all embeddings computed
    2. VRAMManager.clear_between_phases() → GPU cleared
    3. SpecVLM() → models load into freed VRAM
    4. SpecVLM.grade() → inference for all photos
    5. SpecVLM.unload() → GPU cleared for next step
"""

from __future__ import annotations

import math
import os
import json
import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set

import torch
import numpy as np

# Model paths
MODEL_DIR = Path("models/specvlm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Priority gate threshold
DRAFT_CONFIDENCE_THRESHOLD = 0.85

# Batch inference settings
BATCH_SIZE  = 8    # images per single VLM forward pass
MAX_WAIT_MS = 500  # partial-batch flush threshold (ms)

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
    "sharp well-exposed street photo with excellent light and shadow",
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
]

_NEG_PROMPTS: List[str] = [
    "a blurry out of focus snapshot with poor composition",
    "overexposed or underexposed photo with bad framing and no clear subject",
    "cluttered confusing image with no focal point and distracting background",
    "low quality grainy photo with motion blur and flat uninteresting scene",
    "boring snapshot with no visual interest, no intention and no reason to look twice",
]

# Aspect-specific prompts for the verify pass (richer per-dimension breakdown)
_ASPECT_PROMPTS: Dict[str, List[str]] = {
    "Technical":    ["sharp crisp focus excellent technical quality",
                     "blurry soft out of focus poor sharpness"],
    "Composition":  ["rule of thirds leading lines balanced visual hierarchy bold geometry",
                     "cluttered random frame no composition no focal point"],
    "Lighting":     ["light that creates mood, atmosphere, or depth — directional, diffuse, dramatic, or cinematic",
                     "harsh uncontrolled light with blown highlights muddy shadows flat tonal range no depth"],
    "Narrative":    ["decisive moment authentic documentary story visual tension",
                     "no moment no story no visual intention random boring static"],
    # Human/Culture: covers human-centred shots AND strong sense-of-place shots with no people.
    # Negative avoids penalising intentionally empty architectural / liminal frames.
    "Human/Culture":["compelling subject — candid human emotion and street energy, OR powerful sense of place and atmosphere",
                     "no subject no visual intent completely empty purposeless scene with nothing to look at"],
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


def _detect_aspect_ratio(path: str) -> str:
    """Return nearest common ratio string from image dimensions."""
    try:
        from PIL import Image as _PIL
        w, h = _PIL.open(path).size
        ratio = w / h if h else 1.0
        # Map to nearest standard ratio
        _RATIOS = [(1/1,"1:1"),(4/3,"4:3"),(3/2,"3:2"),(16/9,"16:9"),(2/3,"2:3"),(3/4,"3:4"),(9/16,"9:16")]
        return min(_RATIOS, key=lambda x: abs(x[0] - ratio))[1]
    except Exception:
        return "unknown"


def _detect_genre(aspect_scores: Dict[str, float]) -> str:
    """
    Classify the photo genre from CLIP aspect scores.

    Priority order: Liminal → Architectural → Street.
    Both Liminal and Architectural share low Human/Culture; the differentiator
    is whether Lighting/Narrative dominates (liminal) or Composition does
    (architectural).

    Liminal: empty/atmospheric — fog, corridors, underpasses, reflections.
        Human/Culture < 0.35 AND (Lighting > 0.50 OR Narrative > 0.48)
    Architectural: structural/geometric — buildings, symmetry, urban geometry.
        Human/Culture < 0.38 AND Composition > 0.52
    Street: everything else — candid, human-centric, decisive moment.
    """
    human   = aspect_scores.get("Human/Culture", 0.5)
    comp    = aspect_scores.get("Composition",   0.5)
    light   = aspect_scores.get("Lighting",      0.5)
    narr    = aspect_scores.get("Narrative",      0.5)

    # Architectural check first: strong geometry overrides atmospheric mood
    if human < 0.38 and comp > 0.52:
        return "Architectural"
    # Liminal: atmospheric/empty with no strong geometric intent
    if human < 0.35 and (light > 0.50 or narr > 0.48):
        return "Liminal"
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


# ── Priority Gate Controller ───────────────────────────────────────────────────


class PriorityGate:
    """
    Controls whether to skip the 7B verifier based on draft confidence.
    
    Logic:
        - If draft_confidence > 0.88: skip verification, use draft score
        - If draft_confidence <= 0.88: trigger 7B verifier for correction
    """
    
    def __init__(self, threshold: float = DRAFT_CONFIDENCE_THRESHOLD):
        self.threshold = threshold
    
    def should_skip(self, confidence: float) -> bool:
        """Return True if we can skip the 7B verifier."""
        return confidence > self.threshold
    
    def trigger_verification(self, confidence: float) -> bool:
        """Return True if verification should be triggered."""
        return confidence <= self.threshold


# ── VRAM Manager ───────────────────────────────────────────────────────────────


class VRAMManager:
    """
    Manages VRAM between pipeline phases to prevent overflow on laptop GPUs.
    """
    
    @staticmethod
    def clear_between_phases() -> None:
        """
        Force cleanup between Bulk Encoder and Reasoning phases.
        Essential for 4-6GB VRAM laptops.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_vram_usage() -> Dict[str, float]:
        """Return current VRAM usage statistics."""
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
            }
        return {
            "total": 0.0,
            "allocated": 0.0,
            "reserved": 0.0,
        }
    
    @staticmethod
    def ensure_sufficient_vram(min_required: int = 3_000_000_000) -> bool:
        """Check if at least `min_required` bytes are available."""
        if not torch.cuda.is_available():
            return True
        
        usage = VRAMManager.get_vram_usage()
        available = usage["total"] - usage["allocated"]
        return available >= min_required


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


# ── SpecVLM Pipeline ───────────────────────────────────────────────────────────


class SpecVLM:
    """
    Speculative Decoding VLM for high-speed photo grading.
    
    Uses a two-stage approach:
    1. Draft Model (1.5B) - Fast, lower accuracy
    2. Verify Model (7B) - Slow, high accuracy (only when needed)
    
    Memory-efficient design for laptop GPUs (4-6GB VRAM).
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.priority_gate = PriorityGate()
        self._draft_model = None
        self._verify_model = None
        self._loaded = False
        self._verify_disabled = False  # set True if 7B OOM on load

    @property
    def draft_model(self):
        if self._draft_model is None:
            self._load_draft_model()
        return self._draft_model

    @property
    def verify_model(self):
        if self._verify_model is None and not self._verify_disabled:
            self._load_verify_model()
        return self._verify_model
    
    def _load_draft_model(self):
        """Load DeepSeek-R1-Distill-Qwen-1.5B — INT4 if bitsandbytes available, FP16 fallback."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from deepseek_model import (
            DRAFT_MODEL_ID, MODEL_CACHE_DIR,
            _model_weights_exist, _download_model_if_needed,
        )

        local_dir = MODEL_CACHE_DIR / "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"
        if not _model_weights_exist(local_dir):
            _download_model_if_needed(DRAFT_MODEL_ID, local_dir)
        load_from = str(local_dir) if _model_weights_exist(local_dir) else DRAFT_MODEL_ID

        device_map = "auto" if torch.cuda.is_available() else "cpu"
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._draft_model = AutoModelForCausalLM.from_pretrained(
                load_from, quantization_config=quant_config,
                device_map=device_map, torch_dtype=torch.float16,
                trust_remote_code=True, **_FLASH_ATTN_KWARGS,
            )
            print("[specvlm] Draft model loaded: INT4 via bitsandbytes")
        except Exception as e_bnb:
            print(f"[specvlm] INT4 unavailable ({e_bnb}), loading draft in FP16…")
            self._draft_model = AutoModelForCausalLM.from_pretrained(
                load_from, device_map=device_map,
                torch_dtype=torch.float16, trust_remote_code=True,
                **_FLASH_ATTN_KWARGS,
            )
            print("[specvlm] Draft model loaded: FP16")
        self._draft_model.eval()
        self._draft_tokenizer = AutoTokenizer.from_pretrained(
            load_from, trust_remote_code=True, use_fast=True,
        )

        # Load DPO LoRA adapter if it exists (read-only, non-trainable)
        _adapter_dir = Path("models/dpo_adapter")
        if _adapter_dir.exists() and (_adapter_dir / "adapter_config.json").exists():
            try:
                from peft import PeftModel
                self._draft_model = PeftModel.from_pretrained(
                    self._draft_model, str(_adapter_dir), is_trainable=False
                )
                self._draft_model.eval()
                print("[specvlm] DPO LoRA adapter loaded for inference.")
            except Exception as _e:
                print(f"[specvlm] DPO adapter skipped ({_e}) — using base model.")

    def _load_verify_model(self):
        """Load DeepSeek-R1-Distill-Qwen-7B — INT4 if bitsandbytes available, FP16 fallback.

        If loading fails with CUDA OOM, sets _verify_disabled=True so the pipeline
        continues in draft-only mode rather than crashing.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from deepseek_model import (
            VERIFY_MODEL_ID, MODEL_CACHE_DIR,
            _model_weights_exist, _download_model_if_needed,
        )

        local_dir = MODEL_CACHE_DIR / "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"
        if not _model_weights_exist(local_dir):
            _download_model_if_needed(VERIFY_MODEL_ID, local_dir)
        load_from = str(local_dir) if _model_weights_exist(local_dir) else VERIFY_MODEL_ID

        device_map = "auto" if torch.cuda.is_available() else "cpu"

        def _is_oom(exc: Exception) -> bool:
            return "out of memory" in str(exc).lower() or isinstance(
                exc, torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError") else type(None)
            )

        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._verify_model = AutoModelForCausalLM.from_pretrained(
                load_from, quantization_config=quant_config,
                device_map=device_map, torch_dtype=torch.float16,
                trust_remote_code=True, **_FLASH_ATTN_KWARGS,
            )
            print("[specvlm] Verify model loaded: INT4 via bitsandbytes")
        except Exception as e_bnb:
            if _is_oom(e_bnb):
                print(f"[specvlm] ⚠️  7B verify model OOM ({e_bnb}) — falling back to draft-only mode")
                self._verify_disabled = True
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return
            print(f"[specvlm] INT4 unavailable ({e_bnb}), loading verify in FP16…")
            try:
                self._verify_model = AutoModelForCausalLM.from_pretrained(
                    load_from, device_map=device_map,
                    torch_dtype=torch.float16, trust_remote_code=True,
                    **_FLASH_ATTN_KWARGS,
                )
                print("[specvlm] Verify model loaded: FP16")
            except Exception as e_fp16:
                if _is_oom(e_fp16):
                    print(f"[specvlm] ⚠️  7B verify model OOM (FP16 fallback) — falling back to draft-only mode")
                    self._verify_disabled = True
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return
                raise

        self._verify_model.eval()
        self._verify_tokenizer = AutoTokenizer.from_pretrained(
            load_from, trust_remote_code=True, use_fast=True,
        )
    
    def _unload_models(self):
        """Unload models to free VRAM."""
        self._draft_model = None
        self._verify_model = None
        self._draft_tokenizer = None
        self._verify_tokenizer = None
    
    def grade(self, image_path: str) -> SpecVLMResult:
        """
        Grade a single image using speculative decoding.

        Builds a minimal VisualMetadata with a neutral 0.5 CLIP baseline and
        delegates to grade_from_metadata so the prompt and clamping logic are
        consistent with the batch path.
        """
        meta = build_visual_metadata(image_path, clip_score=0.5, aspect_scores={})
        return self.grade_from_metadata(meta)
    
    def grade_batch(self, image_paths: List[str], progress=None) -> List[SpecVLMResult]:
        """Grade multiple images using speculative decoding."""
        results = []
        n = len(image_paths)

        for i, path in enumerate(image_paths):
            result = self.grade(path)
            results.append(result)

            if progress:
                progress((i + 1) / n, f"SpecVLM: {i + 1}/{n}")

        return results

    def grade_batch_scan(
        self,
        image_paths: List[str],
        top_pct: float = 0.20,
        progress=None,
    ) -> List[SpecVLMResult]:
        """
        Low-Latency Scan: two-pass grading.

        Pass 1 — 1.5B draft runs on every image.
        Pass 2 — 7B verifier runs ONLY on the top `top_pct` fraction by draft score,
                  and only when draft_confidence ≤ DRAFT_CONFIDENCE_THRESHOLD.
        Weak-bucket shots fall into the bottom 80% and never touch the 7B model.
        """
        n = len(image_paths)
        if n == 0:
            return []

        # Pass 1: draft all images
        scan_metas = [build_visual_metadata(p, clip_score=0.5, aspect_scores={}) for p in image_paths]
        draft_results: List[Dict[str, Any]] = []
        for i, meta in enumerate(scan_metas):
            draft_results.append(self._draft_inference(meta))
            if progress:
                progress(0.5 * (i + 1) / n, f"Scan draft: {i+1}/{n}")

        # Rank by draft score; only top top_pct are eligible for 7B verification
        top_k: int = max(1, math.ceil(n * top_pct))
        ranked = sorted(range(n), key=lambda idx: draft_results[idx]["score"], reverse=True)
        verify_eligible: Set[int] = set(ranked[:top_k])

        # Pass 2: verify eligible images where the draft is uncertain
        results: List[SpecVLMResult] = []
        for i, path in enumerate(image_paths):
            draft = draft_results[i]
            if (not self._verify_disabled
                    and i in verify_eligible
                    and self.priority_gate.trigger_verification(draft["confidence"])):
                verify = self._verify_inference(scan_metas[i])
                final_score = self._ensemble_scores(
                    draft["score"], verify["score"], draft["confidence"]
                )
                results.append(SpecVLMResult(
                    path=path,
                    score=final_score,
                    confidence=verify["confidence"],
                    reasoning_log=verify["reasoning"],
                    is_verified=True,
                    draft_score=draft["score"],
                    verify_score=verify["score"],
                ))
            else:
                results.append(SpecVLMResult(
                    path=path,
                    score=draft["score"],
                    confidence=draft["confidence"],
                    reasoning_log=draft["reasoning"],
                    is_verified=False,
                    draft_score=draft["score"],
                ))
            if progress:
                progress(0.5 + 0.5 * (i + 1) / n, f"Scan verify: {i+1}/{n}")

        return results
    
    def _draft_inference(self, meta: "VisualMetadata") -> Dict[str, Any]:
        """Run draft model (1.5B) inference on structured VisualMetadata."""
        prompt = self._build_prompt(meta)
        inputs = self._draft_tokenizer(prompt, return_tensors="pt").to(self.draft_model.device)
        with torch.no_grad():
            outputs = self.draft_model.generate(
                **inputs, max_new_tokens=400, temperature=0.15,
                do_sample=True, return_dict_in_generate=True, output_scores=True,
            )
        response = self._draft_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        parsed   = self._parse_response(response, meta.clip_score)
        confidence = self._calculate_confidence(outputs.scores) if parsed["json_ok"] else 0.0
        return {"score": parsed["score"], "reasoning": parsed["reasoning"], "confidence": confidence}

    def _verify_inference(self, meta: "VisualMetadata") -> Dict[str, Any]:
        """Run verify model (7B) inference on structured VisualMetadata."""
        prompt = self._build_prompt(meta, is_verify=True)
        inputs = self._verify_tokenizer(prompt, return_tensors="pt").to(self.verify_model.device)
        with torch.no_grad():
            outputs = self.verify_model.generate(
                **inputs, max_new_tokens=600, temperature=0.05,
                do_sample=True, return_dict_in_generate=True, output_scores=True,
            )
        response = self._verify_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        parsed   = self._parse_response(response, meta.clip_score)
        confidence = self._calculate_confidence(outputs.scores) if parsed["json_ok"] else 0.5
        return {"score": parsed["score"], "reasoning": parsed["reasoning"], "confidence": confidence}
    
    # Per-genre scoring guidance injected into the prompt
    _GENRE_INSTRUCTIONS: Dict[str, str] = {
        "Street": (
            "- Weight Human/Culture and Narrative most heavily — decisive moment and emotional "
            "connection are the core of street photography.\n"
            "- Composition and Lighting amplify but do not replace human story.\n"
            "- Penalise weak focus, blown exposure, or a frame with no subject."
        ),
        "Architectural": (
            "- Weight Composition and Lighting most heavily — geometry, symmetry, shadow "
            "play, and structural depth define quality here.\n"
            "- Human/Culture score is expected to be low; DO NOT penalise the absence of people.\n"
            "- Reward bold perspective, tonal range, and precise framing."
        ),
        "Liminal": (
            "- Weight Lighting and Narrative most heavily — mood, atmosphere, and the "
            "psychological sense of emptiness or transition are the defining qualities.\n"
            "- Human/Culture score will naturally be near zero; NEVER penalise this.\n"
            "- DO NOT treat an empty frame as a weakness — emptiness IS the subject.\n"
            "- Reward fog, reflections, flat corridors, underpasses, and liminal tension.\n"
            "- Penalise only technical failures: noise, poor exposure, or accidental blur."
        ),
    }

    def _build_prompt(self, meta: "VisualMetadata", is_verify: bool = False) -> str:
        """Build the prompt for VLM inference from structured VisualMetadata."""
        genre_label = {
            "Street":       "Street / Documentary",
            "Architectural":"Architectural / Geometric",
            "Liminal":      "Liminal / Atmospheric",
        }.get(meta.photo_genre, meta.photo_genre)

        aspect_lines = "\n".join(
            f"  {k:<15} {round(v*100):>3}%  {'█' * int(v*10)}{'░' * (10-int(v*10))}"
            for k, v in sorted(meta.aspect_scores.items(), key=lambda x: -x[1])
        )
        genre_instr = self._GENRE_INSTRUCTIONS.get(meta.photo_genre, self._GENRE_INSTRUCTIONS["Street"])

        style_note = ""
        try:
            from background_dpo_trainer import load_style_instruction
            style_note = load_style_instruction()
        except Exception:
            pass
        style_prefix = (
            f"[Editor Profile: {style_note}]\n\n" if style_note else ""
        )

        si_constraint = _SUBJECT_INTRUSION_CONSTRAINT if _cd_brief_implies_empty() else ""

        if is_verify and _cd_brief_implies_empty():
            system = (
                "You are a Purist Photo Critic. You cannot edit the image. "
                "Your task is to find the most accurate ORIGINAL captures that satisfy the user's brief. "
                "If the brief is 'Empty', any human presence is a total failure. "
                "Prioritize original composition and natural lighting.\n\n"
            )
        else:
            system = (
                "You are a professional photo editor specializing in street, documentary, "
                "architectural, and liminal photography.\n\n"
            )

        return (
            style_prefix
            + system
            + "Analyze the following visual metadata and provide a chain-of-thought assessment.\n\n"
            "[Visual Metadata]\n"
            f"Filename    : {meta.filename}\n"
            f"Aspect Ratio: {meta.aspect_ratio}\n"
            f"Genre       : {genre_label}\n"
            f"CLIP Score  : {meta.clip_score:.2f}  ({meta.clip_tier})\n\n"
            "Aspect Scores:\n"
            f"{aspect_lines}\n\n"
            "Scoring Instructions (genre-specific):\n"
            f"{genre_instr}\n\n"
            "General Instructions:\n"
            "1. Reason through what each aspect score suggests given the genre above.\n"
            "2. Consider how the aspect ratio relates to compositional intent.\n"
            "3. Identify the single strongest element and the single biggest weakness.\n"
            "4. Assign a refined final score (0.0–1.0), constrained to ±0.07 from CLIP base.\n"
            f"{si_constraint}\n"
            'Respond ONLY with valid JSON: {"score": 0.73, "reasoning_log": "..."}'
        )

    def _batch_prompt(self, metas: List["VisualMetadata"], is_verify: bool = False) -> str:
        """XML-like multi-image prompt — all BATCH_SIZE metadata blocks in one forward pass."""
        style_note = ""
        try:
            from background_dpo_trainer import load_style_instruction
            style_note = load_style_instruction()
        except Exception:
            pass
        style_prefix = f"[Editor Profile: {style_note}]\n\n" if style_note else ""

        blocks = []
        for i, meta in enumerate(metas):
            data = {
                "filename":      meta.filename,
                "aspect_ratio":  meta.aspect_ratio,
                "genre":         meta.photo_genre,
                "clip_score":    meta.clip_score,
                "clip_tier":     meta.clip_tier,
                "aspect_scores": meta.aspect_scores,
            }
            blocks.append(f'<image_data id="{i}">{json.dumps(data)}</image_data>')

        n = len(metas)
        si_constraint = _SUBJECT_INTRUSION_CONSTRAINT if _cd_brief_implies_empty() else ""

        if is_verify and _cd_brief_implies_empty():
            system = (
                "You are a Purist Photo Critic. You cannot edit the image. "
                "Your task is to find the most accurate ORIGINAL captures that satisfy the user's brief. "
                "If the brief is 'Empty', any human presence is a total failure. "
                "Prioritize original composition and natural lighting.\n\n"
            )
        else:
            system = (
                "You are a professional photo editor specializing in street, documentary, "
                "architectural, and liminal photography.\n\n"
            )

        return (
            style_prefix
            + system
            + "Analyze the image metadata blocks below. Each contains CLIP-derived scores.\n\n"
            "Genre scoring guidance:\n"
            "- Street/Documentary: weight Human/Culture and Narrative most heavily.\n"
            "- Architectural/Geometric: weight Composition and Lighting; never penalise low Human/Culture.\n"
            "- Liminal/Atmospheric: weight Lighting and Narrative; emptiness IS the subject.\n"
            + (si_constraint if si_constraint else "")
            + "\n"
            + "\n".join(blocks)
            + "\n\n"
            "For each image assign a refined score (0.0–1.0) constrained to ±0.07 from clip_score, "
            "and a confidence (0.0–1.0) reflecting certainty in your assessment.\n\n"
            f"Respond ONLY with a JSON array of exactly {n} objects:\n"
            '[{"id": 0, "score": 0.73, "confidence": 0.92, "reasoning_log": "..."}, ...]'
        )

    def _parse_response(self, response: str, clip_score: float, max_drift: float = 0.07) -> Dict[str, Any]:
        """Parse DeepSeek response. Clamp refined score to clip_score ± max_drift."""
        try:
            start = response.rfind("{")   # take last { to skip any prompt echo
            end   = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(response[start:end])
                raw_score = float(parsed.get("score", clip_score))
                score = float(np.clip(raw_score, clip_score - max_drift, clip_score + max_drift))
                score = float(np.clip(score, 0.0, 1.0))
                return {
                    "score":    round(score, 3),
                    "reasoning": str(parsed.get("reasoning_log", parsed.get("reasoning", ""))),
                    "json_ok":  True,
                }
        except Exception:
            pass
        import re
        m = re.search(r'"score"\s*:\s*([0-9.]+)', response)
        raw = float(m.group(1)) if m else clip_score
        score = float(np.clip(raw, clip_score - max_drift, clip_score + max_drift))
        return {
            "score":    round(float(np.clip(score, 0.0, 1.0)), 3),
            "reasoning": response[-600:].strip(),
            "json_ok":  False,
        }

    def _parse_batch_response(
        self, response: str, metas: List["VisualMetadata"]
    ) -> List[Dict[str, Any]]:
        """Parse batch JSON array; confidence=0.0 for failed parses so 7B verify fires."""
        import re
        n   = len(metas)
        out: List[Optional[Dict[str, Any]]] = [None] * n

        try:
            start = response.find("[")
            end   = response.rfind("]") + 1
            if start >= 0 and end > start:
                arr = json.loads(response[start:end])
                for item in arr:
                    idx = int(item.get("id", -1))
                    if 0 <= idx < n:
                        clip_sc   = metas[idx].clip_score
                        raw_score = float(item.get("score", clip_sc))
                        score     = float(np.clip(raw_score, clip_sc - 0.07, clip_sc + 0.07))
                        out[idx]  = {
                            "score":      round(float(np.clip(score, 0.0, 1.0)), 3),
                            "confidence": float(item.get("confidence", 0.0)),
                            "reasoning":  str(item.get("reasoning_log", "")),
                            "json_ok":    True,
                        }
        except Exception:
            pass

        # Per-ID regex fallback for any slots still unparsed
        for i in range(n):
            if out[i] is None:
                clip_sc = metas[i].clip_score
                m = re.search(rf'"id"\s*:\s*{i}\s*,\s*"score"\s*:\s*([0-9.]+)', response)
                if m:
                    raw   = float(m.group(1))
                    score = float(np.clip(raw, clip_sc - 0.07, clip_sc + 0.07))
                    out[i] = {
                        "score":      round(float(np.clip(score, 0.0, 1.0)), 3),
                        "confidence": 0.0,
                        "reasoning":  "",
                        "json_ok":    False,
                    }
                else:
                    out[i] = {"score": clip_sc, "confidence": 0.0, "reasoning": "", "json_ok": False}

        return out  # type: ignore[return-value]

    def _draft_only_score(self, draft_score: float, clip_score: float) -> float:
        """
        For unverified photos, blend draft score toward CLIP baseline.

        The 1.5B model tends to saturate at the ±0.12 clamp limits, causing
        lopsided grades. Blending 60/40 toward CLIP anchors these photos
        close to the CLIP calibration without completely ignoring the draft.
        """
        return round(float(np.clip(0.80 * clip_score + 0.20 * draft_score, 0.0, 1.0)), 3)
    
    def _calculate_confidence(self, scores: Tuple[torch.Tensor]) -> float:
        """Calculate confidence from output logits."""
        if not scores:
            return 0.5
        
        # Use average logit magnitude as confidence proxy
        total_logit = 0.0
        count = 0
        
        for score in scores:
            logits = score[0]  # (vocab_size,)
            max_logit = torch.max(logits).item()
            total_logit += max_logit
            count += 1
        
        # Normalize to [0, 1]
        confidence = min(1.0, max(0.0, (total_logit / max(count, 1)) / 10.0))
        return confidence
    
    def _ensemble_scores(
        self, draft_score: float, verify_score: float, draft_confidence: float
    ) -> float:
        """
        Ensemble draft and verify scores.
        
        When draft confidence is high, trust draft more.
        When draft confidence is low, trust verify more.
        """
        # Weight based on draft confidence
        draft_weight = min(1.0, draft_confidence * 2)  # 0.5-1.0 range
        verify_weight = 1.0 - draft_weight
        
        return draft_score * draft_weight + verify_score * verify_weight
    
    def grade_from_metadata(self, meta: "VisualMetadata") -> SpecVLMResult:
        d = self._draft_inference(meta)
        if self._verify_disabled or self.priority_gate.should_skip(d["confidence"]):
            blended = self._draft_only_score(d["score"], meta.clip_score)
            return SpecVLMResult(
                path=meta.path, score=blended, confidence=d["confidence"],
                reasoning_log=d["reasoning"], is_verified=False, draft_score=d["score"],
            )
        v = self._verify_inference(meta)
        final = self._ensemble_scores(d["score"], v["score"], d["confidence"])
        return SpecVLMResult(
            path=meta.path, score=final, confidence=v["confidence"],
            reasoning_log=v["reasoning"], is_verified=True,
            draft_score=d["score"], verify_score=v["score"],
        )

    def _batch_inference(
        self,
        metas: List["VisualMetadata"],
        model,
        tokenizer,
        label: str,
    ) -> List[Dict[str, Any]]:
        """Single forward pass covering up to BATCH_SIZE images; returns parsed list."""
        if not metas:
            return []
        is_verify = (label == "verify")
        prompt  = self._batch_prompt(metas, is_verify=is_verify)
        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=BATCH_SIZE * 200,
                temperature=0.15, do_sample=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed   = self._parse_batch_response(response, metas)
        ok_count = sum(1 for r in parsed if r["json_ok"])
        print(f"[specvlm] {label} batch({len(metas)}): {ok_count}/{len(metas)} parsed OK")
        return parsed

    def process_metadata_batch(
        self,
        metadata_list: List["VisualMetadata"],
        progress=None,
    ) -> List[SpecVLMResult]:
        """
        Batch two-pass reasoning with one forward pass per BATCH_SIZE chunk.

        Pass 1 — 1.5B draft: chunks of BATCH_SIZE, one generate() call per chunk.
        Pass 2 — 7B verify: only images flagged by confidence ≤ threshold, re-chunked.

        Partial chunks (last chunk < BATCH_SIZE) are processed immediately — no waiting
        beyond MAX_WAIT_MS. VRAM: 1.5B unloads + purge_vram() before 7B loads.
        """
        _p = progress or (lambda f, d: None)
        n  = len(metadata_list)
        if n == 0:
            return []

        # ── Pass 1: Draft ─────────────────────────────────────────────────────
        _p(0.56, f"DeepSeek-R1 1.5B — batch draft ({n} photos, batch={BATCH_SIZE})…")
        _ = self.draft_model   # ensure loaded

        draft_results: List[Optional[Dict[str, Any]]] = [None] * n
        chunks  = [metadata_list[i:i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]
        offsets = list(range(0, n, BATCH_SIZE))

        for ci, (chunk, offset) in enumerate(zip(chunks, offsets)):
            for j, parsed in enumerate(
                self._batch_inference(chunk, self._draft_model, self._draft_tokenizer, "draft")
            ):
                draft_results[offset + j] = parsed
            done = min(offset + BATCH_SIZE, n)
            _p(0.56 + 0.14 * (ci + 1) / len(chunks),
               f"Draft batch {ci+1}/{len(chunks)} ({done}/{n})")

        needs_verify = [
            i for i, r in enumerate(draft_results)
            if not self._verify_disabled
            and r is not None
            and self.priority_gate.trigger_verification(r["confidence"])
        ]
        print(f"[specvlm] Batch draft done. {len(needs_verify)}/{n} flagged for 7B verify.")
        _p(0.70, f"Draft done — {len(needs_verify)}/{n} need 7B verification")

        # ── Unload 1.5B + purge before 7B ─────────────────────────────────────
        self._draft_model     = None
        self._draft_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try: torch.cuda.ipc_collect()
            except Exception: pass

        # ── Pass 2: Verify (flagged only) ─────────────────────────────────────
        verify_map: Dict[int, Dict[str, Any]] = {}
        if needs_verify and not self._verify_disabled:
            _p(0.71, f"Loading DeepSeek-R1 7B — verifying {len(needs_verify)} photos…")
            try:
                _ = self.verify_model   # triggers load
                nv_metas   = [metadata_list[i] for i in needs_verify]
                nv_chunks  = [nv_metas[i:i + BATCH_SIZE] for i in range(0, len(nv_metas), BATCH_SIZE)]
                nv_offsets = list(range(0, len(nv_metas), BATCH_SIZE))
                for ci, (chunk, offset) in enumerate(zip(nv_chunks, nv_offsets)):
                    for j, parsed in enumerate(
                        self._batch_inference(chunk, self._verify_model, self._verify_tokenizer, "verify")
                    ):
                        verify_map[needs_verify[offset + j]] = parsed
                    done = min(offset + BATCH_SIZE, len(nv_metas))
                    _p(0.71 + 0.14 * (ci + 1) / len(nv_chunks),
                       f"Verify batch {ci+1}/{len(nv_chunks)} ({done}/{len(nv_metas)})")
            except Exception as e:
                print(f"[specvlm] 7B batch verify failed: {e}")

        # ── Merge ──────────────────────────────────────────────────────────────
        results: List[SpecVLMResult] = []
        for i, meta in enumerate(metadata_list):
            d = draft_results[i] or {"score": meta.clip_score, "confidence": 0.0, "reasoning": ""}
            if i in verify_map:
                v     = verify_map[i]
                final = self._ensemble_scores(d["score"], v["score"], d["confidence"])
                results.append(SpecVLMResult(
                    path=meta.path, score=final, confidence=v["confidence"],
                    reasoning_log=v["reasoning"], is_verified=True,
                    draft_score=d["score"], verify_score=v["score"],
                ))
            else:
                blended = self._draft_only_score(d["score"], meta.clip_score)
                results.append(SpecVLMResult(
                    path=meta.path, score=blended, confidence=d["confidence"],
                    reasoning_log=d["reasoning"], is_verified=False,
                    draft_score=d["score"],
                ))
        return results

    def grade_batch_from_metadata(
        self,
        metadatas: List["VisualMetadata"],
        progress=None,
    ) -> List[SpecVLMResult]:
        """Delegates to process_metadata_batch."""
        return self.process_metadata_batch(metadatas, progress=progress)

    def unload(self) -> None:
        """Unload models to free VRAM."""
        self._unload_models()
        VRAMManager.clear_between_phases()


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
    Map raw discriminants so scores spread naturally across all three grade zones.

    IQR [p25, p75] maps to [0.33, 0.67].  Grade thresholds (Strong ≥0.60,
    Weak ≤0.41) are inside the IQR zone rather than sitting exactly at p25/p75,
    so the middle 50% of photos land clearly in Mid rather than hovering at
    bucket boundaries.  This produces roughly 20% Weak / 58% Mid / 22% Strong
    for a typical mixed-quality batch — a genuinely good photo still needs to
    be above the ~79th percentile to score Strong.
    """
    lo = float(np.percentile(raw, 25))
    hi = float(np.percentile(raw, 75))
    span = max(hi - lo, 1e-4)
    # [lo, hi] → [0.33, 0.67]  (34-point spread, thresholds inside IQR)
    return np.clip((raw - lo) / span * 0.34 + 0.33, 0.0, 1.0)


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

        gate = PriorityGate(threshold=DRAFT_CONFIDENCE_THRESHOLD)

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

        for i, path in enumerate(image_paths):
            draft_score = float(cal_overall[i])
            confidence  = min(1.0, abs(draft_score - 0.5) * 2.0)
            is_verified = False

            # Always populate aspect scores — every photo gets all 5 bars
            aspect_scores: Dict[str, float] = {}
            if have_aspects and cal_aspects is not None:
                aspect_scores = {
                    name: round(float(cal_aspects[i, j]), 3)
                    for j, name in enumerate(aspect_names)
                }

            # Always blend the weighted aspect average into the final score.
            # Architectural / liminal photos have low Human/Culture by design
            # (no people) — down-weight that axis so structural strengths dominate.
            if aspect_scores:
                human_sc = aspect_scores.get("Human/Culture", 0.5)
                comp_sc  = aspect_scores.get("Composition",   0.5)
                is_arch  = human_sc < 0.38 and comp_sc > 0.52

                _W: Dict[str, float] = {
                    "Technical":     1.0,
                    "Composition":   1.2,
                    "Lighting":      1.0,
                    "Narrative":     0.6 if is_arch else 1.0,
                    "Human/Culture": 0.15 if is_arch else 1.0,
                }
                total_w      = sum(_W.get(k, 1.0) for k in aspect_scores)
                aspect_avg   = sum(v * _W.get(k, 1.0) for k, v in aspect_scores.items()) / total_w
                draft_score  = round(0.60 * draft_score + 0.40 * aspect_avg, 3)
                confidence   = min(1.0, abs(draft_score - 0.5) * 2.0)

            # is_verified: only True when the gate explicitly triggered the
            # verify pass (used for badge display purposes only)
            if (aspect_scores
                    and not scan_mode
                    and gate.trigger_verification(confidence)):
                is_verified = True

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
                f"Graded {i + 1}/{n} — {draft_score:.2f}"
                + (" ✓" if is_verified else ""),
            )

        return results

    def unload(self) -> None:
        pass
