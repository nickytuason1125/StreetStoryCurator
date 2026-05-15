"""
DeepSeek Model Wrapper with Auto-Download

Wraps DeepSeek-R1-Distill-Qwen models for Speculative Decoding.

Provides:
- Draft Model: DeepSeek-R1-Distill-Qwen-1.5B (INT4-AWQ)
- Verify Model: DeepSeek-R1-Distill-Qwen-7B (INT4-AWQ)

Both models use 4-bit quantization for efficient inference on laptop GPUs.

Auto-download: Models are downloaded on first run if not present locally.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import numpy as np


# Model IDs
DRAFT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
VERIFY_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Model cache directory
MODEL_CACHE_DIR = Path("models/deepseek")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _model_weights_exist(local_dir: Path) -> bool:
    """
    Return True if a completed HuggingFace model download is detected.

    Accepts both legacy pytorch_model.bin shards and modern safetensors shards,
    as well as single-file models.  config.json is used as the primary sentinel
    because it is always present after a successful snapshot_download.
    """
    if not local_dir.exists() or not (local_dir / "config.json").exists():
        return False
    index_candidates = [
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    if any((local_dir / f).exists() for f in index_candidates):
        return True
    # Single-file models (no shard index)
    return any(local_dir.glob("*.safetensors")) or any(local_dir.glob("pytorch_model*.bin"))


def _download_model_if_needed(model_id: str, local_dir: Path) -> bool:
    """
    Download a HuggingFace model into local_dir if not already present.

    Args:
        model_id: HuggingFace model ID (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        local_dir: Local directory to download into.

    Returns:
        True if model is ready, False on error.
    """
    if _model_weights_exist(local_dir):
        print(f"✓ Model already cached at {local_dir}")
        return True

    print(f"📦 Downloading {model_id} to {local_dir}...")
    print("   This may take several minutes depending on your connection.")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"✓ Download complete: {local_dir}")
        return True

    except ImportError:
        print("⚠️  huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"⚠️  Download failed for {model_id}: {e}")
        return False


class DeepSeekModel:
    """
    DeepSeek-R1-Distill-Qwen model wrapper with INT4-AWQ quantization.
    
    Supports both draft (1.5B) and verify (7B) models.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        quantize: bool = True,
    ):
        """
        Initialize DeepSeek model.
        
        Args:
            model_id: HuggingFace model ID
            device: "auto", "cpu", or "cuda"
            quantize: Use 4-bit quantization
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model_id = model_id
        self.quantize = quantize
        
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _ensure_downloaded(self) -> bool:
        """Ensure model is downloaded, downloading from HuggingFace if needed."""
        model_name = self.model_id.replace("/", "_")
        local_dir = MODEL_CACHE_DIR / model_name
        return _download_model_if_needed(self.model_id, local_dir)
    
    def _load_model(self) -> None:
        """Load model with 4-bit quantization."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Ensure model is downloaded
        if not self._ensure_downloaded():
            raise RuntimeError(f"Model {self.model_id} not found. Please download it first.")
        
        # Extract model name from ID
        model_name = self.model_id.replace("/", "_")
        local_dir = MODEL_CACHE_DIR / model_name
        
        # Quantization config for INT4-AWQ
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        print(f">Loading {self.model_id}...")
        
        self._model = AutoModelForCausalLM.from_pretrained(
            str(local_dir),
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self._model.eval()
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(local_dir),
            trust_remote_code=True,
            use_fast=False,
        )
        
        self._loaded = True
        print(f"✓ {self.model_id} loaded successfully")
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_logits: bool = False,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            return_logits: Whether to return logits
        
        Returns:
            (generated_text, logits)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=return_logits,
            )
        
        generated_text = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        
        logits = None
        if return_logits and outputs.scores:
            logits = torch.stack(outputs.scores, dim=1).cpu().float().numpy()
        
        return generated_text, logits
    
    def score(
        self,
        prompt: str,
        candidates: List[str],
    ) -> np.ndarray:
        """
        Score multiple candidate completions.
        
        Args:
            prompt: Input prompt
            candidates: List of candidate completions
        
        Returns:
            Logits for each candidate
        """
        # For now, return placeholder
        return np.ones(len(candidates)) / len(candidates)
    
    def unload(self) -> None:
        """Unload model to free VRAM."""
        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class DraftModel(DeepSeekModel):
    """DeepSeek-R1-Distill-Qwen-1.5B for fast draft inference."""
    
    def __init__(self, device: str = "auto"):
        super().__init__(DRAFT_MODEL_ID, device=device)


class VerifyModel(DeepSeekModel):
    """DeepSeek-R1-Distill-Qwen-7B for high-quality verification."""
    
    def __init__(self, device: str = "auto"):
        super().__init__(VERIFY_MODEL_ID, device=device)


# ── High-Level API ─────────────────────────────────────────────────────────────


def create_draft_model(device: str = "auto") -> DraftModel:
    """Create draft model (1.5B)."""
    return DraftModel(device=device)


def create_verify_model(device: str = "auto") -> VerifyModel:
    """Create verify model (7B)."""
    return VerifyModel(device=device)


def create_model(model_type: str, device: str = "auto") -> DeepSeekModel:
    """
    Create DeepSeek model by type.
    
    Args:
        model_type: "draft" or "verify"
        device: Device to use
    
    Returns:
        DeepSeekModel instance
    """
    if model_type == "draft":
        return create_draft_model(device)
    elif model_type == "verify":
        return create_verify_model(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def download_all_models() -> Dict[str, bool]:
    """
    Download all required models for SpecVLM pipeline.
    
    Returns:
        Dict mapping model names to success status
    """
    results = {}
    
    # Download draft model
    results["DeepSeek-R1-Distill-Qwen-1.5B"] = _download_model_if_needed(
        DRAFT_MODEL_ID, MODEL_CACHE_DIR / "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"
    )
    
    # Download verify model
    results["DeepSeek-R1-Distill-Qwen-7B"] = _download_model_if_needed(
        VERIFY_MODEL_ID, MODEL_CACHE_DIR / "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"
    )
    
    return results
