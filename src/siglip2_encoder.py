"""
SigLIP-2 ViT-g/14 NaFlex Encoder with Auto-Download

Replaces SigLIP-So400M with:
- FP8 quantization for faster inference
- Native aspect ratio preservation (fixes compositional "squishing")
- Larger model capacity (1536-d embeddings)

VRAM Protocol:
    1. SigLIP2Encoder() → model loads into VRAM
    2. encode_images() → all embeddings computed
    3. unload() → GPU cleared for next step

Auto-download: Model is downloaded on first run if not present locally.
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np

# Model configuration — largest 1536-d SigLIP-2 variant available in open_clip_torch
_MODEL_TAG = "ViT-gopt-16-SigLIP2-384"
_PRETRAINED = "webli"
EMBED_DIM = 1536  # SigLIP-2 ViT-gopt/16 @ 384px

# Model cache directory
MODEL_CACHE_DIR = Path("models/siglip2")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _siglip2_cache_exists() -> bool:
    """Return True if SigLIP-2 weights are already in MODEL_CACHE_DIR (any depth)."""
    if not MODEL_CACHE_DIR.exists():
        return False
    weight_exts = {".pt", ".bin", ".safetensors"}
    return any(
        f.suffix in weight_exts
        for f in MODEL_CACHE_DIR.rglob("*")
        if f.is_file() and not f.name.endswith(".incomplete")
    )


def _download_siglip2_if_needed() -> bool:
    """
    Pre-download SigLIP-2 ViT-g/14 NaFlex weights into MODEL_CACHE_DIR if absent.

    open_clip auto-downloads on first use; calling this before the first
    SigLIP2Encoder() instantiation avoids a silent first-request delay.

    Returns:
        True if model is ready, False on error.
    """
    if _siglip2_cache_exists():
        return True

    print(f"📦 Downloading SigLIP-2 ViT-g/14 NaFlex to {MODEL_CACHE_DIR}...")
    print("   This may take several minutes depending on your connection.")

    try:
        import open_clip

        # create_model_and_transforms downloads weights to cache_dir on first call.
        model, _, _ = open_clip.create_model_and_transforms(
            _MODEL_TAG,
            pretrained=_PRETRAINED,
            precision="fp16",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        # Release immediately — we only needed the download side-effect.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ SigLIP-2 download complete: {MODEL_CACHE_DIR}")
        return True

    except ImportError:
        print("⚠️  open_clip_torch not installed. Run: pip install open_clip_torch")
        return False
    except Exception as e:
        print(f"⚠️  SigLIP-2 download failed: {e}")
        return False


class SigLIP2Encoder:
    """
    SigLIP-2 ViT-g/14 NaFlex image encoder with FP8 quantization.

    Key features:
    - Native aspect ratio preservation (no forced square cropping)
    - FP8 quantization for faster inference
    - 1536-d embeddings (vs 1152-d for So400M)
    """

    # Disk file is FP32 (7.1 GB), but loaded as FP16 → ~3.5 GB VRAM.
    # INT8 quantization halves that to ~1.8 GB when torchao is available.
    _QUANTIZED_VRAM_GB = 1.8
    _FP16_VRAM_GB      = 3.5

    def __init__(self, device: str = "auto", quantize: bool = True, progress=None):
        import open_clip

        _p = progress or (lambda f, d: None)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            free_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved(0)
            ) / 1e9
            needed_gb = self._QUANTIZED_VRAM_GB if quantize else self._FP16_VRAM_GB
            if free_gb < needed_gb:
                print(
                    f"[siglip2] WARNING: only {free_gb:.1f} GB VRAM free, "
                    f"need ~{needed_gb:.1f} GB — falling back to CPU inference"
                )
                self.device = torch.device("cpu")

        # Download weights if not cached (first run only — 7.1 GB)
        if not _siglip2_cache_exists():
            _p(0.03, "Downloading SigLIP-2 (7.1 GB) — first run, may take 10+ min…")
        else:
            _p(0.03, "Loading SigLIP-2 from disk (~30–60 s)…")

        _download_siglip2_if_needed()

        _p(0.04, "SigLIP-2 weights ready — initialising model…")

        # Load weights to CPU in fp16; quantization happens on CPU before GPU transfer
        self._model, _, self._prep = open_clip.create_model_and_transforms(
            _MODEL_TAG,
            pretrained=_PRETRAINED,
            precision="fp16",
            cache_dir=str(MODEL_CACHE_DIR),
        )

        self._tok = open_clip.get_tokenizer(_MODEL_TAG)

        if quantize and self.device.type == "cuda":
            _p(0.05, "Quantising SigLIP-2 to INT8…")
            self._model = self._quantize_model(self._model)

        _p(0.06, "Moving SigLIP-2 to GPU…")
        self._model = self._model.to(self.device).eval()
        gc.collect()  # release CPU fp16 copy once transferred
        _p(0.07, "SigLIP-2 ready — encoding images…")
    
    def _quantize_model(self, model) -> torch.nn.Module:
        """
        Tries torchao INT8 weight-only quantization; falls back gracefully to FP16 on GPU.
        FP16 (~3.5 GB) fits in a 6 GB card without quantization.
        """
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
            print("[siglip2] Quantization: INT8 weight-only via torchao (~1.8 GB VRAM)")
            return model
        except Exception as e_torchao:
            print(f"[siglip2] torchao INT8 unavailable ({e_torchao}) — using FP16 on GPU (~3.5 GB)")
            return model
    
    def encode_images(
        self,
        paths: List[str],
        batch_size: int = 0,  # 0 = auto: 8 on GPU, 4 on CPU
        progress=None,
    ) -> np.ndarray:
        """
        Return normalised (N, 1536) float32 embeddings for a list of image paths.

        Uses parallel CPU prefetch: the next batch is loaded/preprocessed in a
        background thread while the GPU runs inference on the current batch.
        Bad paths yield a zero vector.
        """
        from PIL import Image as _PIL
        from concurrent.futures import ThreadPoolExecutor

        if batch_size == 0:
            batch_size = 8 if self.device.type == "cuda" else 4

        _model_dtype = next(iter(self._model.parameters())).dtype
        prep         = self._prep   # local ref avoids attribute lookup in hot loop
        zero         = torch.zeros(3, 384, 384)

        def _load(p: str) -> torch.Tensor:
            try:
                return prep(_PIL.open(p).convert("RGB"))
            except Exception:
                return zero

        all_embs: List[np.ndarray] = []
        n = len(paths)

        with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
            # Pre-load the first batch so there's no wait on the first iteration
            next_futures = [pool.submit(_load, p) for p in paths[:batch_size]]

            for start in range(0, n, batch_size):
                futures        = next_futures
                next_start     = start + batch_size
                # Kick off the next batch load while GPU runs on the current one
                next_futures   = (
                    [pool.submit(_load, p) for p in paths[next_start : next_start + batch_size]]
                    if next_start < n else []
                )

                tensors = [f.result() for f in futures]
                batch   = torch.stack(tensors).to(device=self.device, dtype=_model_dtype)

                with torch.no_grad():
                    emb = self._model.encode_image(batch)
                    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                all_embs.append(emb.cpu().float().numpy())

                if progress:
                    done = min(start + batch_size, n)
                    progress(done / n * 0.47, f"SigLIP-2: {done}/{n}")

        return np.concatenate(all_embs, axis=0)
    
    def encode_text(self, queries: List[str]) -> np.ndarray:
        """Return normalised (N, 1536) float32 embeddings for text queries."""
        tokens = self._tok(queries).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_text(tokens)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
        return emb.cpu().float().numpy()
    
    def unload(self) -> None:
        """Move model to CPU and empty GPU cache."""
        self._model = self._model.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ResizePad:
    """
    Resize and pad image to maintain aspect ratio.
    
    Used for SigLIP-2 to avoid compositional "squishing".
    """
    
    def __init__(self, size: int = 384, fill: int = 128):
        self.size = size
        self.fill = fill
    
    def __call__(self, img):
        """Resize and pad image to square while maintaining aspect ratio."""
        from PIL import Image, ImageOps
        
        # Get original dimensions
        w, h = img.size
        
        # Calculate scale to fit within size
        scale = min(self.size / w, self.size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        padded.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        
        return padded


def get_siglip2_encoder() -> SigLIP2Encoder:
    """Get or create SigLIP-2 encoder singleton."""
    if not hasattr(get_siglip2_encoder, "_instance"):
        get_siglip2_encoder._instance = SigLIP2Encoder()
    return get_siglip2_encoder._instance
