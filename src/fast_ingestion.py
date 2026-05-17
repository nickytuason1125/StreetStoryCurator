"""
Fast image ingestion — TurboJPEG + pin_memory for async GPU transfer.

TurboJPEG (libjpeg-turbo) decodes JPEG files 2-4× faster than PIL by using
SIMD-optimised C code and releasing the GIL during the decode step.

Non-JPEG formats (PNG, TIFF, WEBP) always fall back to PIL Image.open().

pin_memory=True allocates the host tensor in CUDA-pinned (page-locked) memory,
enabling the GPU DMA engine to copy the next batch while the current one runs
inference — hiding H2D transfer latency behind compute.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision.transforms.functional as TF

_JPEG_EXTS = {".jpg", ".jpeg"}

_tj = None   # singleton TurboJPEG instance (thread-safe after init)


def _get_tj():
    global _tj
    if _tj is None:
        try:
            from turbojpeg import TurboJPEG
            _tj = TurboJPEG()
        except Exception:
            pass
    return _tj


def decode_one(
    path: str,
    target_hw: Optional[tuple[int, int]] = None,
    pin: bool = True,
) -> Optional[torch.Tensor]:
    """
    Decode a single image to (C, H, W) float32 tensor in [0, 1].

    target_hw : (H, W) to resize after decode. None = native resolution.
    pin       : call .pin_memory() so the GPU DMA engine can fetch directly.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext in _JPEG_EXTS:
            tj = _get_tj()
            if tj is not None:
                with open(path, "rb") as fh:
                    raw = fh.read()
                bgr = tj.decode(raw)                          # (H, W, 3) uint8 BGR
                rgb = bgr[:, :, ::-1].copy()                  # BGR→RGB, contiguous
            else:
                from PIL import Image
                rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        else:
            from PIL import Image
            rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

        # numpy → (C, H, W) float tensor without an extra copy
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

        if target_hw is not None:
            t = TF.resize(t, list(target_hw), antialias=True)

        _cuda = torch.cuda.is_available()
        return t.pin_memory() if (pin and _cuda) else t

    except Exception as exc:
        print(f"[fast_ingestion] {Path(path).name}: {exc}")
        return None


def decode_batch(
    paths: list[str],
    target_hw: Optional[tuple[int, int]] = None,
    pin: bool = True,
    n_workers: int = 8,
) -> list[Optional[torch.Tensor]]:
    """
    Decode all paths in parallel using ThreadPoolExecutor.
    Returns list of (C, H, W) float32 pinned tensors (None on failure).
    """
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=min(n_workers, len(paths))) as pool:
        return list(pool.map(lambda p: decode_one(p, target_hw, pin), paths))
