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

import base64
import json
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

print("[fast_ingestion] V3 Vision-to-Text Architecture loaded")

_OLLAMA_URL  = "http://localhost:11434/api/chat"
_PIXEL_MODEL = "qwen2.5vl:3b"


def run_pixel_inspector(image_path: str, technical_score: float = 0.5) -> str:
    """
    Phase 1 — VLM Pixel Inspector.

    Sends the image as base64 to a local qwen2.5vl:3b Ollama instance and
    returns a 2-sentence semantic profile focused on lighting, tension, and mood.
    Context is hard-locked to 2048 tokens to respect the 6 GB VRAM budget.

    Returns "" if Ollama is offline or the image cannot be read.
    """
    try:
        import requests as _req
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        prompt = (
            f"Analyze this street photograph. Its technical score is {technical_score:.2f}. "
            "Write a strict 2-sentence semantic profile focusing on lighting, visual tension, "
            "and mood. No filler."
        )
        resp = _req.post(
            _OLLAMA_URL,
            json={
                "model":   _PIXEL_MODEL,
                "stream":  False,
                "messages": [
                    {"role": "user", "content": prompt, "images": [b64]},
                ],
                "options": {"num_ctx": 2048},
            },
            timeout=90,
        )
        resp.raise_for_status()
        profile = resp.json().get("message", {}).get("content", "").strip()
        print(f"[fast_ingestion] pixel_inspector: {Path(image_path).name} → {len(profile)} chars")
        return profile
    except ConnectionError as _e:
        print(f"[fast_ingestion] Ollama unreachable for pixel_inspector: {_e}")
    except Exception as _e:
        print(f"[fast_ingestion] pixel_inspector failed ({_e})")
    return ""

import torch
import torchvision.transforms.functional as TF

_JPEG_EXTS = {".jpg", ".jpeg"}
from raw_support import RAW_EXTS as _RAW_EXTS

_tj = None   # singleton TurboJPEG instance (thread-safe after init)

# rawpy/libraw is NOT thread-safe. decode_one is called from ThreadPoolExecutors
# (e.g. vision_grading_heads._load_images_parallel, 8 workers); concurrent libraw
# calls fault the process with 0xC0000005. This lock serialises every RAW decode so
# only one libraw call runs at a time (JPEG/PNG stay fully parallel).
import threading as _threading
_RAW_LOCK = _threading.Lock()


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
        elif ext in _RAW_EXTS:
            # Serialise libraw (not thread-safe) AND use the embedded preview
            # (~5.5 MB) instead of a full demosaic (~100 MB) — the 1920px preview is
            # ample for TOPIQ, which downsamples to 512. Falls back to full demosaic
            # only when a RAW has no embedded preview (rare).
            from raw_support import extract_embedded_preview, _rawpy_decode
            with _RAW_LOCK:
                _prev = extract_embedded_preview(path, "RGB")
                # np.array (NOT np.asarray) — PIL's buffer is read-only; the caller
                # does an in-place torch .div_(), and writing to a non-writable
                # numpy-backed tensor is undefined behavior → 0xC0000005. Copy to a
                # writable, contiguous array like the JPEG/PIL paths do.
                rgb = (np.array(_prev, dtype=np.uint8)
                       if _prev is not None else _rawpy_decode(path))
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
