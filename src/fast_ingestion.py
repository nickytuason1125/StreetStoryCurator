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

def run_pixel_inspector(image_path: str, technical_score: float = 0.5) -> str:
    """
    Phase 1 — VLM Pixel Inspector.

    Runs the local vision model over the image and returns a 2-sentence semantic
    profile focused on lighting, tension and mood.

    Returns "" when the vision weights are absent or the image cannot be read.
    That is a supported outcome — the profile is an enrichment, and the ingestion
    path is defined without it.

    This used to POST the same base64 payload to Ollama's /api/chat asking for the
    qwen2.5vl:3b tag. No installer pulled that tag, and no document mentioned it,
    so on every machine but the developer's this returned "" via the exception
    handler. Same model family, same prompt, now in-process through the loader
    critique_engine already owns — which also means one resident copy instead of
    a second runtime holding its own.
    """
    try:
        import critique_engine as _ce
        prompt = (
            f"Analyze this street photograph. Its technical score is {technical_score:.2f}. "
            "Write a strict 2-sentence semantic profile focusing on lighting, visual tension, "
            "and mood. No filler."
        )
        profile = _ce.describe_image(image_path, prompt, max_dimension=1024,
                                     max_tokens=200, temperature=0.2)
        if profile:
            print(f"[fast_ingestion] pixel_inspector: {Path(image_path).name} "
                  f"→ {len(profile)} chars")
        return profile
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


def _scale_den(w: int, h: int, hint: int) -> int:
    """Largest power-of-two downscale whose result still covers `hint` px.

    Both decoders can downscale DURING the decode (libjpeg DCT scaling), which
    is far cheaper than decoding full-size and resizing after. Never go below
    the hint: the caller resizes to its exact target afterwards, and upsampling
    a too-small decode would lose real detail.
    """
    for d in (8, 4, 2):
        if min(w, h) // d >= hint:
            return d
    return 1


def decode_one(
    path: str,
    target_hw: Optional[tuple[int, int]] = None,
    pin: bool = True,
    draft_hint: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Decode a single image to (C, H, W) float32 tensor in [0, 1].

    target_hw  : (H, W) to resize after decode. None = native resolution.
    pin        : call .pin_memory() so the GPU DMA engine can fetch directly.
    draft_hint : long-edge px the caller will actually use. When set, JPEGs are
                 decoded at the largest power-of-two downscale that still covers
                 it, instead of full-size-then-shrink.

    Why draft_hint exists: a cull's dominant cost is JPEG decode (measured
    2026-08-28: 279 ms/img decode versus 3 ms for the quality model itself), and
    every caller immediately shrinks to <=512 px. Decoding 45 MP to throw away
    99% of it was the single largest expense in the pipeline. Measured on this
    repo's own files: 325 -> 83 ms/img (3.9x) for 7.7 MB JPEGs, 772 -> 374 ms
    (2.1x) for 50 MB ones, with 36-64x less RAM per image.

    It is OPT-IN because it is not bit-identical: DCT-domain downscaling gives
    slightly different pixels than full decode + Lanczos (mean |drift| 0.0016-
    0.0047 on a 0-1 scale). This repo has been bitten before by a "harmless"
    change that moved grades across a threshold, so callers opt in and the drift
    is measured, not assumed. FRAMEGRADE_DRAFT_DECODE=0 disables globally.
    """
    import os as _os_dd
    if draft_hint and _os_dd.environ.get("FRAMEGRADE_DRAFT_DECODE", "1").strip() == "0":
        draft_hint = None

    ext = Path(path).suffix.lower()
    try:
        if ext in _JPEG_EXTS:
            tj = _get_tj()
            if tj is not None:
                with open(path, "rb") as fh:
                    raw = fh.read()
                bgr = None
                if draft_hint:
                    # TurboJPEG validates scaling_factor against a set the NATIVE
                    # library reports, so an unsupported ratio raises. Without
                    # this fallback that exception would reach the outer handler
                    # and return None — silently DROPPING the photo from the cull
                    # rather than merely decoding it slower. Never trade a
                    # correct result for a faster one.
                    try:
                        _w, _h, _, _ = tj.decode_header(raw)
                        _d = _scale_den(_w, _h, draft_hint)
                        if _d > 1:
                            bgr = tj.decode(raw, scaling_factor=(1, _d))
                    except Exception:
                        bgr = None
                if bgr is None:
                    bgr = tj.decode(raw)                      # (H, W, 3) uint8 BGR
                rgb = bgr[:, :, ::-1].copy()                  # BGR→RGB, contiguous
            else:
                from PIL import Image
                with Image.open(path) as _im:
                    if draft_hint:
                        # Tells libjpeg to decode at 1/2, 1/4 or 1/8 directly.
                        # Must be called BEFORE the pixels are loaded, or it is
                        # a silent no-op.
                        _im.draft("RGB", (draft_hint, draft_hint))
                    rgb = np.array(_im.convert("RGB"), dtype=np.uint8)
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
