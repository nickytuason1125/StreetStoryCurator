"""
fast_niche_detector.py — instant pre-grade niche recommendation.

Why this exists
---------------
The pre-grade picker wants to auto-recommend a photography niche in **under 3
seconds**, before the user starts a cull. The previous approach ran the full
grading pipeline in scan_mode through the SigLIP-2 subprocess — whose cold load
alone is ~12 s (measured 2026-06-30: 13 s for 4 images, 20 s for 8). It also
held the GPU lock and crashed in the fusion/bbox stage. It could never hit <3 s.

This module is a different mechanism entirely:
  - A small CLIP ViT-B/32 (models/ViT-B-32.pt, image + text, 512-d, ~350 MB) is
    loaded ONCE and kept WARM in the server process (CPU only — never touches the
    GPU, so it can't contend with or crash a grade).
  - Per-niche text anchors are built once from the 20-niche registry's pos_probes
    and cached to disk (cache/clip_niche_anchors.npz).
  - detect() encodes a small, size-adaptive sample of the folder and cosine-
    matches each image to the anchors. Fewer / smaller photos → fewer encodes →
    faster scan ("less data scans faster").

Warm cost (server already has torch imported): clip.load ~2.8 s the FIRST time,
then 0; encoding 8–16 small images on CPU ~1–2 s; text anchors precomputed. A
warm detection finishes well under 3 s. Call warmup() at startup to pay the
one-time load up front.

Public API
----------
  warmup()                         -> bool      # load model + anchors (idempotent)
  is_ready()                       -> bool
  detect(image_paths, sample_limit=0) -> dict | None
        {"preset": <registry slug>, "confidence": float, "scores": {slug: float},
         "sampled": int, "elapsed_ms": int}
"""
from __future__ import annotations

import threading
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT         = Path(__file__).resolve().parent.parent
_MODEL_ROOT   = _ROOT / "models"                       # clip.load finds ViT-B-32.pt here
_ANCHOR_CACHE = _ROOT / "cache" / "clip_niche_anchors.npz"

# Hard cap so even a 10k-image folder stays under budget; the sample is taken
# evenly across the folder so it stays representative. Smaller folders use fewer
# images and finish faster.
_MAX_SAMPLE    = 16
_MIN_SAMPLE    = 4        # never recommend off fewer than this (when available)
_DECODE_SIDE   = 256      # CLIP preprocess downscales to 224 — 256 is plenty, fast to decode
_DECODE_WORKERS = 8       # parallel decode (rawpy/PIL release the GIL during C decode)
_DECODE_BUDGET_S = 2.0    # stop collecting decodes past this so total stays < 3 s
_TEMP          = 100.0    # softmax temperature for the reported confidence

_load_lock  = threading.Lock()
_infer_lock = threading.Lock()

_model       = None
_preprocess  = None
_anchor_mat: Optional[np.ndarray] = None   # (K, 512) L2-normalised
_anchor_keys: list[str] = []
_ready       = False


# ── Model + anchors ──────────────────────────────────────────────────────────

def _registry_hash() -> str:
    """Hash the niche keys + pos_probes so cached anchors rebuild on registry edits."""
    from niche_registry import REGISTRY
    h = hashlib.md5()
    for k in sorted(REGISTRY):
        h.update(k.encode("utf-8"))
        for p in REGISTRY[k].get("pos_probes", []):
            h.update(p.encode("utf-8"))
    return h.hexdigest()[:12]


def _load_model():
    """Load CLIP ViT-B/32 on CPU (warm singleton). Uses the on-disk checkpoint."""
    import torch  # already imported by the server process
    import _clip_compat  # noqa: F401  — restores pkg_resources.packaging before clip
    import clip
    model, preprocess = clip.load(
        "ViT-B/32", device="cpu", download_root=str(_MODEL_ROOT)
    )
    model.eval()
    return model, preprocess


def _build_anchors(model) -> tuple[np.ndarray, list[str]]:
    """Encode each niche's pos_probes, average → L2-normalised (K, 512) anchor matrix."""
    import torch
    import clip
    from niche_registry import REGISTRY

    keys = list(REGISTRY.keys())
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for k in keys:
            probes = REGISTRY[k].get("pos_probes") or [REGISTRY[k]["mode_label"]]
            toks = clip.tokenize(probes, truncate=True)
            emb = model.encode_text(toks).float().numpy()        # (P, 512)
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            v = emb.mean(axis=0)
            v /= (np.linalg.norm(v) + 1e-9)
            rows.append(v.astype(np.float32))
    return np.stack(rows), keys


def _load_anchors(model) -> tuple[np.ndarray, list[str]]:
    """Return cached anchors if the registry is unchanged, else build + cache them."""
    want = _registry_hash()
    if _ANCHOR_CACHE.exists():
        try:
            data = np.load(_ANCHOR_CACHE, allow_pickle=True)
            if str(data.get("hash")) == want:
                return data["mat"].astype(np.float32), list(data["keys"])
        except Exception:
            pass
    mat, keys = _build_anchors(model)
    try:
        _ANCHOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(_ANCHOR_CACHE, mat=mat, keys=np.array(keys), hash=want)
    except Exception:
        pass
    return mat, keys


def warmup() -> bool:
    """Load the model + anchors once. Idempotent; safe to call from any thread."""
    global _model, _preprocess, _anchor_mat, _anchor_keys, _ready
    if _ready:
        return True
    with _load_lock:
        if _ready:
            return True
        try:
            t0 = time.time()
            model, preprocess = _load_model()
            mat, keys = _load_anchors(model)
            _model, _preprocess = model, preprocess
            _anchor_mat, _anchor_keys = mat, keys
            _ready = True
            print(f"[niche_detect] warm — CLIP ViT-B/32 + {len(keys)} anchors "
                  f"in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            print(f"[niche_detect] warmup failed: {e}")
            _ready = False
            return False


def is_ready() -> bool:
    return _ready


def release() -> None:
    """Free the CLIP model from RAM — call this before a grade so the ~0.7 GB it
    holds doesn't compete with SigLIP-2 / Qwen on memory-tight machines. The
    next detect() reloads it on demand (~3 s); the text anchors stay cached on
    disk so only the model weights are re-read."""
    global _model, _preprocess, _ready
    with _load_lock:
        if _model is None and not _ready:
            return
        _model = None
        _preprocess = None
        _ready = False
    import gc as _gc
    _gc.collect()
    print("[niche_detect] released — CLIP model freed for grading")


# ── Decode + sample ──────────────────────────────────────────────────────────

def _decode_small(path: str):
    """Decode any image (incl. RAW/HEIC) to a small RGB PIL image, or None."""
    try:
        from raw_support import RAW_EXTS, extract_embedded_preview
        ext = Path(path).suffix.lower()
        if ext in RAW_EXTS:
            img = extract_embedded_preview(path, "RGB")   # embedded preview, no demosaic
            if img is None:
                return None                               # unreadable RAW → caller skips it
        else:
            from PIL import Image
            img = Image.open(path)
            try:
                img.draft("RGB", (_DECODE_SIDE, _DECODE_SIDE))   # fast partial JPEG decode
            except Exception:
                pass
            img = img.convert("RGB")
        img.thumbnail((_DECODE_SIDE, _DECODE_SIDE))
        return img
    except Exception:
        return None


def _even_sample(paths: list[str], cap: int) -> list[str]:
    """Evenly spaced subsample of up to `cap` paths (fewer files → fewer encodes)."""
    n = len(paths)
    if n <= cap:
        return list(paths)
    idx = np.linspace(0, n - 1, cap).round().astype(int)
    return [paths[i] for i in dict.fromkeys(idx.tolist())]


# ── Detection ────────────────────────────────────────────────────────────────

def detect(image_paths: list[str], sample_limit: int = 0) -> Optional[dict]:
    """Recommend a niche for a folder by CLIP-matching a small adaptive sample.

    Returns a dict with the registry slug, confidence, top scores, and timing,
    or None if the detector is unavailable or nothing could be decoded.
    """
    if not warmup():
        return None
    paths = [p for p in (image_paths or []) if p]
    if not paths:
        return None

    t0 = time.time()
    cap = sample_limit if sample_limit and sample_limit > 0 else _MAX_SAMPLE
    sample = _even_sample(paths, cap)

    import torch
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Decode in parallel and stop collecting once the time budget is hit (keeping
    # a minimum), so even slow-to-decode RAW folders finish well under 3 s.
    tensors, used = [], []
    ex = ThreadPoolExecutor(max_workers=min(_DECODE_WORKERS, len(sample)))
    try:
        futs = {ex.submit(_decode_small, p): p for p in sample}
        for fut in as_completed(futs):
            img = fut.result()
            if img is not None:
                tensors.append(_preprocess(img))
                used.append(futs[fut])
            if (time.time() - t0) > _DECODE_BUDGET_S and len(tensors) >= _MIN_SAMPLE:
                break
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    if not tensors:
        return None

    with _infer_lock:
        with torch.no_grad():
            feats = _model.encode_image(torch.stack(tensors)).float().numpy()   # (N, 512)

    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    sims = feats @ _anchor_mat.T            # (N, K) cosine
    per_niche = sims.mean(axis=0)           # (K,)

    z = per_niche * _TEMP
    z -= z.max()
    e = np.exp(z)
    probs = e / (e.sum() + 1e-9)

    order = np.argsort(-per_niche)
    top = int(order[0])
    return {
        "preset":     _anchor_keys[top],
        "confidence": round(float(probs[top]), 3),
        "scores":     {_anchor_keys[i]: round(float(per_niche[i]), 4) for i in order[:5]},
        "sampled":    len(used),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
