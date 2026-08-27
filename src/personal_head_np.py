"""
personal_head_np.py — torch-free scorer for the PersonalHead taste model.

Why this exists
---------------
The grade worker is deliberately CUDA-free: SigLIP and IQA run in isolated
subprocesses and the parent does no GPU work. But Step 5 called
`personal_head.score()`, and that module imports torch at module scope — 349 MB
(measured) pulled into the parent purely to evaluate a
Linear(D,256)-ReLU-Linear(256,64)-ReLU-Linear(64,1)-Sigmoid MLP. Three matmuls.

That 349 MB sat resident for the whole run, including the window where the
2.71 GB encode subprocess is loading, which is exactly the moment that decides
whether the machine pages. Doing the forward pass in numpy removes it entirely.

Weight mirror
-------------
The trained weights live in a torch .pt (a pickle), so reading them needs torch.
This module keeps a numpy mirror (.npz) beside it and regenerates it whenever
the .pt is newer — so the ONE torch import happens on the first grade after
training, and never again. Staleness is checked by mtime, not existence: after
retraining, a stale mirror would silently score with the old taste model.

nn.Linear stores weight as (out, in), so each layer is `x @ W.T + b`.
Verified numerically identical to personal_head.score() — see
tests/test_personal_head_np.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_WEIGHTS_PATH = Path("cache/personal_head.pt")
_NPZ_PATH     = Path("cache/personal_head.npz")

_LAYER_KEYS = ("net.0.weight", "net.0.bias",
               "net.2.weight", "net.2.bias",
               "net.4.weight", "net.4.bias")

_cache: Optional[dict] = None
_cache_mtime: float = -1.0


def _mirror_is_stale() -> bool:
    """True when the .npz is missing or older than the .pt it mirrors."""
    if not _NPZ_PATH.exists():
        return True
    try:
        return _NPZ_PATH.stat().st_mtime < _WEIGHTS_PATH.stat().st_mtime
    except OSError:
        return True


def export_mirror() -> bool:
    """Regenerate the numpy mirror from the .pt. Imports torch (once)."""
    if not _WEIGHTS_PATH.exists():
        return False
    try:
        import torch                              # the one and only torch use
        # weights_only=True — see the note at the matching load in
        # personal_head.py. A state_dict needs no pickle reduction.
        saved = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        arrays = {k: np.asarray(v.detach().cpu().numpy(), dtype=np.float32)
                  for k, v in saved.items()}
        missing = [k for k in _LAYER_KEYS if k not in arrays]
        if missing:
            print(f"[personal_head_np] mirror not written — .pt lacks {missing}")
            return False
        _NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(_NPZ_PATH), **arrays)
        print(f"[personal_head_np] weight mirror refreshed -> {_NPZ_PATH.name}")
        return True
    except Exception as exc:
        print(f"[personal_head_np] mirror export failed ({exc})")
        return False


def _weights() -> Optional[dict]:
    """Load (and memoise) the mirror, refreshing it if the .pt moved on."""
    global _cache, _cache_mtime
    if not _WEIGHTS_PATH.exists():
        return None
    if _mirror_is_stale() and not export_mirror():
        return None
    try:
        mtime = _NPZ_PATH.stat().st_mtime
        if _cache is not None and mtime == _cache_mtime:
            return _cache
        z = np.load(str(_NPZ_PATH))
        w = {k: z[k] for k in _LAYER_KEYS}
        _cache, _cache_mtime = w, mtime
        return w
    except Exception as exc:
        print(f"[personal_head_np] mirror unreadable ({exc})")
        return None


def head_dim() -> Optional[int]:
    """Input width the trained head expects, or None if there is no usable head.

    Callers use this to detect a QUALITY-TIER MISMATCH: the head is trained on
    one encoder's embedding width (e.g. 1536 for 'Pro'), so running a different
    tier makes those weights meaningless. The torch implementation responds by
    silently discarding them and initialising a RANDOM head — whose output then
    gets blended into every score at the 0.20 floor. That is not a neutral
    fallback, it is noise. Skip the blend instead.
    """
    w = _weights()
    if w is None:
        return None
    try:
        return int(w["net.0.weight"].shape[1])
    except Exception:
        return None


def score(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """(N,) preference scores in [0,1], or None if no usable trained head.

    None means "caller should fall back" — it is NOT a neutral score. Returning
    0.5 here would look like a confident-neutral taste vote and silently flatten
    the blend, so the decision is left to the caller.
    """
    w = _weights()
    if w is None:
        return None
    try:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != w["net.0.weight"].shape[1]:
            print(f"[personal_head_np] dim mismatch: embeddings {x.shape} vs "
                  f"head {w['net.0.weight'].shape[1]} — falling back")
            return None
        h = np.maximum(x @ w["net.0.weight"].T + w["net.0.bias"], 0.0)
        h = np.maximum(h @ w["net.2.weight"].T + w["net.2.bias"], 0.0)
        o = h @ w["net.4.weight"].T + w["net.4.bias"]
        return (1.0 / (1.0 + np.exp(-o))).squeeze(-1).astype(np.float32)
    except Exception as exc:
        print(f"[personal_head_np] forward failed ({exc}) — falling back")
        return None
