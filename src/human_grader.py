"""
human_grader.py -- aesthetic ("human perception") score via the LAION aesthetic predictor.

Sourced through pyiqa's `laion_aes` metric -- the same LAION CLIP+MLP aesthetic model the
module always intended to use, but whose weights pyiqa self-hosts and auto-caches. The
previous implementation downloaded `ava1-l14-linearMSE.pth` from a Hugging Face URL on
every startup; that URL is now GATED (HTTP 401), so loading silently failed and the score
fell back to a constant 0.5. Routing through pyiqa removes the gated dependency entirely
(and drops a redundant manual ViT-L/14 CLIP load), so this cannot break the same way again.

The metric is created lazily on first score and cached as a process singleton -- importing
this module is cheap and never triggers a download. If pyiqa is somehow unavailable the
score degrades to a neutral 0.5 with a single clear warning, never a crash.

Public API (unchanged): get_human_aesthetic_score(img_path) -> float in [0, 1]
"""
from __future__ import annotations

import logging

import torch

_log = logging.getLogger(__name__)

# AVA aesthetic scores roughly span 1-10; the legacy code mapped the meaningful
# 4.0-7.5 band onto 0-1. Kept identical so the 0.2-weighted human_perception term
# and the competition grade thresholds stay on the same scale.
_AES_LO = 4.0
_AES_HI = 7.5

_metric = None        # cached pyiqa InferenceModel, or None if unavailable
_init_done = False    # attempt initialization (and warn) at most once per process


def _get_metric():
    """Lazily build and cache the pyiqa laion_aes metric. Returns None if unavailable."""
    global _metric, _init_done
    if _init_done:
        return _metric
    _init_done = True
    try:
        # pyiqa transitively imports openai-clip for its CLIP-based metrics, so the
        # pkg_resources.packaging shim must run BEFORE pyiqa is imported here too.
        try:
            from . import _clip_compat  # noqa: F401
        except ImportError:
            import _clip_compat  # noqa: F401
        import pyiqa
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _metric = pyiqa.create_metric("laion_aes", device=device)
        if hasattr(_metric, "eval"):
            _metric.eval()
        _log.info("human_grader: pyiqa laion_aes ready on %s", device)
    except Exception as e:
        _metric = None
        _log.warning(
            "human_grader: pyiqa laion_aes unavailable (%s) -- human_perception will "
            "return neutral 0.5. Ensure `pyiqa` is installed (it is in requirements.txt).",
            e,
        )
    return _metric


def get_human_aesthetic_score(img_path) -> float:
    """Aesthetic score in [0, 1] for one image. Returns 0.5 if the metric is unavailable."""
    metric = _get_metric()
    if metric is None:
        return 0.5
    try:
        with torch.no_grad():
            out = metric(str(img_path))
        raw = float(out.item()) if out.numel() == 1 else float(out.flatten()[0].item())
        return max(0.0, min((raw - _AES_LO) / (_AES_HI - _AES_LO), 1.0))
    except Exception as e:
        _log.warning("human_grader: scoring failed for %s (%s) -- returning 0.5", img_path, e)
        return 0.5
