"""Per-photo grading progress must end BELOW the image-quality band.

2026-10-04: SpecVLM mapped its per-photo ticks to 0.51→0.86, but the IQA
phase that follows reports 0.66→0.84. The runner's monotonic clamp then held
the bar at 86% for the entire IQA pass (~25 min on 4,150 photos) — "stuck at
86%" while IQA was scoring ~150 photos a minute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from specvlm_pipeline import SpecVLMPipeline  # noqa: E402

IQA_BAND_START = 0.66


def test_specvlm_progress_stays_below_iqa_band(tmp_path):
    rng = np.random.default_rng(0)
    n = 12
    paths = [str(tmp_path / f"{i}.jpg") for i in range(n)]
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    emits: list[float] = []
    SpecVLMPipeline().grade_images(
        paths,
        progress=lambda f, d: emits.append(float(f)),
        scan_mode=True,
        embeddings=unit(rng.normal(size=(n, 16)).astype(np.float32)),
        pos_text_embs=unit(rng.normal(size=(4, 16)).astype(np.float32)),
        neg_text_embs=unit(rng.normal(size=(4, 16)).astype(np.float32)),
    )
    assert emits, "grader reported no progress"
    assert max(emits) < IQA_BAND_START, max(emits)
    assert emits == sorted(emits)
