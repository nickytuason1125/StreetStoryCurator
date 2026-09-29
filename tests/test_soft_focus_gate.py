"""Soft-Focus Gate (Step 5c) must not grade on a curve.

Before 2026-09-28 the gate tested the fine-art similarity AFTER Step 4c
stretched it to each batch's own min..max, so the top slice of every folder
got +0.15 whatever its real similarity: 0226-20 (raw 0.129) became 0.811 and
was boosted to Strong. On the calibrated Pro encoder the gate now reads the
raw similarity against one fixed threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline_stages import soft_focus_gate  # noqa: E402

THRESH = 0.10


def _stretch(raw):
    raw = np.asarray(raw, dtype=float)
    return (raw - raw.min()) / max(raw.max() - raw.min(), 1e-4)


def _boosted_first(batch_raw, absolute):
    """Score of photo 0 (raw 0.08, base 0.55) when graded inside `batch_raw`."""
    raw = np.array([0.08] + list(batch_raw))
    scores = np.full(len(raw), 0.55)
    out, _ = soft_focus_gate(scores, raw, _stretch(raw), absolute=absolute, raw_thresh=THRESH)
    return out[0]


def test_same_photo_same_verdict_in_any_batch_on_pro():
    weak_batch = [0.00, 0.01, 0.02]          # photo 0 is the most fine-art here
    strong_batch = [0.12, 0.13, 0.14]        # photo 0 is the least fine-art here
    assert _boosted_first(weak_batch, absolute=True) == _boosted_first(strong_batch, absolute=True) == 0.55


def test_legacy_stretched_test_was_batch_relative():
    # Documents WHY the change was needed: the same photo flips with its batch.
    assert _boosted_first([0.00, 0.01, 0.02], absolute=False) > 0.55
    assert _boosted_first([0.12, 0.13, 0.14], absolute=False) == 0.55


def test_absolute_gate_boosts_above_threshold_only_from_mid_band():
    raw = np.array([0.129, 0.129, 0.05])
    scores = np.array([0.62, 0.45, 0.62])
    out, n = soft_focus_gate(scores, raw, _stretch(raw), absolute=True, raw_thresh=THRESH)
    assert n == 1
    np.testing.assert_allclose(out, [0.77, 0.45, 0.62])
    np.testing.assert_allclose(scores, [0.62, 0.45, 0.62])   # input untouched


def test_boost_is_capped_at_one():
    out, _ = soft_focus_gate(np.array([0.95]), np.array([0.2]), np.array([1.0]),
                             absolute=True, raw_thresh=THRESH)
    assert out[0] == 1.0
