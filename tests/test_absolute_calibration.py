"""
A photo's grade must not depend on what it was culled alongside.

_calibrate min-max stretched every batch (min -> 0.10, max -> 0.95), so the same
photo scored 0.10 in a strong batch and 0.95 in a weak one - Weak or Strong from
identical pixels. That is per-batch relative grading on the default path, which
CLAUDE.md:79 forbids: "a photo reaches Strong only on actual fused quality".

It survived because the 2026-06 absolute-grading work removed quantile
calibration from grade_pipeline_v2 and stopped there; this min-max lived in the
CLIP scorer.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_absolute_calibration.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import specvlm_pipeline as sp          # noqa: E402


# Two batches containing the SAME photo (raw discriminant 0.020) among very
# different company.
STRONG_BATCH = np.array([0.020, 0.022, 0.025, 0.028, 0.030])
WEAK_BATCH = np.array([0.020, 0.010, 0.008, 0.005, 0.002])

# Anchors standing in for derived ones, spanning a plausible discriminant range.
ANCHORS = (-0.010, 0.040)


def test_same_photo_scores_the_same_in_any_batch():
    """The defining property. This is what the bug broke."""
    a = sp._calibrate(STRONG_BATCH, anchors=ANCHORS)[0]
    b = sp._calibrate(WEAK_BATCH, anchors=ANCHORS)[0]
    assert a == pytest.approx(b), (
        f"same photo scored {a:.2f} beside strong work and {b:.2f} beside weak")


def test_a_uniformly_strong_batch_manufactures_no_rejects():
    """Culling only good photos must not invent a Weak one."""
    out = sp._calibrate(np.array([0.030, 0.031, 0.032, 0.033]), anchors=ANCHORS)
    assert out.min() > 0.41, f"a strong-only batch produced {out.min():.2f}"


def test_a_uniformly_weak_batch_manufactures_no_keepers():
    out = sp._calibrate(np.array([0.000, 0.001, 0.002, 0.003]), anchors=ANCHORS)
    assert out.max() < 0.60, f"a weak-only batch produced {out.max():.2f}"


def test_ordering_is_preserved():
    raw = np.array([0.005, 0.030, 0.012, 0.022])
    out = sp._calibrate(raw, anchors=ANCHORS)
    assert list(np.argsort(out)) == list(np.argsort(raw))


def test_scores_stay_in_range_far_outside_the_anchors():
    """Anchors are percentiles, so real photos WILL fall outside them."""
    out = sp._calibrate(np.array([-5.0, 5.0]), anchors=ANCHORS)
    assert out.min() >= 0.10 and out.max() <= 0.95


def test_single_image_uses_the_same_scale_as_a_batch():
    """A one-photo run must agree with the same photo inside a batch."""
    alone = sp._calibrate(np.array([0.020]), anchors=ANCHORS)[0]
    in_batch = sp._calibrate(STRONG_BATCH, anchors=ANCHORS)[0]
    assert alone == pytest.approx(in_batch)


def test_without_anchors_it_falls_back_and_says_so(capsys):
    """No anchors must degrade LOUDLY to the old behaviour, never silently.

    Silently grading against a wrong or absent scale is worse than the bug.
    """
    out = sp._calibrate(STRONG_BATCH, anchors=None)
    assert len(out) == len(STRONG_BATCH)
    assert "anchor" in capsys.readouterr().out.lower()
