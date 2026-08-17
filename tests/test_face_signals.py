"""
Face + subject-focus signals.

Two failure modes matter more than accuracy here:

  1. Claiming an eye-state verdict we cannot compute. YuNet returns landmark
     POSITIONS; a closed eye still has an eye landmark. A wrong "eyes closed"
     silently discards a keeper, so the module must refuse to answer.
  2. Emitting a confident focus_ratio for a face too small to measure. The
     Laplacian of a 30px face is resampling noise, and a number that looks
     authoritative is worse than no number.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_face_signals.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import face_signals as fs  # noqa: E402


def _noise(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── the thing we must not fake ──────────────────────────────────────────────
def test_eye_state_is_explicitly_unsupported():
    assert fs.eye_state_available() is False


def test_metrics_never_claim_an_eye_verdict():
    m = fs.face_metrics(_noise(600, 900))
    assert m["eye_state_supported"] is False
    assert not any("eye" in k and "state" not in k for k in m if k != "faces")


# ── degradation ─────────────────────────────────────────────────────────────
def test_none_image_returns_empty_not_an_exception():
    m = fs.face_metrics(None)
    assert m["faces_detected"] == 0 and m["focus_ratio"] is None


def test_no_faces_gives_no_focus_verdict():
    """A landscape must not get subject_in_focus=False — there is no subject."""
    m = fs.face_metrics(_noise(600, 900))
    assert m["faces_detected"] == 0
    assert m["subject_in_focus"] is None and m["focus_ratio"] is None


def test_tiny_and_degenerate_images_do_not_crash():
    for h, w in ((1, 1), (5, 5), (19, 19), (20, 20), (64, 1)):
        m = fs.face_metrics(_noise(h, w))
        assert m["faces_detected"] == 0


def test_detect_faces_never_raises_on_garbage():
    for bad in (None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8)):
        assert fs.detect_faces(bad) == []


def test_unreadable_path_is_flagged(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"not an image")
    m = fs.metrics_for_path(str(p))
    assert m.get("unreadable") is True and m["faces_detected"] == 0


# ── the small-face guard ────────────────────────────────────────────────────
def test_focus_ratio_withheld_for_faces_below_the_size_floor(monkeypatch):
    """Face reported, focus verdict withheld — the distinction the guard exists for."""
    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [10.0, 10.0, 30.0, 30.0], "confidence": 0.9,
         "landmarks": {}, "area_frac": 0.002}])
    m = fs.face_metrics(_noise(600, 900))
    assert m["faces_detected"] == 1
    assert m["focus_ratio"] is None and m["subject_in_focus"] is None


def test_area_floor_withholds_verdict_even_when_pixel_floor_passes(monkeypatch):
    """A 120px face in a 4000px frame clears the pixel floor but is still <1% of
    the frame — the band where 39% of real photos were wrongly flagged soft,
    because skin is smooth rather than because the lens missed."""
    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [10.0, 10.0, 120.0, 120.0], "confidence": 0.9,
         "landmarks": {}, "area_frac": 0.004}])
    m = fs.face_metrics(_noise(2000, 3000))
    assert min(120, 120) >= fs._MIN_FOCUS_FACE_PX, "pixel floor should pass"
    assert m["faces_detected"] == 1
    assert m["focus_ratio"] is None, "area floor must still withhold the verdict"


def test_focus_ratio_computed_for_a_large_face(monkeypatch):
    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [50.0, 50.0, 300.0, 300.0], "confidence": 0.9,
         "landmarks": {}, "area_frac": 0.16}])
    m = fs.face_metrics(_noise(600, 900))
    assert m["focus_ratio"] is not None
    assert isinstance(m["subject_in_focus"], bool)


def test_a_blurred_subject_scores_below_a_sharp_one(monkeypatch):
    """The signal must actually track focus, not just produce a number."""
    import cv2
    rng = np.random.default_rng(7)
    scene = rng.integers(0, 255, (600, 900, 3), dtype=np.uint8)

    sharp = scene.copy()
    soft = scene.copy()
    soft[50:350, 50:350] = cv2.GaussianBlur(soft[50:350, 50:350], (31, 31), 12)

    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [50.0, 50.0, 300.0, 300.0], "confidence": 0.9,
         "landmarks": {}, "area_frac": 0.16}])

    assert (fs.face_metrics(soft)["focus_ratio"]
            < fs.face_metrics(sharp)["focus_ratio"]), "blurring the face must lower the ratio"


def test_largest_face_is_the_one_measured(monkeypatch):
    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [0.0, 0.0, 80.0, 80.0], "confidence": 0.9, "landmarks": {}, "area_frac": 0.01},
        {"box": [100.0, 100.0, 300.0, 300.0], "confidence": 0.9, "landmarks": {}, "area_frac": 0.16},
    ])
    m = fs.face_metrics(_noise(600, 900))
    assert m["faces_detected"] == 2
    assert m["largest_face_frac"] == pytest.approx(0.16)


def test_box_clamped_to_image_bounds(monkeypatch):
    """A box running off the edge must not produce an empty slice or an error."""
    monkeypatch.setattr(fs, "detect_faces", lambda bgr, conf=fs._CONF: [
        {"box": [800.0, 500.0, 400.0, 400.0], "confidence": 0.9,
         "landmarks": {}, "area_frac": 0.2}])
    m = fs.face_metrics(_noise(600, 900))
    assert m["faces_detected"] == 1 and m["face_sharpness"] >= 0.0
