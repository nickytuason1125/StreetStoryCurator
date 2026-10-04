"""Subject sharpness: smeared subjects are caught, pans and shallow focus are not.

The three cases the rule exists to separate (2026-10-03):
    pan            sharp subject, streaked background  → not smeared
    shallow focus  sharp subject, soft background      → not smeared
    missed shot    smeared subject                     → smeared
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
cv2 = pytest.importorskip("cv2")
import subject_sharpness as ss  # noqa: E402

BOX = [0.35, 0.3, 0.65, 0.7]


def _scene(seed=0, h=1000, w=1500):
    """Photo-like texture: blocks of varied contrast plus fine edges."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (h // 25, w // 25), dtype=np.uint8),
                      (w, h), interpolation=cv2.INTER_NEAREST)
    fine = rng.integers(0, 2, (h, w), dtype=np.uint8) * 60
    return cv2.addWeighted(base, 0.8, fine, 0.5, 0)


def _motion(img, k=21, horizontal=True):
    kern = np.zeros((k, k), np.float32)
    if horizontal:
        kern[k // 2, :] = 1.0 / k
    else:
        kern[:, k // 2] = 1.0 / k
    return cv2.filter2D(img, -1, kern)


def _paste_subject(bg, subj):
    h, w = bg.shape
    x1, y1, x2, y2 = int(BOX[0] * w), int(BOX[1] * h), int(BOX[2] * w), int(BOX[3] * h)
    out = bg.copy()
    out[y1:y2, x1:x2] = subj[y1:y2, x1:x2]
    return out


def test_pan_sharp_subject_streaked_background_is_not_smeared():
    sharp = _scene()
    img = _paste_subject(_motion(sharp, 31), sharp)
    m = ss.measure(img, [BOX])
    assert m["source"] == "box"
    assert not ss.is_smeared(m), m


def test_shallow_focus_sharp_subject_soft_background_is_not_smeared():
    sharp = _scene(1)
    img = _paste_subject(cv2.GaussianBlur(sharp, (0, 0), 6), sharp)
    assert not ss.is_smeared(ss.measure(img, [BOX]))


def test_motion_smeared_subject_is_smeared():
    sharp = _scene(2)
    img = _paste_subject(sharp, _motion(sharp, 21))
    m = ss.measure(img, [BOX])
    assert ss.is_smeared(m), m


def test_vertical_motion_blur_is_caught_too():
    sharp = _scene(3)
    img = _paste_subject(sharp, _motion(sharp, 21, horizontal=False))
    assert ss.is_smeared(ss.measure(img, [BOX]))


def test_no_subject_falls_back_to_sharpest_region():
    m = ss.measure(_scene(4), [])
    assert m["source"] == "peak" and not ss.is_smeared(m)
    assert ss.is_smeared(ss.measure(_motion(_scene(4), 21), []))


def test_cap_only_lowers_and_only_when_smeared():
    smeared = {"subject": ss.SMEARED_BELOW - 0.05, "subject_streak": 0.3}
    sharp = {"subject": ss.SMEARED_BELOW + 0.05, "subject_streak": 0.02}
    assert ss.apply_cap(0.70, smeared) == ss.SMEARED_CAP
    assert ss.apply_cap(0.30, smeared) == 0.30          # never pushed lower
    assert ss.apply_cap(0.70, sharp) == 0.70
    assert ss.apply_cap(0.70, None) == 0.70             # no measurement → no cap
    assert ss.apply_cap(0.70, {"subject": None}) == 0.70


def test_cap_lands_in_weak():
    assert ss.SMEARED_CAP < 0.41


def test_pan_is_detected():
    sharp = _scene(5)
    assert ss.is_pan(ss.measure(_paste_subject(_motion(sharp, 31), sharp), [BOX]))


def test_static_scene_and_shallow_focus_are_not_pans():
    sharp = _scene(6)
    assert not ss.is_pan(ss.measure(sharp, [BOX]))
    assert not ss.is_pan(ss.measure(_paste_subject(cv2.GaussianBlur(sharp, (0, 0), 6), sharp), [BOX]))


def test_failed_pan_smeared_subject_is_not_a_pan():
    sharp = _scene(7)
    img = _motion(sharp, 31)             # everything smeared, subject included
    m = ss.measure(img, [BOX])
    assert not ss.is_pan(m) and ss.is_smeared(m)


def test_even_softness_is_not_capped_unless_hopeless():
    """Soft manual-focus / vintage portraits: soft evenly, not smeared."""
    soft_even = {"subject": ss.SMEARED_BELOW - 0.05, "subject_streak": 0.01}
    assert not ss.is_smeared(soft_even)
    assert ss.is_smeared({"subject": ss.DEFOCUS_BELOW - 0.05, "subject_streak": 0.01})
    assert not ss.is_smeared({"subject": ss.SMEARED_BELOW - 0.05})   # direction unknown


def test_motion_smear_reads_directional_and_defocus_reads_even():
    sharp = _scene(8)
    motion = ss.measure(_paste_subject(sharp, _motion(sharp, 21)), [BOX])
    defocus = ss.measure(_paste_subject(sharp, cv2.GaussianBlur(sharp, (0, 0), 3)), [BOX])
    assert motion["subject_streak"] >= ss.MOTION_GAP, motion
    assert defocus["subject_streak"] < ss.MOTION_GAP, defocus


SMALL = [0.05, 0.05, 0.2, 0.25]     # a secondary box away from BOX


def test_sharp_bystander_does_not_rescue_a_smeared_main_subject():
    """DSC09273: main subject smeared, a smaller person behind is sharp."""
    sharp = _scene(9)
    img = _paste_subject(sharp, _motion(sharp, 21))       # main box smeared, rest sharp
    m = ss.measure(img, [BOX, SMALL])                     # main subject listed first
    assert ss.is_smeared(m), m


def test_pan_judges_the_tracked_subject_not_the_biggest_box():
    """DSC08943: in a pan the biggest box is a smeared background ship."""
    sharp = _scene(10)
    img = _motion(sharp, 31)                              # camera move smears everything...
    h, w = img.shape
    x1, y1, x2, y2 = int(SMALL[0] * w), int(SMALL[1] * h), int(SMALL[2] * w), int(SMALL[3] * h)
    img[y1:y2, x1:x2] = sharp[y1:y2, x1:x2]               # ...except the tracked subject
    m = ss.measure(img, [BOX, SMALL])                     # smeared big box listed first
    assert not ss.is_smeared(m) and ss.is_pan(m), m


def test_failed_pan_is_weak_and_never_boosted():
    """DSC08906: background streaked, tracked boat smeared along the pan."""
    failed = {"subject": 0.62, "streak": 0.40, "subject_streak": 0.34}
    good = {"subject": 0.68, "streak": 0.33, "subject_streak": 0.10}
    between = {"subject": 0.63, "streak": 0.30, "subject_streak": 0.18}
    assert ss.is_smeared(failed) and not ss.is_pan(failed)
    assert ss.is_pan(good) and not ss.is_smeared(good)
    assert not ss.is_pan(between) and not ss.is_smeared(between)


def test_score_paths_reports_living_subject(monkeypatch):
    """The taste learner needs to know whether a person/animal is the subject."""
    import types
    fake = types.SimpleNamespace(detect_subjects_from_arrays=lambda items, conf: {
        "a": [{"bbox": BOX, "conf": 0.9, "label": "person"}],
        "b": [{"bbox": BOX, "conf": 0.9, "label": "truck"}]})
    monkeypatch.setitem(sys.modules, "dfine_detector", fake)
    monkeypatch.setattr(ss, "_decode", lambda p: _scene(11))
    out = ss.score_paths(["a", "b"])
    assert out["a"]["living"] is True and out["b"]["living"] is False


def test_strong_directional_smear_is_weak_even_when_detail_survives():
    """2026-10-04: shake on a moving boat left subjects at 0.53-0.63 sharpness
    but smeared 0.20-0.32 in one direction (DSC08951/08953/08932, graded
    Strong). Direction is the tell; a good pan is the only exception."""
    shaken = {"subject": 0.60, "subject_streak": 0.25, "streak": 0.10}
    assert ss.is_smeared(shaken)
    good_pan = {"subject": 0.68, "subject_streak": 0.10, "streak": 0.33}
    assert not ss.is_smeared(good_pan)
    mild = {"subject": 0.70, "subject_streak": 0.15, "streak": 0.05}
    assert not ss.is_smeared(mild)
