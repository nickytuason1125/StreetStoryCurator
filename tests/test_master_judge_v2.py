"""Learner v2: new sharpness/pan/subject signals, whole-shoot holdout,
promotion only on a proven 3-class agreement gain."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import master_judge as mj  # noqa: E402


def _row(i, shoot, stars, score, smear=0.0):
    bd = {k: 0.5 for k in mj.FEATURES}
    bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}
    bd.update({"_subject_sharp": 0.7, "_subject_streak": smear, "_bg_streak": 0.1,
               "_living": 1.0})
    return {"path": f"C:/{shoot}/{i}.ARW", "stars": stars, "score": score,
            "breakdown": bd, "source": "", "from_snapshot": False}


def test_extra_features_are_in_the_design():
    for k in ("_subject_sharp", "_subject_streak", "_bg_streak", "_pan", "_smeared_subject", "_living"):
        assert k in mj.EXTRA and k in mj.DESIGN


def test_missing_flags_default_to_false_not_dropped():
    bd = {k: 0.5 for k in mj.FEATURES}
    bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}
    bd.update({"_subject_sharp": 0.7, "_subject_streak": 0.0, "_bg_streak": 0.0, "_living": 0.0})
    assert np.isfinite(mj.feature_vector(bd)).all()      # _pan / _smeared_subject absent = False


def test_missing_measurement_is_not_imputed():
    bd = {k: 0.5 for k in mj.FEATURES}
    bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}          # no _subject_sharp etc.
    assert not np.isfinite(mj.feature_vector(bd)).all()


def test_holdout_never_splits_a_folder():
    rows = [_row(i, f"s{i % 6}", 1 + i % 5, 0.4 + 0.1 * (i % 5)) for i in range(120)]
    groups = [mj._shoot_day(r["path"]) for r in rows]
    folds = mj._group_folds(groups, k=5, seed=1)
    assert len(folds) == 120 and set(folds) <= set(range(5))
    for g in set(groups):                      # a group lives in exactly one fold
        assert len({f for f, gg in zip(folds, groups) if gg == g}) == 1


def test_one_dominant_shoot_does_not_swallow_the_training_set():
    """Real data: one LX3 folder held 514 of 826 ratings, and a single
    25% 'whole-folder' holdout left 156 photos to train on."""
    rows = ([_row(i, "big", 1 + i % 5, 0.5) for i in range(500)]
            + [_row(1000 + i, f"s{i % 5}", 1 + i % 5, 0.5) for i in range(100)])
    groups = [mj._shoot_day(r["path"]) for r in rows]
    folds = mj._group_folds(groups, k=5, seed=1)
    sizes = [folds.count(f) for f in range(5)]
    assert max(sizes) == 500 and sum(1 for z in sizes if z > 0) >= 3


def test_judge_that_cannot_beat_baseline_is_not_promoted(tmp_path):
    rng = np.random.default_rng(0)
    rows = [_row(i, f"s{i % 6}", int(rng.integers(1, 6)), float(rng.random())) for i in range(150)]
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is False


def test_judge_learns_smear_signal_and_promotes(tmp_path):
    """The incumbent's score carries no smear information (it only knows
    'pretty good, 0.5-0.7'); the photographer marks every smeared frame
    down. A judge that sees _subject_streak must win on held-out shoots."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(240):
        smeared = i % 3 == 0
        stars = 2 if smeared else 5
        rows.append(_row(i, f"s{i % 8}", stars, float(rng.uniform(0.5, 0.7)),
                         smear=0.3 if smeared else 0.02))
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is True and out["agree_ci"][0] > 0


def test_calibration_only_judge_promotes_when_incumbent_is_miscalibrated(tmp_path):
    """Real 2026-10-03 shape: the incumbent RANKS well but scores the
    photographer's Mid frames into Weak. New features add nothing, yet a
    recalibration of the incumbent's own score provably helps and ships."""
    rng = np.random.default_rng(5)
    rows = []
    for i in range(400):
        stars = int(rng.choice([2, 3, 3, 3, 4, 5]))
        score = {2: 0.33, 3: 0.37, 4: 0.55, 5: 0.66}[stars] + float(rng.normal(0, 0.03))
        rows.append(_row(i, f"s{i % 20}", stars, score))
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is True and out["kind"] == "calibration"
    assert out["features"] == ["(machine score)"]
    assert out["agree_judge"] > out["agree_base"] and out["agree_ci"][0] > 0
    assert mj.load(tmp_path / "j.json") is not None


def test_lost_reason_cites_agreement_not_rho(tmp_path):
    rng = np.random.default_rng(0)
    rows = [_row(i, f"s{i % 6}", int(rng.integers(1, 6)), float(rng.random())) for i in range(150)]
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is False and "agreement" in out["reason"]


def test_recorded_blend_weight_is_the_one_grading_applies(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(400):
        stars = int(rng.choice([2, 3, 3, 3, 4, 5]))
        score = {2: 0.33, 3: 0.37, 4: 0.55, 5: 0.66}[stars] + float(rng.normal(0, 0.03))
        rows.append(_row(i, f"s{i % 20}", stars, score))
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["blend_weight"] == round(mj._weight_from(out), 4)
