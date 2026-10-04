"""Scorecard: 3-class agreement between machine grades and the photographer's stars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import accuracy as acc  # noqa: E402


def test_star_mapping_matches_ratings_store():
    import ratings_store as rs
    for s in range(1, 6):
        label = rs.grade_for_stars(s)
        want = 2 if "Strong" in label else 1 if "Mid" in label else 0
        assert acc.stars_to_class(s) == want


def test_score_to_class_uses_given_lines():
    assert acc.score_to_class(0.60, 0.58, 0.39) == 2
    assert acc.score_to_class(0.45, 0.58, 0.39) == 1
    assert acc.score_to_class(0.38, 0.58, 0.39) == 0


def test_agreement_and_confusion():
    true, pred = [2, 1, 0, 1], [2, 2, 0, 1]
    assert acc.agreement(pred, true) == 0.75
    cm = acc.confusion(pred, true)
    assert cm[1][2] == 1 and cm[2][2] == 1 and sum(map(sum, cm)) == 4


def test_paired_bootstrap_detects_a_real_gain_and_not_a_tie():
    true = [0, 1, 2] * 40
    groups = [f"s{i // 12}" for i in range(120)]
    better = list(true)
    worse = [1] * 120
    up = acc.paired_bootstrap(true, worse, better, groups, n=500, seed=1)
    assert up["lo"] > 0
    tie = acc.paired_bootstrap(true, better, better, groups, n=200, seed=1)
    assert tie["delta"] == 0 and tie["lo"] <= 0 <= tie["hi"]


def test_shoot_of_is_parent_folder():
    assert acc.shoot_of(r"F:\DCIM\100MSDCF\DSC1.ARW") == r"F:\DCIM\100MSDCF".lower()
