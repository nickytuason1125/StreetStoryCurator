"""
A report must never dress up a degenerate input as a real result.

The grade-accuracy audit printed `Spearman rho = +nan` for section 3 and then,
straight-faced, `taste learning moved agreement away from your judgements by
+nan`. Both lines are false: nothing was measured. The cause is that every
rated photo carried the SAME personal_score, so the rank vector has zero
variance and the correlation is undefined rather than bad.

Section 2 had the same defect one function away: an empty band yields a NaN
mean, every NaN comparison is False, so `Strong >= Mid >= Weak` reported FAIL
for a catalog that had simply never produced a Weak photo.

These lock in the distinction the report has to make:

  1. Undefined (no variance / no data) is reported as undefined, never as a
     number and never as a verdict.
  2. A genuine result still reads exactly as it did before.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_accuracy_report.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import scripts_accuracy_report as rep  # noqa: E402


# ── 1. The correlation itself ────────────────────────────────────────────────

def test_spearman_is_nan_when_one_side_has_no_variance():
    """A constant vector has no ranking, so no rank correlation exists."""
    assert math.isnan(rep.spearman([0.5] * 6, [1, 2, 3, 4, 5, 5]))


def test_spearman_still_measures_a_real_relationship():
    rho = rep.spearman([0.1, 0.2, 0.3, 0.4, 0.5], [1, 2, 3, 4, 5])
    assert rho == pytest.approx(1.0)


# ── 2. Section 3 rendering: personal shift ───────────────────────────────────

def test_personal_shift_names_the_cause_instead_of_printing_nan():
    """This is the exact shape of the bug: 124 photos, one constant score."""
    ppairs = [(0.5, s) for s in ([1, 2, 3, 4, 5] * 24 + [3, 3, 3, 3])]
    lines = rep.personal_shift_lines(ppairs, rho=0.231)
    text = "\n".join(lines)

    assert "nan" not in text.lower()
    assert "constant" in text.lower()
    assert str(len(ppairs)) in text
    # It must NOT claim taste learning moved anything.
    assert "moved agreement" not in text


def test_personal_shift_reports_a_real_shift_normally():
    ppairs = [(0.1, 1), (0.2, 2), (0.3, 3), (0.4, 4), (0.5, 5)]
    text = "\n".join(rep.personal_shift_lines(ppairs, rho=0.231))

    assert "nan" not in text.lower()
    assert "moved agreement toward" in text
    assert "+1.000" in text


def test_personal_shift_handles_a_missing_overall_rho():
    """Section 1 can be empty; section 3 must not compare against nothing."""
    ppairs = [(0.1, 1), (0.2, 2), (0.3, 3)]
    text = "\n".join(rep.personal_shift_lines(ppairs, rho=None))

    assert "nan" not in text.lower()
    assert "moved agreement" not in text


# ── 3. Section 2 rendering: band monotonicity ────────────────────────────────

def test_band_ordering_is_undetermined_when_a_band_is_empty():
    assert rep.band_ordering([4.02, 3.95, float("nan")]) is None


def test_band_ordering_passes_when_ordered():
    assert rep.band_ordering([4.02, 3.95, 3.50]) is True


def test_band_ordering_fails_when_genuinely_out_of_order():
    assert rep.band_ordering([3.50, 3.95, 4.02]) is False
