"""Stored grades are reused only when they were made the way this run grades.

Before 2026-09-28 any stored row with score >= 0.10 counted as "already
graded": Scan-quality grades (made when free RAM triggered a silent downgrade)
were served as final by later full-quality culls, and grader fixes never
reached photos graded before them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pipeline_stages as ps  # noqa: E402


def _row(sig):
    return {"path": "x.jpg", "score": 0.7, "breakdown": ({"_grade_sig": sig} if sig else {})}


FULL = ps.grade_signature(scan_mode=False, deep_grade=False)
SCAN = ps.grade_signature(scan_mode=True, deep_grade=False)
DEEP = ps.grade_signature(scan_mode=False, deep_grade=True)


def test_full_cull_never_serves_a_scan_grade():
    assert not ps.row_is_reusable(_row(SCAN), scan_mode=False, deep_grade=False)
    assert ps.row_is_reusable(_row(FULL), scan_mode=False, deep_grade=False)


def test_requested_scan_may_reuse_any_current_grade():
    for sig in (FULL, SCAN, DEEP):
        assert ps.row_is_reusable(_row(sig), scan_mode=True, deep_grade=False)


def test_full_and_deep_do_not_substitute_for_each_other():
    assert not ps.row_is_reusable(_row(DEEP), scan_mode=False, deep_grade=False)
    assert not ps.row_is_reusable(_row(FULL), scan_mode=False, deep_grade=True)
    assert ps.row_is_reusable(_row(DEEP), scan_mode=False, deep_grade=True)


def test_older_grader_version_is_regraded():
    old = "2026-01-01|full"
    assert not ps.row_is_reusable(_row(old), scan_mode=False, deep_grade=False)
    assert not ps.row_is_reusable(_row(old), scan_mode=True, deep_grade=False)


def test_unsigned_and_malformed_rows_are_regraded():
    for row in (_row(None), {"path": "x", "score": 0.7}, _row("garbage"),
                {"path": "x", "score": 0.7, "breakdown": "not-a-dict"}):
        assert not ps.row_is_reusable(row, scan_mode=False, deep_grade=False)


def test_signature_names_version_and_mode():
    assert FULL == f"{ps.GRADER_VERSION}|full"
    assert SCAN.endswith("|scan") and DEEP.endswith("|deep")
    # Scan wins over a deep request (scan skips the verifier entirely).
    assert ps.grade_signature(scan_mode=True, deep_grade=True).endswith("|scan")
