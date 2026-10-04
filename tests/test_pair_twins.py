"""RAW+JPEG pairs the camera wrote together are graded ONCE (2026-10-04).

A Sony card shot RAW+JPEG holds every photo twice; the cull graded both, which
doubled every stage. The JPEG twin now takes its RAW's result. Only a pair
written within 2 s counts — a JPEG exported later next to its RAW is a
different picture and keeps its own grade.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import grade_pipeline_v2 as gp  # noqa: E402


def _touch(p: Path, mtime: float) -> str:
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return str(p)


def test_camera_pair_keeps_raw_and_maps_jpeg_twin(tmp_path):
    raw = _touch(tmp_path / "7R200104.ARW", 1000.0)
    jpg = _touch(tmp_path / "7R200104.JPG", 1000.5)
    solo = _touch(tmp_path / "7R200105.JPG", 1001.0)
    kept, twins = gp._pair_twins([raw, jpg, solo])
    assert kept == [raw, solo]
    assert twins == {raw: jpg}


def test_later_export_is_not_a_twin(tmp_path):
    raw = _touch(tmp_path / "DSC1.ARW", 1000.0)
    jpg = _touch(tmp_path / "DSC1.jpg", 5000.0)      # exported an hour later
    kept, twins = gp._pair_twins([raw, jpg])
    assert kept == [raw, jpg] and twins == {}


def test_jpeg_without_its_raw_in_the_batch_is_graded(tmp_path):
    _touch(tmp_path / "A.ARW", 1000.0)               # RAW already graded (cached)
    jpg = _touch(tmp_path / "A.JPG", 1000.0)
    kept, twins = gp._pair_twins([jpg])
    assert kept == [jpg] and twins == {}


def test_opt_out(tmp_path, monkeypatch):
    raw = _touch(tmp_path / "B.ARW", 1000.0)
    jpg = _touch(tmp_path / "B.JPG", 1000.0)
    monkeypatch.setenv("FIRSTCUT_GRADE_PAIRS_ONCE", "0")
    assert gp._pair_twins([raw, jpg]) == ([raw, jpg], {})
