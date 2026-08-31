"""
The human-anchored ruler must be fresh to rule, and must never half-rule.

cache/master_anchors.json (scripts/derive_master_anchors.py) anchors the
absolute score scale at the photographer's own 1-2★ / 4-5★ discriminant
quartiles, ahead of the library-percentile anchors. A stale file (encoder
tier / probe set changed since derivation) grades every photo against the
wrong space — which is worse than the generic scale it displaces — so it
must fall through cleanly to the library scale, loudly.

AND, since 2026-08-30: the rating baseline is placeholder data, so the
preference itself is OPT-IN — without LUMARA_MASTER_ANCHORS=1 the file
is ignored even when fresh.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_master_anchors.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import specvlm_pipeline as sp  # noqa: E402


def _probes():
    rng = np.random.default_rng(42)
    pos = rng.normal(size=(4, 16)).astype(np.float32)
    neg = rng.normal(size=(4, 16)).astype(np.float32)
    return pos, neg


def _write(path: Path, fingerprint: str, lo: float, hi: float) -> None:
    path.write_text(json.dumps({"fingerprint": fingerprint, "lo": lo, "hi": hi}),
                    encoding="utf-8")


def _opt_in(monkeypatch):
    monkeypatch.setenv("LUMARA_MASTER_ANCHORS", "1")


def test_fresh_master_anchors_win(tmp_path, monkeypatch):
    _opt_in(monkeypatch)
    pos, neg = _probes()
    fp = sp.probe_fingerprint(pos, neg)
    master = tmp_path / "master_anchors.json"
    lib = tmp_path / "calibration_anchors.json"
    _write(master, fp, -0.031, 0.016)      # the human-anchored scale
    _write(lib, fp, -0.063, 0.046)         # the library-percentile scale
    monkeypatch.setattr(sp, "_MASTER_ANCHORS_PATH", master)
    monkeypatch.setattr(sp, "_ANCHORS_PATH", lib)
    assert sp.load_anchors(pos, neg) == (-0.031, 0.016)


def test_stale_master_anchors_fall_through_to_library_scale(tmp_path, monkeypatch):
    _opt_in(monkeypatch)
    pos, neg = _probes()
    fp = sp.probe_fingerprint(pos, neg)
    master = tmp_path / "master_anchors.json"
    lib = tmp_path / "calibration_anchors.json"
    _write(master, "stale0000stale000", -0.031, 0.016)
    _write(lib, fp, -0.063, 0.046)
    monkeypatch.setattr(sp, "_MASTER_ANCHORS_PATH", master)
    monkeypatch.setattr(sp, "_ANCHORS_PATH", lib)
    assert sp.load_anchors(pos, neg) == (-0.063, 0.046)


def test_inverted_master_anchors_are_ignored(tmp_path, monkeypatch):
    """A corrupt file (hi <= lo) must never become the scale."""
    _opt_in(monkeypatch)
    pos, neg = _probes()
    fp = sp.probe_fingerprint(pos, neg)
    master = tmp_path / "master_anchors.json"
    lib = tmp_path / "calibration_anchors.json"
    _write(master, fp, 0.016, -0.031)      # inverted — garbage
    _write(lib, fp, -0.063, 0.046)
    monkeypatch.setattr(sp, "_MASTER_ANCHORS_PATH", master)
    monkeypatch.setattr(sp, "_ANCHORS_PATH", lib)
    assert sp.load_anchors(pos, neg) == (-0.063, 0.046)


def test_without_opt_in_master_anchors_are_ignored(tmp_path, monkeypatch):
    """Ratings are placeholder data: even a FRESH master-anchors file must
    not steer the absolute scale unless LUMARA_MASTER_ANCHORS=1."""
    monkeypatch.delenv("LUMARA_MASTER_ANCHORS", raising=False)
    pos, neg = _probes()
    fp = sp.probe_fingerprint(pos, neg)
    master = tmp_path / "master_anchors.json"
    lib = tmp_path / "calibration_anchors.json"
    _write(master, fp, -0.031, 0.016)      # fresh — and still ignored
    _write(lib, fp, -0.063, 0.046)
    monkeypatch.setattr(sp, "_MASTER_ANCHORS_PATH", master)
    monkeypatch.setattr(sp, "_ANCHORS_PATH", lib)
    assert sp.load_anchors(pos, neg) == (-0.063, 0.046)
