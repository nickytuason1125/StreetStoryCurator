"""
A fresh install must NOT grade on a curve.

Calibration anchors live in cache/, which is gitignored and not bundled by the
installer. So a new user had no anchors, load_anchors() returned None, and
_calibrate silently fell back to batch-relative scoring - the exact bug the
anchors exist to fix, shipped to every new user, visible only in a log line
nobody reads.

Worse, anchors are DERIVED FROM an existing library, so a first-time user has
nothing to derive from. Shipped defaults break that chicken-and-egg.

Order of preference, pinned below:
    user-derived cache anchors (exact fingerprint)  >  shipped tier defaults  >  None

Run:  venv\\Scripts\\python.exe -m pytest tests/test_anchor_bootstrap.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import specvlm_pipeline as sp          # noqa: E402

DEFAULTS = _ROOT / "data" / "calibration_defaults.json"


def _probes(dim=1536, n_pos=4, n_neg=3, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n_pos, dim)).astype(np.float32)
    q = rng.normal(size=(n_neg, dim)).astype(np.float32)
    return p, q


# ── the shipped file itself ───────────────────────────────────────────────────

def test_defaults_file_is_shipped_and_tracked():
    """If this file is missing from the repo, every fresh install grades on a curve."""
    assert DEFAULTS.exists(), "data/calibration_defaults.json must ship with the app"


def test_defaults_are_sane():
    d = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    assert d["by_tier"], "no tiers in the defaults file"
    for tier, a in d["by_tier"].items():
        assert a["hi"] > a["lo"], f"{tier}: degenerate anchor span"
        assert -1.0 < a["lo"] < 1.0 and -1.0 < a["hi"] < 1.0, f"{tier}: implausible range"
        assert a["aspects"], f"{tier}: no per-aspect anchors"


# ── fallback order ────────────────────────────────────────────────────────────

def test_fresh_install_gets_defaults_not_none(monkeypatch, tmp_path):
    """The whole point: no cache file must still yield absolute grading."""
    monkeypatch.setattr(sp, "_ANCHORS_PATH", tmp_path / "absent.json")
    pos, neg = _probes()
    got = sp.load_anchors(pos, neg, tier="high")
    assert got is not None, "a fresh install fell back to the curve"
    lo, hi = got
    assert hi > lo


def test_user_anchors_beat_shipped_defaults(monkeypatch, tmp_path):
    """A library-derived scale fits that library better than a generic one."""
    pos, neg = _probes()
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "fingerprint": sp.probe_fingerprint(pos, neg),
        "lo": -0.111, "hi": 0.222,
    }), encoding="utf-8")
    monkeypatch.setattr(sp, "_ANCHORS_PATH", p)
    assert sp.load_anchors(pos, neg, tier="high") == (-0.111, 0.222)


def test_stale_user_anchors_fall_back_to_defaults_not_to_the_curve(monkeypatch, tmp_path):
    """A stale cache must degrade to the generic scale, never to batch-relative."""
    pos, neg = _probes()
    other, other_neg = _probes(seed=9)
    p = tmp_path / "anchors.json"
    p.write_text(json.dumps({
        "fingerprint": sp.probe_fingerprint(other, other_neg),
        "lo": -0.111, "hi": 0.222,
    }), encoding="utf-8")
    monkeypatch.setattr(sp, "_ANCHORS_PATH", p)
    got = sp.load_anchors(pos, neg, tier="high")
    assert got is not None and got != (-0.111, 0.222)


def test_unknown_tier_has_no_default_and_says_so(monkeypatch, tmp_path, capsys):
    """Only tiers we actually derived may claim a default. No silent guessing."""
    monkeypatch.setattr(sp, "_ANCHORS_PATH", tmp_path / "absent.json")
    pos, neg = _probes(dim=999)
    got = sp.load_anchors(pos, neg, tier="not_a_tier")
    assert got is None
    assert "anchor" in capsys.readouterr().out.lower()


def test_aspect_defaults_also_bootstrap(monkeypatch, tmp_path):
    """Per-aspect bars must not stay on a curve while the headline score is fixed."""
    monkeypatch.setattr(sp, "_ANCHORS_PATH", tmp_path / "absent.json")
    ap, an = _probes(n_pos=5, n_neg=5)
    names = ["Composition", "Lighting", "Narrative", "Human/Culture", "Technical"]
    got = sp.load_aspect_anchors(ap, an, names, tier="high")
    assert got is not None and len(got) == len(names)
    assert all(hi > lo for lo, hi in got)
