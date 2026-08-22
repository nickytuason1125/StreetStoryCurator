r"""
Choosing a cohesive set of k photographs from the WHOLE graded library.

What this replaces
------------------
creative_director.py:1123 took the single highest-scoring photo and kept only
its 40 nearest neighbours. Of 5,634 graded Strong photos the model saw 12, and
the 12 were chosen for RESEMBLING ONE FRAME -- immediately before the sequencer
was asked to find contrast. If that top scorer was an outlier, the whole story
was built inside a cul-de-sac.

Measured facts this encodes
---------------------------
  personal_score vs aesthetic score correlation: 0.010 over 5,634 photos.
  Taste is independent information, not a tinted copy of quality -- but it is
  trained on 124 ratings with alignment 0.52 against a 0.50 coin flip, so it
  leads only where it is confident.

  Cohesion as a THRESHOLD has no operating range: floors of 0.55 and 0.80 both
  returned 10 every time, 0.85 and 0.88 returned 1. A cliff, not a curve. It is
  a weighted term here, never a gate.

Run:  venv\Scripts\python.exe -m pytest tests/test_story_selector.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import story_selector as ss  # noqa: E402


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _library(specs):
    """specs: list of (path, vec, score, personal). Returns (rows, M)."""
    rows = [{"path": p, "score": s, "personal_score": ps} for p, v, s, ps in specs]
    M = np.stack([_unit(v) for _, v, _, _ in specs])
    return rows, M


# ── taste is a baseline, weighted by its own confidence ──────────────────────

def test_neutral_taste_barely_counts():
    """std over the real library is 0.075, so most photos sit near 0.5. A head
    with no opinion must not steer anything."""
    assert ss.taste_weight(0.50) == pytest.approx(ss.TASTE_FLOOR, abs=1e-6)


def test_confident_taste_leads():
    assert ss.taste_weight(1.0) == pytest.approx(ss.TASTE_CEIL, abs=1e-6)
    assert ss.taste_weight(0.0) == pytest.approx(ss.TASTE_CEIL, abs=1e-6)


def test_taste_weight_is_symmetric_about_neutral():
    assert ss.taste_weight(0.75) == pytest.approx(ss.taste_weight(0.25))


def test_merit_prefers_the_head_when_it_is_confident():
    """Two photos, equal aesthetic score; the one the head likes must win."""
    liked = ss.merit(score=0.70, personal=0.95)
    disliked = ss.merit(score=0.70, personal=0.05)
    assert liked > disliked


def test_merit_ignores_a_neutral_head():
    near = ss.merit(score=0.70, personal=0.50)
    assert near == pytest.approx(0.70, abs=0.02)


# ── selection ────────────────────────────────────────────────────────────────

def test_k_is_honoured():
    specs = [("/p%d.jpg" % i, np.random.RandomState(i).rand(8), 0.6 + i * 0.001, 0.5)
             for i in range(40)]
    rows, M = _library(specs)
    q = _unit(np.ones(8))
    for k in range(4, 11):
        idx, diag = ss.select(q, rows, M, k=k)
        assert len(idx) == k, diag


def test_near_duplicates_are_never_both_chosen():
    base = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    twin = base + np.array([0, 0.02, 0, 0, 0, 0, 0, 0])   # cosine ~0.9998
    specs = [("/a.jpg", base, 0.99, 0.5), ("/b.jpg", twin, 0.98, 0.5)]
    specs += [("/o%d.jpg" % i, np.random.RandomState(100 + i).rand(8), 0.6, 0.5)
              for i in range(10)]
    rows, M = _library(specs)
    idx, _ = ss.select(_unit(base), rows, M, k=4)
    chosen = {rows[i]["path"] for i in idx}
    assert not {"/a.jpg", "/b.jpg"} <= chosen, "picked the same photograph twice"


def test_bundled_demo_images_are_never_selected():
    """56 files under dataset_images/ are the app's own assets and were being
    selected into user stories."""
    good = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    specs = [("C:/app/dataset_images/carousel_01.jpg", good, 0.99, 0.9)]
    specs += [("/mine%d.jpg" % i, np.random.RandomState(i).rand(8), 0.7, 0.5)
              for i in range(10)]
    rows, M = _library(specs)
    idx, _ = ss.select(_unit(good), rows, M, k=4)
    assert all("dataset_images" not in rows[i]["path"] for i in idx)


def test_the_whole_library_is_considered_not_one_neighbourhood():
    """The bug this replaces: a strong, relevant frame far from the top scorer
    must still be reachable."""
    cluster = [("/cluster%d.jpg" % i,
                np.array([1.0, 0.01 * i, 0, 0, 0, 0, 0, 0]), 0.95, 0.5)
               for i in range(30)]
    far = [("/far.jpg", np.array([0, 0, 0, 0, 0, 0, 0, 1.0]), 0.99, 0.99)]
    rows, M = _library(cluster + far)
    q = _unit(np.array([0, 0, 0, 0, 0, 0, 0, 1.0]))     # brief points AT the outlier
    idx, _ = ss.select(q, rows, M, k=4)
    assert any(rows[i]["path"] == "/far.jpg" for i in idx), (
        "a relevant frame outside the dominant cluster was unreachable")


def test_diagnostics_report_cohesion_rather_than_enforcing_it():
    specs = [("/p%d.jpg" % i, np.random.RandomState(i).rand(8), 0.7, 0.5)
             for i in range(20)]
    rows, M = _library(specs)
    idx, diag = ss.select(_unit(np.ones(8)), rows, M, k=5)
    assert "cohesion_mean" in diag and "cohesion_min" in diag
    assert 0.0 <= diag["cohesion_mean"] <= 1.0


def test_too_few_candidates_returns_what_exists_with_a_reason():
    # Distinct directions, not random vectors: np.random.rand lives in the
    # positive orthant, where everything is >0.88 similar to everything else,
    # so the duplicate guard would (correctly) reject them and this would test
    # the wrong thing.
    specs = [("/p%d.jpg" % i, np.eye(8)[i], 0.7, 0.5) for i in range(3)]
    rows, M = _library(specs)
    idx, diag = ss.select(_unit(np.ones(8)), rows, M, k=7)
    assert len(idx) == 3
    assert diag.get("reason"), "returning fewer than k must be explained"


def test_empty_library_is_survivable():
    idx, diag = ss.select(_unit(np.ones(8)), [], np.zeros((0, 8), dtype=np.float32), k=5)
    assert idx == []
    assert diag.get("reason")


def test_scratchpad_and_temp_images_are_excluded():
    """The library also contained 8 rows under a previous session's
    Temp/claude/.../scratchpad/demo_images. Those are not the user's
    photographs either, and one slipped into a story with score 1.00 because
    the filter only matched dataset_images."""
    bs = chr(92)
    good = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    win = bs.join(["C:", "Users", "x", "AppData", "Local", "Temp", "claude",
                   "abc", "scratchpad", "d.jpg"])
    junk = [
        ("C:/Users/x/AppData/Local/Temp/claude/abc/scratchpad/demo_images/c1.jpg",
         good, 1.0, 0.9),
        (win, good, 1.0, 0.9),
    ]
    mine = [("/mine%d.jpg" % i, np.eye(8)[i % 8], 0.7, 0.5) for i in range(8)]
    rows, M = _library(junk + mine)
    idx, _ = ss.select(_unit(good), rows, M, k=4)
    for i in idx:
        q = rows[i]["path"].lower().replace(bs, "/")
        assert "scratchpad" not in q and "/temp/" not in q, rows[i]["path"]
