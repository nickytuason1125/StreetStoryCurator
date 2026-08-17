"""
Behaviour locks for the stages extracted out of ``run_v2``.

These stages were lifted verbatim out of a 2,400-line function, so the risk they
carry is not "does the new code work" but "does it still do exactly what the
inlined version did". Each test below pins a rule that was previously reachable
only by running a whole cull:

  1. Duplicate clusters pick their winner on the FINAL score.
  2. Grade buckets are ABSOLUTE, not relative to the batch.
  3. Unreadable (zero-vector) rows leave the pipeline instead of scoring 0.00.
  4. Every stage degrades rather than raising — a broken stage must not be able
     to empty a photographer's cull.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_pipeline_stages.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pipeline_stages as ps          # noqa: E402

STRONG, MID, WEAK = "Strong ✅", "Mid ⚠️", "Weak ❌"


def _grade(scores, paths=None):
    """assign_grades with the production thresholds and labels."""
    arr = np.asarray(scores, dtype=float)
    paths = paths or [f"img{i}.jpg" for i in range(len(arr))]
    return ps.assign_grades(arr, paths, np,
                            strong_thresh=0.60, mid_thresh=0.41,
                            strong_label=STRONG, mid_label=MID, weak_label=WEAK)


# ── 1. duplicate clusters ────────────────────────────────────────────────────

def test_cluster_winner_is_highest_final_score():
    """★ goes to the best frame; everyone else is told which file beat them."""
    cluster_ids = [7, 7, 7]
    scores = np.array([0.40, 0.80, 0.55])
    flags = ps.mark_duplicate_groups(cluster_ids, scores, ["a.jpg", "b.jpg", "c.jpg"])

    assert flags[1].startswith("★"), "highest score must win the cluster"
    assert "Best of 3 similar shots" in flags[1]
    assert "0.80" in flags[1]
    for loser in (0, 2):
        assert flags[loser].startswith("\U0001f501")
        assert "b.jpg is better" in flags[loser]
    assert "+0.40" in flags[0]      # 0.80 - 0.40
    assert "+0.25" in flags[2]      # 0.80 - 0.55


def test_unclustered_and_singleton_photos_get_no_flag():
    """cid -1 is 'not a duplicate'; a cluster of one is not a duplicate either."""
    flags = ps.mark_duplicate_groups([-1, -1, 3], np.array([0.9, 0.2, 0.5]),
                                     ["a.jpg", "b.jpg", "c.jpg"])
    assert flags == ["", "", ""]


def test_sim_flags_length_always_matches_input():
    flags = ps.mark_duplicate_groups([], np.array([]), [])
    assert flags == []


def test_sim_flag_failure_degrades_to_empty_flags():
    """Advisory UI text must never cost the grades computed above it."""
    class Explodes:
        def __getitem__(self, i):
            raise RuntimeError("boom")

    flags = ps.mark_duplicate_groups([1, 1], Explodes(), ["a.jpg", "b.jpg"])
    assert flags == ["", ""], "a failed stage must return empty, not raise"


# ── 2. absolute grade buckets ────────────────────────────────────────────────

def test_thresholds_are_inclusive_at_the_boundary():
    _, grades = _grade([0.60, 0.59, 0.41, 0.40])
    assert grades == [STRONG, MID, MID, WEAK]


def test_buckets_are_absolute_not_relative():
    """A uniformly excellent shoot must not be forced to grow a Weak tail.

    This is the regression that quantile calibration caused: it pinned ~25%
    Strong / 20% Weak on every run regardless of actual quality.
    """
    _, all_good = _grade([0.90, 0.88, 0.85, 0.83, 0.81])
    assert all_good == [STRONG] * 5

    _, all_bad = _grade([0.20, 0.18, 0.15, 0.13, 0.11])
    assert all_bad == [WEAK] * 5


def test_nan_scores_become_weak_not_crash():
    scores, grades = _grade([float("nan"), 0.75])
    assert scores[0] == pytest.approx(0.15)
    assert grades == [WEAK, STRONG]


def test_scores_are_clamped_and_rounded():
    scores, _ = _grade([5.0, -3.0, 0.6666])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.10)
    assert scores[2] == pytest.approx(0.67)


def test_grades_align_one_to_one_with_paths():
    paths = [f"p{i}.jpg" for i in range(4)]
    scores, grades = _grade([0.7, 0.5, 0.3, 0.95], paths)
    assert len(grades) == len(paths) == len(scores)


# ── 3. unreadable rows leave the pipeline ────────────────────────────────────

def test_zero_vector_rows_are_dropped():
    """A RAW whose preview could not be read must not be judged as a bad photo."""
    paths = ["ok1.jpg", "dead.RW2", "ok2.jpg"]
    embs = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    out_paths, out_embs, n = ps.drop_unreadable_rows(paths, embs, np)

    assert out_paths == ["ok1.jpg", "ok2.jpg"]
    assert n == 2
    assert out_embs.shape == (2, 2)
    assert np.allclose(out_embs[0], [1.0, 0.0])


def test_all_readable_returns_input_untouched():
    paths = ["a.jpg", "b.jpg"]
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    out_paths, out_embs, n = ps.drop_unreadable_rows(paths, embs, np)
    assert out_paths is paths and out_embs is embs and n == 2


def test_length_mismatch_is_left_alone():
    """Defensive: a shape disagreement means something upstream is wrong —
    dropping rows on a bad index mapping would corrupt the run silently."""
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    embs = np.array([[1.0, 0.0]], dtype=np.float32)
    out_paths, out_embs, n = ps.drop_unreadable_rows(paths, embs, np)
    assert out_paths == paths and n == 3

    out_paths, out_embs, n = ps.drop_unreadable_rows(paths, None, np)
    assert out_paths == paths and out_embs is None and n == 3


# ── 4. EXIF ──────────────────────────────────────────────────────────────────

def test_exif_timestamps_preserve_order_and_length(tmp_path):
    """pool.map order is what aligns timestamps to paths — pin it."""
    files = []
    for i in range(5):
        f = tmp_path / f"{i}.jpg"
        f.write_bytes(b"\xff\xd8not-an-image")
        files.append(str(f))
    out = ps.read_exif_timestamps(files)
    assert len(out) == 5
    assert all(t == 0.0 for t in out)


def test_exif_on_empty_list_does_not_hang():
    assert ps.read_exif_timestamps([]) == []


# ── 5. gate result bookkeeping ───────────────────────────────────────────────

def test_gate_result_counts_all_three_failure_kinds():
    g = ps.GateResult(
        survivors=["a.jpg"],
        blur_disqualified={"b.jpg"},
        yolo_disqualified={"c.jpg", "d.jpg"},
        technical_disq={"e.jpg": "flat"},
    )
    assert g.n_failed == 4


def test_empty_gate_result_means_no_op():
    """The gate's failure path returns a bare GateResult; it must read as
    'nothing was disqualified', never as 'everything failed'."""
    g = ps.GateResult()
    assert g.n_failed == 0
    assert g.survivors == []
