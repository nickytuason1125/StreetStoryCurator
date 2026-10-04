r"""
Pinned-frame splice (2026-09-22).

The user can pin frames from the previous build; those frames MUST reappear
in the next regenerated sequence no matter what rotation, the people gate or
the Art Director decided. _splice_pinned() is the pure core of that guarantee;
these tests pin down its contract:

  - a pinned frame that selection dropped is swapped in for the LOWEST-scored
    non-pinned pick,
  - a pinned frame that selection already kept is NOT duplicated,
  - short-of-target runs get the pin APPENDED, not swapped,
  - pins are resolved across BOTH the surviving pool and the pre-filter
    snapshot (a pin the person-kill excluded still comes back),
  - the four parallel seq_* lists stay index-parallel afterwards,
  - no pins -> the call is a no-op.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creative_director import _splice_pinned  # noqa: E402


def _mk(n, seed=0):
    rng = np.random.default_rng(seed)
    paths = [f"C:/lib/img_{i}.jpg" for i in range(n)]
    embs = [rng.normal(size=16).astype(np.float32) for _ in range(n)]
    scores = [0.9 - 0.05 * i for i in range(n)]
    aspects = [{"Composition": 0.5 + i / 100} for i in range(n)]
    return paths, embs, scores, aspects


def test_dropped_pin_swaps_in_for_lowest_scored():
    pool, embs, scores, aspects = _mk(5)
    seq_paths = pool[:3]
    seq_embs = embs[:3]
    seq_scores = scores[:3]
    seq_aspects = aspects[:3]
    n = _splice_pinned(
        seq_paths, seq_embs, seq_scores, seq_aspects,
        ["C:/lib/img_4.jpg"], 3,
        lookup_sources=((pool, embs, scores, aspects),),
    )
    assert n == 1
    assert "C:/lib/img_4.jpg" in seq_paths
    # Lowest-scored pick (img_2, 0.80) was displaced; img_0 survived.
    assert "C:/lib/img_2.jpg" not in seq_paths
    assert "C:/lib/img_0.jpg" in seq_paths
    assert len(seq_paths) == 3
    # Parallel lists stay index-parallel.
    k = seq_paths.index("C:/lib/img_4.jpg")
    assert np.allclose(seq_embs[k], embs[4])
    assert seq_scores[k] == pytest.approx(scores[4])
    assert seq_aspects[k] == aspects[4]


def test_already_selected_pin_is_not_duplicated():
    pool, embs, scores, aspects = _mk(5)
    seq_paths = list(pool[:3])
    before = list(seq_paths)
    n = _splice_pinned(
        seq_paths, list(embs[:3]), list(scores[:3]), list(aspects[:3]),
        [pool[0]], 3, lookup_sources=((pool, embs, scores, aspects),),
    )
    assert n == 0
    assert seq_paths == before


def test_short_sequence_appends_pin():
    pool, embs, scores, aspects = _mk(5)
    seq_paths = list(pool[:2])
    seq_embs = list(embs[:2]); seq_scores = list(scores[:2])
    seq_aspects = list(aspects[:2])
    n = _splice_pinned(
        seq_paths, seq_embs, seq_scores, seq_aspects,
        [pool[4]], 3, lookup_sources=((pool, embs, scores, aspects),),
    )
    assert n == 1
    assert len(seq_paths) == 3
    assert seq_paths[2] == pool[4]
    assert len(seq_embs) == len(seq_scores) == len(seq_aspects) == 3


def test_pin_excluded_from_pool_still_restored_via_snapshot():
    pool, embs, scores, aspects = _mk(5)
    # The person gate removed img_3 from the surviving pool entirely;
    # the pre-filter snapshot still has it.
    kept = [i for i in range(5) if i != 3]
    filtered_paths = [pool[i] for i in kept]
    filtered_embs = [embs[i] for i in kept]
    filtered_scores = [scores[i] for i in kept]
    filtered_aspects = [aspects[i] for i in kept]
    seq_paths = list(filtered_paths[:3])
    seq_embs = list(filtered_embs[:3])
    seq_scores = list(filtered_scores[:3])
    seq_aspects = list(filtered_aspects[:3])
    n = _splice_pinned(
        seq_paths, seq_embs, seq_scores, seq_aspects,
        [pool[3]], 3,
        lookup_sources=(
            (filtered_paths, filtered_embs, filtered_scores, filtered_aspects),
            (pool, embs, scores, aspects),
        ),
    )
    assert n == 1
    assert pool[3] in seq_paths
    k = seq_paths.index(pool[3])
    assert np.allclose(seq_embs[k], embs[3])


def test_duplicate_pins_resolved_once():
    pool, embs, scores, aspects = _mk(5)
    seq_paths = list(pool[:3])
    seq_embs = list(embs[:3]); seq_scores = list(scores[:3])
    seq_aspects = list(aspects[:3])
    n = _splice_pinned(
        seq_paths, seq_embs, seq_scores, seq_aspects,
        [pool[4], pool[4]], 3, lookup_sources=((pool, embs, scores, aspects),),
    )
    assert n == 1
    assert seq_paths.count(pool[4]) == 1


def test_no_pins_is_a_noop():
    pool, embs, scores, aspects = _mk(4)
    seq_paths = list(pool[:3])
    before = list(seq_paths)
    assert _splice_pinned(
        seq_paths, list(embs[:3]), list(scores[:3]), list(aspects[:3]),
        None, 3, lookup_sources=((pool, embs, scores, aspects),),
    ) == 0
    assert seq_paths == before


def test_path_variant_resolution():
    """Backslash/forward-slash variants of the same path still resolve."""
    pool, embs, scores, aspects = _mk(4)
    seq_paths = list(pool[:3])
    seq_embs = list(embs[:3]); seq_scores = list(scores[:3])
    seq_aspects = list(aspects[:3])
    # str(Path(x).resolve()) normalises separators to the platform kind, so
    # a backslash variant of an existing pool path must match exactly.
    n = _splice_pinned(
        seq_paths, seq_embs, seq_scores, seq_aspects,
        ["C:\\lib\\img_3.jpg"], 3, lookup_sources=((pool, embs, scores, aspects),),
    )
    assert n == 1
    assert pool[3] in seq_paths
