"""Post-kill-switch FIFO replenish (2026-09-22).

Live failure this pins: 'orange and reds' → HARD_FILTER_PEOPLE=true → the
early starvation guard FIFO-retired the 10 OLDEST used frames → the D-FINE
person kill disqualified 11 of those 12 → a 1-photo sequence. The used-history
file had already been rewritten by the early guard, so the Step-4a FIFO had
nothing left to retire. The fix: retire the NEXT-oldest batch from the
ORIGINAL (pre-rewrite) used entries and admit the survivors.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _mk(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"photo_{i:03d}.jpg"
        p.write_bytes(b"x")
        paths.append(str(p))
    return paths


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=1536).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_replenish_skips_already_retired_and_takes_next_oldest(tmp_path):
    """The 10 oldest were already retired by the early guard — the helper
    must continue DOWN the original history, not restart from the top."""
    from creative_director import _post_kill_replenish
    paths = _mk(tmp_path, 10)
    avoid_res = {str(Path(p).resolve()) for p in paths}
    original = [{"path": paths[i], "ts": 1000 + i} for i in range(10)]
    early_evicted = {str(Path(paths[0]).resolve()),
                     str(Path(paths[1]).resolve()),
                     str(Path(paths[2]).resolve())}
    snap_embs = [_unit(i) for i in range(10)]
    retire, addback = _post_kill_replenish(
        original, early_evicted, avoid_res, paths, snap_embs,
        need=3, pool_embs=[_unit(99)])
    # next-oldest = photos 3,4,5 (0,1,2 already retired by the early guard)
    assert {Path(paths[j]).name for j in addback} == {
        "photo_003.jpg", "photo_004.jpg", "photo_005.jpg"}
    assert set(retire) == {str(Path(paths[j]).resolve()) for j in (3, 4, 5)}


def test_replenish_drops_burst_twins_of_surviving_pool(tmp_path):
    """A replenished frame nearly identical to a surviving pool photo must
    NOT be admitted — rotation must not reintroduce duplicates."""
    from creative_director import _post_kill_replenish
    paths = _mk(tmp_path, 3)
    avoid_res = {str(Path(p).resolve()) for p in paths}
    original = [{"path": paths[i], "ts": 1000 + i} for i in range(3)]
    pool_twin = _unit(42)
    snap_embs = [pool_twin.copy(), _unit(7), _unit(8)]   # photo_000 is a twin
    retire, addback = _post_kill_replenish(
        original, set(), avoid_res, paths, snap_embs,
        need=3, pool_embs=[pool_twin])
    assert 0 not in addback, "twin of a surviving frame was re-admitted"
    assert set(addback) == {1, 2}


def test_replenish_needs_zero_returns_empty(tmp_path):
    from creative_director import _post_kill_replenish
    paths = _mk(tmp_path, 4)
    retire, addback = _post_kill_replenish(
        [{"path": paths[0], "ts": 1}], set(),
        {str(Path(paths[0]).resolve())}, paths, [_unit(0)],
        need=0, pool_embs=[])
    assert retire == [] and addback == []


def test_replenish_ignores_entries_outside_avoid(tmp_path):
    from creative_director import _post_kill_replenish
    paths = _mk(tmp_path, 3)
    original = [{"path": paths[0], "ts": 1}]   # never excluded — frees nothing
    retire, addback = _post_kill_replenish(
        original, set(), {str(Path(paths[2]).resolve())}, paths,
        [_unit(i) for i in range(3)], need=2, pool_embs=[])
    assert retire == [] and addback == []


def test_wire_post_kill_replenish_exists_in_pipeline():
    """Source-level guard: the replenish must be wired after the YOLO kill
    and BEFORE the final FIFO safety net."""
    cd_src = (Path(__file__).resolve().parent.parent /
              "src" / "creative_director.py").read_text(encoding="utf-8")
    assert "_post_kill_replenish(" in cd_src
    assert "POST-KILL REPLENISH" in cd_src
    replenish_pos = cd_src.find("POST-KILL REPLENISH:")
    fifo_pos = cd_src.find("pool starved (")
    assert 0 < replenish_pos < fifo_pos, \
        "replenish must run before the last-resort FIFO block"
