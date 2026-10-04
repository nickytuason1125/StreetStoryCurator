r"""
FIFO used-history rotation (timestamped entries): a saved story must not
reshow its photos until EVERY frame has had a turn -- and when the pool
starves, the OLDEST-SAVED entries retire first, never the recent ones.

Run:  venv\Scripts\python.exe -m pytest tests/test_used_rotation.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import creative_director as cd


def _e(path, ts):
    return {"path": path, "ts": ts}


def _paths(n):
    return [f"p{i}.jpg" for i in range(n)]


def test_evicts_oldest_saved_first_not_most_recent():
    paths = _paths(6)
    # ts ascending = save order: p4/p5 were saved MOST recently
    used = [{"path": "p4.jpg", "ts": 3.0}, {"path": "p5.jpg", "ts": 3.1},
            {"path": "p0.jpg", "ts": 1.0}]
    ev, rem, pool = cd._fifo_evict_used(used, [1, 2, 3], 5, paths)
    names = [e["path"] for e in ev]
    assert names == ["p0.jpg", "p4.jpg"], "oldest ts retires first"
    rem_names = [e["path"] for e in rem]
    assert "p5.jpg" in rem_names               # most recent save stays marked
    assert sorted(pool) == [0, 1, 2, 3, 4]     # meets min_pool: evicted p0/p4 re-enter


def test_stops_as_soon_as_the_request_can_be_met():
    paths = _paths(6)
    used = [{"path": f"p{i}.jpg", "ts": float(i)} for i in range(6)]
    ev, rem, pool = cd._fifo_evict_used(used, [], 2, paths)
    assert [e["path"] for e in ev] == ["p0.jpg", "p1.jpg"]   # only 2, not all 6
    assert [e["path"] for e in rem] == ["p2.jpg", "p3.jpg", "p4.jpg", "p5.jpg"]
    assert sorted(pool) == [0, 1]


def test_used_paths_no_longer_in_the_library_are_left_marked():
    paths = _paths(4)
    used = [{"path": "gone.jpg", "ts": 0.5}, {"path": "p1.jpg", "ts": 1.0}]
    ev, rem, pool = cd._fifo_evict_used(used, [0], 2, paths)
    assert [e["path"] for e in ev] == ["p1.jpg"]   # only the real candidate retires
    assert any(e["path"] == "gone.jpg" for e in rem)
    assert sorted(pool) == [0, 1]


def test_full_rotation_cycle_returns_every_frame_exactly_once():
    # Two consecutive starved runs over a 6-frame library. Between runs the
    # ROUTER re-marks reused frames with a NEW timestamp (that is the real
    # flow), so the second eviction must pick the still-oldest cycle.
    paths = _paths(6)
    used = [{"path": f"p{i}.jpg", "ts": float(i)} for i in range(6)]
    ev1, rem1, pool1 = cd._fifo_evict_used(used, [], 3, paths)
    # router re-mark: the frames shown in cycle 1 get fresh timestamps
    used2 = list(rem1) + [{"path": e["path"], "ts": 10.0 + i}
                          for i, e in enumerate(ev1)]
    ev2, rem2, pool2 = cd._fifo_evict_used(used2, [], 3, paths)
    n1 = {e["path"] for e in ev1}
    n2 = {e["path"] for e in ev2}
    assert n1 == {"p0.jpg", "p1.jpg", "p2.jpg"}
    assert n2 == {"p3.jpg", "p4.jpg", "p5.jpg"}   # next-oldest cycle
    assert not (n1 & n2)                          # a frame never repeats early
    assert n1 | n2 == set(paths)                  # full coverage
