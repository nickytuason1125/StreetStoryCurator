"""Tests for stage_runner timings, model residency, and calibration math."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage_runner import StageTimer
import model_residency as res


def test_stage_timer_marks_close_in_order():
    t = StageTimer()
    t.mark("a")
    time.sleep(0.05)
    t.mark("b")
    time.sleep(0.05)
    snap = t.snapshot()
    names = [s["stage"] for s in snap["stages"]]
    assert names == ["a", "b"]   # first mark opens the timer; no phantom setup
    assert all(s["seconds"] >= 0.04 for s in snap["stages"])
    assert snap["total"] == round(sum(s["seconds"] for s in snap["stages"]), 2)


def test_stage_timer_snapshot_closes_open_stage():
    t = StageTimer()
    t.mark("only")
    time.sleep(0.05)
    snap = t.snapshot()
    assert snap["stages"][-1]["seconds"] >= 0.04
    # snapshot is terminal: re-snapshot doesn't double-count
    snap2 = t.snapshot()
    assert snap2["total"] == snap["total"]


def test_residency_register_release():
    calls = []
    res.register("m1", lambda: calls.append("m1"), 1.0)
    assert "m1" in res.status()
    assert res.release("m1") is True
    assert calls == ["m1"]
    assert "m1" not in res.status()


def test_residency_release_all_and_never_raises():
    res.register("bad", lambda: 1 / 0, 1.0)   # unload that raises
    res.register("good", lambda: None, 2.0)
    assert res.release_all() >= 1
    assert res.status() == {}


def test_residency_evict_until_lru_order():
    order = []
    res.register("old", lambda: order.append("old"), 5.0)
    time.sleep(0.01)
    res.register("new", lambda: order.append("new"), 5.0)
    evicted = res.evict_until(2.0, free_ram_fn=lambda: 0.5)
    assert evicted == 2
    assert order == ["old", "new"]   # LRU: oldest evicted first
    assert res.status() == {}


def test_calibrate_compute_thresholds():
    from calibrate_subjects import compute_thresholds
    sims = [0.0] * 4 + [0.1] * 4          # p50=0.05, p90=0.1
    d = compute_thresholds(sims)
    assert d["p50"] == pytest.approx(0.05)
    assert d["p90"] == pytest.approx(0.1)
    assert d["max"] == pytest.approx(0.1)


def test_threshold_for_missing_calibration_returns_none(tmp_path, monkeypatch):
    import calibrate_subjects as cal
    monkeypatch.setattr(cal, "_CAL_PATH", tmp_path / "missing.json")
    assert cal.threshold_for("vehicles") is None
