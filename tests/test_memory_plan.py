"""MemoryPlan unit tests — the OOM admission/degradation ladder.

The ladder is the app's answer to "not enough RAM": degrade to a Scan
(measured ~2.0 GB) instead of refusing, and refuse only when even a Scan
cannot fit. These tests pin that behaviour so it cannot regress.

Run:  venv/Scripts/python.exe -m pytest tests/test_memory_plan.py -v
"""
import os
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest

from src import memory_plan as mp


def _stub_floor(monkeypatch, gb: float) -> None:
    """Stub the encoder's hard floor without importing the heavy module."""
    monkeypatch.setitem(
        sys.modules, "siglip2_encoder",
        types.SimpleNamespace(_default_ram_floor_gb=lambda: gb))


# ── the numbers themselves ───────────────────────────────────────────────────

def test_scan_needs_less_than_full():
    assert mp._need_scan_gb() < mp._need_full_gb(500)


def test_scan_charge_is_the_measured_two_gb():
    # The 2026-09-06 over-charge bug: scan was billed the IQA-inclusive 3.8 GB
    # and a low-RAM machine was refused a scan it could actually run.
    assert mp._need_scan_gb() == pytest.approx(2.0)


def test_gate_override_does_not_touch_the_encoder_floor(monkeypatch):
    # FIRSTCUT_MIN_RAM_GB used to alias BOTH the cull gate and the encoder's
    # batch-reduction floor — escaping one silently disabled the other.
    monkeypatch.setenv("FIRSTCUT_MIN_RAM_GB", "99")
    import run_profile
    assert run_profile.required_ram_gb(10) == 99.0            # gate: honoured
    prof = run_profile.RunProfile(run_profile.spec_for("low"), gpu=False)
    assert prof.ram_soft_gb < 99.0                            # encoder: independent


# ── the ladder ───────────────────────────────────────────────────────────────

def test_full_plan_when_ram_is_plenty(monkeypatch):
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 12.0)
    plan = mp.plan_for(500, requested_scan=False)
    assert plan["plan"] == "full"
    assert plan["degraded"] is False
    assert plan["scan_mode"] is False


def test_degrades_to_scan_when_tight(monkeypatch):
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 2.5)
    plan = mp.plan_for(500, requested_scan=False)
    assert plan["plan"] == "scan"
    assert plan["scan_mode"] is True
    assert plan["degraded"] is True
    assert plan["note"]           # the UI is told what happened and why


def test_scan_request_never_counts_as_degraded(monkeypatch):
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 2.5)
    plan = mp.plan_for(500, requested_scan=True)
    assert plan["plan"] == "scan" and plan["degraded"] is False


def test_refuses_when_even_reduced_scan_cannot_fit(monkeypatch):
    _stub_floor(monkeypatch, 1.2)                     # minimal rung = 1.5 GB
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 0.9)
    assert mp.plan_for(500, requested_scan=False) is None
    assert mp.plan_for(10, requested_scan=True) is None


def test_degrades_to_reduced_batch_scan_in_the_old_gate_band(monkeypatch):
    # 1.4–2.0 GB was admitted by the OLD pre-spawn gate (floor + 0.2) — the
    # ladder must not be stricter than that. Stub floor 1.2 → rung at 1.5 GB.
    _stub_floor(monkeypatch, 1.2)
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 1.55)
    monkeypatch.setenv("SIGLIP_ENC_BATCH", "8")
    plan = mp.plan_for(500, requested_scan=False)
    assert plan["plan"] == "scan+reduced-batch"
    assert plan["degraded"] is True and plan["scan_mode"] is True
    assert os.environ.get("SIGLIP_ENC_BATCH") == "2"   # set for the runner child


def test_unmeasurable_ram_fails_open_but_loud(monkeypatch):
    monkeypatch.setattr(mp, "free_ram_gb", lambda: None)
    plan = mp.plan_for(500, requested_scan=False)
    assert plan is not None                # availability first
    assert plan["degraded"] is False
    assert plan["note"]                    # …but it says so


# ── stage-boundary checkpoints ───────────────────────────────────────────────

def test_failpoint_raises_memoryerror(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_OOM_FAILPOINT", "encode")
    with pytest.raises(MemoryError, match="failpoint"):
        mp.memory_checkpoint("encode")


def test_failpoint_ignores_other_stages(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_OOM_FAILPOINT", "judge")
    _stub_floor(monkeypatch, 1.2)
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 8.0)
    mp.memory_checkpoint("encode")         # must NOT raise


def test_checkpoint_below_hard_floor_raises_recoverable(monkeypatch):
    _stub_floor(monkeypatch, 1.2)
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 0.5)
    with pytest.raises(MemoryError, match="resume"):
        mp.memory_checkpoint("encode")


def test_checkpoint_tight_band_shrinks_batch(monkeypatch):
    _stub_floor(monkeypatch, 1.2)
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 1.9)   # hard < free < hard+0.8
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    notices = []
    mp.memory_checkpoint("encode", progress=lambda f, d: notices.append(d))
    assert os.environ.get("SIGLIP_ENC_BATCH") == "2"
    assert any("Low memory" in n for n in notices)


def test_checkpoint_comfortable_is_silent(monkeypatch):
    _stub_floor(monkeypatch, 1.2)
    monkeypatch.setattr(mp, "free_ram_gb", lambda: 8.0)
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    mp.memory_checkpoint("save")
    assert os.environ.get("SIGLIP_ENC_BATCH") != "2"


def test_free_ram_gates_on_the_scarcer_of_ram_and_commit(monkeypatch):
    # The 2026-09-07 crash: physical pages were "available" but the COMMIT
    # limit was exhausted ("The paging file is too small" while importing
    # scipy). free_ram_gb must return whichever is scarcer.
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (5.0, 1.0))
    assert mp.free_ram_gb() == pytest.approx(1.0)


def test_commit_headroom_exposed_for_diagnostics(monkeypatch):
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (5.0, 1.0))
    assert mp.commit_headroom_gb() == pytest.approx(1.0)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        mp.memory_checkpoint("nonsense")
