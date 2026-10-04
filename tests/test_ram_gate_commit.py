"""RAM gate = commit headroom by default (2026-10-04).

Gating on physical free RAM made culls wait for Windows to page other apps
out "by itself" — 160 s of a 223 s golden run with 2.5 GB held elsewhere;
the commit gate ran the same set in 56 s with identical grades.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import memory_plan as mp  # noqa: E402


def test_default_gate_is_commit_headroom(monkeypatch):
    monkeypatch.delenv("FIRSTCUT_RAM_GATE", raising=False)
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (0.4, 18.0))   # phys tight, commit ample
    assert mp.free_ram_gb() == 18.0
    assert mp.physical_free_gb() == 0.4


def test_physical_gate_still_available(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_RAM_GATE", "physical")
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (0.4, 18.0))
    assert mp.free_ram_gb() == 0.4


def test_commit_exhaustion_still_gates(monkeypatch):
    monkeypatch.delenv("FIRSTCUT_RAM_GATE", raising=False)
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (6.0, 0.7))    # small pagefile, full
    assert mp.free_ram_gb() == 0.7


def test_phase_a_uses_physical_memory(monkeypatch):
    import grade_pipeline_v2 as gp
    monkeypatch.setattr(mp, "_global_memory_status", lambda: (1.0, 18.0))
    monkeypatch.delenv("FIRSTCUT_PHASE_A", raising=False)
    gp._phase_a_start(["a.jpg"])
    assert gp._PHASE_A.get("proc") is None       # 1.0 GB physical < 3.0: not started


def test_power_throttling_opt_out_never_raises():
    import proc_qos
    assert proc_qos.opt_out_power_throttling() in (True, False)
