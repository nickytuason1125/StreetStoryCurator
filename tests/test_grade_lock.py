"""
grade_lock regression tests — the cross-process arm of the 409 gates.

The 2026-09-07 duplicate-stack incident: two server stacks each had their own
in-process `_grading_active` flag, so a clear landing on the IDLE stack deleted
catalog.json under the OTHER stack's grade, and two grade_runners ran the same
request file. routers/grading.py and routers/misc.py now also consult
cache/grading.lock via src/grade_lock.grade_in_progress(). These tests pin the
lock semantics that make that safe:

  1. No lock file → not held (fail-open).
  2. Our own PID → held (grade_worker child of this server).
  3. A dead PID → stale: not held AND the file is removed, so a crashed grade
     can never wedge the gate.
  4. A PID now owned by a non-Python process (PID reuse) → stale: not held and
     removed.

Hermetic — no models, no subprocesses, no server. Run:

    venv\\Scripts\\python.exe -m pytest tests/test_grade_lock.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.grade_lock import grade_in_progress, lock_path, sweep_stale  # noqa: E402


def _lock(tmp_path: Path, pid: int) -> Path:
    p = lock_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid), encoding="utf-8")
    return p


def test_missing_lock_is_not_held(tmp_path):
    assert grade_in_progress(tmp_path) is False


def test_self_pid_is_held(tmp_path):
    _lock(tmp_path, os.getpid())
    assert grade_in_progress(tmp_path) is True


def test_dead_pid_lock_is_stale_and_removed(tmp_path):
    p = _lock(tmp_path, 99999999)  # no such process on any realistic box
    assert grade_in_progress(tmp_path) is False
    assert not p.exists(), "stale lock must be removed on read"


def test_non_python_pid_is_treated_as_stale(tmp_path):
    import psutil
    victim = next(
        p for p in psutil.process_iter(["name"])
        if p.info["name"] and "python" not in p.info["name"].lower()
    )
    p = _lock(tmp_path, victim.pid)
    assert grade_in_progress(tmp_path) is False
    assert not p.exists(), "PID-reused lock must be removed on read"


def test_garbage_contents_fail_open(tmp_path):
    p = lock_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not-a-pid", encoding="utf-8")
    assert grade_in_progress(tmp_path) is False


# ── sweep_stale: startup hygiene (server_impl.lifespan) ─────────────────────
def test_sweep_stale_removes_dead_pid_lock(tmp_path):
    p = lock_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("99999999", encoding="utf-8")
    assert sweep_stale(tmp_path) is True
    assert not p.exists()


def test_sweep_stale_leaves_own_pid_lock(tmp_path):
    p = lock_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(os.getpid()), encoding="utf-8")
    assert sweep_stale(tmp_path) is False
    assert p.exists(), "own/live lock must never be swept"


def test_sweep_stale_missing_lock_is_noop(tmp_path):
    assert sweep_stale(tmp_path) is False
