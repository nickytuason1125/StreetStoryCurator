"""
grade_lock.py — cross-process "grade in progress" gate.

grade_worker.py writes cache/grading.lock (containing its PID) for the life of
a grade and unlinks it in its finally. But the server's 409 guards only saw
the in-process `_grading_active` flag — so when a second server stack was
alive (the 2026-09-07 duplicate-stack incident: two launchers, two backends
fighting over port 8000), catalog/clear could delete catalog.json underneath
the OTHER stack's merge_write, and a second grade_runner was spawned for the
same request file.

Every API gate that used to consult only the in-process flag now also calls
grade_in_progress() here: held = a live PID that is not ours. A lock naming a
dead (or PID-reused, non-Python) process is stale and is removed on read, so
a crashed grade can never wedge the gate forever.

Fail-open: any error reading or verifying the lock reports "not held" — this
guard is a marker layered on top of the in-process flag, and per the same rule
grade_worker applies to its own lock writes, it must never block a grade.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lock_path(data_dir=None) -> Path:
    """Where the marker lives — honour the caller's data dir, else repo root."""
    base = Path(data_dir) if data_dir else ROOT
    return Path(base) / "cache" / "grading.lock"


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _is_stale(pid: int, p: Path) -> bool:
    """True — and the lock is removed — iff the PID is provably dead or was
    reused by a non-Python process. Unverifiable → False (fail-open)."""
    try:
        import psutil
    except Exception:
        return False
    try:
        name = (psutil.Process(pid).name() or "").lower()
    except psutil.NoSuchProcess:
        _remove(p)  # crashed grade left the marker behind — clear it
        return True
    except Exception:
        return False
    if "python" not in name:
        _remove(p)  # PID reused by an unrelated process — stale
        return True
    return False


def grade_in_progress(data_dir=None) -> bool:
    """True if a live foreign grade process holds cache/grading.lock.

    Stale locks (dead PID, or PID reused by a non-Python process) are removed
    on sight. Unreadable/absent/iverifiable → False (fail-open).
    """
    p = lock_path(data_dir)
    try:
        pid = int(p.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        return False
    except Exception:
        return False
    if pid == os.getpid():
        return True  # grade_worker child of this very server
    return not _is_stale(pid, p)


def sweep_stale(data_dir=None) -> bool:
    """Startup hygiene: remove cache/grading.lock iff provably stale.

    Returns True if a stale lock was removed. A live lock, our own, or an
    unreadable one is left untouched — the server must never destroy a marker
    it cannot prove is dead (a grade running in a REAL child of this process
    is protected by the pid == os.getpid() branch in grade_in_progress; a
    foreign live lock stays for the 409 gates to honour).
    """
    p = lock_path(data_dir)
    try:
        pid = int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    if pid == os.getpid():
        return False
    return _is_stale(pid, p)
