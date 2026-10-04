"""Deterministic work counters for the performance guard (2026-10-04).

Wall-clock time on a desktop swings with whatever else is running (the same
600-photo cull measured 156 s and 515 s on one day). The COUNT of expensive
operations does not: decodes per photo, model loads, detector images, worker
starts. scripts/perf_guard.py compares these exactly against a recorded
baseline, so a structural slowdown — a stage that starts decoding every photo
again — fails loudly no matter how busy the machine is.

Off unless FIRSTCUT_WORK_COUNTERS names a directory (zero cost in normal
culls). Every process — grade runner, encode/iqa/detect workers — writes its
own JSON file there on flush(); the guard sums them. Workers leave through
os._exit, which skips atexit, so they call flush() explicitly.
"""
from __future__ import annotations

import os
import threading
from collections import Counter

_DIR = os.environ.get("FIRSTCUT_WORK_COUNTERS", "").strip()
_LOCK = threading.Lock()
_COUNTS: Counter = Counter()


def enabled() -> bool:
    return bool(_DIR)


def bump(name: str, n: int = 1) -> None:
    if not _DIR:
        return
    with _LOCK:
        _COUNTS[name] += n


def flush(tag: str = "") -> None:
    """Write this process's counts (cumulative) to <dir>/<pid>[-tag].json."""
    if not _DIR:
        return
    import json
    try:
        os.makedirs(_DIR, exist_ok=True)
        with _LOCK:
            data = dict(_COUNTS)
        name = f"{os.getpid()}{('-' + tag) if tag else ''}.json"
        tmp = os.path.join(_DIR, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, os.path.join(_DIR, name))
    except Exception:
        pass   # measurement must never break a cull


def collect(directory: str) -> dict:
    """Sum every process file in `directory` (used by the guard)."""
    import json
    total: Counter = Counter()
    try:
        for fn in os.listdir(directory):
            if fn.endswith(".json"):
                with open(os.path.join(directory, fn), encoding="utf-8") as f:
                    total.update(json.load(f))
    except FileNotFoundError:
        pass
    return dict(total)


def install_exit_flush(tag: str = "") -> None:
    """Flush on ANY exit of this process: atexit for normal exits, and a
    wrapper around os._exit, which the workers use and which skips atexit."""
    if not _DIR:
        return
    import atexit
    atexit.register(flush, tag)
    _orig = os._exit
    if getattr(_orig, "_wc_wrapped", False):
        return

    def _exit(code):
        flush(tag)
        _orig(code)
    _exit._wc_wrapped = True
    os._exit = _exit
