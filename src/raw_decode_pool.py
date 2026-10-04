"""
Process-pool RAW preview decode (2026-09-23).

Why a process pool and not more threads: early_exit_gate.py already learned
this the hard way — LibRaw (which rawpy wraps) is NOT reliably thread-safe,
and decoding RAW files concurrently across threads native-crashed the grade
worker with 0xC0000005 (see its `_technical_inspect` docstring). Separate OS
PROCESSES each get their own LibRaw instance with no shared native state, so
that crash class cannot occur here. Only compact bytes cross the process
boundary — never a live rawpy/PIL object — so IPC stays cheap.

This module exists because encode_worker.py's decode-prefetch pipeline used a
single decode thread for every file. That is fine for JPEGs (PIL draft-mode
decode is milliseconds) but became the real bottleneck on a large RAW-only
cull: thousands of sequential rawpy.imread() + extract_thumb() calls, each
opening and parsing a full RAW container one at a time, landing inside the
single progress band capped at 46% ("Analyzing N photos...") — reported as
the cull "stalling at 46%" on a 4,000-RAW folder.

Deliberately its OWN module, not a function inside encode_worker.py: on
Windows, spawning a process needs to import the module the target function
lives in. If that function lived in encode_worker.py, every decode worker
would re-run encode_worker's top-level `import torch` / `import
onnxruntime` — paying that cost per worker for no reason. Kept here, a
worker only needs `rawpy`/`numpy`.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor


def _decode_raw_thumb(path: str):
    """Runs in a worker PROCESS. Returns (kind, payload):
        ("jpeg",  bytes)                         embedded thumb was JPEG (common case)
        ("array", (raw_bytes, shape, dtype_str))  rare BITMAP-format thumb
        (None,    "error text")                   unreadable / no embedded thumb
    """
    try:
        import rawpy
        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return "jpeg", thumb.data
        import numpy as np
        arr = np.asarray(thumb.data)
        return "array", (arr.tobytes(), arr.shape, str(arr.dtype))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def worker_count() -> int:
    """How many decode processes to run. This is LibRaw-parse + I/O bound,
    not compute bound, and each worker's own Python startup is a real cost
    paid once — so this stays well below core count rather than maxing it
    out. FIRSTCUT_RAW_DECODE_WORKERS overrides."""
    override = os.environ.get("FIRSTCUT_RAW_DECODE_WORKERS", "")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(4, (os.cpu_count() or 4) - 1))


_pool: "ProcessPoolExecutor | None" = None
_assigned_pids: set = set()


def get_pool() -> ProcessPoolExecutor:
    """Lazily create the pool once and reuse it for the whole encode — a
    fresh pool per batch would pay process-startup cost thousands of times
    over a 4,000-photo cull."""
    global _pool
    if _pool is None:
        _n = worker_count()
        _pool = ProcessPoolExecutor(max_workers=_n)
        print(f"[raw_decode_pool] started {_n} RAW decode worker process(es)", flush=True)
    return _pool


def tie_workers_to_job() -> None:
    """Best-effort: assign every currently-spawned pool worker to this
    process's Windows Job Object, so a killed/crashed encode_worker takes its
    decode processes down with it instead of orphaning them — the same class
    of bug win_job.py exists to fix for encode_worker/iqa_worker themselves
    (2026-07-21 incident: an orphaned worker silently held ~9 GB RAM).
    Windows normally places a process's children into its own job
    automatically when jobs nest, so this is largely a defensive double-check
    rather than the only line of defense. Any failure — non-Windows, no
    pywin32, pool internals unavailable — is a silent no-op, matching
    win_job's own best-effort contract."""
    if _pool is None:
        return
    try:
        import win_job
        procs = getattr(_pool, "_processes", None) or {}
        for pid in list(procs.keys()):
            if pid not in _assigned_pids:
                win_job.assign(pid)
                _assigned_pids.add(pid)
    except Exception:
        pass


def reset_pool() -> None:
    """Tear down a broken/hung pool so the next get_pool() call starts fresh.
    Callers fall back to serial decode for whatever was in flight."""
    global _pool, _assigned_pids
    if _pool is not None:
        try:
            _pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    _pool = None
    _assigned_pids = set()


def shutdown_pool() -> None:
    global _pool, _assigned_pids
    if _pool is not None:
        try:
            _pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
    _pool = None
    _assigned_pids = set()
