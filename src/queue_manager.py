"""
Background Annotation Queue Manager.

Event-driven design: waits on asyncio.Queue instead of polling LanceDB every 30 s.
grade_pipeline_v2 / server.py pushes image paths into the queue upon grading completion.
The annotation daemon acquires gpu_lock before loading the GGUF model so it never
runs concurrently with the grader and never busts the 5.5 GB VRAM ceiling.

Legacy start() / stop() are kept as no-ops for any callers that have not yet been
updated to the new interface.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import model_registry as _mr
_GGUF_PATH = _mr.vision_gguf_path()
_LOCK_PATH = Path(__file__).resolve().parent.parent / "cache" / "grading.lock"


def _grade_worker_running() -> bool:
    """True only if a grade worker is ACTIVELY grading (lock file present AND its
    PID alive). A stale lock (worker hard-killed) returns False so annotations are
    never blocked forever. This gates annotation independently of gpu_lock: when a
    grade's SSE stream disconnects (window closed) gpu_lock is released early while
    the worker keeps grading, so gpu_lock alone would let the annotation model
    (Ollama qwen2.5vl:3b, ~3-4 GB) load and OOM the still-running grade."""
    try:
        if not _LOCK_PATH.exists():
            return False
        _pid = int((_LOCK_PATH.read_text(encoding="utf-8").strip() or "0"))
        if _pid <= 0:
            return False
        import psutil
        return psutil.pid_exists(_pid)
    except Exception:
        return False


def _free_ram_gb() -> float:
    """Currently available system RAM, or a permissive default if psutil fails."""
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 99.0


def _annotate_min_ram_gb() -> float:
    """Free-RAM floor below which annotation waits. FRAMEGRADE_ANNOTATE_MIN_RAM_GB."""
    import os
    try:
        return float(os.environ.get("FRAMEGRADE_ANNOTATE_MIN_RAM_GB", "3.0"))
    except (TypeError, ValueError):
        return 3.0


def _annotation_backend_available() -> bool:
    """Return True if the local Qwen2.5-VL GGUF is on disk.

    This used to also accept "Ollama answers on 11434", which made the daemon
    start on a machine that had a running Ollama but none of FrameGrade's own
    weights — every annotation then failed one at a time, in the background,
    where nobody saw it. One backend, one check.
    """
    return _GGUF_PATH.exists()


def _annotate_one(path: str) -> None:
    """Synchronous annotation call — safe to run in a thread-pool executor."""
    try:
        import lance_store as _ls
        import critique_engine as _ce
        image_hash = Path(path).stem
        if not image_hash:
            return
        result = _ce.run_audit_annotation(image_hash)
        factors = result.get("score_factors", [])
        _ls.update_annotations(path, factors)
        if result.get("error"):
            print(f"[qm] Annotation error for {Path(path).name}: {result['error']}")
    except Exception as _e:
        print(f"[qm] Annotation error for {path}: {_e}")


async def start_async(queue: asyncio.Queue, gpu_lock: asyncio.Lock) -> None:
    """
    Async annotation daemon.

    Blocks at `await queue.get()` (0% CPU when idle).
    Acquires gpu_lock before calling the GGUF model so grading and annotation
    are never in VRAM simultaneously.
    """
    if not _annotation_backend_available():
        print("[qm] annotation model not installed — daemon idle")
        return

    print("[qm] Async annotation daemon started")
    loop = asyncio.get_running_loop()
    while True:
        path = await queue.get()
        try:
            # Never annotate while a grade worker is actively grading — the
            # annotation model would collide with the grade's Qwen and OOM it.
            # (gpu_lock is insufficient: a disconnected grade stream releases it
            # early while the worker keeps running — see _grade_worker_running.)
            _spins = 0
            while _grade_worker_running() and _spins < 1800:   # cap ~30 min
                await asyncio.sleep(1.0)
                _spins += 1
            # RAM backpressure. A finished cull hands this daemon one job per
            # graded photo, and each annotation keeps a 1.5-4 GB VL model
            # resident — a long sustained tail landing right after the heaviest
            # part of the run. Waiting for headroom changes only WHEN a photo is
            # annotated, never WHICH, and the bounded spin means a permanently
            # tight machine still makes progress rather than stalling forever.
            _floor  = _annotate_min_ram_gb()
            _rspins = 0
            while _free_ram_gb() < _floor and _rspins < 600:    # cap ~10 min
                if _rspins == 0:
                    print(f"[qm] RAM {_free_ram_gb():.1f} GB < {_floor:.1f} GB floor "
                          f"— deferring annotation of {Path(path).name}")
                await asyncio.sleep(1.0)
                _rspins += 1
            async with gpu_lock:
                await loop.run_in_executor(None, _annotate_one, path)
        except Exception as _e:
            print(f"[qm] Daemon error for {path}: {_e}")
        finally:
            queue.task_done()


# ── Legacy threading interface (no-ops) ──────────────────────────────────────
# Kept so any caller that still calls start()/stop() does not error out.

def start() -> None:
    pass


def stop() -> None:
    pass
