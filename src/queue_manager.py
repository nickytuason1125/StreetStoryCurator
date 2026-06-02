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

_GGUF_PATH = Path(__file__).resolve().parent.parent / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"


def _gguf_or_ollama_available() -> bool:
    """Return True if Qwen2.5-VL GGUF exists OR Ollama is reachable."""
    if _GGUF_PATH.exists():
        return True
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


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
    if not _gguf_or_ollama_available():
        print("[qm] No GGUF or Ollama backend — annotation daemon idle")
        return

    print("[qm] Async annotation daemon started")
    loop = asyncio.get_running_loop()
    while True:
        path = await queue.get()
        try:
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
