"""
Pipeline Calibration Warmup

Writes cache/.warmup_done on first run so the startup sequence does not
repeat this check on every launch.

Model loading (SigLIP-2, Qwen) is intentionally NOT done here:
  - BitsAndBytes INT4 hard-crashes the process on Windows when loaded from a
    background daemon thread before CUDA is properly initialised by the main
    grading path.
  - SigLIP-2 loaded here races with the user-triggered grade run → OOM crash.

Models load on-demand the first time the user clicks Grade.
"""
from __future__ import annotations
from pathlib import Path

_SENTINEL       = Path("cache/.warmup_done")
_warmup_running = False
_warmup_done    = False


def is_done() -> bool:
    global _warmup_done
    if _warmup_done:
        return True
    if _SENTINEL.exists():
        _warmup_done = True
        return True
    return False


def get_status() -> dict:
    return {
        "warmup_done":    is_done(),
        "warmup_running": _warmup_running,
    }


def run_warmup() -> None:
    """Write the sentinel and return — no model loading."""
    global _warmup_running, _warmup_done
    if is_done() or _warmup_running:
        return
    _warmup_running = True
    try:
        _mark_done()
        print("[warmup] Sentinel written — models load on first Grade run")
    except Exception as _e:
        print(f"[warmup] Sentinel write failed: {_e}")
    finally:
        _warmup_running = False


def _mark_done() -> None:
    global _warmup_done
    try:
        _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        _SENTINEL.touch()
    except OSError:
        pass
    _warmup_done = True


def reset_sentinel() -> None:
    """Remove the warmup sentinel so it runs again on next startup."""
    global _warmup_done
    try:
        if _SENTINEL.exists():
            _SENTINEL.unlink()
    except OSError:
        pass
    _warmup_done = False
