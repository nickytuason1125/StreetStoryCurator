"""
ram_probe.py — per-stage RAM instrument for the grading pipeline.

Every RAM guard in this codebase reads system-wide free RAM
(psutil.virtual_memory()); nothing measured what THIS process actually holds,
so a stage that spiked 1 GB looked identical to one that spiked 10 MB.

Two numbers are reported, and they mean different things:

  peak_wset  process-wide high-water set by the OS. Monotonic — it only ever
             rises — so it is the truthful "worst moment of the whole run",
             but it CANNOT be divided up between stages.
  stage max  sampled by a background thread every SAMPLE_S seconds and charged
             to whichever stage label was current at that instant. This is what
             attributes cost to a stage, and it catches transients that a
             boundary-only sample would miss entirely.

An earlier version attributed peak_wset growth to the last stage label. That
was misleading: a label stays current until the next progress call, so a label
covering a long span (the early-exit gate label runs from ~1.5% to ~50%)
absorbed every unrelated allocation in that window. The sampler fixes the
measurement; the label windows are still only as fine as the pipeline's own
progress calls, so read a stage row as "peak RSS while this stage was current".

Wiring is one line — run_v2 wraps its progress callback:

    _p = ram_probe.wrap_progress(progress)

Every stage already calls _p(frac, desc), so no call site changes and the desc
text reaching the SSE/UI is untouched.

FRAMEGRADE_RAM_TRACE=0 disables. Every path is wrapped in try/except:
instrumentation must never break a grade.
"""
from __future__ import annotations

import os
import threading

_ENABLED  = os.environ.get("FRAMEGRADE_RAM_TRACE", "1").strip() != "0"
_SAMPLE_S = 0.4

_stage_max: dict = {}          # label -> peak RSS (GB) seen while it was current
_order: list = []              # labels in first-seen order
_current: str = "startup"
_lock = threading.Lock()
_sampler: threading.Thread = None
_stop = threading.Event()


def rss_now() -> tuple:
    """(rss_gb, peak_wset_gb, sys_free_gb). Zeros if psutil is unavailable."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", mi.rss)   # peak_wset is Windows-only
        return (mi.rss / 1e9, peak / 1e9, psutil.virtual_memory().available / 1e9)
    except Exception:
        return (0.0, 0.0, 0.0)


# Wall-clock seconds charged to each stage. RAM told us WHERE the memory went;
# nothing ever measured where the TIME goes, so optimisation targets were being
# chosen by intuition. Same attribution rule as _stage_max: time is charged to
# whichever label was current, so a stage is only as precise as the pipeline's
# own progress calls.
_stage_secs: dict = {}
_stage_t0: float = 0.0


def _record(rss: float) -> None:
    with _lock:
        if _current not in _stage_max:
            _stage_max[_current] = rss
            _order.append(_current)
        elif rss > _stage_max[_current]:
            _stage_max[_current] = rss


def _sample_loop() -> None:
    while not _stop.wait(_SAMPLE_S):
        try:
            _record(rss_now()[0])
        except Exception:
            pass


def _ensure_sampler() -> None:
    global _sampler
    if _sampler is not None:
        return
    try:
        _stop.clear()
        _sampler = threading.Thread(target=_sample_loop, daemon=True)
        _sampler.start()
    except Exception:
        _sampler = None


def wrap_progress(progress):
    """Return a callable with _p's exact (frac, desc) signature that traces RAM."""
    _inner = progress or (lambda f, d: None)
    if not _ENABLED:
        return _inner
    _ensure_sampler()

    def _wrapped(frac: float, desc: str = "") -> None:
        global _current, _stage_t0
        try:
            import time as _t
            rss, peak, free = rss_now()
            _record(rss)                       # close out the stage that just ended
            with _lock:
                _now = _t.monotonic()
                if _stage_t0:                  # bank the elapsed time on the OLD label
                    _stage_secs[_current] = _stage_secs.get(_current, 0.0) + (_now - _stage_t0)
                _stage_t0 = _now
                _current = desc or f"{frac:.3f}"
            _record(rss)                       # open the new one
            # ASCII-safe: stage labels carry non-cp1252 characters (ellipsis,
            # bullets) and a console encode error here would silently kill the
            # whole trace. The label is only echoed for humans; _current keeps
            # the original text.
            _safe = (desc[:48]).encode("ascii", "replace").decode("ascii")
            print(f"[ram] {frac:.3f} rss={rss:.2f} peak={peak:.2f} free={free:.2f}  {_safe}",
                  flush=True)
        except Exception:
            pass                               # never break a grade for instrumentation
        _inner(frac, desc)

    return _wrapped


def summary() -> list:
    """Print process-wide peak plus the stages with the highest resident RSS."""
    if not _ENABLED:
        return []
    try:
        _stop.set()
        rss, peak, free = rss_now()
        _record(rss)
        with _lock:
            rows = [(lbl, _stage_max.get(lbl, 0.0)) for lbl in _order]
        top = sorted(rows, key=lambda r: r[1], reverse=True)[:8]
        # Time breakdown — the thing that actually tells you what to optimise.
        try:
            import time as _t2
            global _stage_t0
            if _stage_t0:
                _stage_secs[_current] = _stage_secs.get(_current, 0.0) + (_t2.monotonic() - _stage_t0)
            _tot = sum(_stage_secs.values()) or 1.0
            print(f"[ram] -- TIME: {_tot:.0f}s total --", flush=True)
            for lbl, sec in sorted(_stage_secs.items(), key=lambda r: -r[1])[:8]:
                _s = lbl.encode("ascii", "replace").decode("ascii")[:52]
                print(f"[ram]     {sec:6.1f}s  {100*sec/_tot:4.1f}%  {_s}", flush=True)
        except Exception:
            pass
        # ASCII only. Windows consoles default to cp1252, and a UnicodeEncodeError
        # here would be swallowed by the except below — printing nothing at all
        # unless PYTHONIOENCODING=utf-8 happened to be set.
        print(f"[ram] -- process peak {peak:.2f} GB (whole run), free now {free:.2f} GB --",
              flush=True)
        print("[ram]    peak RSS while each stage was current:", flush=True)
        for label, mx in top:
            _safe = label.encode("ascii", "replace").decode("ascii")
            print(f"[ram]     {mx:5.2f} GB  {_safe[:58]}", flush=True)
        return [{"stage": lbl, "peak_rss_gb": round(mx, 3)} for lbl, mx in top]
    except Exception:
        return []


def reset() -> None:
    """Clear the trace and stop the sampler.

    The old version set _stop and dropped the thread reference WITHOUT joining.
    _ensure_sampler() then calls _stop.clear() for the next run — and if the
    previous sampler was still inside `_stop.wait(_SAMPLE_S)` when that clear
    landed, it never observed the set and looped forever. run_v2 calls reset()
    once per folder, so a multi-folder grade leaked one psutil-polling thread
    per folder, each still writing into the shared _stage_max. Joining first
    makes the handoff deterministic; the join is bounded so instrumentation can
    never stall a grade.
    """
    global _stage_max, _order, _current, _sampler, _stage_secs, _stage_t0
    _stage_secs = {}
    _stage_t0 = 0.0
    _stop.set()
    _old = _sampler
    if _old is not None:
        try:
            _old.join(timeout=_SAMPLE_S * 3)
        except Exception:
            pass
    with _lock:
        _stage_max = {}
        _order = []
        _current = "startup"
    _sampler = None
