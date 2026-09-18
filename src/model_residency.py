"""model_residency — the one coordinator for RESIDENT models (2026-09-16).

Before this module, model lifecycle was six ad-hoc mechanisms scattered
across modules (RAM floors, VRAM re-checks, keep-resident env vars, unload
ladders, release-on-finish) — each protecting ONE model, none aware of the
others. This registry makes residency a single decision point:

    residency.register(name, unload_fn, ram_gb)   # modules call this at load
    residency.touch(name)                          # mark recently used
    residency.release(name)                        # targeted unload
    residency.evict_until(free_ram_needed_gb)      # LRU eviction under pressure
    residency.release_all()                        # end-of-run teardown

Registered unload functions must never raise (wrap internally). Estimates are
approximate — eviction order is LRU, sizing is best-effort. Not thread-safe
by design: all current callers run on the background executor's single lane."""

import time
from collections import OrderedDict

_entries = OrderedDict()   # name -> {"unload": fn, "ram_gb": float, "last_used": ts}


def register(name: str, unload_fn, ram_gb: float = 0.0) -> None:
    """Register (or refresh) a resident model with its unload handle."""
    _entries[name] = {"unload": unload_fn, "ram_gb": ram_gb,
                      "last_used": time.time()}
    _entries.move_to_end(name)


def touch(name: str) -> None:
    if name in _entries:
        _entries[name]["last_used"] = time.time()
        _entries.move_to_end(name)


def release(name: str) -> bool:
    e = _entries.pop(name, None)
    if e is None:
        return False
    try:
        e["unload"]()
        print(f"[residency] released {name} (~{e['ram_gb']} GB)", flush=True)
        return True
    except Exception as exc:
        print(f"[residency] release of {name} failed: {exc}", flush=True)
        return False


def release_all() -> int:
    n = 0
    for name in list(_entries.keys()):
        if release(name):
            n += 1
    return n


def evict_idle(max_idle_s: float) -> list:
    """Release every registered model idle for longer than `max_idle_s`.

    The server RAM watchdog calls this each tick so residency is ONE mechanism:
    any model that registers itself at load time gets idle-unloaded, instead of
    the watchdog hardcoding module names. Returns the names released.
    """
    import time as _t
    now = _t.time()   # registry stamps last_used with time.time() — must match
    stale = [n for n, e in _entries.items() if now - e["last_used"] > max_idle_s]
    return [n for n in stale if release(n)]


def evict_until(free_ram_needed_gb: float, free_ram_fn=None) -> int:
    """LRU-evict resident models until free RAM can fund `needed_gb`, or
    nothing is left. `free_ram_fn` injects the measurement (defaults to
    memory_plan.free_ram_gb). Returns how many models were evicted."""
    import gc
    if free_ram_fn is None:
        try:
            import memory_plan as _mp
            def free_ram_fn(): return _mp.free_ram_gb()
        except Exception:
            return 0
    evicted = 0
    while _entries:
        try:
            free = free_ram_fn()
        except Exception:
            free = None
        if free is None or free >= free_ram_needed_gb:
            break
        oldest = next(iter(_entries))
        release(oldest)
        evicted += 1
        gc.collect()
    return evicted


def status() -> dict:
    return {name: {"ram_gb": e["ram_gb"],
                   "age_s": round(time.time() - e["last_used"], 1)}
            for name, e in _entries.items()}
