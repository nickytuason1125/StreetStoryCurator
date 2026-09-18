"""AdaptivePermits: the live-retunable concurrency gate behind the thumbnail
pool (src/adaptive_permits.py, 2026-09-14). The failure class this prevents:
a pool frozen at a boot-RAM snapshot stays narrow for ever even after the
machine frees up - the slow-grid incident of 2026-09-14."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adaptive_permits import AdaptivePermits  # noqa: E402


def test_acquire_release_roundtrip():
    p = AdaptivePermits(2)
    assert p.acquire(timeout=1)
    assert p.acquire(timeout=1)
    assert p.held == 2
    p.release(); p.release()
    assert p.held == 0


def test_starved_acquire_returns_false_not_hang():
    p = AdaptivePermits(1)
    p.acquire(timeout=1)
    t0 = time.monotonic()
    assert p.acquire(timeout=0.2) is False, "starved acquire must return, not block"
    assert time.monotonic() - t0 < 2.0
    p.release()


def test_raising_limit_wakes_waiters():
    """THE behaviour the slow-grid fix depends on: waiters parked during a
    memory squeeze must proceed the moment the limit rises - no restart."""
    p = AdaptivePermits(1)
    p.acquire(timeout=1)
    got = []
    def _waiter():
        if p.acquire(timeout=5):
            got.append(True)
            p.release()
    th = threading.Thread(target=_waiter); th.start()
    time.sleep(0.2)
    assert not got, "waiter must still be parked under the tight limit"
    p.set_limit(2)                      # RAM freed -> pool re-widens live
    th.join(timeout=5)
    assert got, "raising the limit must wake the parked waiter"


def test_lowering_limit_never_revokes_held_permits():
    p = AdaptivePermits(4)
    held = [p.acquire(timeout=1) for _ in range(3)]
    assert all(held)
    p.set_limit(1)                      # tighten: new acquires wait...
    assert p.acquire(timeout=0.2) is False
    for _ in held: p.release()
    assert p.acquire(timeout=1), "...but the gate works again once drained"


def test_limit_floor_is_one():
    p = AdaptivePermits(0)
    assert p.limit >= 1, "a zero/negative limit must clamp to a workable gate"
    assert p.acquire(timeout=1)
