"""Adaptive concurrency permits for the thumbnail pools (2026-09-14).

Why: the thumbnail worker pools used to be sized ONCE at server boot from a
free-RAM snapshot and frozen for the process lifetime. A boot at 2-3 GB free
pinned the grid to 2 concurrent decodes FOR EVER - even after Chrome closed
and 5 GB freed up, until the next restart. The machine gets faster; the pool
never noticed.

The fix: create the pools at the HARDWARE ceiling (worker count never changes)
and gate each decode with a permit whose LIMIT is retuned every 30 s from the
live free-RAM table. Same table the pools always used - it just became live:

    free >= 8 GB -> 8 permits   >= 5 -> 6   >= 3 -> 4   below -> 2

A permit is a counted gate, not a thread: the executor keeps its full ceiling,
and each decode waits for a permit before touching the disk. Waiting decodes
resume automatically the moment the limit rises (Chrome closed, encoder warm),
so the grid heals itself without a restart.
"""
from __future__ import annotations

import threading
import time


class AdaptivePermits:
    """A counted gate with a limit that can be retuned while waiters wait.

    acquire() blocks until a permit is free under the CURRENT limit (bounded
    by `timeout`), release() returns one and wakes waiters. Raising the limit
    wakes every waiter immediately - a grid parked at 2-wide during a memory
    squeeze re-widens the moment RAM frees, with no restart.
    """

    def __init__(self, limit: int, name: str = "") -> None:
        self._cv = threading.Condition()
        self._name = name
        self._limit = max(1, int(limit))
        self._held = 0

    @property
    def limit(self) -> int:
        with self._cv:
            return self._limit

    @property
    def held(self) -> int:
        with self._cv:
            return self._held

    def set_limit(self, limit: int) -> None:
        """Retune live. Raising wakes waiters (they re-check immediately);
        lowering never revokes a held permit - existing decodes finish, new
        ones wait for the tighter budget."""
        with self._cv:
            self._limit = max(1, int(limit))
            self._cv.notify_all()

    def acquire(self, timeout: float = 30.0) -> bool:
        """Wait up to `timeout` seconds for a permit. False = starved (caller
        decides: skip for background work, degrade for user-facing work)."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cv:
            while True:
                if self._held < self._limit:
                    self._held += 1
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=min(1.0, remaining))

    def release(self) -> None:
        with self._cv:
            self._held = max(0, self._held - 1)
            self._cv.notify_all()
