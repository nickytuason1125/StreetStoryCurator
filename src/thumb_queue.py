"""One work queue for thumbnail generation.

Replaces the two thread pools (on-demand + prewarm), the adaptive RAM permits
and the in-flight dedup registry that grew around them (2026-09-06 .. 09-15).
Those layers each patched a symptom of one cause: nothing ordered the work, so
background jobs and off-screen requests competed with the tiles the user was
looking at. This queue orders it:

- URGENT jobs (a browser is waiting on /api/thumb) always run before
  BACKGROUND ones (folder-open prewarm), and newest-first among themselves:
  the most recent request is the tile most likely still on screen.
- One job per key. A second request for a key that is queued or running gets
  the same Future; an urgent request for a queued background job promotes it.
- Background jobs are dropped (their Future resolves None) when `skip_background`
  says so — during a grade, or when RAM is short. Urgent jobs always run.

Rendering itself is the caller's function; this module knows nothing about
images, so it can be tested with plain callables.
"""
from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future
from typing import Callable, Optional


class _Job:
    __slots__ = ("key", "arg", "future", "urgent", "started")

    def __init__(self, key: str, arg, urgent: bool):
        self.key = key
        self.arg = arg
        self.future: Future = Future()
        self.urgent = urgent
        self.started = False


class ThumbQueue:
    def __init__(self, render: Callable, workers: int = 6,
                 skip_background: Optional[Callable[[], bool]] = None,
                 name: str = "thumb"):
        self._render = render
        self._skip_background = skip_background or (lambda: False)
        self._cv = threading.Condition()
        self._urgent: deque = deque()      # LIFO: pop() from the right
        self._background: deque = deque()  # FIFO: popleft()
        self._jobs: dict = {}              # key -> _Job (queued or running)
        for i in range(max(1, workers)):
            threading.Thread(target=self._worker, daemon=True,
                             name=f"{name}-{i}").start()

    def request(self, key: str, arg=None, urgent: bool = True) -> Future:
        """Queue `render(arg)` under `key`; returns the job's Future."""
        with self._cv:
            job = self._jobs.get(key)
            if job is not None:
                if urgent and not job.urgent and not job.started:
                    # Promote: the stale background entry is skipped when popped.
                    job.urgent = True
                    self._urgent.append(job)
                    self._cv.notify()
                return job.future
            job = _Job(key, key if arg is None else arg, urgent)
            self._jobs[key] = job
            (self._urgent if urgent else self._background).append(job)
            self._cv.notify()
            return job.future

    def pending(self) -> int:
        with self._cv:
            return len(self._jobs)

    def _next(self) -> _Job:
        with self._cv:
            while True:
                while self._urgent:
                    job = self._urgent.pop()
                    if not job.started:
                        job.started = True
                        return job
                while self._background:
                    job = self._background.popleft()
                    if job.started or job.urgent:
                        continue   # already running, or promoted (in _urgent)
                    job.started = True
                    return job
                self._cv.wait()

    def _worker(self) -> None:
        while True:
            job = self._next()
            result = None
            try:
                if job.urgent or not self._skip_background():
                    result = self._render(job.arg)
            except Exception:
                result = None      # a failed render is a missing thumb, not a hang
            finally:
                with self._cv:
                    if self._jobs.get(job.key) is job:
                        del self._jobs[job.key]
                job.future.set_result(result)
