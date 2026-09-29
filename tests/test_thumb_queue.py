"""Contract tests for src/thumb_queue.ThumbQueue.

The queue exists because unordered thumbnail work made card dumps crawl
(2026-09-28): background prewarm and off-screen requests competed with the
tiles the user was looking at. These tests lock the ordering and dedup rules.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from thumb_queue import ThumbQueue  # noqa: E402


class _Gate:
    """A render function that blocks until released and records call order."""

    def __init__(self):
        self.order: list = []
        self.calls: dict = {}
        self.release = threading.Event()
        self.first_started = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, key):
        with self._lock:
            self.calls[key] = self.calls.get(key, 0) + 1
        self.first_started.set()
        self.release.wait(5)
        with self._lock:
            self.order.append(key)
        return f"done:{key}"


def _blocked_queue(**kw):
    """One worker, parked on a 'blocker' job so later submissions queue up."""
    gate = _Gate()
    q = ThumbQueue(gate, workers=1, **kw)
    q.request("blocker")
    assert gate.first_started.wait(5)
    return q, gate


def test_urgent_runs_before_background_and_newest_first():
    q, gate = _blocked_queue()
    futs = [q.request("bg1", urgent=False), q.request("bg2", urgent=False),
            q.request("u1"), q.request("u2")]
    gate.release.set()
    for f in futs:
        f.result(5)
    assert gate.order == ["blocker", "u2", "u1", "bg1", "bg2"]


def test_duplicate_requests_share_one_render():
    q, gate = _blocked_queue()
    a, b = q.request("x"), q.request("x")
    assert a is b
    gate.release.set()
    assert a.result(5) == "done:x"
    assert gate.calls["x"] == 1


def test_urgent_request_promotes_queued_background_job():
    q, gate = _blocked_queue()
    bg = [q.request(f"bg{i}", urgent=False) for i in range(3)]
    promoted = q.request("bg2")            # user scrolled to it
    assert promoted is bg[2]
    gate.release.set()
    for f in bg:
        f.result(5)
    assert gate.order[:2] == ["blocker", "bg2"]
    assert gate.calls["bg2"] == 1          # never rendered twice


def test_background_skipped_when_told_but_urgent_always_runs():
    skip = threading.Event()
    skip.set()
    q, gate = _blocked_queue(skip_background=skip.is_set)
    bg = q.request("bg", urgent=False)
    u = q.request("u")
    gate.release.set()
    assert u.result(5) == "done:u"
    assert bg.result(5) is None
    assert "bg" not in gate.calls


def test_failed_render_resolves_none_and_frees_the_key():
    calls = []

    def boom(key):
        calls.append(key)
        raise RuntimeError("corrupt file")

    q = ThumbQueue(boom, workers=1)
    assert q.request("bad").result(5) is None
    assert q.request("bad").result(5) is None   # retried, not stuck
    assert calls == ["bad", "bad"]
    assert q.pending() == 0
