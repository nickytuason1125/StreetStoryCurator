"""grading._dir_ok's retry must use asyncio.sleep, never time.sleep.

grade_photos_v2_stream resolves each requested folder through _dir_ok's retry
(a USB/SD card can be momentarily asleep at submit time). A blocking
time.sleep() there freezes the WHOLE FastAPI event loop for up to ~4s per
unreachable folder — every other client's SSE progress, thumbnail requests,
and health polling stall too, not just this request. Proven two ways:
  1. real time.sleep is poisoned — the test fails outright if the retry ever
     calls it instead of asyncio.sleep
  2. a coroutine scheduled concurrently with the retry keeps making real
     progress (ticks recorded), proving the event loop was never blocked
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (circular with routers.grading — load first)
import routers.grading as grading  # noqa: E402


def test_dir_ok_retries_without_blocking_the_event_loop(monkeypatch, tmp_path):
    missing = str(tmp_path / "not_a_real_folder")

    def _poisoned_sleep(seconds):
        raise AssertionError(
            f"time.sleep({seconds}) was called — _dir_ok's retry must use "
            f"asyncio.sleep so it doesn't freeze the whole event loop")
    monkeypatch.setattr(time, "sleep", _poisoned_sleep)

    real_sleep = asyncio.sleep
    sleep_calls: list[float] = []

    async def _fast_sleep(seconds):
        sleep_calls.append(seconds)
        await real_sleep(0)   # yield control to the loop; don't actually wait 2s
    monkeypatch.setattr(grading.asyncio, "sleep", _fast_sleep)

    ticks: list[int] = []

    async def _ticker():
        """A totally independent coroutine. If _dir_ok's retry blocked the
        loop (e.g. via time.sleep), this could not interleave and run."""
        for i in range(5):
            await real_sleep(0)
            ticks.append(i)

    async def _run():
        return await asyncio.gather(grading._dir_ok(missing), _ticker())

    result, _ = asyncio.run(_run())

    assert result is False, "a folder that never appears must resolve to False"
    assert sleep_calls == [2.0, 2.0], f"expected exactly 2 retry sleeps of 2.0s, got {sleep_calls}"
    assert ticks == [0, 1, 2, 3, 4], "the ticker must run to completion interleaved with the retry"


def test_dir_ok_recovers_if_the_folder_appears_mid_retry(monkeypatch, tmp_path):
    """The exact USB/SD-card wake-up case this retry exists for."""
    target = tmp_path / "card_waking_up"

    real_sleep = asyncio.sleep
    calls = {"n": 0}

    async def _sleep_then_create(seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            target.mkdir()   # the card "wakes up" during the first wait
        await real_sleep(0)
    monkeypatch.setattr(grading.asyncio, "sleep", _sleep_then_create)
    monkeypatch.setattr(time, "sleep", lambda s: (_ for _ in ()).throw(
        AssertionError("must not use time.sleep")))

    result = asyncio.run(grading._dir_ok(str(target)))
    assert result is True
    assert calls["n"] == 1, "should stop retrying as soon as the folder appears"
