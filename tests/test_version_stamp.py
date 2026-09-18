"""Regression tests for the /api/version build stamp (2026-09-14).

The original bug: the stamp embedded time() evaluated at REQUEST time, so
every call returned a different "build" and the UI's version handshake
reloaded the window in an infinite loop. The stamp must be STABLE for the
life of the server process and change only across restarts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server_impl  # noqa: F401,E402  — must come first: routers.* are mounted by it
from routers.system import backend_version  # noqa: E402


def _build_of() -> str:
    import asyncio, json as _json
    resp = asyncio.new_event_loop().run_until_complete(backend_version())
    return _json.loads(resp.body)["build"]


def test_stamp_is_stable_across_calls():
    """Two calls seconds apart must return the IDENTICAL stamp."""
    import time
    b1 = _build_of()
    time.sleep(1.1)
    b2 = _build_of()
    assert b1 == b2, f"stamp changed between calls: {b1} -> {b2}"


def test_stamp_changes_only_when_start_epoch_changes():
    """Patching the captured start epoch must change the stamp — this proves
    the stamp is wired to process lifetime, not to the clock."""
    import routers.system as rs
    b1 = _build_of()
    old = rs._STARTED_EPOCH_CACHE
    try:
        rs._STARTED_EPOCH_CACHE = old + 10_000
        b2 = _build_of()
    finally:
        rs._STARTED_EPOCH_CACHE = old
    assert b1 != b2, "stamp did not react to a changed start epoch"
    assert b1.rpartition("-")[0] == b2.rpartition("-")[0], "mtime component moved"


def test_stamp_format_is_parseable():
    b = _build_of()
    mtime_part, _, start_part = b.rpartition("-")
    assert mtime_part.isdigit() and start_part.isdigit(), f"malformed stamp: {b}"
