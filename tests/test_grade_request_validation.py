"""GradeRequest.folder_path validation.

The reachability retry (three probes, up to ~4s of blocking time.sleep) used
to live in this Pydantic field_validator — a synchronous classmethod that
FastAPI runs inline while parsing the request body, so it froze the WHOLE
async event loop for every other client while it waited out a USB/SD-card
nap. That retry now lives only in routers.grading._dir_ok, which can
genuinely await. This validator does cheap synchronous path normalization
only — no existence check, no sleep, ever.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (mounts routers — must load before routers.library directly)
from routers.library import GradeRequest  # noqa: E402


def test_construction_never_sleeps(monkeypatch, tmp_path):
    """A GradeRequest for a folder that doesn't exist must build instantly —
    no retry, no blocking wait — regardless of whether the folder is real."""
    def _poisoned_sleep(seconds):
        raise AssertionError(
            f"GradeRequest construction called time.sleep({seconds}) — the "
            f"reachability retry must not live in the synchronous validator")
    monkeypatch.setattr(time, "sleep", _poisoned_sleep)

    missing = str(tmp_path / "does_not_exist_yet")
    t0 = time.monotonic()
    req = GradeRequest(folder_path=missing)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"construction took {elapsed:.2f}s — something is still blocking"
    # Normalized (resolved), but existence is NOT validated here anymore —
    # that's routers.grading._dir_ok's job, async, at request-handling time.
    assert Path(req.folder_path) == Path(missing).resolve(strict=False)


def test_real_folder_still_normalizes(tmp_path):
    req = GradeRequest(folder_path=str(tmp_path))
    assert Path(req.folder_path) == tmp_path.resolve()


def test_garbage_path_still_rejected():
    """Format validation (not existence) is still enforced."""
    with pytest.raises(Exception):
        GradeRequest(folder_path="\0invalid\0path")
