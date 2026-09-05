"""
C5 chaos invariant: the hard DISK gate refuses a cull when the app drive has
< 1 GB free, because persistence (catalog merge_write + Lance compaction)
dies with OSError Errno 28 on a full disk — the work completes but its
results vanish.

The gate lives INSIDE the grade generator (an SSE error event, not an HTTP
status), so this test drives the generator directly — same deterministic
pattern as test_grade_singleflight.py — with shutil.disk_usage patched to
report a full drive. Asserts: the refusal arrives as an SSE error event AND
the single-flight flag is cleared afterwards.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl                          # noqa: E402
from routers import grading as grading_mod  # noqa: E402
from routers.library import GradeRequest    # noqa: E402


@pytest.fixture(autouse=True)
def _flag_clear():
    server_impl._grading_active.clear()
    yield
    server_impl._grading_active.clear()


def test_disk_full_refuses_via_sse_and_clears_flag(monkeypatch, tmp_path):
    import shutil, run_profile, psutil
    # RAM gate passes deterministically; DISK gate reports 0.5 GB on the app drive
    monkeypatch.setattr(run_profile, "required_ram_gb",
                        lambda n_photos=0, scan_mode=False: 0.1)
    monkeypatch.setattr(psutil, "virtual_memory",
                        lambda: types.SimpleNamespace(available=1_000 * 1_000_000_000))
    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=500_000_000,
                                                        total=1_000_000_000_000))
    monkeypatch.setattr(grading_mod, "_release_annotation_model", lambda: None)
    monkeypatch.setattr(grading_mod, "_precull_ram_sweep", lambda: None)
    monkeypatch.setitem(sys.modules, "fast_niche_detector",
                        types.SimpleNamespace(release=lambda: None))

    async def drive():
        resp = await grading_mod.grade_photos_v2_stream(
            GradeRequest(folder_path=str(tmp_path), scan_mode=True))
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(drive())
    assert chunks, "generator must emit the disk refusal as an SSE event"
    text = "".join(chunks)
    assert "disk free" in text and "Refused" in text, text[:200]
    assert not server_impl._grading_active.is_set(), (
        "single-flight flag must clear after the disk refusal"
    )
    payload = json.loads(text.removeprefix("data: ").strip())
    assert "error" in payload