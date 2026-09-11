"""
The grade single-flight cluster (M2/M3/M4) as permanent regression tests.

These encode the three race-condition fixes as deterministic assertions that
need no RAM, no models, and no subprocesses:

  1. A second grade request while _grading_active is set is refused with 409
     BEFORE any folder resolution or model work (the handler's first gate).
  2. The handler SETS the flag at response-return time — not lazily inside
     the SSE body. The flag used to be set only inside the generator, AFTER
     `await gpu_lock.acquire()` — a yield point — so a concurrent request
     could slip between the check and the set. Proven here by getting the
     409 while request A's stream exists but its body has not been consumed.
  3. The generator's finally CLEARS the flag and RELEASES gpu_lock even when
     the prelude explodes. The prelude used to run outside the try: a failure
     there leaked gpu_lock (and would now wedge the guard until restart).
  4. A RAM-gate refusal (503) must NOT leave the flag set.
  5. catalog/clear is refused with 409 while a grade holds the flag (M3) —
     a clear racing merge_write is last-writer-wins silent data loss.

The handlers are plain async functions, driven directly on an event loop —
no TestClient, no HTTP. The RAM gate is patched to be deterministic on any
machine (the real gate depends on live free memory, which differs everywhere).

The generator is never allowed past its prelude: _precull_ram_sweep is
patched to raise (or stubbed no-op where noted), so no grade_runner
subprocess is ever spawned and no model is ever loaded. Run:

    venv\\Scripts\\python.exe -m pytest tests/test_grade_singleflight.py -v
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl                                   # noqa: E402  (mounts routers)
from routers import grading as grading_mod           # noqa: E402
from routers import misc as misc_mod                 # noqa: E402
from routers.library import GradeRequest             # noqa: E402


@pytest.fixture(autouse=True)
def _flag_clear():
    """Never let a failing test leave the global single-flight flag set —
    that would poison every later test in the session."""
    server_impl._grading_active.clear()
    yield
    server_impl._grading_active.clear()


@pytest.fixture(autouse=True)
def _lock_isolated(tmp_path, monkeypatch):
    """The cross-process 409 arm reads the REAL cache/grading.lock — a live
    cull on this machine (e.g. the 2026-09-07 22:48 run) would correctly 409
    every grade/clear these tests attempt. Unit tests must not depend on
    whether the developer happens to be grading, so point the lock path at a
    per-test directory: no lock file → grade_in_progress() is False, and no
    live system state can leak into the assertions.

    Patch BOTH import identities: routers do `from src.grade_lock import ...`
    while this module imported the bare `grade_lock` (both sys.path entries
    exist), and they are distinct module objects."""
    import grade_lock as top_level_lock
    from src import grade_lock as src_level_lock
    _fake = lambda data_dir=None: Path(tmp_path) / "grading.lock"  # noqa: E731
    monkeypatch.setattr(top_level_lock, "lock_path", _fake)
    monkeypatch.setattr(src_level_lock, "lock_path", _fake)


def _patch_gate_to_pass(monkeypatch):
    """Make the RAM gate deterministic: it measures live free memory, which
    differs per machine (and this repo's dev box runs at 85-97% usage).
    Belt and braces: pin required_ram_gb low AND pin psutil's free figure
    high — either alone passes the gate, so a patch-timing surprise in a
    fresh import context can never turn these into a 503."""
    import psutil, run_profile
    monkeypatch.setattr(
        run_profile, "required_ram_gb",
        lambda n_photos=0, scan_mode=False: 0.1,
    )
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=1_000 * 1_000_000_000),
    )


def _patch_prelude(monkeypatch, boom: bool):
    """No annotation model, no niche detector import, and a RAM sweep that
    either no-ops or explodes — the generator must never reach the real
    subprocess spawn in these tests."""
    monkeypatch.setattr(grading_mod, "_release_annotation_model", lambda: None)
    if boom:
        def _boom():
            raise RuntimeError("prelude exploded")
        monkeypatch.setattr(grading_mod, "_precull_ram_sweep", _boom)
    else:
        monkeypatch.setattr(grading_mod, "_precull_ram_sweep", lambda: None)
    # The generator does `import fast_niche_detector` inside its prelude —
    # stub it so the real module (a torch/CLIP import) is never loaded.
    monkeypatch.setitem(
        sys.modules, "fast_niche_detector",
        types.SimpleNamespace(release=lambda: None),
    )


# ── 1. The guard is the FIRST thing the handler does ────────────────────────
def test_second_grade_refused_409_while_flag_set():
    server_impl._grading_active.set()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            grading_mod.grade_photos_v2_stream(GradeRequest(folder_path=str(_ROOT)))
        )
    assert exc_info.value.status_code == 409


# ── 2. THE M2 PROOF: 409 while request A's stream is pending ───────────────
def test_second_grade_refused_while_stream_pending(monkeypatch, tmp_path):
    """Request A has RETURNED its StreamingResponse (the old code only set
    the flag inside the body, after an await — the window this closes).
    Request B arriving in that window must get 409, not a second runner."""
    _patch_gate_to_pass(monkeypatch)
    _patch_prelude(monkeypatch, boom=True)
    monkeypatch.setattr(grading_mod, "gpu_lock", asyncio.Lock())

    req_a = GradeRequest(folder_path=str(tmp_path), scan_mode=True)
    req_b = GradeRequest(folder_path=str(tmp_path), scan_mode=True)

    async def drive():
        resp_a = await grading_mod.grade_photos_v2_stream(req_a)
        # The set must already have happened at response-return time —
        # the old code set it lazily inside the body, after an await.
        assert server_impl._grading_active.is_set(), (
            "handler must set the single-flight flag before returning the stream"
        )
        with pytest.raises(HTTPException) as exc_info:
            await grading_mod.grade_photos_v2_stream(req_b)
        assert exc_info.value.status_code == 409
        # Now consume A's body: the prelude explodes, and the finally must
        # clean up (this is also the client-disconnect path's contract).
        with pytest.raises(RuntimeError, match="prelude exploded"):
            async for _chunk in resp_a.body_iterator:
                pass

    asyncio.run(drive())
    assert not server_impl._grading_active.is_set(), (
        "generator finally must clear the flag after the prelude failure"
    )


# ── 3. Prelude failure releases gpu_lock (the leak the old code had) ───────
def test_prelude_failure_releases_gpu_lock(monkeypatch, tmp_path):
    _patch_gate_to_pass(monkeypatch)
    _patch_prelude(monkeypatch, boom=True)
    lock = asyncio.Lock()
    monkeypatch.setattr(grading_mod, "gpu_lock", lock)

    async def drive():
        resp = await grading_mod.grade_photos_v2_stream(
            GradeRequest(folder_path=str(tmp_path), scan_mode=True)
        )
        with pytest.raises(RuntimeError, match="prelude exploded"):
            async for _chunk in resp.body_iterator:
                pass

    asyncio.run(drive())
    assert not lock.locked(), (
        "gpu_lock must be released when the generator's prelude fails — "
        "a leaked lock deadlocks every future grade"
    )


# ── 4. A refused cull (RAM gate 503) must not wedge the guard ──────────────
def test_ram_gate_refusal_leaves_flag_clear(monkeypatch, tmp_path):
    _patch_prelude(monkeypatch, boom=False)
    import psutil, run_profile
    monkeypatch.setattr(
        run_profile, "required_ram_gb",
        lambda n_photos=0, scan_mode=False: 999.0,   # refused on any machine
    )
    # The gate measures the SCARCER of physical free RAM and commit headroom
    # through GlobalMemoryStatusEx (memory_plan._global_memory_status) — since
    # that refactor, patching psutil.virtual_memory alone has no effect, and
    # the plan's LAST rung prices off the encoder hard floor, not
    # required_ram_gb. Starve both: no measurement value and a huge need →
    # even "scan+reduced-batch" cannot fit → the handler must 503.
    monkeypatch.setattr("src.memory_plan._global_memory_status", lambda: (0.5, 0.5))
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: types.SimpleNamespace(available=500_000_000),   # 0.5 GB free
    )
    resp = asyncio.run(
        grading_mod.grade_photos_v2_stream(
            GradeRequest(folder_path=str(tmp_path), scan_mode=True)
        )
    )
    assert resp.status_code == 503
    assert "Not enough RAM" in json.loads(resp.body)["error"]
    assert not server_impl._grading_active.is_set(), (
        "a 503 refusal must never leave the single-flight flag set"
    )


# ── 5. M3: catalog/clear vs a running grade ─────────────────────────────────
def test_catalog_clear_refused_409_while_grade_running():
    server_impl._grading_active.set()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(misc_mod.clear_catalog())
    assert exc_info.value.status_code == 409


def test_catalog_clear_proceeds_when_idle(monkeypatch, tmp_path):
    """Hermetic happy path: the module is pointed at a throwaway catalog and
    the backup is stubbed, so the photographer's real library is untouched."""
    import catalog_store
    fake = tmp_path / "catalog.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(misc_mod, "_CATALOG_PATH", fake)
    monkeypatch.setattr(catalog_store, "back_up", lambda *a, **k: None)
    assert asyncio.run(misc_mod.clear_catalog()) == {"ok": True}