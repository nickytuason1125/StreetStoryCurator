"""System routes — moved verbatim from server_impl.py (Milestone 4 split).

Decorators retargeted app -> router; every bare name that used to live in
server_impl resolves lazily through the module __getattr__ below (PEP 562),
so request-time access always sees the fully-initialised app without
circular imports. FastAPI names are imported eagerly because decorator-time
evaluation (parameter defaults like Query(...)) runs at import.
"""
from fastapi import (
    APIRouter, Body, Depends, File, Form, HTTPException, Query, Request,
    Response, UploadFile,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, validator, model_validator

from server_impl import (  # shared state & helpers
    _GRADE_MIN_RAM_GB, _SMI_CACHE, _SMI_CACHE_S, _SMI_CACHE_TS, os, sys,
)

router = APIRouter()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.get("/api/config")
async def get_config():
    """Return runtime configuration flags consumed by the frontend."""
    try:
        from frontier_config import is_force_frontier
        ff = is_force_frontier()
    except ImportError:
        ff = False
    from __version__ import __version__ as _version
    return JSONResponse({"force_frontier": ff, "version": _version})


def _ram_payload() -> dict:
    """Shared RAM snapshot for /api/system/ram and /api/events (see system_ram)."""
    import server_impl as _si
    _free = _si._accurate_free_gb()
    _commit = _si._accurate_commit_headroom_gb()
    try:
        import psutil as _ps
        _vm = _ps.virtual_memory()
        _phys_free = round(_vm.available / 1e9, 2)
        _total = round(_vm.total / 1e9, 1)
        _pct = round(_vm.percent, 1)
    except Exception:
        _phys_free = _total = _pct = None
    return {
        # Commit-aware — the gates' figure. What "free" means for grading.
        "ram_free_gb":   round(_free, 2) if _free is not None else _phys_free,
        "ram_total_gb":  _total,
        "ram_percent":   _pct,
        "ram_min_gb":    _GRADE_MIN_RAM_GB,
        # Diagnostics: why the number is what it is.
        "ram_phys_free_gb":  _phys_free,
        "ram_commit_free_gb": round(_commit, 2) if _commit is not None else None,
        "commit_limited": bool(
            _free is not None and _phys_free is not None and _free < _phys_free - 0.25
        ),
    }


@router.get("/api/system/ram")
async def system_ram():
    """Live system-memory snapshot for the UI's RAM readiness indicator.

    Deliberately tiny (psutil only — no torch / model imports) so the frontend can
    poll it every couple of seconds and reflect Task Manager in real time.
    `percent` is memory in use (== Task Manager's headline %); `free` is the
    'Available' figure. `min_gb` is the hard cull gate (below it grading is 503).

    SYNC ACCURACY (2026-09-14): `free` is now the SAME figure the grade gates
    use — min(physical available, commit headroom) — via server_impl's
    _accurate_free_gb. The badge can no longer show "3.8 GB free" while the
    pagefile limit makes the grader refuse the same moment. The scarcer of the
    two is what gates care about; the raw physical figure and the commit
    headroom are both included so the UI (or a tooltip) can explain WHY a
    figure is low.
    """
    return JSONResponse(_ram_payload())


@router.get("/api/events")
async def api_events():
    """THE sync channel (2026-09-14): one SSE stream pushing RAM, grade state,
    and the catalog revision — each only when it CHANGES.

    Before this, the UI polled four endpoints on independent timers and could
    act on contradictory snapshots (the 45%-reload bug was exactly that). Now
    the server is the single source of truth and volunteers every change:
      {"events": {"ram":   {...same shape as /api/system/ram...},
                  "grade": {...same shape as /api/grade/state...}}}
    A missing key means "unchanged since the last event" — no news IS no news.
    Heartbeats every ~15 s keep proxies from idling the connection out; the UI's
    EventSource reconnects automatically if the stream drops, and its slow
    polls remain as a belt-and-braces fallback.
    """
    import asyncio as _aio
    import json as _json_ev
    import time as _t_ev

    async def _gen():
        from routers.grading import _grade_state_payload
        import asyncio as _aio_ev
        _loop = _aio_ev.get_running_loop()
        _last: dict = {}
        _last_emit = _t_ev.monotonic()
        while True:
            try:
                # Off-loop: psutil reads + lock-file stat/PID checks are blocking
                # calls, and this loop shares the asyncio event loop with the
                # grade's own SSE progress stream — blocking here delays every
                # connected client's progress ticks. Compute in the pool.
                _candidate = await _loop.run_in_executor(None, lambda: {
                    "ram": _ram_payload(),
                    "grade": _grade_state_payload(),
                })
            except Exception:
                _candidate = {}
            out = {}
            for _k, _v in _candidate.items():
                try:
                    _s = _json_ev.dumps(_v, sort_keys=True, default=str)
                except Exception:
                    continue
                if _last.get(_k) != _s:
                    _last[_k] = _s
                    out[_k] = _v
            try:
                if out:
                    yield f"data: {_json_ev.dumps({'events': out})}\n\n"
                    _last_emit = _t_ev.monotonic()
                elif _t_ev.monotonic() - _last_emit > 15:
                    yield ": heartbeat\n\n"
                    _last_emit = _t_ev.monotonic()
            except Exception:
                return   # client gone — end the stream cleanly
            await _aio.sleep(2)

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _server_started_epoch() -> int:
    """Server start time, captured ONCE (2026-09-14 hotfix).

    BUG this fixes: the build stamp originally embedded time() evaluated at
    REQUEST time — every call to /api/version returned a different stamp, so
    the UI's version handshake saw a "new backend" every 60 s and reloaded the
    window in an infinite loop (thumbnails never finished loading, the grid
    kept resetting). A stamp must change ONLY when the server restarts or the
    code changes — hence module-level capture.
    """
    global _STARTED_EPOCH_CACHE
    try:
        if not _STARTED_EPOCH_CACHE:
            import time as _t0
            _STARTED_EPOCH_CACHE = int(_t0.time())
        return _STARTED_EPOCH_CACHE
    except Exception:
        return 0

_STARTED_EPOCH_CACHE = 0


@router.get("/api/version")
async def backend_version():
    """Build stamp for the UI↔backend version handshake.

    The UI fetches this on mount, on window focus, and every 60 s; a changed
    stamp means the server process was restarted with newer code (or restarted
    at all), so the frontend hard-reloads rather than talking a stale contract
    to a fresh backend (or displaying a gallery against a dead one). The stamp
    mixes the newest source mtime with the server's START time — both captured
    once at boot/request-module-load, so the stamp is STABLE for the life of
    the process.
    """
    from pathlib import Path as _P
    stamp = 0.0
    _unit = _P(__file__).parent.parent
    for _f in ("server_impl.py", "grade_worker.py", "grade_runner.py",
               "src/memory_plan.py", "routers/grading.py", "routers/library.py"):
        try:
            stamp = max(stamp, (_unit / _f).stat().st_mtime)
        except Exception:
            pass
    _started = _server_started_epoch()
    return JSONResponse({
        "build": f"{int(stamp)}-{_started}",
        "started_epoch": _started,
    })


@router.get("/api/models/status")
async def model_status():
    """Return current grader mode and model availability for the frontend indicator."""
    from pathlib import Path as _P

    # Qwen2.5-VL-3B is the primary grader — HuggingFace nests safetensors in subdirs
    qwen_dir  = _P("models/qwen_vlm")
    draft_ok  = any(qwen_dir.rglob("*.safetensors")) if qwen_dir.exists() else False
    # Pre-quantised INT4 checkpoint — when absent, the first cull pauses ~2-5 min
    # at ~52% to quantise once (the frontend shows a disclaimer for this).
    int4_dir   = _P("models/qwen_vlm_int4")
    int4_cached = (int4_dir / "config.json").exists() and any(int4_dir.glob("*.safetensors"))
    # SpecVLM CLIP weights (fallback grader)
    spec_dir  = _P("models/specvlm")
    verify_ok = any(spec_dir.glob("*.safetensors")) if spec_dir.exists() else False
    # Registry, not a filename. This was the FOURTH module hardcoding the
    # same weight file; after the text model changed, a correct install
    # reported the judge missing and a stale one reported it present.
    try:
        import model_registry as _mr_health
        judge_ok = _mr_health.text_gguf_path().exists()
    except Exception:
        judge_ok = False
    # phi4-mini-reasoning is not in model_registry and never ships, so this
    # reported a permanently-missing model. Retired rather than fixed: the
    # field stays for API compatibility and is simply False.
    phi4_ok   = False

    try:
        import sys, os
        src_dir = os.path.join(os.path.dirname(__file__), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from grade_pipeline_v2 import _grader_status, _qwen_singleton, _qwen_loading
        last         = dict(_grader_status)
        qwen_warm    = _qwen_singleton is not None
        qwen_loading = _qwen_loading
    except Exception:
        last         = {"mode": "idle", "verify_used": False, "photos_last": 0, "error": None}
        qwen_warm    = False
        qwen_loading = False

    # Qwen background-download progress from model_loader
    qwen_dl_pct: int | None = None
    try:
        from model_loader import _DOWNLOAD_STATUS as _DL
        _qs = _DL.get("qwen_vlm", "pending")
        if isinstance(_qs, str) and _qs.startswith("downloading:"):
            qwen_dl_pct = int(_qs.split(":")[1])
    except Exception:
        pass

    # Pipeline calibration warmup status
    warmup_done    = False
    warmup_running = False
    try:
        from warmup_runner import get_status as _ws
        _wst = _ws()
        warmup_done    = _wst["warmup_done"]
        warmup_running = _wst["warmup_running"]
    except Exception:
        pass

    # Sidecar model residency (2026-09-16): the Qwen-VL critique + text-judge
    # GGUFs live in a disposable sidecar process. Expose what's warm so the
    # UI/telemetry can see it. None = no sidecar running at all.
    sidecar_loaded = None
    try:
        import sys, os
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        src_dir = os.path.abspath(src_dir)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from sidecar_client import probe_health as _sc_ph, read_marker as _sc_rm
        _m = _sc_rm()
        if _m:
            _h = _sc_ph(int(_m["port"]), timeout=0.8)
            if _h:
                sidecar_loaded = _h.get("loaded")
    except Exception:
        pass

    # GPU / VRAM telemetry — via nvidia-smi, NOT torch.cuda. This endpoint is polled
    # by the frontend; torch.cuda.get_device_properties/memory_reserved would give
    # this long-lived SERVER process a CUDA context, which can race the grade worker's
    # isolated GPU subprocesses. nvidia-smi is a separate process — zero CUDA state
    # here — and memory.free reports true free VRAM across all processes.
    #
    # nvidia-smi costs a full Windows process spawn (~50-150 ms) per call, and the
    # pre-grade modal polls this endpoint every 3 s. The result is cached for
    # _SMI_CACHE_S so a polling UI costs ~zero: VRAM numbers moving within a
    # 2.5 s window is imperceptible in the readiness indicator.
    vram_free_gb  = None
    vram_total_gb = None
    gpu_name      = last.get("gpu_name")
    compute_device = "unknown"
    global _SMI_CACHE, _SMI_CACHE_TS   # noqa: F841 — module-level cache vars below
    try:
        import time as _t_smi
        _now = _t_smi.monotonic()
        _have_gpu_info = False
        if _SMI_CACHE is not None and (_now - _SMI_CACHE_TS) < _SMI_CACHE_S:
            # Cache hit: use the cached numbers. (This branch used to fall
            # through to the nvidia-smi spawn below with _smi/_sp undefined —
            # a NameError swallowed by the except, leaving compute_device
            # "unknown", which the UI rendered as a CPU warning. The whole
            # point of the cache is to NOT spawn nvidia-smi here.)
            (_tot, _free, _nm) = _SMI_CACHE
            vram_total_gb = round(_tot / 1024.0, 1)
            vram_free_gb  = round(_free / 1024.0, 1)
            if not gpu_name:
                gpu_name = _nm
            _have_gpu_info = True
        else:
            import subprocess as _sp, shutil as _sh
            _smi = _sh.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
            _out = _sp.run(
                [_smi, "--query-gpu=memory.total,memory.free,name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if _out.returncode == 0 and _out.stdout.strip():
                _tot, _free, _nm = [x.strip() for x in _out.stdout.strip().splitlines()[0].split(",")]
                try:
                    _SMI_CACHE      = (float(_tot), float(_free), _nm)
                    _SMI_CACHE_TS   = _now
                except Exception:
                    pass
                vram_total_gb = round(float(_tot) / 1024.0, 1)   # MiB -> GB
                vram_free_gb  = round(float(_free) / 1024.0, 1)
                if not gpu_name:
                    gpu_name = _nm
                _have_gpu_info = True
        if _have_gpu_info:
            _sd = last.get("siglip_device", "unknown")
            _qd = last.get("qwen_device",   "unknown")
            if _qd == "gpu" or _sd == "gpu":
                compute_device = "gpu"
            elif _qd == "cpu" or _sd == "cpu":
                compute_device = "cpu"
            else:
                compute_device = "gpu"   # GPU present, assume GPU until proven otherwise
        else:
            compute_device = "cpu"
    except Exception:
        pass

    # System RAM telemetry — lets the UI tell the user whether it's clear to grade
    # BEFORE they start. _GRADE_MIN_RAM_GB mirrors the hard cull gate in
    # /api/grade/v2/stream (below that the grade returns 503).
    ram_free_gb  = None
    ram_total_gb = None
    try:
        import psutil as _ps_ram
        _vm = _ps_ram.virtual_memory()
        ram_free_gb  = round(_vm.available / 1e9, 1)
        ram_total_gb = round(_vm.total / 1e9, 1)
    except Exception:
        pass

    return JSONResponse({
        "draft_available":    draft_ok,
        "verify_available":   verify_ok,
        "judge_available":    judge_ok,
        "phi4_available":     phi4_ok,
        "last_mode":          last["mode"],
        "last_verify_used":   last["verify_used"],
        "last_error":         last["error"],
        "qwen_warm":          qwen_warm,
        "qwen_loading":       qwen_loading,
        "qwen_int4_cached":   int4_cached,
        "qwen_download_pct":  qwen_dl_pct,
        "warmup_done":        warmup_done,
        "warmup_running":     warmup_running,
        "sidecar_loaded":     sidecar_loaded,
        "compute_device":     compute_device,  # "gpu" | "cpu" | "unknown"
        "vram_free_gb":       vram_free_gb,
        "vram_total_gb":      vram_total_gb,
        "gpu_name":           gpu_name,
        "ram_free_gb":        ram_free_gb,
        "ram_total_gb":       ram_total_gb,
        "ram_min_gb":         _GRADE_MIN_RAM_GB,
    })


@router.post("/api/models/preload")
async def preload_vision_engine():
    """
    Preload is intentionally a no-op.
    Loading BnB INT4 Qwen from a ThreadPoolExecutor thread before CUDA is
    initialised by the grading path causes a fatal C-level crash on Windows.
    Models load on-demand inside run_v2() which runs in a dedicated daemon thread.
    """
    try:
        from grade_pipeline_v2 import _qwen_singleton
        if _qwen_singleton is not None:
            return JSONResponse({"status": "already_warm"})
        return JSONResponse({"status": "will_load_on_grade"})
    except Exception:
        return JSONResponse({"status": "will_load_on_grade"})


@router.post("/api/models/warmup/reset")
async def reset_warmup():
    """Delete the warmup sentinel so calibration re-runs on next startup."""
    try:
        from warmup_runner import reset_sentinel
        reset_sentinel()
        return JSONResponse({"status": "sentinel_cleared"})
    except Exception as exc:
        print(f"[warmup/reset] failed: {exc}", flush=True)
        return JSONResponse({"status": "error", "detail": "Reset failed — see the server log."}, status_code=500)


@router.get("/api/models/download-status")
async def model_download_status():
    """Return the current auto-download status for all SpecVLM model weights."""
    from model_loader import get_download_status
    return JSONResponse(get_download_status())


