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
    return JSONResponse({"force_frontier": ff})


@router.get("/api/system/ram")
async def system_ram():
    """Live system-memory snapshot for the UI's RAM readiness indicator.

    Deliberately tiny (psutil only — no torch / model imports) so the frontend can
    poll it every couple of seconds and reflect Task Manager in real time.
    `percent` is memory in use (== Task Manager's headline %); `free` is the
    'Available' figure. `min_gb` is the hard cull gate (below it grading is 503)."""
    try:
        import psutil as _ps
        _vm = _ps.virtual_memory()
        return JSONResponse({
            "ram_free_gb":  round(_vm.available / 1e9, 1),
            "ram_total_gb": round(_vm.total / 1e9, 1),
            "ram_percent":  round(_vm.percent, 1),
            "ram_min_gb":   _GRADE_MIN_RAM_GB,
        })
    except Exception:
        return JSONResponse({"ram_free_gb": None, "ram_total_gb": None,
                             "ram_percent": None, "ram_min_gb": _GRADE_MIN_RAM_GB})


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
        if _SMI_CACHE is not None and (_now - _SMI_CACHE_TS) < _SMI_CACHE_S:
            (_tot, _free, _nm) = _SMI_CACHE
            vram_total_gb = round(_tot / 1024.0, 1)
            vram_free_gb  = round(_free / 1024.0, 1)
            if not gpu_name:
                gpu_name = _nm
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


