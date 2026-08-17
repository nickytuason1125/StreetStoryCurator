import suppress_console  # patches subprocess/multiprocessing/asyncio/BLAS before anything else imports them
import os
import sys
# Prevent any joblib/loky worker process from spawning (flashes a cmd window on Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# Force matplotlib to the non-interactive Agg backend before pymoo or any other library
# imports it.  pymoo's Display callback pulls in matplotlib on the first minimize() call;
# the default TkAgg/Qt5Agg backend creates a GUI handle that Windows briefly shows as a
# CMD prompt flash.  Agg renders to memory only -- no window, no flash.
os.environ.setdefault("MPLBACKEND", "Agg")
# Suppress the "unauthenticated requests" noise from HuggingFace hub without going full offline
# (HF_HUB_OFFLINE=1 breaks timm/open_clip local cache resolution).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
try:
    import joblib.parallel as _jp
    _jp.DEFAULT_BACKEND = "threading"
except Exception:
    pass

import asyncio
import uvicorn, signal, sys, time, threading
import requests as _requests
# Force UTF-8 output so emoji in print() don't crash on cp1252 terminals/threads.
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import List

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── MODULE PATH DIAGNOSTIC ───────────────────────────────────────────────────
# Detects whether we are running the local project copy or a global pip-installed clone.
try:
    import grade_pipeline_v2 as _path_probe
    print("=" * 60)
    print("EXECUTING FROM PHYSICAL PATH:", _path_probe.__file__)
    print("sys.path[0]:                 ", sys.path[0])
    print("=" * 60)
    del _path_probe
except Exception as _pp_err:
    print(f"[server] Module path probe failed: {_pp_err}")

# ── Frozen (PyInstaller) path resolution ────────────────────────────────────
if getattr(sys, 'frozen', False):
    # Running as PyInstaller onedir bundle — exe lives at curator-api/curator-api.exe
    # Set CWD to exe dir so relative paths (models/, frontend/dist/, cache/) resolve.
    _EXE_DIR = Path(sys.executable).parent
    os.chdir(_EXE_DIR)
    # Redirect writable cache to user's AppData (Program Files is read-only)
    _DATA_DIR = Path(os.environ.get('CURATOR_DATA_DIR', str(_EXE_DIR)))
else:
    _EXE_DIR = Path(__file__).parent
    _DATA_DIR = _EXE_DIR

# ---------------------------------------------------------------------------
# Path-safety helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
    ".bmp", ".gif", ".heic", ".heif",
    ".arw", ".cr2", ".cr3", ".nef", ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw",
})

def _safe_image_path(raw: str) -> Path:
    """Resolve symlinks, normalise, and verify the path is an existing image file.

    User photos may live anywhere on disk — no prefix restriction is applied.
    Raises HTTPException on traversal tricks (``..``), symlink escapes, missing
    files, or non-image extensions.
    """
    try:
        p = Path(raw).resolve(strict=False)
    except (ValueError, OSError):
        raise HTTPException(400, "Invalid path")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "File not found")
    if p.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(400, "Not an image file")
    return p


def _safe_dir_path(raw: str) -> Path:
    """Resolve symlinks, normalise, and verify the path is an existing directory."""
    try:
        p = Path(raw).resolve(strict=False)
    except (ValueError, OSError):
        raise HTTPException(400, "Invalid path")
    if not p.is_dir():
        raise HTTPException(400, "Not a valid directory")
    return p


# ── crash.log size guard ──────────────────────────────────────────────────────
# Every grading subprocess (grade_runner, encode_worker, iqa_worker) has its
# stdout/stderr redirected into crash.log, and the pipeline prints one line per
# photo per stage plus a [ram] line per progress tick. Nothing ever truncated
# it, so on a machine that culls regularly the file grows without bound.
#
# The launcher holds crash.log open in APPEND mode for the whole session, so a
# rename/rotate would hit a Windows sharing violation. Truncating in place is
# compatible with an append-mode writer (every write seeks to the current end),
# so the tail is preserved and the head is dropped.
_CRASH_LOG_MAX_MB  = 25.0
_CRASH_LOG_KEEP_MB = 4.0


def _trim_crash_log(path: str) -> None:
    """Keep crash.log's most recent _CRASH_LOG_KEEP_MB once it exceeds the cap."""
    try:
        if os.path.getsize(path) <= _CRASH_LOG_MAX_MB * 1024 * 1024:
            return
        _keep = int(_CRASH_LOG_KEEP_MB * 1024 * 1024)
        with open(path, "rb") as _f:
            _f.seek(-_keep, os.SEEK_END)
            _tail = _f.read()
        # Drop a partial first line so the retained head is well-formed.
        _nl = _tail.find(b"\n")
        if 0 <= _nl < len(_tail) - 1:
            _tail = _tail[_nl + 1:]
        with open(path, "wb") as _f:
            _f.write(b"--- crash.log truncated (size cap reached); older entries dropped ---\n")
            _f.write(_tail)
        print(f"[server] crash.log truncated to ~{_CRASH_LOG_KEEP_MB:.0f} MB", flush=True)
    except Exception as _e_trim:
        print(f"[server] crash.log trim skipped: {_e_trim}", flush=True)


RECENTLY_GENERATED: set = set()
MAX_HISTORY = 25
LAST_SEQUENCE: list = []   # paths from the most recent generation — used as avoid_paths

# ── Creative Direction — used-path persistence ────────────────────────────────
_USED_CD_PATHS_FILE = Path("cache/used_cd_paths.json")

def _load_used_cd_paths() -> set:
    """Return the set of source-image paths already used in a saved CD sequence."""
    try:
        if _USED_CD_PATHS_FILE.exists():
            import json as _j
            return set(_j.loads(_USED_CD_PATHS_FILE.read_text(encoding="utf-8")))
    except Exception:
        pass
    return set()

def _save_used_cd_paths(used: set) -> None:
    import json as _j
    _USED_CD_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USED_CD_PATHS_FILE.write_text(_j.dumps(sorted(used), indent=2), encoding="utf-8")

# GPU mutex — serialises all VRAM-using operations (grading + annotation daemon).
# Initialised in lifespan() once the event loop is running.
gpu_lock:         asyncio.Lock  | None = None
annotation_queue: asyncio.Queue | None = None

# ── Persistent grade worker ───────────────────────────────────────────────────
# A single long-lived subprocess keeps SigLIP-2 + text embeddings resident in
# memory between grading runs, avoiding the 15-30 s cold-load penalty on repeat
# grades.  If the worker dies (crash / OOM kill), it is transparently respawned
# on the next Grade click via _ensure_worker().
import multiprocessing as _mpw
_worker_proc:  "_mpw.Process | None" = None
_worker_req_q: "_mpw.Queue  | None" = None
_worker_resp_q:"_mpw.Queue  | None" = None
_worker_lock   = threading.Lock()


def _ensure_worker():
    """Return (req_q, resp_q), spawning or respawning the worker as needed."""
    global _worker_proc, _worker_req_q, _worker_resp_q
    with _worker_lock:
        if _worker_proc is not None and not _worker_proc.is_alive():
            print("[server] Grade worker died — respawning on next request")
            try:
                _worker_proc.kill()
            except Exception:
                pass
            _worker_proc = _worker_req_q = _worker_resp_q = None

        if _worker_proc is None:
            import grade_worker as _gw
            req_q  = _mpw.Queue()
            resp_q = _mpw.Queue()
            proc   = _mpw.Process(
                target=_gw.grade_worker_loop,
                args=(req_q, resp_q),
                daemon=True,
            )
            proc.start()
            _worker_proc  = proc
            _worker_req_q = req_q
            _worker_resp_q = resp_q
            print(f"[server] Grade worker started (pid={proc.pid})")

        return _worker_req_q, _worker_resp_q

# Background pre-computation
# Keyed by folder path so stale clusters from a previous grade never bleed through.
GLOBAL_CLUSTER_CACHE: dict = {}          # {"folder": str, "labels": ndarray, "paths": list}
_BG_EXECUTOR    = ThreadPoolExecutor(max_workers=1)
# Two separate executors so on-demand thumbnail requests (serve_thumb) are
# never queued behind background pre-warm jobs.
def _thumb_pool_sizes() -> tuple[int, int]:
    """Size the thumbnail worker pools by total RAM so concurrent RAW decodes
    don't spike memory on small machines. Returns (on_demand, prewarm)."""
    try:
        import psutil as _ps
        gb = _ps.virtual_memory().total / 1e9
        ondemand = 8 if gb >= 24 else 6 if gb >= 12 else 4 if gb >= 8 else 2
        prewarm  = 2 if gb >= 16 else 1
        return ondemand, prewarm
    except Exception:
        return 6, 2

_THUMB_OD_WORKERS, _THUMB_PW_WORKERS = _thumb_pool_sizes()
_THUMB_ONDEMAND = ThreadPoolExecutor(max_workers=_THUMB_OD_WORKERS)  # high-priority, browser-facing
_THUMB_PREWARM  = ThreadPoolExecutor(max_workers=_THUMB_PW_WORKERS)  # low-priority background warm-up

# Set while a grade is streaming. Background thumbnail PREWARM (bulk RAW decodes)
# is skipped while this is set so it doesn't spike RAM next to the SigLIP-2 / Qwen
# load; on-demand thumbnails (what the user is actually looking at) still run.
_grading_active = threading.Event()


def _release_annotation_model() -> None:
    """Drop the resident annotation model so it does not compete with the grade's
    SigLIP-2 load on RAM-tight machines. Called at grade start; annotations reload
    on demand afterwards, and they are gpu_lock-serialised with grades, so nothing
    is actively using it during a cull.

    This used to ask Ollama over HTTP to evict its models, which on a machine
    without Ollama — i.e. every correct install — meant a 2-second connection
    timeout at the start of every single grade, to free memory nothing was using.
    The annotation model now lives in-process, so releasing it is a function call.
    """
    try:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
        import critique_engine as _ce
        _ce.unload()
    except Exception:
        pass

# Hard floor of free system RAM (GB) below which a grade is refused (503). The
# grader-status endpoint reports this so the UI can warn BEFORE the user starts.
# 1.8: the HF SigLIP loader commits only ~1 GB, so culls run at low free RAM;
# Qwen has its own preflight that degrades to CLIP scoring when RAM is tight.
_GRADE_MIN_RAM_GB = float(os.environ.get("FRAMEGRADE_MIN_RAM_GB", "1.8"))

# ── Frontier 2026: legacy V1 analyzer replaced by _FrontierStub ───────────────
# lightweight_analyzer.py was renamed to *.legacy_backup — it cannot be imported.
# All V1 API endpoints that called get_analyzer() will raise RuntimeError, which
# is intentional.  V2 pipeline routes are unaffected.

class _FrontierStub:
    """Null-object stub replacing the removed legacy V1 LightweightStreetScorer."""
    cache: dict = {}          # safe empty cache — callers use .get(k, default)
    _ort_sessions = None      # guarded by 'if analyzer._ort_sessions is None:' checks
    _niche_clf    = None      # guarded by 'if clf else {}' checks

    class _MethodStub:
        """Callable that raises on call; sub-attrs return count=0 stubs."""
        count = 0
        def __call__(self, *a, **kw):
            raise RuntimeError(
                "Legacy V1 analyzer permanently removed in Frontier 2026. "
                "Use the SpecVLM pipeline: POST /api/grade/v2/stream"
            )
        def __getattr__(self, name: str):
            return _FrontierStub._MethodStub()

    def __getattr__(self, name: str):
        return self._MethodStub()


_analyzer_instance: _FrontierStub | None = None
_analyzer_lock = threading.Lock()


def get_analyzer() -> _FrontierStub:
    global _analyzer_instance
    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                _analyzer_instance = _FrontierStub()
    return _analyzer_instance


def _get_editorial_fns():
    from editorial_renderer import generate_magazine_carousel, render_editorial_carousel
    return generate_magazine_carousel, render_editorial_carousel

def _bg_model_prefetch():
    """Download missing models, then run pipeline calibration warmup."""
    try:
        from model_loader import ensure_all_models_downloaded
        ensure_all_models_downloaded()
    except Exception as exc:
        print(f"⚠️  Background model prefetch error: {exc}")
    # Chain calibration warmup — uses top Strong photos from LanceDB history
    # to pre-populate Inductor + BnB CUDA kernel caches on disk.
    try:
        from warmup_runner import run_warmup
        run_warmup()
    except Exception as exc:
        print(f"⚠️  Pipeline warmup error: {exc}")


def _auto_tune_hardware() -> None:
    """Silently adapt the model knobs to the machine's RAM/VRAM headroom.

    Invisible + safe: each value uses os.environ.setdefault (an explicit user env
    always wins) and is wrapped in try/except. On an ample machine the picks equal
    today's defaults, so behaviour is unchanged. MUST run before the grade worker
    is spawned so the worker (and its SigLIP subprocess) inherit the env."""
    ram_total = ram_free = None
    try:
        import psutil as _ps
        _vm = _ps.virtual_memory()
        ram_total, ram_free = _vm.total / 1e9, _vm.available / 1e9
    except Exception:
        pass
    vram_total = None
    try:
        import torch as _t
        if _t.cuda.is_available():
            vram_total = _t.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass

    # Qwen VRAM reserve + batch ceiling — more conservative on small GPUs so the
    # auto-derived batch (qwen_vlm_grader) leaves headroom and avoids VRAM OOM.
    if vram_total is not None:
        if vram_total <= 6.5:
            os.environ.setdefault("QWEN_VRAM_RESERVE", "0.8")
            os.environ.setdefault("QWEN_BS_CEIL", "4")
        elif vram_total <= 8.5:
            os.environ.setdefault("QWEN_VRAM_RESERVE", "0.6")
            os.environ.setdefault("QWEN_BS_CEIL", "6")
        else:
            os.environ.setdefault("QWEN_VRAM_RESERVE", "0.5")
            os.environ.setdefault("QWEN_BS_CEIL", "8")

    # SigLIP clean-fail floor — a LAST-RESORT guard against a genuine OOM, NOT a
    # capacity gate. (Earlier this session it was set to 5.0 to match a measured
    # ~4.5 GB encode "spike", but that number was inflated by reclaimable cache +
    # the profiler's own torch — the encode actually runs fine at ~4 GB free, as it
    # always did. A 5.0 floor REGRESSED working grades, refusing them at 4.2 GB.)
    # Keep it just above the grade-stream gate (1.8) so it only trips when RAM is
    # truly critical; real OOM protection now comes from the leaner baseline
    # (Ollama evict, niche release, prewarm pause) + the port-kill respawn fix.
    _tier = os.environ.get("SIGLIP_TIER", "high").strip().lower()
    os.environ.setdefault(
        "SIGLIP_MIN_FREE_RAM_GB",
        {"high": "2.0", "mid": "1.8", "low": "1.5"}.get(_tier, "2.0"),
    )

    def _r(x):
        return round(x, 1) if isinstance(x, (int, float)) else x
    print(f"[autotune] RAM {_r(ram_total)}/{_r(ram_free)} GB free, VRAM {_r(vram_total)} GB "
          f"→ QWEN_BS_CEIL={os.environ.get('QWEN_BS_CEIL')} "
          f"QWEN_VRAM_RESERVE={os.environ.get('QWEN_VRAM_RESERVE')} "
          f"SIGLIP_MIN_FREE_RAM_GB={os.environ.get('SIGLIP_MIN_FREE_RAM_GB')} "
          f"THUMB_POOLS={_THUMB_OD_WORKERS}/{_THUMB_PW_WORKERS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_lock, annotation_queue
    # Hardware auto-tune FIRST — sets env defaults the grade worker inherits.
    _auto_tune_hardware()
    # Initialise VRAM mutex and annotation queue now that the event loop is running.
    gpu_lock         = asyncio.Lock()
    annotation_queue = asyncio.Queue()

    # Allow CuDNN to auto-tune conv kernels on first batch — faster on fixed-size inputs.
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("[server] cudnn.benchmark enabled")
    except Exception:
        pass

    # KEEP catalog.json across restarts so a finished-but-unviewed or
    # interrupted session survives an app relaunch (e.g. after a window OOM-kill):
    # the frontend's "Resume last session?" banner recovers it, and the user can
    # Discard for a clean slate. (Previously this was deleted on startup, which
    # made post-crash Resume impossible — the whole point of the durability work.)
    try:
        _cat = _DATA_DIR / "cache" / "catalog.json"
        if _cat.exists():
            print("[server] Startup: catalog.json present — Resume available")
    except OSError:
        pass
    # Storage housekeeping — runs once on startup, non-blocking.
    threading.Thread(target=_evict_preview_cache, daemon=True, name="preview-evict").start()
    threading.Thread(target=_cleanup_old_zips,    daemon=True, name="zip-cleanup").start()

    # NOTE: the niche detector's CLIP model is loaded LAZILY (on the first
    # /api/recommend-niche call, i.e. when a folder is selected) rather than at
    # startup — keeping ~0.7 GB out of the baseline footprint on memory-tight
    # machines. It is also released at the start of each grade (see the grade
    # stream) so it never competes with SigLIP-2 / Qwen for RAM. First detect
    # pays a ~3 s load; the spinner already covers it.
    # Background model prefetch is intentionally DISABLED.
    # On Windows, BitsAndBytes INT4 + pyiqa loaded from a daemon thread before
    # CUDA is initialised by the main grading path causes a fatal C-level crash
    # (hard process kill, no traceback). All models load on-demand when Grade is
    # clicked — preloading from a thread here only crashes the startup.
    # _t = threading.Thread(target=_bg_model_prefetch, daemon=True, name="model-prefetch")
    # _t.start()

    # ── Pre-load LanceDB (Rust DLL) on the server thread ─────────────────────
    # On Windows, loading a Rust DLL for the FIRST TIME from a nested daemon
    # thread (main → server-thread → grade-daemon) can hit the Windows DLL
    # loader lock and cause a fatal C-level process kill with no traceback.
    # Loading it here (in the uvicorn server thread, one level from main)
    # puts the DLL in the OS loader cache before any grade thread needs it.
    # Also pre-opens the LanceDB table so the grade thread finds _tbl != None
    # and skips lancedb.connect() entirely.
    # Pre-open the LanceDB table.
    # IMPORTANT: lancedb imports pyarrow which loads native Arrow C++ DLLs.
    # On Windows, loading these DLLs from inside an asyncio coroutine (the IOCP
    # event loop thread) causes a fatal access violation.  Running it in a thread
    # pool executor avoids the DLL loader conflict.
    def _preopen_lancedb():
        try:
            import shutil as _shutil_ldb
            import lance_store as _ls
            try:
                _ls._open_table()
                print("[server] LanceDB table pre-opened OK")
            except Exception as _e_open:
                _shutil_ldb.rmtree(_ls._DB_DIR, ignore_errors=True)
                print(f"[server] LanceDB corrupt ({_e_open}) — deleted for fresh start")
        except Exception as _e_ldb:
            print(f"[server] LanceDB pre-load warning: {_e_ldb}")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _preopen_lancedb)
    except Exception as _e_ldb_boot:
        print(f"[server] LanceDB executor launch failed: {_e_ldb_boot}")

    # Start event-driven async annotation daemon (replaces 30-second polling).
    try:
        import queue_manager as _qm
        asyncio.create_task(_qm.start_async(annotation_queue, gpu_lock))
    except Exception as _qm_err:
        print(f"[server] Queue manager start failed: {_qm_err}")

    # Pre-start the persistent grade worker so first Grade click is instant.
    try:
        threading.Thread(target=_ensure_worker, daemon=True, name="worker-prestart").start()
    except Exception as _e_pw:
        print(f"[server] Grade worker pre-start failed: {_e_pw}")

    yield
    # Shut down the persistent grade worker gracefully.
    try:
        if _worker_req_q is not None:
            _worker_req_q.put({"_stop": True})
        if _worker_proc is not None:
            _worker_proc.join(timeout=5.0)
            if _worker_proc.is_alive():
                _worker_proc.kill()
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)

class _LazyAnalyzer:
    """Proxy that forwards attribute access to the real analyzer once loaded."""
    def __getattr__(self, name):
        return getattr(get_analyzer(), name)

analyzer = _LazyAnalyzer()

_APP_PORT = int(os.environ.get("CURATOR_PORT", "8000"))
# Scoped CORS — the packaged app is SAME-ORIGIN with its API (pywebview loads the
# SPA from this server), so it needs no CORS at all; the extra entries are only for
# the Vite dev server. allow_origins=["*"] previously let ANY website the user
# visited read local-photo responses cross-origin — removed.
_CORS_ORIGINS = [
    f"http://127.0.0.1:{_APP_PORT}", f"http://localhost:{_APP_PORT}",
    "http://127.0.0.1:5173", "http://localhost:5173",   # Vite dev
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security: isolate the local API from other origins ───────────────────────
# This server binds 127.0.0.1, but every website the user visits can still reach
# it. Without this guard a malicious page could read local photos (/api/photo?
# path=…), enumerate folders (/api/browse-folder) and CSRF the state-changing
# endpoints. Defense = Fetch-Metadata Resource Isolation + Host pinning:
#   1. Host header must be localhost — blocks DNS-rebinding (a hostile domain
#      re-resolving to 127.0.0.1 to look same-origin).
#   2. Reject /api/* when Sec-Fetch-Site is cross-site/cross-origin. WebView2 is
#      Chromium and always sends Sec-Fetch-Site; it is a forbidden header, so page
#      JS cannot forge it. Same-origin (the app) and same-site (Vite dev, different
#      port) are allowed; a request with no such header (non-browser) is allowed
#      through the Host check only — browsers are the only cross-origin threat.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}

@app.middleware("http")
async def _security_isolation(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host and host not in _ALLOWED_HOSTS:
            return JSONResponse(status_code=403, content={"error": "Forbidden host"})
        if (request.headers.get("sec-fetch-site") or "") in ("cross-site", "cross-origin"):
            return JSONResponse(
                status_code=403,
                content={"error": "Cross-origin access to the local API is blocked."},
            )
    return await call_next(request)


@app.middleware("http")
async def cache_control_middleware(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path

    if path.startswith("/api/"):
        # API responses: never cache
        response.headers["Cache-Control"] = "no-store"

    elif path.startswith("/assets/") or path.startswith("/thumbs/"):
        # Vite content-hashed assets + thumbnails: cache aggressively
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

    else:
        # HTML / SPA fallback / root — force fresh fetch every time
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        # Strip conditional-request headers so browser can't 304 a stale shell
        for _h in ("ETag", "Last-Modified"):
            if _h in response.headers:
                del response.headers[_h]

    return response


# Thumbnail cache served as static files
THUMB_DIR = _DATA_DIR / "cache" / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/thumbs", StaticFiles(directory=str(THUMB_DIR)), name="thumbs")

# Eye feature overlays served at /static/eye_feature_overlays/
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
(_STATIC_DIR / "eye_feature_overlays").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def shutdown(signum, frame):
    # Flush analyzer cache before exit so no in-flight results are lost
    if _analyzer_instance is not None:
        try:
            _analyzer_instance._save_cache()
        except Exception:
            pass
    sys.exit(0)
import threading as _threading
if _threading.current_thread() is _threading.main_thread():
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_PREVIEW_DIR = _DATA_DIR / "cache" / "previews"
_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

_HEIC_EXTS = frozenset({".heic", ".heif"})

_PREVIEW_MAX = 200  # keep newest N previews; delete oldest beyond this

def _evict_preview_cache() -> None:
    """Keep only the _PREVIEW_MAX most-recently-accessed previews; delete the rest."""
    try:
        files = sorted(_PREVIEW_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[_PREVIEW_MAX:]:
            try:
                old.unlink()
            except OSError:
                pass
        evicted = max(0, len(files) - _PREVIEW_MAX)
        if evicted:
            print(f"[server] Preview cache eviction: removed {evicted} old preview(s)")
    except Exception as _e_ev:
        print(f"[server] Preview cache eviction skipped: {_e_ev}")


_OUTPUT_DIR_ZIP = _DATA_DIR / "output"
_ZIP_MAX_AGE_DAYS = 30

def _cleanup_old_zips() -> None:
    """Delete ZIP exports older than _ZIP_MAX_AGE_DAYS days from the output directory."""
    import time as _time
    cutoff = _time.time() - _ZIP_MAX_AGE_DAYS * 86400
    removed = 0
    try:
        for z in _OUTPUT_DIR_ZIP.rglob("*.zip"):
            try:
                if z.stat().st_mtime < cutoff:
                    z.unlink()
                    removed += 1
            except OSError:
                pass
        if removed:
            print(f"[server] ZIP cleanup: removed {removed} export(s) older than {_ZIP_MAX_AGE_DAYS} days")
    except Exception as _e_zip:
        print(f"[server] ZIP cleanup skipped: {_e_zip}")


def _gen_preview(path: str) -> Path | None:
    """Return a JPEG preview for RAW/HEIC files; None for browser-renderable formats."""
    import hashlib
    src = Path(path).resolve()
    ext = src.suffix.lower()
    if ext not in _RAW_EXTS and ext not in _HEIC_EXTS:
        return None  # browser can render JPEG/PNG/WebP directly

    safe = hashlib.md5(str(src).encode()).hexdigest()[:10] + ".jpg"
    dest = _PREVIEW_DIR / safe
    if dest.exists():
        return dest

    try:
        from PIL import Image as _PILImg
        if ext in _HEIC_EXTS:
            # pillow-heif registers itself as a PIL plugin when imported
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            img = _PILImg.open(str(src)).convert("RGB")
        else:
            import rawpy, io
            with rawpy.imread(str(src)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        img = _PILImg.open(io.BytesIO(thumb.data))
                    else:
                        img = _PILImg.fromarray(thumb.data)
                except rawpy.LibRawNoThumbnailError:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                    img = _PILImg.fromarray(rgb)
            img = img.convert("RGB")
        img.save(str(dest), "JPEG", quality=90)
        # Async-safe fire-and-forget: evict oldest previews if cache is getting large.
        threading.Thread(target=_evict_preview_cache, daemon=True, name="preview-evict").start()
        return dest
    except Exception:
        return None


@app.get("/api/config")
async def get_config():
    """Return runtime configuration flags consumed by the frontend."""
    try:
        from frontier_config import is_force_frontier
        ff = is_force_frontier()
    except ImportError:
        ff = False
    return JSONResponse({"force_frontier": ff})


@app.get("/api/system/ram")
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


@app.get("/api/models/status")
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
    judge_ok  = _P("models/deepseek-r1-8b-q5.gguf").exists()
    phi4_ok   = _P("models/phi4-mini-reasoning-q4.gguf").exists()

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
    vram_free_gb  = None
    vram_total_gb = None
    gpu_name      = last.get("gpu_name")
    compute_device = "unknown"
    try:
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


@app.post("/api/models/preload")
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


@app.post("/api/models/warmup/reset")
async def reset_warmup():
    """Delete the warmup sentinel so calibration re-runs on next startup."""
    try:
        from warmup_runner import reset_sentinel
        reset_sentinel()
        return JSONResponse({"status": "sentinel_cleared"})
    except Exception as exc:
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


@app.get("/api/models/download-status")
async def model_download_status():
    """Return the current auto-download status for all SpecVLM model weights."""
    from model_loader import get_download_status
    return JSONResponse(get_download_status())


@app.get("/api/thumb")
async def serve_thumb(path: str = Query(...)):
    """Create or return a thumbnail (WEBP) for grid display.

    Generation runs in the dedicated _THUMB_ONDEMAND pool (off the asyncio event
    loop, so it never stalls SSE grade progress / health / annotation requests)
    and uses the SAME cache filename as the background prewarm (_gen_one_thumb) —
    so a prewarmed thumbnail is served instantly and is never regenerated.
    """
    import hashlib
    p = _safe_image_path(path)
    src = Path(p).resolve()
    safe_name = hashlib.md5(str(src).encode()).hexdigest()[:10] + ".webp"
    thumb_path = THUMB_DIR / safe_name
    if thumb_path.exists():
        return FileResponse(str(thumb_path))
    # RAM guard: while a cull is running, do NOT decode a fresh thumbnail on demand.
    # Fresh RAW decodes (rawpy) + the full-preview fallback below spike RAM and
    # compete with the grade's SigLIP encode (~3.5 GB) on a memory-tight machine.
    # Cached thumbs still serve instantly (above); uncached ones return 204 so the
    # grid shows a placeholder and they fill in once grading finishes.
    if _grading_active.is_set():
        return Response(status_code=204)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_THUMB_ONDEMAND, _gen_one_thumb, str(src))
    if thumb_path.exists():
        return FileResponse(str(thumb_path))
    # RAW/HEIC that produced no thumbnail (e.g. no embedded preview) → render a
    # full preview as a last resort.
    if src.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        preview = await loop.run_in_executor(None, _gen_preview, str(src))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    raise HTTPException(404, "Thumbnail could not be created")


@app.get("/api/photo")
async def serve_photo(path: str = Query(...)):
    p = _safe_image_path(path)
    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        import asyncio
        preview = await asyncio.get_running_loop().run_in_executor(None, _gen_preview, str(p))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    return FileResponse(str(p))


@app.post("/api/browse-folder")
async def browse_folder(body: dict):
    """Browse one or more folders — immediate, non-recursive scan of each directory.

    Accepts either:
      { "folder_path": "C:/…" }
    or
      { "folder_paths": ["C:/…", "D:/…"] }

    Returns combined unique folders and images.
    """
    raw_paths = body.get("folder_paths") or body.get("folder_path")
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if not raw_paths:
        return {"folders": [], "images": [], "files": []}

    folders_set = set()
    images_set = set()

    for raw in raw_paths:
        try:
            dirpath = _safe_dir_path(raw)
        except HTTPException:
            # skip invalid entries but continue
            continue
        try:
            for p in dirpath.iterdir():
                try:
                    if p.is_dir():
                        folders_set.add(str(p))
                    elif p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                        images_set.add(str(p))
                except PermissionError:
                    pass
        except PermissionError:
            pass

    folders = sorted(folders_set)
    images = sorted(images_set)
    return {"folders": folders, "images": images, "files": []}


def _read_exif(path: str) -> dict:
    """Delegates to src/exif_reader.read_exif.

    This was 130 lines inline in this file, so it could only be exercised by
    booting the whole server — and it carried three defects nobody caught:
    aperture formatted at one significant digit (f/1.4 rendered "f/1", f/11
    rendered "f/1e+01"), every ExposureTime forced through a Fraction so a 2.5s
    exposure rendered "5/2s", and RAW files returning nothing at all because PIL
    cannot open them and a bare `except` made that look like "no EXIF".
    """
    from src.exif_reader import read_exif
    return read_exif(path)


_RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw"}

def _gen_one_thumb(path: str, low_priority: bool = False) -> None:
    """Generate a single thumbnail into the cache directory (thread-safe). Optimized for speed.

    Shared by the background prewarm AND the on-demand /api/thumb handler, which
    now use the SAME cache filename — so writes go to a per-thread temp file and
    are atomically os.replace()'d into place, preventing a half-written WEBP from
    being served or two concurrent writers corrupting the same file.

    low_priority=True marks background prewarm jobs; these are skipped while a
    grade is running so their RAW decodes don't spike RAM next to the grader.
    """
    if low_priority and _grading_active.is_set():
        return
    try:
        from PIL import Image as _PILImg
        import hashlib as _hl
        src = Path(path).resolve()
        if not src.exists() or src.suffix.lower() not in _IMAGE_EXTS:
            return
        safe = _hl.md5(str(src).encode()).hexdigest()[:10] + ".webp"
        dest = THUMB_DIR / safe
        if dest.exists():
            return

        # Smaller target size for faster processing (grid display only)
        THUMB_SIZE = (200, 200)

        def _save(img) -> None:
            """Atomically write `img` to `dest` via a unique temp file."""
            tmp = dest.with_name(f"{dest.stem}.{os.getpid()}_{threading.get_ident()}.tmp.webp")
            try:
                img.save(str(tmp), "WEBP", quality=60, method=3)  # skip optimize for speed
                os.replace(str(tmp), str(dest))
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass

        if src.suffix.lower() in _RAW_EXTS:
            # Embedded preview ONLY — never demosaic the full sensor array (memory-safe).
            # If a RAW has no embedded preview, skip it cleanly rather than postprocess.
            try:
                import rawpy, io
                with rawpy.imread(str(src)) as raw:
                    thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = _PILImg.open(io.BytesIO(thumb.data))
                else:
                    img = _PILImg.fromarray(thumb.data)
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)  # faster than LANCZOS
                _save(img)
            except Exception as _e_raw_thumb:
                print(f"[thumb] RAW read error, skipping {src.name}: {_e_raw_thumb}")
                return
        elif src.suffix.lower() in _HEIC_EXTS:
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            with _PILImg.open(src) as img:
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)
                _save(img)
        else:
            # JPEG fast path: try embedded EXIF thumbnail first (<1 ms vs ~100 ms)
            if src.suffix.lower() in {".jpg", ".jpeg"}:
                try:
                    import piexif, io as _io
                    _exif = piexif.load(str(src))
                    _tb   = _exif.get("thumbnail")
                    if _tb and len(_tb) > 512:
                        with _PILImg.open(_io.BytesIO(_tb)) as img:
                            img = img.convert("RGB")
                            img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)
                            _save(img)
                        return
                except Exception:
                    pass
            # Draft-mode decode: PIL tells libjpeg to decode at 1/2, 1/4 or 1/8 scale
            # (4–8× faster for large JPEGs; no-op for PNG/WebP).
            with _PILImg.open(src) as img:
                img.draft("RGB", THUMB_SIZE)
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)
                _save(img)
    except Exception:
        pass


@app.post("/api/list-folder")
async def list_folder(body: dict):
    """Return image paths instantly — no EXIF, no blocking I/O on the hot path."""
    import asyncio
    folder = _safe_dir_path(body.get("folder_path", ""))

    exts = _IMAGE_EXTS

    def _scan():
        return sorted(
            str(p) for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

    loop = asyncio.get_running_loop()
    paths = await loop.run_in_executor(None, _scan)

    # Pre-warm ALL thumbnails in the background — no cap.
    # The low-priority executor (2 workers) processes them without blocking
    # on-demand requests from the browser.
    for p in paths:
        _THUMB_PREWARM.submit(_gen_one_thumb, p, True)   # low_priority — paused during grades

    # Return empty EXIF — frontend loads it lazily via /api/exif when needed.
    photos = [{"path": p, "exif": {}} for p in paths]
    return {"paths": paths, "photos": photos, "count": len(paths)}


@app.get("/api/exif")
async def get_exif(path: str = Query(...)):
    """Lazy EXIF loader — called by the frontend when a photo is selected."""
    import asyncio
    p = _safe_image_path(path)
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _read_exif, str(p))
    return data


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

class GradeRequest(BaseModel):
    folder_path: str = ""
    folder_paths: list[str] = []   # multi-folder support; takes priority when non-empty
    preset: str = "Classic Street"
    deep_review: bool = False
    deep_grade: bool = False       # Deep Grade: use Qwen VLM for scoring (default OFF = SigLIP zero-shot)
    force_rescan: bool = False
    scan_mode: bool = False        # Low-Latency Scan: top 20% only get 7B verification
    mogco_target: int = 5          # story sequence length (1–10)
    sample_limit: int = 0          # >0 caps niche-detection scan to a sample (0 = use default)

    @field_validator("folder_path")
    @classmethod
    def validate_folder_path(cls, v: str) -> str:
        if not v:
            return v
        try:
            p = Path(v).resolve(strict=False)
        except (ValueError, OSError):
            raise ValueError("Invalid path")
        if not p.is_dir():
            raise ValueError("Path is not a valid directory")
        return str(p)


def _run_vlm_deep_review(results: list) -> None:
    """
    Background task: editorial rationale notes for gated photos only.
    Gate: top 15% (score > 0.65) + borderline band (0.45–0.55).
    VLMRationaleGenerator never emits numeric scores — metric engine stays
    the sole source of truth.  Runs in _BG_EXECUTOR off the event loop.
    """
    try:
        from vlm_niche_detector import VLMRationaleGenerator, DEEP_REVIEW_TOP, DEEP_REVIEW_LOW, DEEP_REVIEW_HIGH
        vlm = get_analyzer()._ensure_vlm()
        if vlm is None or vlm.llm is None:
            return
        candidates = [
            r[0] for r in results
            if (r[1].get("score", 0) > DEEP_REVIEW_TOP
                or DEEP_REVIEW_LOW <= r[1].get("score", 0) <= DEEP_REVIEW_HIGH)
        ]
        if not candidates:
            return
        generator = VLMRationaleGenerator(vlm.llm)
        generator.generate_batch_sync(candidates)
    except Exception:
        pass   # never crash the background thread


def _precompute_clusters(folder: str, results: list) -> None:
    """Background task: K-Means on embeddings so /api/generate is instant."""
    global GLOBAL_CLUSTER_CACHE
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from joblib import parallel_backend
        _analyzer = get_analyzer()
        valid = [
            r for r in results
            if r[1].get("score", 0) > 0.20
            and r[1].get("grade") != "Error \u274c"
            and "\U0001f501" not in r[1].get("sim_flag", "")
        ]
        if len(valid) < 5:
            return
        embs = np.array([
            _analyzer.cache.get(r[0], {}).get("embedding", r[1].get("embedding", []))
            for r in valid
        ], dtype=np.float64)
        if embs.ndim != 2 or embs.shape[1] == 0:
            return
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs  = embs / (norms + 1e-9)
        k = min(10, len(valid))
        # Use threading backend so joblib/loky never spawns a new process
        # (which would flash a cmd window on Windows).
        with parallel_backend('threading', n_jobs=1):
            labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(
                embs.astype(np.float32)
            )
        GLOBAL_CLUSTER_CACHE = {
            "folder":  folder,
            "labels":  labels,
            "paths":   [r[0] for r in valid],
        }
    except Exception:
        pass   # never crash the background thread


@app.post("/api/grade")
async def grade_photos(req: GradeRequest):
    import asyncio
    global GLOBAL_CLUSTER_CACHE
    if not os.path.isdir(req.folder_path):
        raise HTTPException(400, "Invalid folder path")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: analyzer.analyze_folder(req.folder_path, preset=req.preset, force_rescan=True),
        )

        # Load any VLM critique results already on disk from a prior deep_review run.
        from pathlib import Path as _Path
        import json as _json
        _grade_cache_path = _DATA_DIR / "cache" / "vlm_rationale_cache.json"
        _vlm_grades: dict = (
            _json.loads(_grade_cache_path.read_text(encoding="utf-8"))
            if _grade_cache_path.exists() else {}
        )

        gallery = [{
            "path":        r[0],
            "grade":       r[1]["grade"],
            "score":       r[1]["score"],
            "critique":    r[1]["critique"],
            "breakdown":   r[1]["breakdown"],
            "nima_score":  r[1].get("nima_score"),
            "sim_flag":    r[1].get("sim_flag",   ""),
            "cluster_id":  r[1].get("cluster_id", -1),
            "faces":       r[1].get("faces", 0),
            "rationale": _vlm_grades.get(r[0]),   # None until background task completes
        } for r in results]
        strong = sum(1 for g in gallery if "Strong" in g["grade"])
        mid    = sum(1 for g in gallery if "Mid"    in g["grade"])
        weak   = sum(1 for g in gallery if "Weak"   in g["grade"])

        # Invalidate stale cache for a new folder, then kick off background tasks.
        GLOBAL_CLUSTER_CACHE = {}
        _BG_EXECUTOR.submit(_precompute_clusters, req.folder_path, results)
        if req.deep_review:
            _BG_EXECUTOR.submit(_run_vlm_deep_review, results)

        return JSONResponse({"status": "success", "total": len(gallery),
                             "strong": strong, "mid": mid, "weak": weak,
                             "data": gallery})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ollama/status")
async def ollama_status():
    """
    Local text/vision model availability. Response shape is unchanged:
    {alive: bool, models: [{name, size_vram, size_total, until}]}

    The path keeps its name so the frontend's 15-second poll keeps working, but
    it no longer speaks to an Ollama daemon — there isn't one. `alive` now means
    "a local model is installed", and `models` lists those files with their
    on-disk size. size_vram is 0 because llama_cpp's residency is decided per
    call by the offload ladder, not held as a server-side fact.
    """
    def _sync() -> dict:
        try:
            import sys as _s, os as _o
            _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
            import local_llm as _llm
            import critique_engine as _ce

            models = []
            for _p in (_llm.model_path(),
                       Path("models/qwen2.5-vl-2b-instruct-q4_k_m.gguf")):
                if _p.exists():
                    models.append({"name": _p.name, "size_vram": 0,
                                   "size_total": _p.stat().st_size, "until": ""})
            return {"alive": bool(_llm.available() or _ce.vision_available()),
                    "models": models}
        except Exception as _e:
            return {"alive": False, "models": [], "error": str(_e)}

    result = await asyncio.get_running_loop().run_in_executor(None, _sync)
    return JSONResponse(result)


@app.post("/api/grade/v2/stream")
async def grade_photos_v2_stream(req: GradeRequest):
    """
    V2 pipeline: SigLIP → Q-Align → PersonalHead → MOGCO-II.
    Same SSE format as /api/grade/stream for drop-in frontend compatibility.
    Supports multi-folder: grades each folder, then runs MOGCO-II once across all.
    """
    import json as _json
    from fastapi.responses import StreamingResponse

    # ── System RAM gate ─────────────────────────────────────────────────────
    # Lowered 3.0 → 1.8 GB: the HF SigLIP loader commits only ~1 GB (was ~9.5 GB
    # under open_clip), so culls run at far lower free RAM now. Qwen grading has
    # its own preflight that falls back to CLIP scoring if RAM is tight, so a low
    # but non-zero RAM cull still completes (in degraded mode) rather than 503.
    try:
        import psutil as _psutil
        _free_gb = _psutil.virtual_memory().available / 1e9
        if _free_gb < _GRADE_MIN_RAM_GB:
            return JSONResponse(
                status_code=503,
                content={"error": f"Not enough RAM to grade safely — only {_free_gb:.1f} GB free, need at least ~2 GB. "
                         "Close a couple of apps and retry."},
            )
    except Exception:
        pass

    # NOTE: there was an Ollama health gate here that 503'd every non-scan grade
    # when http://localhost:11434 did not answer. Nothing installed Ollama — not
    # Setup.ps1, not requirements.txt — and no user-facing doc mentioned it, while
    # CLAUDE.md rule 5 promised a fully offline app. So a correct, complete install
    # could not grade a single photo. Grading never needed it either: the default
    # path is SigLIP zero-shot plus TOPIQ, and the modules that did call Ollama now
    # run the same models locally through llama_cpp. Removed, not made optional.

    # Resolve all valid folders — folder_paths (multi) takes priority over folder_path
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders:
        if req.folder_path and os.path.isdir(req.folder_path):
            all_folders = [str(Path(req.folder_path).resolve())]
        else:
            raise HTTPException(400, "No valid folder path provided")

    async def _stream_with_lock():
        # Hold gpu_lock for the full grading run so the annotation daemon
        # cannot load its GGUF concurrently and bust the 5.5 GB VRAM ceiling.
        _gl = gpu_lock
        if _gl is not None:
            await _gl.acquire()
        # Free RAM held by background work before the grade's heavy model loads:
        #   - the niche detector's CLIP (~0.7 GB), reloads on the next folder-select
        #   - pause the background thumbnail prewarm (RAW decodes spike RAM)
        try:
            import fast_niche_detector as _fnd_rel
            _fnd_rel.release()
        except Exception:
            pass
        _release_annotation_model()   # free the ~1.5-4 GB annotation model
        _grading_active.set()
        try:
            import tempfile as _tf, subprocess as _sp

            # ── Run the grade as a CLEAN subprocess (grade_runner.py) ───────────
            # NOT a multiprocessing spawn child. A plain process running the pipeline
            # completes reliably where the multiprocessing worker died 0xC0000005 at
            # GPU-process boundaries (a GPU child exiting faulted the mp-spawn parent).
            # If this process crashes it is a separate OS process — the server and
            # window are unaffected; the catalog checkpoint + Resume recover the work.
            # SigLIP and IQA each still run in their OWN sub-subprocesses; this runner
            # does no direct GPU work.
            _fd, _req_path = _tf.mkstemp(suffix=".gradereq.json"); os.close(_fd)
            _prog_path = _req_path + ".progress.jsonl"
            open(_prog_path, "w", encoding="utf-8").close()
            with open(_req_path, "w", encoding="utf-8") as _rf:
                _json.dump({
                    "folders":      all_folders,
                    "preset":       req.preset,
                    "force_rescan": req.force_rescan,
                    "scan_mode":    req.scan_mode,
                    "deep_grade":   req.deep_grade,
                    "catalog_path": str(_CATALOG_PATH),
                    "data_dir":     str(_DATA_DIR),
                    "mogco_target": 0,   # cull only; Story sequencing is its own endpoint
                }, _rf)

            _runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grade_runner.py")
            _flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
            _renv = dict(os.environ); _renv["PYTHONIOENCODING"] = "utf-8"
            # Route the runner's [v2] progress prints to crash.log (utf-8) for
            # debuggability — its result stream goes via the progress file above.
            _crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
            _trim_crash_log(_crash_path)
            _rlog = open(_crash_path, "a", encoding="utf-8", errors="replace")
            import win_job as _wj
            _proc = _wj.popen(
                [sys.executable, _runner, _req_path, _prog_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                creationflags=_flags, close_fds=True,
                stdin=_sp.DEVNULL, stdout=_rlog, stderr=_rlog, env=_renv,
            )
            try: _rlog.close()   # child keeps its inherited fd
            except Exception: pass
            print(f"[server] Grade runner subprocess pid={_proc.pid}", flush=True)

            _loop = asyncio.get_running_loop()
            _pos = 0
            _done = False

            def _read_new():
                # Binary read from the last byte offset; only consume up to the last
                # complete newline so a half-written progress line is never mis-parsed.
                nonlocal _pos
                try:
                    with open(_prog_path, "rb") as _pf:
                        _pf.seek(_pos)
                        _data = _pf.read()
                    _cut = _data.rfind(b"\n")
                    if _cut < 0:
                        return ""
                    _pos += _cut + 1
                    return _data[:_cut + 1].decode("utf-8", "replace")
                except Exception:
                    return ""

            def _emit(_line):
                nonlocal _done
                _line = _line.strip()
                if not _line:
                    return None
                try:
                    _msg = _json.loads(_line)
                except Exception:
                    return None
                if _msg.get("done") and annotation_queue is not None:
                    for _g in _msg.get("data", []):
                        _gpath = _g.get("path", "")
                        if _gpath and float(_g.get("score", 0.0)) > 0.0 and not _g.get("has_annotations"):
                            annotation_queue.put_nowait(_gpath)
                if _msg.get("done") or _msg.get("error"):
                    _done = True
                return _line

            while True:
                _chunk = await _loop.run_in_executor(None, _read_new)
                for _ln in _chunk.splitlines():
                    _out = _emit(_ln)
                    if _out is not None:
                        yield f"data: {_out}\n\n"
                if _done:
                    break
                if _proc.poll() is not None:
                    # Runner exited — drain any final lines, then if we never saw a
                    # done/error result it crashed: surface the recoverable checkpoint.
                    _tailchunk = await _loop.run_in_executor(None, _read_new)
                    for _ln in _tailchunk.splitlines():
                        _out = _emit(_ln)
                        if _out is not None:
                            yield f"data: {_out}\n\n"
                    if not _done:
                        print(f"[server] Grade runner exited without result: code={_proc.returncode}", flush=True)
                        _recovered = 0
                        try:
                            if _CATALOG_PATH.exists():
                                _recovered = len(_json.loads(_CATALOG_PATH.read_text(encoding="utf-8")).get("photos", []))
                        except Exception:
                            pass
                        yield f"data: {_json.dumps({'error': f'Grade process exited unexpectedly (code {_proc.returncode}) — {_recovered} grades were checkpointed and can be recovered. Check crash.log', 'recovered': _recovered})}\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        finally:
            # This finally runs on normal completion AND on client disconnect
            # (GeneratorExit at a yield). Previously the subprocess kept running
            # after a disconnect while gpu_lock was released here — so a second
            # grade could start concurrently and bust the VRAM ceiling. Terminate
            # the runner if it's still alive, and always clean the temp files.
            try:
                if _proc.poll() is None:
                    _proc.terminate()
                    try:
                        _proc.wait(timeout=5)
                    except Exception:
                        _proc.kill()
                    print(f"[server] Grade runner pid={_proc.pid} terminated on stream close", flush=True)
            except Exception:
                pass
            for _tmp in (_req_path, _prog_path):
                try: os.unlink(_tmp)
                except Exception: pass
            _grading_active.clear()   # resume background thumbnail prewarm
            if _gl is not None:
                _gl.release()

    return StreamingResponse(_stream_with_lock(), media_type="text/event-stream")


@app.post("/api/regrade")
async def regrade_photos(req: GradeRequest):
    """
    Force a full re-grade: clears catalog.json, runs the full IQA pipeline
    (force_rescan=True), and rebuilds the catalog. SSE streaming, same format
    as /api/grade/v2/stream.
    """
    try:
        os.remove(str(_CATALOG_PATH))
        print("[regrade] Purged catalog.json before re-grade")
    except FileNotFoundError:
        pass
    except Exception as _e:
        print(f"[regrade] catalog purge warning: {_e}")
    return await grade_photos_v2_stream(req.model_copy(update={"force_rescan": True, "scan_mode": False}))


@app.post("/api/scan")
async def scan_photos(req: GradeRequest):
    """
    Low-latency scan: clears catalog.json, runs embedding + IQA without full
    SpecVLM verification (scan_mode=True), and rebuilds the catalog. SSE streaming,
    same format as /api/grade/v2/stream.
    """
    return await grade_photos_v2_stream(req.model_copy(update={"force_rescan": True, "scan_mode": True}))


@app.post("/api/personal/update")
async def personal_update(payload: dict):
    """
    Update the PersonalHead MLP when the user moves a photo between grade buckets.

    Body: { path1, grade1, path2, grade2 }
    Fetches embeddings from LanceDB and runs a Margin Ranking Loss update step.
    """
    try:
        import numpy as np
        import personal_head as ph
        import lance_store   as ls

        path1, grade1 = payload["path1"], payload["grade1"]
        path2, grade2 = payload["path2"], payload["grade2"]

        rows = ls.query_by_paths([path1, path2])
        by_path = {r["path"]: r for r in rows}

        if path1 not in by_path or path2 not in by_path:
            return JSONResponse({"ok": False, "error": "paths not found in LanceDB"})

        emb1 = by_path[path1]["embedding"]
        emb2 = by_path[path2]["embedding"]
        loss = await run_in_threadpool(ph.update, emb1, grade1, emb2, grade2)

        # Refresh personal scores for all stored photos
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = await run_in_threadpool(ph.score, all_embs)
            ls.update_personal_scores({r["path"]: float(s) for r, s in zip(all_rows, new_pers)})

        # Queue DPO preference events for background soul-alignment training
        try:
            import background_dpo_trainer as _dpo
            # path1 moved from grade2 → grade1 means path1 now has grade1
            # Queue: what changed grade, old → new
            _dpo.get_trainer().queue_event(path1, grade2, grade1)
        except Exception:
            pass  # DPO is best-effort; never block the main update

        return JSONResponse({"ok": True, "loss": round(loss, 5)})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/personal/star")
async def personal_star(payload: dict):
    """
    Train PersonalHead from a star rating on a single photo.

    Stars map to grade labels:
        4-5 → Strong ✅   3 → Mid ⚠️   1-2 → Weak ❌   0 → skip

    A contrastive photo with a different grade is pulled from LanceDB to form
    a preference pair for MarginRankingLoss.  The DPO queue is also updated so
    BackgroundDPOTrainer can fire once 20 events accumulate.
    """
    import random as _random
    try:
        path  = str(payload.get("path", "")).strip()
        stars = int(payload.get("stars", 0))
        if not path or stars == 0:
            return JSONResponse({"ok": True, "skipped": True})

        # Persist to the durable ratings store FIRST — this is the taste
        # baseline and must survive re-culls/catalog rebuilds even if the
        # PersonalHead training below fails.
        try:
            import ratings_store as _rs
            _rs.set_rating(path, stars)
        except Exception as _e_rs:
            print(f"[star] durable ratings_store write failed: {_e_rs}")

        star_grade = "Strong ✅" if stars >= 4 else ("Mid ⚠️" if stars == 3 else "Weak ❌")
        _RANK = {"Strong ✅": 2, "Mid ⚠️": 1, "Weak ❌": 0}
        star_rank = _RANK[star_grade]

        import numpy as np
        import personal_head as ph
        import lance_store   as ls

        rows = ls.query_by_paths([path])
        if not rows:
            return JSONResponse({"ok": False, "error": "path not found in LanceDB"})

        this_row   = rows[0]
        this_emb   = this_row["embedding"]
        this_grade = this_row.get("grade") or "Mid ⚠️"

        # Queue DPO event: auto grade → user star grade
        try:
            import background_dpo_trainer as _dpo
            _dpo.get_trainer().queue_event(path, this_grade, star_grade)
        except Exception as _e_dpo:
            print(f"[star] DPO queue skipped: {_e_dpo}")

        # PersonalHead pair update: find a contrastive photo
        all_rows    = ls.query_all()
        contrastive = [r for r in all_rows
                       if r["path"] != path
                       and _RANK.get(r.get("grade") or "Mid ⚠️", 1) != star_rank]

        loss = 0.0
        if contrastive:
            other      = _random.choice(contrastive[:40])
            other_emb  = other["embedding"]
            other_grade = other.get("grade") or "Mid ⚠️"
            loss = await run_in_threadpool(ph.update, this_emb, star_grade, other_emb, other_grade)

            # Refresh personal scores across all stored photos
            if all_rows:
                all_embs = np.stack([r["embedding"] for r in all_rows])
                new_pers = await run_in_threadpool(ph.score, all_embs)
                ls.update_personal_scores(
                    {r["path"]: float(s) for r, s in zip(all_rows, new_pers)}
                )

        # Auto-retrain the whole baseline every _RETRAIN_EVERY new ratings —
        # incremental pair-updates drift toward recent ratings; a periodic full
        # fit on the durable store keeps the head representative as the baseline
        # grows toward hundreds. Fire-and-forget so it never blocks the rating.
        retrained = None
        try:
            global _ratings_since_retrain
            _ratings_since_retrain += 1
            if _ratings_since_retrain >= _RETRAIN_EVERY:
                _ratings_since_retrain = 0
                import asyncio as _aio
                _aio.create_task(run_in_threadpool(_retrain_personal_baseline))
                retrained = "scheduled"
        except Exception as _e_rt:
            print(f"[star] auto-retrain schedule skipped: {_e_rt}")

        return JSONResponse({"ok": True, "star_grade": star_grade,
                             "loss": round(loss, 5), "retrain": retrained})
    except Exception as e:
        raise HTTPException(500, str(e))


# Full-baseline retrain plumbing — gathers every durable rating + its embedding
# and fits the PersonalHead from scratch (stable for 100s of ratings).
_RETRAIN_EVERY = 25
_ratings_since_retrain = 0


def _gather_rating_samples() -> list:
    import numpy as np
    import ratings_store as _rs, lance_store as _ls
    ratings = _rs.load()
    if not ratings:
        return []
    rows = {r["path"]: r for r in _ls.query_all(min_score=0.0)}
    _g = lambda s: "Strong ✅" if s >= 4 else ("Mid ⚠️" if s == 3 else "Weak ❌")
    out = []
    for p, s in ratings.items():
        r = rows.get(p)
        if r is not None and r.get("embedding") is not None:
            out.append((np.asarray(r["embedding"], dtype=np.float32), _g(int(s))))
    return out


def _retrain_personal_baseline() -> dict:
    import numpy as np
    import personal_head as ph
    samples = _gather_rating_samples()
    if not samples:
        return {"n": 0}
    stats = ph.fit(samples)
    try:
        import lance_store as _ls
        rows = _ls.query_all()
        if rows:
            embs = np.stack([r["embedding"] for r in rows])
            pers = ph.score(embs)
            _ls.update_personal_scores({r["path"]: float(s) for r, s in zip(rows, pers)})
    except Exception:
        pass
    print(f"[personal] baseline retrained: {stats}")
    return stats


@app.post("/api/personal/retrain")
async def personal_retrain(payload: dict = None):
    """Manually retrain the PersonalHead on the full durable rating baseline."""
    stats = await run_in_threadpool(_retrain_personal_baseline)
    return JSONResponse({"ok": True, **stats})


@app.post("/api/update_preference")
async def update_preference(payload: dict):
    """
    Update preference by providing a winner and a loser image path.
    Body: { "winner_path": str, "loser_path": str }
    Runs a MarginRankingLoss update on the PersonalHead and refreshes stored scores.
    """
    try:
        import numpy as np
        import personal_head as ph
        import lance_store as ls
        winner = payload.get("winner_path")
        loser = payload.get("loser_path")
        if not winner or not loser:
            return JSONResponse({"ok": False, "error": "winner_path and loser_path required"})

        rows = ls.query_by_paths([winner, loser])
        by_path = {r["path"]: r for r in rows}
        if winner not in by_path or loser not in by_path:
            return JSONResponse({"ok": False, "error": "paths not found in LanceDB"})

        emb_w = by_path[winner]["embedding"]
        emb_l = by_path[loser]["embedding"]

        loss = await run_in_threadpool(ph.update, emb_w, 1, emb_l, 0)

        # Refresh personal scores for all stored photos (LanceDB)
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = await run_in_threadpool(ph.score, all_embs)
            ls.update_personal_scores({r["path"]: float(s) for r, s in zip(all_rows, new_pers)})

        return JSONResponse({"ok": True, "loss": round(loss, 5)})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/manage/sort-files")
async def sort_files(payload: dict):
    """
    Move graded photos into Strong / Mid / Weak subdirectories.
    Body: { folder_path, gallery: [...], copy: bool }
    """
    try:
        from grade_pipeline_v2 import sort_files as _sort
        result = _sort(
            payload["folder_path"],
            payload["gallery"],
            copy=bool(payload.get("copy", False)),
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/creative-direction/stream")
async def creative_direction_stream(payload: dict):
    """
    SSE stream — Purist Creative Direction pipeline.

    Selects the best original captures for a 5-image Story Sequence.
    No pixel modification is performed. Output files are copies of originals.

    Payload:
        anchor_path  str  – reference image path (used for metadata only)
        folder_path  str  – base folder (locates Final_Portfolio output)
        style_prompt str  – style brief for the DeepSeek-R1 Agent Rule Set
        n_target     int  – target sequence length (5–10, default 7)
    """
    import asyncio, json, numpy as _np
    from fastapi.responses import StreamingResponse

    anchor_path     = (payload.get("anchor_path") or "").strip()
    folder_path     = (payload.get("folder_path") or "").strip()
    style_prompt    = (payload.get("style_prompt") or "").strip()
    n_target        = int(payload.get("n_target", 7))
    n_target        = max(3, min(10, n_target))
    peg_image_hash  = (payload.get("peg_image_hash") or "").strip() or None
    mode            = (payload.get("mode") or "story").strip().lower()
    if mode == "auto":
        try:
            from creative_director_agent import classify_mode
            mode = classify_mode(style_prompt)
        except Exception as _e_classify:
            print(f"[server] mode='auto' classification skipped ({_e_classify}) — defaulting to story")
            mode = "story"
    if mode not in ("story", "competition"):
        mode = "story"

    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    def _push(msg: dict):
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    def _progress(frac: float, desc: str):
        _push({"progress": round(frac, 3), "desc": desc})

    def _run():
        try:
            import numpy as np
            import json

            # ── Fetch all graded images (Strong + Mid + Weak) ─────────────────
            _progress(0.01, "Loading graded images…")
            strong_paths:  list[str] = []
            embeddings:    list      = []
            scores:        list      = []
            aspect_scores: list      = []
            IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}

            # Tier 0: catalog.json — always-available grade cache written by frontend
            try:
                if _CATALOG_PATH.exists():
                    _cat = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
                    _photos = _cat.get("photos", [])
                    if folder_path:
                        from pathlib import Path as _Path
                        _fp = str(_Path(folder_path).resolve())
                        _photos = [p for p in _photos if p.get("path", "").startswith(_fp)]
                    _photos = [p for p in _photos if float(p.get("score", 0)) > 0]
                    if _photos:
                        _photos.sort(key=lambda p: float(p.get("score", 0)), reverse=True)
                        strong_paths  = [p["path"]                          for p in _photos]
                        embeddings    = [np.zeros(1536, dtype=np.float32)   for _ in _photos]
                        scores        = [float(p.get("score", 0.5))         for p in _photos]
                        aspect_scores = [
                            p["breakdown"] if isinstance(p.get("breakdown"), dict)
                            else (json.loads(p["breakdown"]) if isinstance(p.get("breakdown"), str) else {})
                            for p in _photos
                        ]
                        _progress(0.02, f"Found {len(strong_paths)} graded images (catalog)")
            except Exception as _e:
                print(f"[cd] catalog.json read failed: {_e}")

            # Tier 1: LanceDB — primary source for embeddings; also fallback when catalog empty.
            # Always run to enrich embeddings even when catalog.json already provided paths —
            # catalog.json has no embeddings, so creative mode needs LanceDB for visual diversity.
            try:
                import lance_store as ls
                if strong_paths:
                    _lance_rows = ls.query_by_paths(strong_paths)
                else:
                    _lance_rows = ls.query_all(min_score=0.0)
                    if folder_path:
                        from pathlib import Path as _Path
                        fp = str(_Path(folder_path).resolve())
                        _lance_rows = [c for c in _lance_rows if c["path"].startswith(fp)]
                    _lance_rows = [c for c in _lance_rows if float(c.get("score", 0)) > 0]
                if _lance_rows:
                    if not strong_paths:
                        # Catalog was empty — use LanceDB as primary source
                        _lance_rows.sort(key=lambda c: float(c.get("score", 0)), reverse=True)
                        strong_paths  = [c["path"]                                  for c in _lance_rows]
                        scores        = [float(c.get("score", 0.5))                 for c in _lance_rows]
                        aspect_scores = [
                            c["breakdown"] if isinstance(c.get("breakdown"), dict)
                            else (json.loads(c["breakdown"]) if isinstance(c.get("breakdown"), str) else {})
                            for c in _lance_rows
                        ]
                        embeddings = [np.array(c["embedding"], dtype=np.float32) for c in _lance_rows]
                        _progress(0.02, f"Found {len(strong_paths)} graded images (LanceDB)")
                    else:
                        # Catalog provided paths — enrich embeddings from LanceDB
                        _emb_map = {c["path"]: np.array(c["embedding"], dtype=np.float32) for c in _lance_rows}
                        embeddings = [_emb_map.get(p, np.zeros(1536, dtype=np.float32)) for p in strong_paths]
                        n_real = sum(1 for e in embeddings if np.any(e != 0))
                        print(f"[cd] Embeddings enriched from LanceDB: {n_real}/{len(embeddings)} real")
            except Exception as e:
                print(f"[cd] LanceDB query failed: {e}")

            # Tier 2: Strong/ subfolder on disk (fallback when LanceDB is empty)
            if not strong_paths and folder_path:
                from pathlib import Path as _Path
                strong_dir = _Path(folder_path) / "Strong"
                if strong_dir.exists():
                    strong_paths = [
                        str(f) for f in sorted(strong_dir.iterdir())
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    ]
                    embeddings = [np.zeros(1536, dtype=np.float32) for _ in strong_paths]
                    scores     = [0.75] * len(strong_paths)

            # Tier 3: Scan folder directly for any images (cap at 50)
            # Extension alone isn't enough — a renamed/truncated garbage file with a
            # .jpg suffix passes an extension check trivially. The graded paths above
            # (catalog.json / LanceDB) already went through grade_pipeline_v2's
            # early-exit technical gate, which is why this is the one candidate
            # source that needs its own readability check before it can end up
            # copied into the user's Final_Portfolio output.
            if not strong_paths and folder_path:
                from pathlib import Path as _Path
                fp = _Path(folder_path)
                if fp.exists():
                    candidates = sorted(
                        f for f in fp.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )
                    all_imgs = []
                    for f in candidates:
                        if len(all_imgs) >= 50:
                            break
                        try:
                            from PIL import Image as _Image
                            with _Image.open(f) as _im:
                                _im.verify()
                        except Exception:
                            print(f"[cd] Tier 3 folder scan: skipping unreadable file {f.name}")
                            continue
                        all_imgs.append(str(f))
                    if all_imgs:
                        strong_paths = all_imgs
                        embeddings   = [np.zeros(1536, dtype=np.float32) for _ in strong_paths]
                        scores       = [0.5] * len(strong_paths)
                        _progress(0.02, f"Using {len(strong_paths)} folder images (grade folder for better selection)")

            if not strong_paths:
                _push({"error": "No images found. Grade your folder first."})
                return

            _progress(0.03, f"Found {len(strong_paths)} images for creative direction")

            # ── Release SigLIP-2 singleton before Creative Mode LLMs load ─────
            try:
                from grade_pipeline_v2 import release_grading_models
                release_grading_models()
            except Exception as _e_rel:
                print(f"[server] release_grading_models skipped: {_e_rel}")
                # Belt-and-suspenders: a release failure must never leave
                # stale GPU tensors resident before Story Mode's own GPU
                # work (contact-sheet critique GGUF) starts loading.
                try:
                    from vram_manager import VRAMManager
                    VRAMManager.purge_vram()
                except Exception:
                    pass

            # ── Run pipeline ──────────────────────────────────────────────────
            from creative_director import run_creative_direction

            avoid_paths = sorted(_load_used_cd_paths())

            result = run_creative_direction(
                strong_paths      = strong_paths,
                embeddings        = embeddings,
                scores            = scores or None,
                aspect_scores_list= aspect_scores or None,
                anchor_path       = anchor_path,
                output_dir        = folder_path or str(
                    Path(anchor_path).parent if anchor_path else _DATA_DIR / "cache"
                ),
                style_prompt      = style_prompt,
                n_target          = n_target,
                avoid_paths       = avoid_paths,
                progress          = _progress,
                peg_image_hash    = peg_image_hash,
                mode              = mode,
            )

            # Auto-mark generated images as used so next generation picks different ones.
            # Explicit save (save-sequence) persists to Story_<ts>/; this just rotates the pool.
            if result.get("outputs"):
                new_used = {
                    o["source_path"] for o in result["outputs"]
                    if o.get("success") and o.get("source_path")
                }
                if new_used:
                    updated = _load_used_cd_paths() | new_used
                    # Reset when the whole pool has cycled through
                    if len(updated) >= len(strong_paths):
                        updated = new_used
                    _save_used_cd_paths(updated)

            _push({"done": True, "data": result})

        except Exception as exc:
            _push({"error": str(exc)})
        finally:
            import gc as _gc
            _gc.collect()

    _BG_EXECUTOR.submit(_run)

    async def _event_stream():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if "done" in msg or "error" in msg:
                break

    return StreamingResponse(
        _event_stream(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/creative-direction/list-portfolio")
async def list_portfolio(payload: dict):
    """
    Return the list of stylized images in Final_Portfolio/ for a given folder.
    Body: { folder_path }
    """
    folder = (payload.get("folder_path") or "").strip()
    if not folder:
        return JSONResponse({"images": []})
    try:
        port_dir = Path(folder) / "Final_Portfolio"
        if not port_dir.is_dir():
            return JSONResponse({"images": []})
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted(
            str(f) for f in port_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        return JSONResponse({"images": images, "dir": str(port_dir)})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/creative-direction/save-sequence")
async def save_cd_sequence(payload: dict):
    """
    Copy stylized outputs to a timestamped Story folder and write a manifest.
    Marks the source images as used so they are excluded from future sequences.

    Body: { outputs: [{source_path, output_path, params, success}], base_dir: str }
    """
    import shutil
    from datetime import datetime

    outputs  = payload.get("outputs", [])
    base_dir = (payload.get("base_dir") or "").strip()

    successes = [o for o in outputs if o.get("success") and o.get("output_path")]
    if not successes:
        return JSONResponse({"ok": False, "error": "No successful outputs to save"})

    # Resolve base dir
    if not base_dir:
        base_dir = str(Path(successes[0]["output_path"]).parent.parent)
    base_dir_p = Path(base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_dir = base_dir_p / f"Story_{timestamp}"
    story_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    ROLE_ORDER = ["opener", "subject", "detail", "contrast", "closer"]
    sorted_out = sorted(
        successes,
        key=lambda o: ROLE_ORDER.index(o.get("params", {}).get("role", "")) if o.get("params", {}).get("role", "") in ROLE_ORDER else 99
    )

    for i, item in enumerate(sorted_out):
        src = Path(item["output_path"])
        if not src.exists():
            continue
        role = item.get("params", {}).get("role", "unknown")
        dest = story_dir / f"{i+1:02d}_{role}_{src.name}"
        shutil.copy2(str(src), str(dest))
        manifest.append({
            "seq": i + 1,
            "role":        role,
            "source_path": item.get("source_path", ""),
            "output_path": str(dest),
            "score":       item.get("params", {}).get("mogco_objectives", {}).get("set_cohesion", 0),
        })

    import json as _j
    (story_dir / "manifest.json").write_text(_j.dumps(manifest, indent=2))

    # Mark source paths as used
    source_paths = {o["source_path"] for o in successes if o.get("source_path")}
    used = _load_used_cd_paths() | source_paths
    _save_used_cd_paths(used)

    return JSONResponse({
        "ok":        True,
        "story_dir": str(story_dir),
        "count":     len(manifest),
        "used_total": len(used),
    })


@app.post("/api/creative-direction/clear-used")
async def clear_used_cd_paths():
    """Reset the used-image history so all photos are eligible again."""
    _save_used_cd_paths(set())
    return JSONResponse({"ok": True, "used_total": 0})


@app.get("/api/creative-direction/used-count")
async def get_used_cd_count():
    """Return how many source images are currently excluded from future sequences."""
    return JSONResponse({"count": len(_load_used_cd_paths())})


@app.post("/api/grade/stream")
async def grade_photos_stream(req: GradeRequest):
    """Streams grading progress as SSE, then emits the full result as the final event."""
    import asyncio, json
    from fastapi.responses import StreamingResponse

    global GLOBAL_CLUSTER_CACHE

    # Resolve which folders to grade — multi-folder takes priority
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders:
        if req.folder_path and os.path.isdir(req.folder_path):
            all_folders = [req.folder_path]
        else:
            raise HTTPException(400, "No valid folder path provided")

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()

    def _progress(frac: float, desc: str = "") -> None:
        loop.call_soon_threadsafe(
            aqueue.put_nowait, {"progress": round(frac, 3), "desc": desc}
        )

    async def _run() -> None:
        global GLOBAL_CLUSTER_CACHE
        try:
            # Grade each folder; combine all results
            n = len(all_folders)
            combined: list = []
            for i, fp in enumerate(all_folders):
                def _folder_progress(frac: float, desc: str = "", _i=i, _n=n) -> None:
                    _progress((_i + frac) / _n, desc)
                folder_results = await loop.run_in_executor(
                    None,
                    lambda _fp=fp: analyzer.analyze_folder(
                        _fp, preset=req.preset,
                        force_rescan=True, progress=_folder_progress,
                    ),
                )
                combined.extend(folder_results)
            results = combined

            from pathlib import Path as _Path
            import json as _json
            _grade_cache_path = _DATA_DIR / "cache" / "vlm_rationale_cache.json"
            _vlm_grades: dict = (
                _json.loads(_grade_cache_path.read_text(encoding="utf-8"))
                if _grade_cache_path.exists() else {}
            )
            gallery = [{
                "path":        r[0],
                "grade":       r[1]["grade"],
                "score":       r[1]["score"],
                "critique":    r[1]["critique"],
                "breakdown":   r[1]["breakdown"],
                "nima_score":  r[1].get("nima_score"),
                "sim_flag":    r[1].get("sim_flag", ""),
                "cluster_id":  r[1].get("cluster_id", -1),
                "faces":       r[1].get("faces", 0),
                "rationale": _vlm_grades.get(r[0]),
            } for r in results]
            strong = sum(1 for g in gallery if "Strong" in g["grade"])
            mid    = sum(1 for g in gallery if "Mid"    in g["grade"])
            weak   = sum(1 for g in gallery if "Weak"   in g["grade"])
            GLOBAL_CLUSTER_CACHE = {}
            _BG_EXECUTOR.submit(_precompute_clusters, all_folders[0], results)
            if req.deep_review:
                _BG_EXECUTOR.submit(_run_vlm_deep_review, results)

            # Run MOGCO beam search now that DuckDB is populated.
            # Enrich paths with gallery metadata so the frontend can render directly.
            mogco_sequence: list = []
            try:
                from mogco_sequencer import run_mogco_sequence
                beam = await loop.run_in_executor(None, run_mogco_sequence)
                if beam.get("paths"):
                    info_by_path = {g["path"]: g for g in gallery}
                    for path, slot, obj in zip(
                        beam["paths"], beam["slots"], beam["beam_objectives"]
                    ):
                        frame = dict(info_by_path.get(path, {"path": path}))
                        frame["slot"]             = slot
                        frame["mogco_objectives"] = obj
                        frame["engine"]           = "mogco-beam"
                        mogco_sequence.append(frame)
            except Exception:
                pass  # MOGCO failure never blocks grading result

            await aqueue.put({
                "done": True, "total": len(gallery),
                "strong": strong, "mid": mid, "weak": weak, "data": gallery,
                "mogco_sequence": mogco_sequence,
            })
        except Exception as exc:
            await aqueue.put({"error": str(exc)})

    asyncio.create_task(_run())

    async def _generate():
        while True:
            try:
                msg = await asyncio.wait_for(aqueue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield "data: {\"ping\":true}\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("done") or msg.get("error"):
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/detect_niches")
async def detect_niches(payload: dict):
    photos = payload.get("photos", [])
    if not photos:
        return []
    input_data = [(p["path"], {"breakdown": p.get("breakdown", {}), "faces": p.get("faces", 0)}) for p in photos]
    return analyzer._detect_top_niches(input_data, top_n=5)


@app.post("/api/niches/build-anchors")
async def build_niche_anchors():
    """(Re)build NicheClassifier visual prototypes from the current cache."""
    import asyncio
    loop = asyncio.get_running_loop()
    built = await loop.run_in_executor(None, get_analyzer()._build_niche_anchors)
    clf   = get_analyzer()._niche_clf
    return {
        "built":   built,
        "anchors": clf.anchor_info if clf else {},
    }


# ---------------------------------------------------------------------------
# Generate sequence
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def generate_carousel(payload: dict):
    global RECENTLY_GENERATED, LAST_SEQUENCE
    try:
        photos = payload.get("photos", [])
        seed   = payload.get("seed") or int(time.time() * 1000) % (2**31)
        if not photos or len(photos) < 5:
            raise ValueError("Need at least 5 photos to generate a sequence.")

        # DEBUG LOG: Capture state before filtering
        total_photos = len(photos)
        before_filter_count = len(photos)
        before_filter_available = [p for p in photos if p["path"] not in RECENTLY_GENERATED]
        before_filter_count_available = len(before_filter_available)
        
        # Filter out recently generated paths to guarantee unique regenerations
        # (user-marked avoid_paths are handled by the sequencer, not global history)
        available = [p for p in photos if p["path"] not in RECENTLY_GENERATED]
        if len(available) < 5:
            print(f"[DEBUG generate] POOL EXHAUSTED - RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, available={len(available)}, resetting...")
            available = photos          # pool exhausted — reset and start fresh
            RECENTLY_GENERATED.clear()
            LAST_SEQUENCE = []          # stale avoidances would starve the fresh pool
        else:
            print(f"[DEBUG generate] Pool stats: total={total_photos}, RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, available={len(available)}")

        input_data = [
            (p["path"], {
                "score":    p["score"],
                "grade":    p["grade"],
                "embedding": analyzer.cache.get(p["path"], {}).get("embedding", p.get("embedding", [])),
                "breakdown": p.get("breakdown", {}),
                "sim_flag":  p.get("sim_flag", ""),
                "exif_ts":   analyzer.cache.get(p["path"], {}).get("exif_ts") or p.get("exif_ts") or 0.0,
            })
            for p in available
        ]
        _override = payload.get("subject_type") or payload.get("genre")
        _valid    = {"street", "nature", "portrait", "architecture"}
        # user_genre is None when the user chose "Any" — no genre filter applied in sequencer.
        # auto-detected type is used only for the response label.
        user_genre    = _override if (_override in _valid) else None
        detected_type = user_genre or analyzer.detect_subject_type(available)
        subject_type  = user_genre   # None = "Any" → sequence_story skips genre thresholds
        _pacing_valid  = {"Classic Street", "Travel / Documentary", "Minimalist / Art", "Custom"}
        pacing_preset  = payload.get("pacing_preset") if payload.get("pacing_preset") in _pacing_valid else None

        # Inject pre-computed cluster labels if the cache is warm for this folder
        _cache        = GLOBAL_CLUSTER_CACHE
        _folder       = payload.get("folder", "")
        cached_labels = None
        if _cache.get("folder") == _folder and _cache.get("labels") is not None:
            # Build a path→label lookup and align to input_data order
            import numpy as _np
            _lbl_map = dict(zip(_cache["paths"], _cache["labels"].tolist()))
            cached_labels = _np.array(
                [_lbl_map.get(r[0], -1) for r in input_data], dtype=_np.int32
            )

        # Merge server-tracked last sequence with any paths the user manually
        # marked as "used" in the frontend — both are excluded from the next pick.
        _user_avoid  = payload.get("avoid_paths", [])
        _avoid_set   = list(dict.fromkeys(LAST_SEQUENCE + _user_avoid))
        
        # DEBUG LOG: Capture avoid list state
        avoid_set_size = len(_avoid_set)
        last_seq_size = len(LAST_SEQUENCE)
        user_avoid_size = len(_user_avoid)

        # locked_slots: {slot_index_str: path} — positions that must not change
        _locked_slots = payload.get("locked_slots") or {}

        # DEBUG LOG: Print state before calling sequencer
        print(f"[DEBUG generate] total={total_photos}, RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, LAST_SEQUENCE={last_seq_size}, user_avoid={user_avoid_size}, avoid_set={avoid_set_size}, available_for_sequencer={len(input_data)}")
        print(f"[DEBUG generate] avoid_paths: {len(_avoid_set)} unique paths to avoid")

        seq_paths, rationale, seq_type = analyzer.sequence_story(
            input_data, target=5, seed=seed, subject_type=subject_type,
            avoid_paths=_avoid_set, pacing_preset=pacing_preset,
            cached_labels=cached_labels, locked_slots=_locked_slots,
        )

        # Surface error when not enough photos passed genre thresholds
        if not seq_paths:
            err = rationale[0] if rationale else "Not enough qualifying images."
            return JSONResponse({"sequence": [], "subject_type": detected_type, "error": err})

        # Track the last sequence so the next Regenerate avoids the same picks
        LAST_SEQUENCE = list(seq_paths)
        
        # DEBUG LOG: Print generated sequence info
        print(f"[DEBUG generate] Generated sequence: {len(seq_paths)} photos")
        print(f"[DEBUG generate] LAST_SEQUENCE updated to: {LAST_SEQUENCE}")
        print(f"[DEBUG generate] RECENTLY_GENERATED now has {len(RECENTLY_GENERATED)} paths")

        # Record generated paths; trim history to MAX_HISTORY
        RECENTLY_GENERATED.update(seq_paths)
        if len(RECENTLY_GENERATED) > MAX_HISTORY:
            trimmed = list(RECENTLY_GENERATED)[-MAX_HISTORY:]
            RECENTLY_GENERATED.clear()
            RECENTLY_GENERATED.update(trimmed)
            print(f"[DEBUG generate] RECENTLY_GENERATED trimmed to {MAX_HISTORY} paths")

        carousel = []
        for i, path in enumerate(seq_paths):
            info = next((p for p in photos if p["path"] == path), {})
            carousel.append({
                **info,
                "rationale": rationale[i] if i < len(rationale) else "Strong candidate.",
            })
        return JSONResponse({"sequence": carousel, "subject_type": detected_type})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sequence")
async def mogco_sequence_simple(payload: dict):
    """
    Clean single-endpoint MOGCO sequencer for Tauri IPC and external callers.

    Payload fields (all optional):
        vibe_prompt : str   – reserved for future text-to-vector vibe encoding
        target      : int   – frames to select (default 5)
        min_score   : float – quality floor for DuckDB query (default 0.45)
        beam_width  : int   – beam paths (default 4)

    Returns raw beam result: { paths, slots, global_score, beam_objectives }
    """
    import asyncio
    try:
        target     = int(payload.get("target", 5))
        min_score  = float(payload.get("min_score", 0.45))
        beam_width = int(payload.get("beam_width", 4))
        # vibe_prompt reserved — encode to vector here when text encoder is added
        from mogco_sequencer import run_mogco_sequence
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_mogco_sequence(
                vibe_vec=None,
                target=target,
                min_score=min_score,
                beam_width=beam_width,
            ),
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sequence/album")
async def generate_album_sequence(payload: dict):
    """
    Build a multi-event album sequence from all graded images in the cache.

    Groups photos into temporal events (default 15-min gap), consolidates burst
    clusters within each event, then applies role-based pacing (SHOT_ROLES) to
    select the best 'frames' photos per event.

    Payload (all optional):
        gap_threshold : int   – seconds between events (default 900)
        frames        : int   – photos per event (default 5)
    """
    try:
        from sequence_engine import segment_events, consolidate_bursts, assign_roles

        gap_threshold = int(payload.get("gap_threshold", 900))
        frames        = int(payload.get("frames", 5))

        # Build flat list of records from the graded cache, injecting path
        records = []
        for path, data in analyzer.cache.items():
            if not data.get("embedding") or data.get("grade", "") == "Error ❌":
                continue
            records.append({
                "path":      path,
                "score":     float(data.get("score", 0)),
                "grade":     data.get("grade", ""),
                "embedding": data.get("embedding", []),
                "breakdown": data.get("breakdown", {}),
                "exif_ts":   float(data.get("exif_ts") or 0.0),
                "sim_flag":  data.get("sim_flag", ""),
            })

        if len(records) < frames:
            return JSONResponse({"error": f"Need at least {frames} graded images, got {len(records)}."})

        events      = segment_events(records, gap_threshold=gap_threshold)
        album       = []
        for i, event_group in enumerate(events):
            heroes   = consolidate_bursts(event_group)
            sequence = assign_roles(heroes, target=frames)
            album.append({
                "event_id":  f"evt_{i}",
                "start_ts":  event_group[0].get("exif_ts"),
                "frames": [
                    {
                        "path":       h["path"],
                        "score":      h.get("score", 0),
                        "grade":      h.get("grade", ""),
                        "burst_size": h.get("burst_size", 1),
                    }
                    for h in sequence
                ],
                "pacing": "Role-constrained + diversity-enforced",
            })

        return JSONResponse({"album": album, "events_detected": len(events)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# MOGCO sequencer — multi-objective Pareto selection backed by DuckDB
# ---------------------------------------------------------------------------

@app.post("/api/sequence/mogco")
async def mogco_sequence_endpoint(payload: dict):
    """
    MOGCO sequencer — two modes selectable via ``mode`` field:

    mode="beam"  (default)
        Queries DuckDB directly, applies an optional vibe/style filter, then runs
        beam search (width configurable) scoring quality + role_fit + visual_flow.
        Fastest for large libraries; DuckDB does the heavy lifting.

    mode="pareto"
        Greedy per-slot Pareto-front selection across 5 objectives.
        Requires embeddings from the frontend payload or JSON cache.

    Shared payload fields:
        photos        list  – photo records from the graded gallery
        target        int   – frames to select (default 5)
        subject_type  str   – genre hint passed to Pareto mode (default 'street')
        seed          int   – RNG seed (Pareto mode only, default 42)

    Beam-mode extra fields:
        mode          str   – "beam" | "pareto" (default "beam")
        vibe_path     str   – path of a reference photo; its DuckDB embedding is
                              used to filter candidates by style similarity
        vibe_thresh   float – minimum cosine similarity to the vibe photo (default 0.60)
        beam_width    int   – parallel beam paths (default 4)
        min_score     float – hard quality floor for DB query (default 0.45)
    """
    import asyncio
    import numpy as np

    try:
        photos      = payload.get("photos", [])
        target      = int(payload.get("target", 5))
        stype       = payload.get("subject_type") or "street"
        seed        = int(payload.get("seed", 42))
        mode        = payload.get("mode", "beam")
        vibe_path   = payload.get("vibe_path")
        vibe_thresh = float(payload.get("vibe_thresh", 0.60))
        beam_width  = int(payload.get("beam_width", 4))
        min_score   = float(payload.get("min_score", 0.45))

        if len(photos) < target:
            return JSONResponse({"sequence": [], "error": f"Need at least {target} photos."})

        from photo_cache import get_photo_cache
        db_cache = get_photo_cache()

        # ── Beam mode — DuckDB does the query, beam search does the rest ──────
        if mode == "beam":
            # Resolve vibe embedding from DuckDB or JSON cache
            vibe_vec = None
            if vibe_path:
                vibe_rows = db_cache.get_by_paths([vibe_path])
                if vibe_rows and len(vibe_rows[0]["embedding"]) > 0:
                    vibe_vec = vibe_rows[0]["embedding"]
                else:
                    raw = analyzer.cache.get(vibe_path, {}).get("embedding", [])
                    if raw:
                        vibe_vec = np.array(raw, dtype=np.float64)

            from mogco_sequencer import run_mogco_sequence
            beam_result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: run_mogco_sequence(
                    vibe_vec=vibe_vec,
                    vibe_thresh=vibe_thresh,
                    target=target,
                    beam_width=beam_width,
                    min_score=min_score,
                ),
            )

            if "error" in beam_result and not beam_result.get("paths"):
                return JSONResponse({"sequence": [], **beam_result})

            # Merge with frontend photo data for the carousel
            info_by_path = {p["path"]: p for p in photos}
            carousel = []
            for path, slot, obj in zip(
                beam_result["paths"],
                beam_result["slots"],
                beam_result["beam_objectives"],
            ):
                base = dict(info_by_path.get(path, {"path": path}))
                base["slot"]             = slot
                base["mogco_objectives"] = {
                    "flow":     obj.get("flow", 0),
                    "quality":  obj.get("quality", 0),
                    "role_fit": obj.get("role_fit", 0),
                }
                base["engine"] = "mogco-beam"
                carousel.append(base)

            return JSONResponse({
                "sequence":      carousel,
                "subject_type":  stype,
                "engine":        "mogco-beam",
                "global_score":  beam_result.get("global_score"),
                "vibe_active":   vibe_vec is not None,
            })

        # ── Pareto mode — pre-fetch embeddings, greedy Pareto selection ───────
        db_records = db_cache.get_by_paths([p["path"] for p in photos])
        db_by_path = {r["path"]: r for r in db_records}

        candidates: list[dict] = []
        for p in photos:
            path = p["path"]
            db   = db_by_path.get(path)
            if db is not None and len(db["embedding"]) > 0:
                emb = db["embedding"]
            else:
                raw = analyzer.cache.get(path, {}).get("embedding", [])
                emb = np.array(raw, dtype=np.float64)
            if len(emb) == 0 or np.linalg.norm(emb) < 1e-6:
                continue
            candidates.append({
                "path":      path,
                "score":     float(p.get("score", db["score"] if db else 0.0)),
                "grade":     p.get("grade", ""),
                "breakdown": p.get("breakdown", db["breakdown"] if db else {}),
                "embedding": emb,
                "exif_ts":   float(p.get("exif_ts") or (db["exif_ts"] if db else 0.0)),
                "sim_flag":  p.get("sim_flag", ""),
            })

        if len(candidates) < target:
            return JSONResponse({
                "sequence": [],
                "error": f"Only {len(candidates)} photos have valid embeddings (need {target}).",
            })

        from mogco_engine import mogco_sequence
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: mogco_sequence(candidates, target=target, stype=stype, rng_seed=seed),
        )

        info_by_path = {p["path"]: p for p in photos}
        carousel = []
        for frame in result:
            base = dict(info_by_path.get(frame["path"], {}))
            base["slot"]             = frame.get("slot", "")
            base["mogco_objectives"] = frame.get("mogco_objectives", {})
            base["engine"]           = "mogco-pareto"
            carousel.append(base)

        return JSONResponse({
            "sequence":     carousel,
            "subject_type": stype,
            "engine":       "mogco-pareto",
            "db_hits":      len(db_records),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Art Director — prompt-driven sequence generation
# ---------------------------------------------------------------------------

_DIRECTOR_GENRE_MAP: dict = {
    "portrait":     ["portrait", "face", "person", "people", "character", "subject"],
    "street":       ["street", "urban", "city", "candid", "reportage", "bystander"],
    "architecture": ["architecture", "building", "geometric", "structure", "facade"],
    "nature":       ["nature", "landscape", "outdoor", "wildlife", "scenic"],
}

_DIRECTOR_MOOD_MAP: dict = {
    "melancholic":  ["melancholic", "melancholy", "somber", "dark", "brooding", "quiet", "lonely"],
    "dramatic":     ["dramatic", "intense", "powerful", "bold", "charged", "striking"],
    "minimalist":   ["minimalist", "minimal", "simple", "clean", "sparse", "austere", "zen"],
    "humanist":     ["humanist", "intimate", "candid", "emotional", "empathetic", "warm", "tender"],
    "cinematic":    ["cinematic", "film", "noir", "atmospheric", "moody", "filmic"],
    "documentary":  ["documentary", "reportage", "journalistic", "photojournalism", "real", "raw"],
    "editorial":    ["editorial", "magazine", "commercial", "polished", "professional", "fashion"],
    "competition":  ["competition", "award", "submit", "contest", "prize", "jury", "festival"],
}

# Per-mood weight deltas applied to each slot's (comp, human, mood/light) weights
_MOOD_DELTA: dict = {
    "melancholic":  {"comp": -0.05, "human": -0.10, "mood": +0.15},
    "dramatic":     {"comp": +0.00, "human": +0.15, "mood": +0.05},
    "minimalist":   {"comp": +0.20, "human": -0.15, "mood": -0.05},
    "humanist":     {"comp": -0.05, "human": +0.20, "mood": -0.10},
    "cinematic":    {"comp": +0.05, "human": -0.10, "mood": +0.20},
    "documentary":  {"comp": -0.05, "human": +0.10, "mood": +0.00},
    "editorial":    {"comp": +0.15, "human": -0.05, "mood": +0.00},
    "competition":  {"comp": +0.10, "human": +0.05, "mood": +0.05},
}


class _CLIPTextSearch:
    """
    CLIP text-to-image search engine.

    Encodes a text query into the CLIP embedding space (ViT-B/32, trained on
    400M image-text pairs) and ranks candidate photos by cosine similarity.
    Image embeddings are computed on-demand and cached in memory for the
    lifetime of the server process — subsequent calls for the same photo path
    are instant.

    This is the standard industry approach for semantic photo search:
    text and image share the same 512-dim embedding space so similarity
    directly reflects how well a photo matches the description.
    """

    def __init__(self) -> None:
        import clip as _clip
        import torch as _torch
        self._device     = _torch.device("cpu")
        self._model, self._prep = _clip.load(
            "ViT-B/32", device=self._device, download_root="./models"
        )
        self._model.eval()
        # path → normalised (512,) CLIP image embedding
        self._img_cache: dict[str, "np.ndarray"] = {}

    def _img_emb(self, path: str) -> "np.ndarray | None":
        if path in self._img_cache:
            return self._img_cache[path]
        try:
            import torch as _torch
            from PIL import Image as _PImage
            img    = _PImage.open(path).convert("RGB")
            tensor = self._prep(img).unsqueeze(0).to(self._device)
            with _torch.no_grad():
                emb = self._model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            arr = emb.cpu().numpy()[0]
            self._img_cache[path] = arr
            return arr
        except Exception:
            return None

    def rank(self, paths: list[str], query: str) -> list[tuple[str, float]]:
        """Return [(path, similarity)] sorted by descending cosine similarity."""
        import clip as _clip
        import torch as _torch
        tokens = _clip.tokenize([query], truncate=True).to(self._device)
        with _torch.no_grad():
            txt = self._model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        txt_arr = txt.cpu().numpy()[0]

        results: list[tuple[str, float]] = []
        for p in paths:
            emb = self._img_emb(p)
            sim = float(np.dot(emb, txt_arr)) if emb is not None else 0.0
            results.append((p, sim))
        return sorted(results, key=lambda x: -x[1])


_clip_searcher: "_CLIPTextSearch | None" = None


def _get_clip_searcher() -> "_CLIPTextSearch":
    global _clip_searcher
    if _clip_searcher is None:
        _clip_searcher = _CLIPTextSearch()
    return _clip_searcher


def _clip_rank_by_brief(input_data: list, brief: str) -> list:
    """
    Re-rank (path, data) pairs by CLIP text-image similarity to the brief.
    Falls back to original order silently if CLIP is unavailable.
    """
    try:
        searcher = _get_clip_searcher()
        paths    = [d[0] for d in input_data]
        ranked   = searcher.rank(paths, brief)          # [(path, sim), …]
        order    = {path: i for i, (path, _) in enumerate(ranked)}
        return sorted(input_data, key=lambda x: order.get(x[0], len(ranked)))
    except Exception:
        return input_data


def _parse_director_prompt(prompt: str) -> dict:
    """
    Extract genre, mood biases, and frame count from a natural language brief.
    Returns custom SHOT_ROLES weights + director's note.
    """
    import re, copy
    from collections import OrderedDict

    text = prompt.lower()

    # Genre detection — first match wins
    genre = None
    for g, kws in _DIRECTOR_GENRE_MAP.items():
        if any(kw in text for kw in kws):
            genre = g
            break

    # Mood detection — can be multiple
    moods = [m for m, kws in _DIRECTOR_MOOD_MAP.items() if any(kw in text for kw in kws)]

    # Frame count extraction
    target = 5
    m = re.search(r'\b(\d+)\s*(?:photo|frame|image|shot|picture)', text)
    if m:
        target = max(3, min(10, int(m.group(1))))

    # Start from base SHOT_ROLES weights
    base = OrderedDict([
        ("opener",   {"comp_weight": 0.4, "human_weight": 0.1, "mood_weight": 0.3, "diversity_penalty": 0.2}),
        ("subject",  {"comp_weight": 0.2, "human_weight": 0.5, "mood_weight": 0.1, "diversity_penalty": 0.2}),
        ("detail",   {"comp_weight": 0.5, "human_weight": 0.1, "mood_weight": 0.2, "diversity_penalty": 0.2}),
        ("contrast", {"comp_weight": 0.2, "human_weight": 0.2, "mood_weight": 0.4, "diversity_penalty": 0.2}),
        ("closer",   {"comp_weight": 0.3, "human_weight": 0.1, "mood_weight": 0.5, "diversity_penalty": 0.2}),
    ])

    # Accumulate mood deltas across all detected moods
    for mood in moods:
        delta = _MOOD_DELTA.get(mood, {})
        dc, dh, dm = delta.get("comp", 0), delta.get("human", 0), delta.get("mood", 0)
        for role in base.values():
            role["comp_weight"]  = max(0.05, role["comp_weight"]  + dc)
            role["human_weight"] = max(0.05, role["human_weight"] + dh)
            role["mood_weight"]  = max(0.05, role["mood_weight"]  + dm)
            # Re-normalise so weights sum to (1 - diversity_penalty)
            total = role["comp_weight"] + role["human_weight"] + role["mood_weight"]
            budget = 1.0 - role["diversity_penalty"]
            scale  = budget / total if total > 0 else 1.0
            role["comp_weight"]  *= scale
            role["human_weight"] *= scale
            role["mood_weight"]  *= scale

    # Compose director's note
    mood_desc  = " + ".join(moods) if moods else "balanced"
    genre_desc = genre or "auto-detected genre"
    note = f"Reading brief as **{mood_desc}** with **{genre_desc}** focus."
    if "competition" in moods:
        note += " Competition mode: maximising technical quality and compositional impact."
    note += f" Selecting {target} frames calibrated for this narrative arc."

    return {
        "genre":            genre,
        "target":           target,
        "custom_shot_roles": base,
        "director_note":    note,
        "style_tags":       moods,
    }


_DIRECTOR_POOL_DIR = _DATA_DIR / "cache" / "director_pool"


@app.post("/api/director")
async def director_sequence(payload: dict):
    """
    Art Director: parse a natural language brief and generate a curated sequence
    from the graded photo pool (or uploaded competition photos).
    """
    import asyncio as _aio
    try:
        prompt = str(payload.get("prompt", "")).strip()
        photos = payload.get("photos", [])

        if not prompt:
            return JSONResponse({"error": "Please describe the sequence you want."})

        parsed = _parse_director_prompt(prompt)
        target = int(payload.get("target") or parsed["target"])

        if not photos or len(photos) < target:
            return JSONResponse({
                "error": f"Need at least {target} graded photos. "
                         "Grade your folder first or upload competition photos."
            })

        input_data = [
            (p["path"], {
                "score":     float(p.get("score", 0)),
                "grade":     p.get("grade", ""),
                "embedding": analyzer.cache.get(p["path"], {}).get("embedding", p.get("embedding", [])),
                "breakdown": p.get("breakdown", {}),
                "sim_flag":  p.get("sim_flag", ""),
                "exif_ts":   float(analyzer.cache.get(p["path"], {}).get("exif_ts") or p.get("exif_ts") or 0.0),
            })
            for p in photos
        ]

        # CLIP text-image ranking: encodes brief → 512-dim embedding, ranks photos
        # by cosine similarity (CLIP trained on 400M image-text pairs).
        if prompt:
            input_data = await _aio.get_running_loop().run_in_executor(
                None, lambda: _clip_rank_by_brief(input_data, prompt)
            )

        loop = _aio.get_running_loop()
        seq_paths, rationale, seq_type = await loop.run_in_executor(
            None,
            lambda: analyzer.sequence_story(
                input_data,
                target=target,
                subject_type=parsed["genre"],
                avoid_paths=[],
                seed=int(time.time() * 1000) % (2 ** 31),
                custom_shot_roles=parsed["custom_shot_roles"],
            ),
        )

        if not seq_paths:
            err = rationale[0] if rationale else "No qualifying images for this brief."
            return JSONResponse({"error": err})

        sequence = []
        for i, path in enumerate(seq_paths):
            info = next((p for p in photos if p["path"] == path), {})
            sequence.append({**info, "slot_label": rationale[i] if i < len(rationale) else f"Frame {i+1}"})

        return JSONResponse({
            "sequence":      sequence,
            "director_note": parsed["director_note"],
            "genre":         seq_type,
            "style_tags":    parsed["style_tags"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/director/upload-grade")
async def director_upload_grade(
    files: list[UploadFile] = File(...),
    preset: str = Form("Classic Street"),
):
    """
    Upload image files for competition use, grade them, and return scored results.
    Files are saved to cache/director_pool/ so thumbnails remain accessible.
    """
    import asyncio as _aio, shutil

    if not files:
        raise HTTPException(400, "No files provided.")

    # Clear previous pool and create fresh batch folder
    batch_dir = _DIRECTOR_POOL_DIR
    if batch_dir.exists():
        shutil.rmtree(batch_dir, ignore_errors=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    for f in files:
        safe = Path(f.filename or "upload").name
        dest = batch_dir / safe
        dest.write_bytes(await f.read())

    # Grade using existing pipeline
    loop = _aio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: analyzer.analyze_folder(str(batch_dir), preset=preset),
    )

    gallery = [
        {
            "path":       r[0],
            "grade":      r[1]["grade"],
            "score":      r[1]["score"],
            "critique":   r[1].get("critique", ""),
            "breakdown":  r[1]["breakdown"],
            "sim_flag":   r[1].get("sim_flag", ""),
            "cluster_id": r[1].get("cluster_id", -1),
            "embedding":  r[1].get("embedding", []),
        }
        for r in results
    ]

    return JSONResponse({"photos": gallery, "total": len(gallery)})


@app.post("/api/director/clear-pool")
async def director_clear_pool():
    """Delete uploaded competition photos from cache/director_pool/."""
    import shutil
    if _DIRECTOR_POOL_DIR.exists():
        shutil.rmtree(_DIRECTOR_POOL_DIR, ignore_errors=True)
    return JSONResponse({"cleared": True})


@app.post("/api/clear_history")
async def clear_generation_history():
    global RECENTLY_GENERATED
    RECENTLY_GENERATED.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Niche recommendation
# ---------------------------------------------------------------------------

@app.post("/api/recommend")
async def analyze_niche(payload: dict):
    import math

    results = payload.get("photos", [])
    if not results:
        return {"preset": "Classic Street", "confidence": 0, "reason": "No images provided."}

    # ── Extract named breakdown values by label (not positional index) ────────
    LABEL_MAP = {
        "Technical":"tech","News Sharpness":"tech","Cleanliness":"tech","Execution":"tech",
        "Detail Retention":"tech","Exposure":"tech","Sharpness & Detail":"tech",
        "Composition":"comp","Framing":"comp","Context":"comp","Geometry & Balance":"comp",
        "Negative Space":"comp","Framing Instinct":"comp","Layered Depth":"comp",
        "Lighting":"light","Atmosphere":"light","Natural Light":"light","Mood & Tone":"light",
        "Tonal Purity":"light","Contrast Purity":"light","Available Light":"light",
        "Natural Light Quality":"light",
        "Decisive Moment":"auth","Cultural Depth":"auth","Journalistic Integrity":"auth",
        "Narrative Suggestion":"auth","Conceptual Weight":"auth","Reduction":"auth",
        "Authenticity":"auth","Immediacy":"auth","Environmental Truth":"auth",
        "Subject Isolation":"human","Sense of Place":"human","Human Impact":"human",
        "Character Presence":"human","Emotional Resonance":"human","Scale Element":"human",
        "Human/Culture":"human","Presence":"human","Scale & Life":"human",
    }

    totals = {"tech":0.0,"comp":0.0,"light":0.0,"auth":0.0,"human":0.0}
    counts = {"tech":0,"comp":0,"light":0,"auth":0,"human":0}
    scores_all, faces_list = [], []
    n_items = 0

    for item in results:
        d = item if isinstance(item, dict) else (item[1] if isinstance(item,(list,tuple)) else {})
        b = d.get("breakdown", {})
        for label, val in b.items():
            key = LABEL_MAP.get(label)
            if key:
                totals[key] += float(val)
                counts[key] += 1
        scores_all.append(float(d.get("score", 0.0)))
        faces_list.append(int(d.get("faces", 0)))
        n_items += 1

    if n_items == 0:
        return {"preset": "Classic Street", "confidence": 0, "reason": "No scoreable images."}

    avg = {k: (totals[k]/counts[k] if counts[k] else 0.5) for k in totals}
    t, c, l, a, h = avg["tech"], avg["comp"], avg["light"], avg["auth"], avg["human"]

    # ── Derived signals ───────────────────────────────────────────────────────
    valid_scores  = [s for s in scores_all if s > 0]
    avg_score     = sum(valid_scores) / len(valid_scores) if valid_scores else 0.5
    score_std     = math.sqrt(sum((x-avg_score)**2 for x in valid_scores)/len(valid_scores)) if valid_scores else 0.0
    avg_faces     = sum(faces_list) / n_items
    frac_with_faces = sum(1 for f in faces_list if f > 0) / n_items
    strong_frac   = sum(1 for s in scores_all if s > 0.65) / n_items
    weak_frac     = sum(1 for s in scores_all if s < 0.45) / n_items

    # Interaction terms that are the actual discriminant features
    tech_x_auth   = t * a            # high = documentary/press; low tech only = snapshot
    comp_x_light  = c * l            # high = landscape/cinematic
    human_x_auth  = h * a            # high = travel/humanist; low = landscape/minimalist
    low_tech_flag = max(0.0, 0.45 - t)   # how far below the "deliberate" threshold
    no_people     = max(0.0, 0.30 - h)   # how strongly people are absent
    people_heavy  = max(0.0, h - 0.55)   # how strongly people dominate

    # ── Discriminant scoring ──────────────────────────────────────────────────
    # Each archetype is scored on the cross-dimension patterns that uniquely
    # identify it, with penalties for patterns that contradict it.
    def clamp(x): return max(0.0, min(1.0, x))

    raw = {}

    # SNAPSHOT: raw immediacy signature — low tech OR high variance, high auth
    # Key: (low_tech OR high_std) AND high_auth
    snapshot_tech_signal = clamp(low_tech_flag * 2.2)
    snapshot_var_signal  = clamp(score_std * 4.0)
    snapshot_trigger     = clamp(max(snapshot_tech_signal, snapshot_var_signal * 0.8))
    raw["Snapshot / Point-and-Shoot"] = (
        0.40 * snapshot_trigger * a +
        0.25 * clamp(weak_frac * 1.5) +
        0.20 * clamp(frac_with_faces) +
        0.15 * clamp(1.0 - strong_frac * 1.5)
    ) - clamp((t - 0.60) * 2.0) * 0.35    # penalise if actually technically sharp

    # STREET - EDITORIAL: balanced auth + comp + human, all above floor
    # Use min-of-three so any below-threshold dimension suppresses the score without
    # collapsing to near-zero when two dimensions are just above threshold.
    street_balance = min(clamp(a - 0.30), clamp(c - 0.30), clamp(h - 0.30))
    raw["Classic Street"] = (
        0.40 * clamp(street_balance * 5.0) +   # needs all three
        0.25 * clamp(a) +
        0.20 * clamp(h) +
        0.15 * clamp(c)
    ) - clamp(low_tech_flag * 1.5) * 0.20      # slight penalty for very low tech

    # WORLD PRESS DOC: tech + auth is the signature, human context required
    # Key: high tech AND high auth, people present
    raw["Photojournalism"] = (
        0.40 * clamp(tech_x_auth * 2.0) +
        0.30 * clamp((t - 0.50) * 3.0) +        # tech must be genuinely high
        0.20 * clamp(h) +
        0.10 * clamp(a)
    ) - clamp((0.50 - t) * 3.0) * 0.40          # hard penalty if tech is low

    # TRAVEL EDITOR: auth + human together, place/environment matters
    # Key: both auth AND human high (cultural immersion pattern)
    raw["Travel Editor"] = (
        0.45 * clamp(human_x_auth * 2.2) +
        0.25 * clamp(l) +
        0.20 * clamp(frac_with_faces) +
        0.10 * clamp(c)
    ) - clamp(no_people * 2.5) * 0.35           # penalise if few people

    # HUMANIST / EVERYDAY: people-dominant, warmth, auth — highest human of all
    # Key: human is the single dominant dimension, auth supports it
    raw["Humanist/Everyday"] = (
        0.50 * clamp(people_heavy * 3.5) +       # human must be very high
        0.25 * clamp(human_x_auth * 2.0) +
        0.15 * clamp(frac_with_faces) +
        0.10 * clamp(avg_faces / 2.0)
    ) - clamp(no_people * 3.0) * 0.50           # hard penalty without people

    # CINEMATIC / EDITORIAL: light is the dominant axis, mood over sharpness
    # Key: light far above other dimensions
    light_dominance = clamp(l - max(t, c, a, h) + 0.10)
    raw["Cinematic/Editorial"] = (
        0.45 * clamp(l * 1.4) +
        0.30 * clamp(light_dominance * 3.0) +
        0.15 * clamp(comp_x_light * 1.5) +
        0.10 * clamp(1.0 - abs(h - 0.45))       # some human but not dominant
    ) - clamp((0.55 - l) * 3.0) * 0.40          # hard penalty if light is not high

    # LANDSCAPE WITH ELEMENTS: high light + high comp, very low human
    # Key: comp_x_light interaction AND absence of people
    raw["Landscape with Elements"] = (
        0.40 * clamp(comp_x_light * 2.0) +
        0.30 * clamp(no_people * 3.0) +          # no people is a positive signal here
        0.20 * clamp(l * 1.3) +
        0.10 * clamp(c * 1.3)
    ) - clamp(frac_with_faces * 2.0) * 0.40     # penalise if faces appear often

    # MINIMALIST / URBEX: comp dominates everything, low people, controlled palette
    # Key: comp is the single highest dimension, auth/human low
    comp_dominance = clamp(c - max(t, l, a, h) + 0.10)
    raw["Minimalist/Urbex"] = (
        0.45 * clamp(c * 1.4) +
        0.30 * clamp(comp_dominance * 3.5) +
        0.15 * clamp(no_people * 2.0) +
        0.10 * clamp(t)
    ) - clamp(frac_with_faces * 1.5) * 0.30     # penalise faces

    # FINE ART / CONTEMPORARY: comp + intentionality, not purely about people or place
    # Key: comp high, auth genuinely low (staged/conceptual, not candid street)
    # clamp(max(0, 0.50-a) * 3) only rewards when auth is truly low; drops to 0 at auth >= 0.50
    raw["Fine Art/Contemporary"] = (
        0.40 * clamp(c * 1.3) +
        0.25 * clamp(l) +
        0.20 * clamp(max(0.0, 0.50 - a) * 3.0) +
        0.15 * clamp(t)
    ) - clamp(a - 0.65) * 0.45                  # harder penalty if clearly candid

    # London Street: atmosphere + human, urban mood, between street and cinematic
    # Needs both human presence AND decent light — penalise if either is absent
    raw["London Street"] = (
        0.35 * clamp(l * 1.2) +
        0.30 * clamp(human_x_auth * 1.8) +
        0.20 * clamp(h) +
        0.15 * clamp(a)
    ) - clamp(no_people * 2.0) * 0.35           # penalise if very few people
    - clamp((0.40 - l) * 3.0) * 0.25            # penalise if light quality is poor

    # ── Normalise scores to [0, 1] ────────────────────────────────────────────
    min_r = min(raw.values())
    max_r = max(raw.values())
    spread = max(max_r - min_r, 0.01)
    normalised = {name: (v - min_r) / spread for name, v in raw.items()}

    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
    best_preset = ranked[0][0]
    # Confidence = how far the winner leads the runner-up (not just that it won).
    # Gap of 0.40+ → 99%; gap of 0.20 → ~50%; gap of 0.08 → ~20%.
    _gap = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    _confidence = int(round(min(_gap * 2.5, 1.0) * 99))

    REASONS = {
        "Snapshot / Point-and-Shoot":
            "Batch shows raw immediacy — imperfect technique, high candid energy, variable quality. The moment is the priority.",
        "Classic Street":
            "Decisive moments, deliberate framing, and human presence in balance. Classic street photography benchmark.",
        "Photojournalism":
            "High technical sharpness combined with strong authenticity and human impact. Aligns with documentary standards.",
        "Travel Editor":
            "Strong cultural presence and authentic immersion. People and place work together across the batch.",
        "Humanist/Everyday":
            "People dominate the frame throughout. Warm, candid, dignity-driven — the human subject is the story.",
        "Cinematic/Editorial":
            "Light is the dominant force. Atmospheric, mood-driven, with cinematic colour and tonal direction.",
        "Landscape with Elements":
            "Natural light and compositional depth without human subjects. Foreground-layered environmental storytelling.",
        "Minimalist/Urbex":
            "Composition is the single strongest signal. Clean reduction, negative space, and structural purity.",
        "Fine Art/Contemporary":
            "Compositional intent over candid capture. Conceptual framing and tonal control elevate it beyond documentation.",
        "London Street":
            "Urban atmosphere and human presence in soft, directional light. Between street photography and cinematic mood.",
    }

    # ── Per-niche actionable guidance ──────────────────────────────────────────
    GUIDANCE = {
        "Classic Street": {
            "submit":  ["World Street Photography Awards", "Burn Magazine", "6 Mois"],
            "market":  "Editorial agencies (Panos, VII), documentary publishers, photobook imprints, festival circuits (Visa Pour l'Image).",
            "study":   ["Vivian Maier", "Alex Webb", "Daido Moriyama"],
        },
        "Travel Editor": {
            "submit":  ["Travel Photographer of the Year", "National Geographic Open Call", "Wanderlust Photo Awards", "Condé Nast Traveler"],
            "market":  "Travel magazines, tourism boards, airline in-flight media, hotel and hospitality brands.",
            "study":   ["Steve McCurry", "Ami Vitale", "Jonas Bendiksen"],
        },
        "Photojournalism": {
            "submit":  ["World Press Photo", "POYi", "Pictures of the Year International", "Bayeux-Calvados Award", "W. Eugene Smith Grant"],
            "market":  "Wire agencies (AP, Reuters, Getty), daily newspapers, long-form digital editorial, documentary book publishers.",
            "study":   ["James Nachtwey", "Lynsey Addario", "Sebastião Salgado"],
        },
        "Cinematic/Editorial": {
            "submit":  ["LensCulture Art Photography Awards", "Sony World Photography (Creative)", "1854 Media Awards", "IPA Advertising"],
            "market":  "Advertising agencies, film and TV production, fashion editorial, luxury brand campaigns.",
            "study":   ["Gregory Crewdson", "Philip-Lorca diCorcia", "Saul Leiter"],
        },
        "Fine Art/Contemporary": {
            "submit":  ["Paris Photo", "Rencontres d'Arles", "LensCulture Emerging Talent", "Foam Talent Call", "Aperture Summer Open"],
            "market":  "Gallery representation, museum acquisitions, art collectors, photobook publishers (Mack, Loose Joints, SPBH).",
            "study":   ["Wolfgang Tillmans", "Alec Soth", "Stephen Shore"],
        },
        "Minimalist/Urbex": {
            "submit":  ["Mono Awards", "Tokyo International Photo Awards", "B&W Spider Awards", "Chromatic Awards (Architecture)"],
            "market":  "Interior design publications, architectural practices, fine art print collectors, corporate art acquisitions.",
            "study":   ["Fan Ho", "Michael Kenna", "Hiroshi Sugimoto"],
        },
        "London Street": {
            "submit":  ["Street Foto San Francisco", "Sony World Photography (Street)", "Street Photo Prize"],
            "market":  "UK cultural institutions, editorial press, documentary photobooks, urban lifestyle brands.",
            "study":   ["Nick Turpin", "Matt Stuart", "Jesse Marlow"],
        },
        "Snapshot / Point-and-Shoot": {
            "submit":  ["Dazed Photography Awards", "It's Nice That", "Shoot Film Co Annual", "Superchief Gallery Open"],
            "market":  "Youth and lifestyle brands, music press, zine and independent publishers, social-first editorial.",
            "study":   ["Nan Goldin", "Wolfgang Tillmans", "Ryan McGinley"],
        },
        "Landscape with Elements": {
            "submit":  ["Landscape Photographer of the Year (UK)", "GDT European Wildlife", "Nature TTL Photographer of the Year", "Outdoor Photographer of the Year"],
            "market":  "Calendar publishers, tourism and national park bodies, outdoor gear brands, fine art print galleries.",
            "study":   ["Michael Kenna", "Charlie Waite", "Art Wolfe"],
        },
        "Humanist/Everyday": {
            "submit":  ["Taylor Wessing Portrait Prize", "Head On Portrait Prize", "Sony World Photography (Portraits)", "Humanity Photo Awards"],
            "market":  "NGO and charity publishers, editorial press (colour supplements), portrait documentary books, cultural foundations.",
            "study":   ["Dorothea Lange", "Mary Ellen Mark", "Platon"],
        },
    }

    # ── Dimension-specific coaching ────────────────────────────────────────────
    # Improve: keyed to the photographer's weakest average dimension.
    DIM_IMPROVE = {
        "tech":  "Technical execution is your floor to raise — sharper focus and cleaner exposure separate keepers from near-misses. Shoot in better light or use a faster shutter.",
        "comp":  "Compositional intentionality is what separates your shots from everyone else's — look for geometric tension, layering, and negative space before you press the shutter.",
        "light": "Light quality transforms good subjects into great photographs. Extend your sessions into early morning and late afternoon. Overcast diffusion is underrated.",
        "auth":  "Wait one beat longer. The decisive moment is usually a half-second ahead of where most photographers fire — resist the urge to shoot on the approach.",
        "human": "Close the distance. Proximity and genuine presence create the human connection missing from these frames. Engage before you raise the camera.",
    }

    # Strength: keyed to the strongest average dimension.
    DIM_STRENGTH = {
        "tech":  "Technical precision is your competitive floor — sharp, clean frames give editors nothing to reject on technical grounds.",
        "comp":  "Compositional instinct is your signature — your frames show geometry and intentionality that stop the edit.",
        "light": "Light is your strongest tool — atmospheric, directional, mood-driven exposures recur consistently across the batch.",
        "auth":  "Decisive-moment capture is where you stand out — peak gesture, unguarded expression, unrepeatable timing.",
        "human": "Human presence and cultural depth are your clearest signal — subjects feel authentic, unposed, and alive.",
    }

    dim_avgs   = {"tech": t, "comp": c, "light": l, "auth": a, "human": h}
    weakest    = min(dim_avgs, key=dim_avgs.get)
    strongest  = max(dim_avgs, key=dim_avgs.get)
    guidance   = GUIDANCE.get(best_preset, {})

    return {
        "preset":     best_preset,
        "confidence": _confidence,
        "reason":     REASONS.get(best_preset, "Best match for this batch's visual signature."),
        "ranking":    [{"preset": n, "score": round(s, 3)} for n, s in ranked],
        "submit":     guidance.get("submit", []),
        "market":     guidance.get("market", ""),
        "study":      guidance.get("study", []),
        "improve":    DIM_IMPROVE.get(weakest, ""),
        "strength":   DIM_STRENGTH.get(strongest, ""),
        "weakest":    weakest,
        "strongest":  strongest,
    }


# ---------------------------------------------------------------------------
# Pre-grade niche detection
# ---------------------------------------------------------------------------
# Lets the pre-grade picker auto-select the ideal niche BEFORE the full cull.
# Runs a fast scan_mode pass (CLIP composition scores — no IQA, no Ollama) over
# the folder through the persistent worker, then scores niches with the same
# engine as /api/recommend. Non-streaming and gpu_lock-serialised, so it never
# races a real grade for the GPU or the shared response queue, and never touches
# gallery state.
#
# The recommender works in a 10-archetype space; the picker speaks the 20-niche
# registry (src/niche_registry.py). This lossy map projects the archetype onto
# the closest registry slug so setPreset() actually matches a dropdown option.
# Images to scan for detection — a representative sample, capped so the pass
# finishes in ~seconds on a warm worker no matter how large the folder is.
_DETECT_SAMPLE = 24

_ARCHETYPE_TO_NICHE = {
    "Classic Street":             "classic_street",
    "Snapshot / Point-and-Shoot": "classic_street",
    "Photojournalism":            "photojournalism",
    "Travel Editor":              "travel_cultural",
    "Humanist/Everyday":          "documentary",
    "Cinematic/Editorial":        "liminal",
    "Landscape with Elements":    "landscape",
    "Minimalist/Urbex":           "minimalist",
    "Fine Art/Contemporary":      "fine_art",
    "London Street":              "urban_city",
}


async def _scan_folder_for_data(all_folders: list, preset: str, sample_limit: int) -> list:
    """Fast scan_mode pass through the persistent worker; returns the graded
    photo dicts (with breakdowns) for niche scoring, or [] if the worker dies /
    produces nothing. force_rescan=False so an already-graded folder resolves
    instantly on re-open."""
    import queue as _std_queue
    _gl = gpu_lock
    if _gl is not None:
        await _gl.acquire()
    try:
        loop = asyncio.get_running_loop()
        req_q, resp_q = await loop.run_in_executor(None, _ensure_worker)
        req_q.put({
            "folders":      all_folders,
            "preset":       preset or "Classic Street",
            "force_rescan": False,
            "scan_mode":    True,     # fast CLIP pass — no IQA, no Ollama gate
            "catalog_path": str(_CATALOG_PATH),
            "data_dir":     str(_DATA_DIR),
            "mogco_target": 0,
            "sample_limit": sample_limit,     # cap to a representative subset → ~seconds
            "detect_only":  True,             # never clears/writes catalog.json
        })

        def _try_get():
            try:
                return resp_q.get(timeout=1.0)
            except _std_queue.Empty:
                return None

        while True:
            try:
                msg = await asyncio.wait_for(loop.run_in_executor(None, _try_get), timeout=20.0)
            except asyncio.TimeoutError:
                msg = None
            if msg is None:
                if _worker_proc is None or not _worker_proc.is_alive():
                    return []                      # worker died — caller falls back
                continue
            if msg.get("error"):
                return []
            if msg.get("done"):
                return msg.get("data", []) or []
    finally:
        if _gl is not None:
            _gl.release()


@app.post("/api/recommend-niche")
async def recommend_niche(req: GradeRequest):
    """Instant pre-grade niche recommendation for the picker.

    Uses the warm CPU CLIP ViT-B/32 detector (src/fast_niche_detector.py) over a
    small, size-adaptive sample of the folder — no GPU, no grading pipeline, and
    no gpu_lock, so it can never stall previews or crash a grade, and returns in
    well under 3 s. Returns a registry slug the dropdown can select directly;
    falls back to classic_street so the picker is never blocked."""
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders and req.folder_path and os.path.isdir(req.folder_path):
        all_folders = [str(Path(req.folder_path).resolve())]
    if not all_folders:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "No valid folder to scan."}

    # Gather candidate images (non-recursive, same rule as /api/browse-folder).
    img_paths: list[str] = []
    for d in all_folders:
        try:
            for p in Path(d).iterdir():
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    img_paths.append(str(p))
        except (PermissionError, OSError):
            continue
    if not img_paths:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "No images found to scan."}
    img_paths.sort()

    try:
        from fast_niche_detector import detect as _detect_niche
        loop = asyncio.get_running_loop()
        rec = await loop.run_in_executor(
            None, lambda: _detect_niche(img_paths, req.sample_limit)
        )
    except Exception as e:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": f"Detection failed: {e}"}
    if not rec:
        return {"preset": "classic_street", "confidence": 0, "detected": False,
                "reason": "Could not read images — pick a niche manually."}
    rec["detected"] = True
    return rec


# ---------------------------------------------------------------------------
# Export magazine carousel
# ---------------------------------------------------------------------------

@app.post("/api/export/magazine")
async def export_magazine(payload: dict):
    try:
        images = payload.get("images", [])
        if len(images) < 5:
            raise HTTPException(400, "Need 5 images")
        clean_data = [
            {"path": i["path"], "rationale": i.get("rationale", ""), "presenter": "Curator"}
            for i in images
        ]
        generate_magazine_carousel, _ = _get_editorial_fns()
        zip_path = generate_magazine_carousel(clean_data)
        return FileResponse(zip_path, media_type="application/x-zip-compressed",
                            filename="Magazine_Carousel.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Export full-res photos by grade
# ---------------------------------------------------------------------------

@app.post("/api/export/grades")
async def export_by_grade(payload: dict):
    import shutil
    photos    = payload.get("photos", [])       # [{path, grade}, ...]
    dest      = payload.get("dest", "").strip()
    grades    = set(payload.get("grades", []))  # e.g. ["Strong ✅", "Mid ⚠️"]

    if not dest:
        raise HTTPException(400, "dest folder is required")
    if not grades:
        raise HTTPException(400, "at least one grade must be selected")

    dest_root = Path(dest)
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(400, f"Cannot create destination folder: {e}")

    copied, skipped, errors = 0, 0, []
    for item in photos:
        src_path = item.get("path", "")
        grade    = item.get("grade", "")
        if grade not in grades:
            skipped += 1
            continue
        src = Path(src_path)
        if not src.exists():
            errors.append(src_path)
            continue
        # Subfolder per grade, strip emoji for safe dir name. Grade strings are
        # already "{Word} {emoji}" (e.g. "Mid ⚠️") — replacing the emoji WITH
        # the word duplicated it ("Mid Mid"); just strip the emoji instead.
        safe_grade = grade.replace("✅", "").replace("⚠️", "").replace("❌", "").strip()
        out_dir = dest_root / safe_grade
        out_dir.mkdir(exist_ok=True)
        dest_file = out_dir / src.name
        # Avoid silent overwrite — append suffix if name collides
        counter = 1
        while dest_file.exists():
            dest_file = out_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        try:
            shutil.copy2(src, dest_file)
            copied += 1
        except Exception as e:
            errors.append(f"{src_path}: {e}")

    return {
        "copied":  copied,
        "skipped": skipped,
        "errors":  errors,
        "dest":    str(dest_root),
    }


# ---------------------------------------------------------------------------
# Editorial endpoint (slot-based selection + render)
# ---------------------------------------------------------------------------

@app.post("/api/editorial")
async def generate_editorial(payload: dict, fmt: str = Query("portrait")):
    import random
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    from datetime import datetime

    items_raw      = payload.get("photos", [])
    excluded_paths = set(payload.get("excluded_paths", []))

    if not items_raw:
        raise HTTPException(400, "No photos provided")

    scored = [
        {"path": p["path"], "score": p.get("score", 0), "grade": p.get("grade", ""),
         "breakdown": p.get("breakdown", {}),
         "emb": np.array(analyzer.cache.get(p["path"], {}).get("embedding", [0.0] * 384))}
        for p in items_raw
        if p["path"] not in excluded_paths and p.get("score", 0) > 0
    ]
    if len(scored) < 5:
        # Reset exclusions if pool too small
        scored = [
            {"path": p["path"], "score": p.get("score", 0), "grade": p.get("grade", ""),
             "breakdown": p.get("breakdown", {}),
             "emb": np.array(analyzer.cache.get(p["path"], {}).get("embedding", [0.0] * 384))}
            for p in items_raw if p.get("score", 0) > 0
        ]
    if len(scored) < 5:
        raise HTTPException(400, "Need at least 5 scoreable photos")

    scored.sort(key=lambda x: x["score"], reverse=True)
    pool = scored[:max(int(len(scored) * 0.7), 15)]
    rng  = random.Random(random.randint(0, 999_999))

    slot_roles = [
        {"Composition": 0.5, "Technical": 0.3, "Lighting": 0.2},
        {"human": 0.5, "auth": 0.4, "comp": 0.1},
        {"tech": 0.6, "comp": 0.3, "light": 0.1},
        {"light": 0.6, "auth": 0.3, "comp": 0.1},
        {},
    ]

    def _role_score(it, weights):
        b    = it.get("breakdown", {})
        vals = list(b.values())
        if not vals: return it["score"]
        pos  = {"tech": 0, "comp": 1, "light": 2, "auth": 3, "human": 4,
                "Composition": 1, "Technical": 0, "Lighting": 2}
        s = 0.0
        for k, w in weights.items():
            if k in b:                            s += b[k] * w
            elif k in pos and pos[k] < len(vals): s += vals[pos[k]] * w
        return s

    selected, used = [], set()
    for weights in slot_roles:
        candidates = [s for s in pool if s["path"] not in used]
        if not candidates: break
        if not weights:
            if selected:
                sel_embs = np.stack([s["emb"] for s in selected])
                best, best_d = None, -1.0
                for cand in candidates:
                    d = 1.0 - float(_cos(cand["emb"].reshape(1, -1), sel_embs).min())
                    if d > best_d: best_d, best = d, cand
                pick = best
            else:
                pick = rng.choice(candidates)
        else:
            ranked = sorted(candidates,
                            key=lambda s: _role_score(s, weights) + rng.uniform(0, 0.10),
                            reverse=True)
            pick = rng.choice(ranked[:min(4, len(ranked))])
        selected.append(pick)
        used.add(pick["path"])

    slot_labels = ["Opening", "Human Moment", "Detail", "Mood", "Closing"]
    for i, s in enumerate(selected):
        s["rationale"] = analyzer.cache.get(s["path"], {}).get("rationale", "") or slot_labels[i]

    out_dir = Path("output/editorial") / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        _, render_editorial_carousel = _get_editorial_fns()
        out_paths, zip_path = render_editorial_carousel(selected, out_dir, fmt=fmt)
    except Exception as e:
        raise HTTPException(500, str(e))

    return JSONResponse([
        {"path": p, "source_path": selected[i]["path"],
         "score": selected[i]["score"], "grade": selected[i]["grade"],
         "rationale": selected[i]["rationale"], "zip": zip_path}
        for i, p in enumerate(out_paths)
    ])


# ---------------------------------------------------------------------------
# Native folder picker (used by Edge app mode — no pywebview js_api available)
# ---------------------------------------------------------------------------

@app.get("/api/pick-folder")
async def pick_folder_dialog():
    """Opens a native OS folder-picker dialog and returns the chosen path."""
    import asyncio, subprocess, sys, os, tempfile, ctypes

    # Use ctypes to call Windows API directly - no subprocess needed
    def _show_dialog():
        try:
            # Windows API constants
            BIF_RETURNONLYFSDIRS = 0x00000001
            BIF_NEWDIALOGSTYLE = 0x00000040
            
            # Define BROWSEINFO structure
            class BROWSEINFO(ctypes.Structure):
                _fields_ = [
                    ('hwndOwner', ctypes.c_void_p),
                    ('pidlRoot', ctypes.c_void_p),
                    ('pszDisplayName', ctypes.c_char_p),
                    ('lpszTitle', ctypes.c_char_p),
                    ('ulFlags', ctypes.c_uint),
                    ('lpfn', ctypes.c_void_p),
                    ('lParam', ctypes.c_void_p),
                    ('iImage', ctypes.c_int)
                ]
            
            # Get the folder path using Windows API
            ctypes.windll.shell32.Shell32_SHBrowseForFolderW.argtypes = [ctypes.POINTER(BROWSEINFO)]
            ctypes.windll.shell32.Shell32_SHBrowseForFolderW.restype = ctypes.c_void_p
            
            # Use a simpler approach - create a temporary Python script that uses tkinter
            # but runs with pythonw.exe to avoid console window
            script = '''
import tkinter as tk
import tkinter.filedialog as fd
import sys
import os

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', True)
root.focus_force()

# Try to use the newer dialog style
try:
    p = fd.askdirectory(
        title='Select Photo Folder',
        parent=root,
        initialdir=os.path.expanduser('~')
    )
except:
    p = fd.askdirectory(title='Select Photo Folder', parent=root)

root.destroy()
print(p if p else '', end='')
'''
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_script = f.name
            
            try:
                # Use pythonw.exe to run the script
                _py = sys.executable
                if os.name == "nt" and _py.lower().endswith("python.exe"):
                    _pyw = _py[:-10] + "pythonw.exe"
                    if os.path.exists(_pyw):
                        _py = _pyw
                
                proc = subprocess.Popen(
                    [_py, temp_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    close_fds=True
                )
                stdout, _ = proc.communicate(timeout=120)
                path = stdout.decode('utf-8').strip()
                return path if path else None
            finally:
                try:
                    os.unlink(temp_script)
                except:
                    pass
                    
        except Exception as e:
            return None

    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _show_dialog)
    return JSONResponse({"path": path})


# ---------------------------------------------------------------------------
# XMP / JSON sidecar export
# ---------------------------------------------------------------------------

@app.post("/api/export/metadata")
async def export_metadata_endpoint(payload: dict):
    """
    Write XMP/JSON sidecars for a list of graded photos.
    payload: { photos: [{path, grade, score, critique, breakdown, nima_score}],
               dest: optional output folder }
    Returns list of {source, sidecar} pairs.
    """
    from engine_utils import export_metadata
    photos  = payload.get("photos", [])
    raw_dest = payload.get("dest") or None
    dest = str(_safe_dir_path(raw_dest)) if raw_dest else None
    if not photos:
        raise HTTPException(400, "No photos provided")
    results = []
    for p in photos:
        try:
            src = _safe_image_path(p["path"])
            sidecar = export_metadata(str(src), p, out_dir=dest)
            results.append({"source": p["path"], "sidecar": sidecar})
        except Exception as e:
            results.append({"source": p["path"], "error": str(e)})
    return JSONResponse({"exported": len([r for r in results if "sidecar" in r]),
                         "results": results})


# ---------------------------------------------------------------------------
# Incremental folder watch
# ---------------------------------------------------------------------------

_folder_watcher: "FolderWatcher | None" = None   # type: ignore[name-defined]
_watched_folder: str = ""

@app.post("/api/watch/start")
async def watch_start(payload: dict):
    """
    Start watching a folder for new/modified images.
    New arrivals are auto-added to the vector DB (non-blocking background task).
    payload: { folder: str }
    """
    from engine_utils import FolderWatcher, LocalVectorDB

    global _folder_watcher, _watched_folder
    folder = payload.get("folder", "").strip()
    preset = payload.get("preset", "Classic Street")
    if not folder or not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")

    if _folder_watcher is not None:
        _folder_watcher.stop()

    vec_db = LocalVectorDB()
    _folder_watcher = FolderWatcher(analyzer=get_analyzer(), db=vec_db, preset=preset)
    _folder_watcher.watch(folder)
    _watched_folder = folder
    return {"status": "watching", "folder": folder}


@app.post("/api/watch/stop")
async def watch_stop():
    global _folder_watcher, _watched_folder
    if _folder_watcher:
        _folder_watcher.stop()
        _folder_watcher = None
    _watched_folder = ""
    return {"status": "stopped"}


@app.get("/api/watch/status")
async def watch_status():
    return {"watching": _folder_watcher is not None, "folder": _watched_folder}


# ---------------------------------------------------------------------------
# Vector DB search
# ---------------------------------------------------------------------------

@app.post("/api/search/similar")
async def search_similar(payload: dict):
    """
    Find visually similar images in the vector DB for a given source path.
    payload: { path: str, limit: int = 20 }
    """
    from engine_utils import LocalVectorDB
    path  = payload.get("path", "")
    limit = int(payload.get("limit", 20))
    emb   = analyzer.cache.get(path, {}).get("embedding")
    if emb is None:
        raise HTTPException(404, f"No embedding cached for {path!r} — grade it first")
    vec_db = LocalVectorDB()
    results = vec_db.search(emb, limit=limit)
    return JSONResponse({"query": path, "results": results})


# ---------------------------------------------------------------------------
# Reference bank (exemplar indexing)
# ---------------------------------------------------------------------------

@app.get("/api/exemplar-count")
def exemplar_count():
    return {"count": analyzer._ref_bank.count}


@app.get("/api/nima-status")
def nima_status():
    """Returns whether the NIMA ONNX has been generated and is loaded."""
    from pathlib import Path as _P
    onnx_exists = _P("models/onnx/nima.onnx").exists()
    session_loaded = (
        analyzer._ort_sessions is not None and
        "nima" in analyzer._ort_sessions
    )
    return {"available": session_loaded, "onnx_exists": onnx_exists}


async def _warm_and_run(fn):
    """Ensure ONNX is loaded, then run fn() in a thread executor."""
    import asyncio
    if analyzer._ort_sessions is None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, analyzer._ensure_sessions)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


@app.post("/api/index-exemplars")
async def index_exemplars(payload: dict):
    """Replace the entire bank with embeddings from a folder."""
    folder = payload.get("folder_path", "")
    if not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")
    count = await _warm_and_run(
        lambda: analyzer._ref_bank.build(
            folder,
            analyzer._ort_sessions["composition"],
            analyzer._comp_input,
        )
    )
    return {"status": "indexed", "count": count}


@app.post("/api/add-exemplars")
async def add_exemplars(payload: dict):
    """Append new exemplars from a folder to the existing bank (deduped)."""
    folder = payload.get("folder_path", "")
    if not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")
    added, skipped = await _warm_and_run(
        lambda: analyzer._ref_bank.add(
            folder,
            analyzer._ort_sessions["composition"],
            analyzer._comp_input,
        )
    )
    return {"status": "added", "added": added, "skipped": skipped,
            "total": analyzer._ref_bank.count}


@app.post("/api/clear-exemplars")
def clear_exemplars():
    """Remove all exemplars from the bank."""
    analyzer._ref_bank.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@app.post("/api/clear_cache")
def clear_cache():
    if os.path.exists(str(_DATA_DIR / "cache" / "light_scores.json")):
        os.remove(str(_DATA_DIR / "cache" / "light_scores.json"))
        analyzer.cache.clear()
        return {"status": "cleared"}
    analyzer.cache.clear()
    return {"status": "empty"}


# ---------------------------------------------------------------------------
# Pacing presets
# ---------------------------------------------------------------------------

@app.get("/api/presets")
def get_presets():
    from sequence_engine import PacingManager
    return PacingManager().presets

@app.post("/api/presets/save")
def save_preset(payload: dict):
    from sequence_engine import PacingManager
    pm = PacingManager()
    pm.save_custom_weights(
        payload.get("name", "Custom"),
        payload.get("weights", {}),
    )
    return {"status": "saved"}


# ---------------------------------------------------------------------------
# Serve React frontend (catch-all — must be last)
# ---------------------------------------------------------------------------

DIST = _EXE_DIR / "frontend" / "dist"

import json

@app.get("/api/saved-sequences")
async def get_saved_sequences():
    """Return list of saved sequences."""
    sequences_file = _DATA_DIR / "cache" / "saved_sequences.json"
    if not sequences_file.exists():
        return {"sequences": []}
    try:
        with open(sequences_file, "r") as f:
            return json.load(f)
    except Exception:
        return {"sequences": []}


@app.post("/api/save-sequence")
async def save_sequence(payload: dict):
    """Save a sequence to disk."""
    name = payload.get("name")
    sequence = payload.get("sequence", [])
    if not name or not sequence:
        raise HTTPException(400, "Name and sequence required")

    sequences_file = _DATA_DIR / "cache" / "saved_sequences.json"
    sequences_file.parent.mkdir(exist_ok=True)
    
    try:
        with open(sequences_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"sequences": []}
    
    # Remove existing sequence with same name
    data["sequences"] = [s for s in data["sequences"] if s["name"] != name]
    data["sequences"].append({"name": name, "sequence": sequence})
    
    with open(sequences_file, "w") as f:
        json.dump(data, f, indent=2)
    
    return {"success": True, "message": f"Sequence '{name}' saved"}


_CATALOG_PATH = _DATA_DIR / "cache" / "catalog.json"

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}

@app.get("/api/catalog")
async def get_catalog():
    if not _CATALOG_PATH.exists():
        return JSONResponse({"exists": False}, headers=_NO_CACHE_HEADERS)
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        return JSONResponse({"exists": True, **data}, headers=_NO_CACHE_HEADERS)
    except Exception:
        return JSONResponse({"exists": False}, headers=_NO_CACHE_HEADERS)

@app.post("/api/catalog/save")
async def save_catalog(payload: dict):
    photos  = payload.get("photos", [])
    folders = payload.get("folders", [])
    _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CATALOG_PATH.write_text(
        json.dumps({
            "photos":    photos,
            "folders":   folders,
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True}

@app.post("/api/catalog/clear")
async def clear_catalog():
    if _CATALOG_PATH.exists():
        _CATALOG_PATH.unlink()
    return {"ok": True}


@app.post("/api/flags/lock")
async def toggle_lock(payload: dict):
    """Toggle lock flag for a photo."""
    path = payload.get("path", "")
    lock_file = _DATA_DIR / "cache" / "photo_flags.json"
    lock_file.parent.mkdir(exist_ok=True)
    try:
        if lock_file.exists():
            with open(lock_file, "r") as f:
                data = json.load(f)
        else:
            data = {"locked": []}
        if path in data["locked"]:
            data["locked"].remove(path)
        else:
            data["locked"].append(path)
        with open(lock_file, "w") as f:
            json.dump(data, f, indent=2)
        return {"success": True, "locked": path in data["locked"]}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/flags/used")
async def toggle_used(payload: dict):
    """Toggle used flag for a photo."""
    path = payload.get("path", "")
    used_file = _DATA_DIR / "cache" / "photo_flags.json"
    used_file.parent.mkdir(exist_ok=True)
    try:
        if used_file.exists():
            with open(used_file, "r") as f:
                data = json.load(f)
        else:
            data = {"used": []}
        if path in data["used"]:
            data["used"].remove(path)
        else:
            data["used"].append(path)
        with open(used_file, "w") as f:
            json.dump(data, f, indent=2)
        return {"success": True, "used": path in data["used"]}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/api/flags/load")
async def load_flags():
    """Load all photo flags."""
    flags_file = _DATA_DIR / "cache" / "photo_flags.json"
    try:
        if flags_file.exists():
            with open(flags_file, "r") as f:
                return json.load(f)
        return {"locked": [], "used": []}
    except Exception:
        return {"locked": [], "used": []}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Ingest a local image file for use as a Creative Director peg.

    Saves the file to data/ingestion_queue/, hashes it with MD5,
    and inserts a stub LanceDB record so vector_search can locate it.
    Returns {"status": "success", "hash": file_hash, "path": saved_path}.
    """
    import hashlib as _hl

    # Bounded read — cap memory so an oversized upload can't OOM the process on a
    # RAM-tight machine (was an unbounded await file.read()).
    _MAX_IMG_BYTES = 64 * 1024 * 1024   # 64 MB
    _chunks, _size = [], 0
    while True:
        _c = await file.read(1024 * 1024)
        if not _c:
            break
        _size += len(_c)
        if _size > _MAX_IMG_BYTES:
            raise HTTPException(413, "Image too large (max 64 MB).")
        _chunks.append(_c)
    data      = b"".join(_chunks)
    file_hash = _hl.md5(data).hexdigest()

    queue_dir = _DATA_DIR / "data" / "ingestion_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in _IMAGE_EXTS:
        raise HTTPException(400, "Unsupported image format")
    dest = queue_dir / f"{file_hash}{suffix}"
    dest.write_bytes(data)

    # Phase 1 — VLM Pixel Inspector (Ollama qwen2.5vl:3b)
    # Runs before LanceDB insertion so the semantic profile is stored in breakdown.
    semantic_profile = ""
    try:
        import asyncio as _aio
        from fast_ingestion import run_pixel_inspector
        loop = _aio.get_running_loop()
        semantic_profile = await loop.run_in_executor(
            None, run_pixel_inspector, str(dest), 0.0
        )
    except Exception as _e_pi:
        print(f"[upload] pixel_inspector skipped: {_e_pi}")

    # Insert stub LanceDB record — zero-vector placeholder so peg lookup resolves.
    # semantic_profile is embedded in breakdown JSON so downstream Art Director can use it.
    import json as _json_mod
    breakdown_stub = _json_mod.dumps({"semantic_profile": semantic_profile})
    try:
        import lance_store as _ls
        _ls.upsert_batch([{
            "path":           str(dest),
            "embedding":      [0.0] * 1536,
            "score":          0.0,
            "personal_score": 0.0,
            "grade":          "Pending",
            "reasoning_log":  semantic_profile,
            "breakdown":      breakdown_stub,
            "exif_ts":        0.0,
        }])
        print(f"[upload] stub record inserted: {dest.name}  hash={file_hash}")
    except Exception as _e_ldb:
        print(f"[upload] LanceDB stub insert failed: {_e_ldb}")

    # TODO: Trigger full grading pipeline here

    return JSONResponse({"status": "success", "hash": file_hash, "path": str(dest)})


@app.get("/api/heatmap/technical/{image_hash:path}")
async def heatmap_technical(image_hash: str):
    """Return a Base64-encoded RGBA PNG blur heatmap for the given image path."""
    import asyncio
    from pipeline.heatmaps import generate_technical_heatmap

    p = _safe_image_path(image_hash)
    readable_path = str(p)

    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        preview = _gen_preview(str(p))
        if preview:
            readable_path = str(preview)
        else:
            raise HTTPException(422, "Could not decode RAW/HEIC for heatmap")

    try:
        loop = asyncio.get_running_loop()
        b64 = await loop.run_in_executor(None, generate_technical_heatmap, readable_path)
        return JSONResponse({"b64": b64})
    except Exception as exc:
        raise HTTPException(500, f"Heatmap generation failed: {exc}")


@app.get("/api/search/semantic")
async def search_semantic(q: str = Query(default="", description="Free-text photo description")):
    """
    Natural-language semantic search over the LanceDB embedding index.

    Uses SigLIP-2's text tower to encode the query and returns the top-20
    most visually similar photos.  Reuses the grading singleton if loaded
    (no extra VRAM); falls back to a CPU encoder otherwise.
    """
    import asyncio as _aio

    if not q.strip():
        raise HTTPException(400, "Query parameter 'q' is required and must be non-empty")

    try:
        from creative_director import semantic_search as _ss
        loop    = _aio.get_running_loop()
        results = await loop.run_in_executor(None, lambda: _ss(query=q, limit=20))
        return JSONResponse({"results": results, "query": q})
    except Exception as exc:
        raise HTTPException(500, f"Semantic search failed: {exc}")


@app.post("/api/critique/details")
async def critique_details(req: Request):
    """
    On-demand deep text critique for a single selected photo.

    Sends the image to Qwen2.5-VL via Ollama using GENERATE_DEEP_TEXT_PROMPT —
    text only, no scores, no bounding boxes.  RAG context from
    cache/rag_concepts.json is injected automatically.

    Body: { "image_path": str, "mode": "story" | "competition" }
    Returns: { "narrative_arc": str, "geometry_composition": str }
    """
    data = await req.json()
    raw_path = data.get("image_path", "").strip()
    mode     = data.get("mode", "story")

    if not raw_path:
        raise HTTPException(400, "image_path is required")

    try:
        img_path = str(_safe_image_path(raw_path))
    except HTTPException:
        raise
    except Exception as _e:
        raise HTTPException(400, f"Invalid image path: {_e}")

    from qwen_vlm_grader import execute_vlm_text_deep_dive
    result = await run_in_threadpool(execute_vlm_text_deep_dive, img_path, mode)

    if result is None:
        raise HTTPException(503, "The critique model is not installed. Run the model "
                                 "downloader to fetch it — grading is unaffected.")

    return JSONResponse(result)


@app.get("/api/critique/jury/{image_hash:path}")
async def jury_critique(image_hash: str):
    """
    Generate a 3-paragraph DeepSeek-R1:8b jury critique for a single image.

    Runs the GGUF inference in an isolated subprocess so a C-level crash or
    OOM in llama-cpp-python kills only the child, not this server process.

    image_hash : MD5 stem used as the LanceDB path identifier.
    Returns {"critique": str, "think": str}.
    """
    import sys as _sys
    import asyncio as _aio
    import json as _json
    import subprocess as _sp

    if not image_hash.strip():
        raise HTTPException(400, "image_hash is required")

    # Inline script executed in the child process.
    # Uses sys.path.insert so it can import from src/ without package install.
    _critique_script = r"""
import sys, json
sys.path.insert(0, 'src')
image_hash = sys.argv[1]
try:
    from critique_engine import run_jury_critique
    result = run_jury_critique(image_hash=image_hash)
    print(json.dumps(result), flush=True)
except Exception as _e:
    import traceback
    print(json.dumps({
        "error":   f"{type(_e).__name__}: {_e}",
        "critique": "",
        "think":   traceback.format_exc(),
    }), flush=True)
"""

    try:
        # Use pythonw.exe on Windows — windowless Python never flashes a console.
        _py = _sys.executable
        if os.name == "nt" and _py.lower().endswith("python.exe"):
            _pyw = _py[:-10] + "pythonw.exe"
            if os.path.exists(_pyw):
                _py = _pyw
        _cflags = _sp.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = await _aio.create_subprocess_exec(
            _py, "-c", _critique_script, image_hash,
            stdout=_aio.subprocess.PIPE,
            stderr=_aio.subprocess.PIPE,
            cwd=str(Path(__file__).parent),
            creationflags=_cflags,
        )

        try:
            stdout_b, stderr_b = await _aio.wait_for(proc.communicate(), timeout=90)
        except _aio.TimeoutError:
            # Kill and reap the child so it doesn't linger as a zombie
            try:
                proc.kill()
                await _aio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            print("[jury] subprocess timed out after 90 s — killed and reaped")
            return JSONResponse(
                {"error": "Critique timed out (90 s). Model may still be loading — try again.", "critique": "", "think": ""},
                status_code=504,
            )

        stderr_txt = stderr_b.decode(errors="replace").strip()
        if stderr_txt:
            # Forward child stderr to server log for debugging (truncated)
            print(f"[jury subprocess] {stderr_txt[:800]}")

        stdout_txt = stdout_b.decode(errors="replace").strip()
        if not stdout_txt:
            code = proc.returncode
            print(f"[jury] subprocess exited {code} with no stdout")
            return JSONResponse(
                {"error": f"Critique engine exited ({code}) with no output.", "critique": "", "think": stderr_txt[:400]},
                status_code=500,
            )

        # The child may emit debug prints before the final JSON line — find the last one.
        result: dict = {}
        for _line in reversed(stdout_txt.splitlines()):
            _line = _line.strip()
            if _line.startswith("{"):
                try:
                    result = _json.loads(_line)
                    break
                except _json.JSONDecodeError:
                    continue

        if not result:
            return JSONResponse(
                {"error": f"Could not parse critique output: {stdout_txt[:200]}", "critique": "", "think": ""},
                status_code=500,
            )

        if result.get("error") and not result.get("critique"):
            print(f"[jury] critique error from child: {result['error'][:200]}")
            return JSONResponse({"error": result["error"], "critique": "", "think": result.get("think", "")})

        return JSONResponse(result)

    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        return JSONResponse(
            {"error": f"Server error: {type(exc).__name__}: {exc}", "critique": "", "think": ""},
            status_code=500,
        )


class ModelPullRequest(BaseModel):
    model_name: str

@app.post("/api/models/pull")
async def pull_model_stream(req: ModelPullRequest):
    """Stream the model downloader's progress as ndjson.

    This used to proxy Ollama's /api/pull. Ollama is gone, so it proxied a port
    nothing was listening on. It now runs scripts/fetch_models.py, which asks
    tier_select which encoder this machine will actually use and fetches only
    that — the difference between ~0.8 GB on a CPU laptop and the 20+ GB the old
    unconditional prefetch would have pulled.

    A subprocess, not an in-process call: downloads take minutes, and the event
    loop must stay free to serve the UI that is displaying this progress.
    """
    def _stream():
        # Imported here, not at module scope: suppress_console patches subprocess
        # at import line 1, and a local import picks up the patched module.
        import subprocess
        proc = None
        try:
            cmd = [sys.executable, str(Path(__file__).parent / "scripts" / "fetch_models.py"),
                   "--json"]
            if req.model_name in ("optional", "all"):
                cmd.append("--with-optional" if req.model_name == "optional" else "--all")
            proc = subprocess.Popen(
                cmd, cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:                      # type: ignore[union-attr]
                line = line.strip()
                if line.startswith("{"):
                    yield (line + "\n").encode()
            proc.wait(timeout=30)
        except Exception as exc:
            import json as _j
            yield (_j.dumps({"name": "plan", "status": "fail",
                             "message": str(exc)[:200]}) + "\n").encode()
        finally:
            # A disconnected client must not leave a multi-GB download orphaned.
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@app.get("/api/health/engine")
async def health_engine():
    """Report which optional AI model files are present on disk.

      - status "online"  : at least one model file is installed
      - status "offline" : none are
      - missing_models   : the file names that are absent

    This used to be a hybrid check that first asked Ollama over HTTP which model
    tags it had pulled. That made a network round-trip on a 10-second UI poll to
    answer a question about the local filesystem, and every entry it could return
    named a third-party service the installer never set up. Every name here is now
    a .gguf file, which the frontend already distinguishes from a service tag, so
    the "install Ollama" branch of the banner is unreachable by construction.

    Grading does not depend on any of these. They power critique, annotations and
    Story Mode; the response is advisory, never a gate.
    """
    def _check_sync() -> dict:
        # From model_registry, not a fourth copy of the literals. The list here
        # named the 2B GGUFs, which do not exist on the Hub, so this endpoint
        # reported two permanently-missing files and the UI offered to download
        # something unobtainable. That is precisely why the registry exists.
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
        import model_registry as _mr
        missing = [m.dest.name for m in _mr.missing_gguf()]
        present = len(_mr.GGUF_MODELS) - len(missing)
        return {"status": "online" if present else "offline",
                "missing_models": missing}

    result = await asyncio.get_running_loop().run_in_executor(None, _check_sync)
    return JSONResponse(result)


@app.get("/api/annotations/{image_hash:path}")
async def get_annotations(image_hash: str):
    """Return has_annotations, score_factors, and eye_overlay_url for a single image."""
    # Resolve eye overlay URLs for canvas_renderer.py outputs
    _overlay_base = Path(__file__).parent / "static" / "eye_feature_overlays"
    _verified     = _overlay_base / f"verified_{image_hash}.png"
    _critique     = _overlay_base / f"critique_{image_hash}.png"
    eye_overlay_url: str = ""
    if _verified.exists():
        eye_overlay_url = f"/static/eye_feature_overlays/verified_{image_hash}.png"
    elif _critique.exists():
        eye_overlay_url = f"/static/eye_feature_overlays/critique_{image_hash}.png"

    try:
        import lance_store as _ls
        all_rows = _ls.query_all(min_score=0.0)
        record   = next(
            (r for r in all_rows
             if Path(r["path"]).stem == image_hash or image_hash in Path(r["path"]).stem),
            None,
        )
        if record is None:
            return JSONResponse({
                "has_annotations": "",
                "score_factors":   [],
                "eye_overlay_url": eye_overlay_url,
            })
        import json as _json
        _sf_raw = record.get("score_factors", "")
        try:
            _sf = _json.loads(_sf_raw) if _sf_raw else []
        except Exception:
            _sf = []
        return JSONResponse({
            "has_annotations": record.get("has_annotations", ""),
            "score_factors":   _sf,
            "eye_overlay_url": eye_overlay_url,
        })
    except Exception as _e:
        return JSONResponse({
            "has_annotations": "",
            "score_factors":   [],
            "eye_overlay_url": eye_overlay_url,
            "error": str(_e),
        })


@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """
    Upload a PDF reference document. Extracts text, runs LLM concept extraction,
    appends phrases to cache/rag_concepts.json.
    """
    import tempfile as _tmp, shutil as _sh
    try:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported")

        # Save upload to a temp file (bounded) then hand off to pdf_rag.ingest_pdf.
        # Cap at 200 MB so an oversized upload can't fill the disk / OOM the parser.
        suffix = ".pdf"
        _MAX_PDF_BYTES = 200 * 1024 * 1024
        with _tmp.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            _written = 0
            while True:
                _c = await file.read(1024 * 1024)
                if not _c:
                    break
                _written += len(_c)
                if _written > _MAX_PDF_BYTES:
                    tf.close(); Path(tf.name).unlink(missing_ok=True)
                    raise HTTPException(413, "PDF too large (max 200 MB).")
                tf.write(_c)
            tmp_path = tf.name

        from pdf_rag import ingest_pdf as _ingest
        phrases = await run_in_threadpool(_ingest, tmp_path, file.filename)

        Path(tmp_path).unlink(missing_ok=True)

        from pdf_rag import list_pdfs as _list

        # Bust the SigLIP-2 text embedding cache so next grade run re-encodes
        # with the new PDF phrases baked into the positive rubric.
        try:
            import sys as _sys
            _gp = _sys.modules.get("grade_pipeline_v2")
            if _gp is not None:
                _gp._text_emb_cache.clear()
                print("[server] RAG upload: cleared SigLIP text embedding cache")
        except Exception as _e_cache:
            print(f"[server] Cache clear skipped: {_e_cache}")

        return JSONResponse({"ok": True, "phrases": phrases, "pdfs": _list()})
    except HTTPException:
        raise
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/rag/concepts")
async def rag_concepts():
    """Return all stored concept phrases and PDF metadata."""
    from pdf_rag import load_concepts as _lc, list_pdfs as _lp
    return JSONResponse({"phrases": _lc(), "pdfs": _lp()})


@app.delete("/api/rag/clear")
async def rag_clear():
    """Remove all stored concept phrases and uploaded PDF records."""
    from pdf_rag import clear_concepts as _cc
    await run_in_threadpool(_cc)
    try:
        import sys as _sys
        _gp = _sys.modules.get("grade_pipeline_v2")
        if _gp is not None:
            _gp._text_emb_cache.clear()
    except Exception:
        pass
    return JSONResponse({"ok": True})


@app.post("/api/reasoning_overlay")
async def reasoning_overlay(req: Request):
    """
    Render photographer-style reasoning annotations onto the image server-side.
    Body: { path: str }
    Returns: { overlay_url: str } or { error: str }
    """
    try:
        body      = await req.json()
        raw_path  = body.get("path", "")
        if not raw_path:
            return JSONResponse({"error": "path required"}, status_code=400)
        img_path  = sanitize_path(raw_path)

        import lance_store as _ls
        rows = _ls.query_all(min_score=0.0)
        record = next(
            (r for r in rows if str(Path(r["path"])) == str(Path(img_path))),
            None,
        )
        if record is None:
            return JSONResponse({"error": "image not found in DB"}, status_code=404)

        bd = record.get("breakdown") or {}
        if isinstance(bd, str):
            import json as _json
            try:
                bd = _json.loads(bd)
            except Exception:
                bd = {}

        reasoning_log = record.get("reasoning_log", "")
        if not reasoning_log and bd:
            from src.specvlm_pipeline import _build_reasoning as _br, _detect_genre as _dg
            reasoning_log = _br(
                float(record.get("score", 0.5)),
                bd,
                bool(record.get("is_verified", False)),
                grade=record.get("grade", ""),
                genre=_dg(bd),
            )
        if not reasoning_log:
            return JSONResponse({"error": "no reasoning data for this image"}, status_code=404)

        from canvas_renderer import render_reasoning_overlay as _rro
        overlay_url = await run_in_threadpool(
            _rro,
            img_path,
            reasoning_log,
            bd,
            float(record.get("score", 0.5)),
            record.get("grade", ""),
        )
        return JSONResponse({"overlay_url": overlay_url})
    except Exception as _e:
        import traceback as _tb
        _tb.print_exc()
        return JSONResponse({"error": str(_e)}, status_code=500)


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Path-traversal guard: `%2e%2e/…` in the :path param would otherwise let
    # DIST / full_path escape the web root and serve arbitrary files on disk
    # (confirmed: GET /%2e%2e/%2e%2e/README.md returned the repo file). Resolve
    # and require the result to stay inside DIST; anything else falls through to
    # the SPA index.
    _dist_root = DIST.resolve()
    try:
        candidate = (DIST / full_path).resolve()
        candidate.relative_to(_dist_root)
    except (ValueError, OSError):
        candidate = None
    if candidate is not None and candidate.exists() and candidate.is_file():
        # Hashed assets (JS/CSS with content-hash filenames) are safe to cache forever.
        # index.html must never be cached — always revalidate so a new build is picked up instantly.
        suffix = candidate.suffix.lower()
        if suffix in (".js", ".css") and any(
            c in candidate.stem for c in ("-", "_")
        ):
            headers = {"Cache-Control": "public, max-age=31536000, immutable"}
        else:
            headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return FileResponse(str(candidate), headers=headers)
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(
            str(index),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    raise HTTPException(404, "Frontend not built. Run: cd frontend && npm run build")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
