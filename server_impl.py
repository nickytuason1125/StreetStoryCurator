import suppress_console  # patches subprocess/multiprocessing/asyncio/BLAS before anything else imports them
import os
import re
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
    # PyInstaller 6.x puts every bundled data file under <exe dir>/_internal,
    # NOT beside the executable the way 5.x did. sys._MEIPASS points at that
    # directory, so it is what CWD must be for relative reads of models/,
    # frontend/dist/ and the shipped calibration anchors to resolve. Pointing
    # CWD at the exe directory — correct under the old layout, and what this
    # did — lands one level above all of it, and the app starts up unable to
    # find its own models.
    _BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    _EXE_DIR = _BUNDLE_DIR
    os.chdir(_BUNDLE_DIR)
    # Writable state stays OUT of the bundle: _internal is reinstalled wholesale
    # on update and may sit under Program Files, which is read-only.
    _DATA_DIR = Path(os.environ.get(
        'CURATOR_DATA_DIR',
        str(Path(os.environ.get('LOCALAPPDATA', Path.home())) / 'FrameGrade')))
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
else:
    _EXE_DIR = Path(__file__).parent
    _DATA_DIR = _EXE_DIR

# The catalog's location — shared state, so it lives with the rest of it.
#
# It was defined ONLY in routers/misc.py while routers/grading.py referenced it
# five times. grading.py's module __getattr__ falls back to server_impl, which
# did not have it either, so every reference raised NameError at runtime. The
# route registers, imports succeed, and the failure appears only when someone
# actually grades — which is why the split's parity checks never saw it.
_CATALOG_PATH = _DATA_DIR / "cache" / "catalog.json"

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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically: temp file in the same dir, then os.replace().

    Used for every state file a user depends on across sessions (catalog
    Resume, saved sequences, photo flags). The old direct write_text/open('w')
    pattern truncated the destination BEFORE writing, so a crash or power loss
    mid-write destroyed the only copy — taking the Resume-after-crash feature
    down with it. os.replace is atomic on Windows and POSIX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))


RECENTLY_GENERATED: set = set()
MAX_HISTORY = 25
LAST_SEQUENCE: list = []   # paths from the most recent generation — used as avoid_paths

# ── Creative Direction — used-path persistence ────────────────────────────────
# Resolved under _DATA_DIR (not CWD-relative) so the frozen/ CURATOR_DATA_DIR
# configuration lands in the same store as every other cache file.
_USED_CD_PATHS_FILE = _DATA_DIR / "cache" / "used_cd_paths.json"

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
    # Atomic replace: a crash mid-write must not destroy the used-history.
    _atomic_write_text(_USED_CD_PATHS_FILE, _j.dumps(sorted(used), indent=2))

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

# nvidia-smi telemetry cache for /api/models/status (see the endpoint for why).
_SMI_CACHE: "tuple | None" = None    # (total_mib, free_mib, gpu_name)
_SMI_CACHE_TS: float = 0.0
_SMI_CACHE_S: float = 2.5

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

_MODELS_SENTINEL = Path(__file__).parent / "models" / ".models_ready"


def _models_ready() -> bool:
    """True when an encoder is genuinely installed — not merely claimed.

    The sentinel alone is not trusted, because it was TRACKED IN GIT: a fresh
    clone arrived carrying models/.models_ready and zero weights, so a
    sentinel-only check short-circuited the prefetch and nothing ever downloaded.
    The file is untracked now, but a stale or hand-copied one must not be able to
    convince the app that a machine is provisioned when it is not — the failure is
    silent and lands on the user, which is the worst place for it.

    So the sentinel is a fast path and tier_select is the authority.
    """
    try:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
        import tier_select
        import run_profile
        return any(tier_select.available(t) for t in run_profile.TIERS)
    except Exception:
        return _MODELS_SENTINEL.exists()      # can't verify — trust it, don't loop


def _bg_model_prefetch():
    """Fetch what grading needs on first run, then warm the pipeline caches.

    This was commented out at the call site, and rightly so in its old form: it
    called model_loader.ensure_all_models_downloaded(), which fetched every model
    unconditionally — over 20 GB, including a 6.8 GB VLM for an opt-in feature and
    the giant encoder for a machine that may run the 768-d one. That cannot
    complete on a laptop, so leaving it disabled was safer than running it.

    It now runs scripts/fetch_models.py, which asks tier_select which encoder this
    machine will actually use and fetches only that (~0.8 GB on a CPU laptop).
    REQUIRED group only: the optional critique/Story-Mode models are several GB
    and the user asks for those explicitly from the UI, which streams the same
    script through /api/models/pull.

    Subprocess, not an in-process import: this must not hold a reference to torch
    or CUDA in the server process, which is the ancestor of the grade worker.
    """
    try:
        if _models_ready():
            return
        import subprocess
        print("[models] first run — fetching what this machine needs")
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "scripts" / "fetch_models.py")],
            cwd=str(Path(__file__).parent), timeout=3600,
        )
    except Exception as exc:
        # Never fatal. A failed prefetch leaves the app running and the UI can
        # retry through /api/models/pull; aborting startup would strand the user
        # with no way to trigger the download at all.
        print(f"[models] background prefetch error: {exc}")
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
    # Background model prefetch — re-enabled, and ONLY safe because it changed
    # shape. It was disabled because the old version called
    # model_loader.ensure_all_models_downloaded() directly in this daemon thread,
    # which imports BitsAndBytes INT4 and instantiates a pyiqa metric — i.e. it
    # initialised CUDA in the server process before the grading path did, and
    # Windows then killed the process at C level with no traceback.
    #
    # _bg_model_prefetch now shells out to scripts/fetch_models.py, so no torch,
    # no CUDA and no BitsAndBytes ever enter THIS process. The crash cause is
    # removed rather than tolerated. It is also a no-op once models/.models_ready
    # exists, so it costs a stat() on every launch after the first.
    #
    # Do NOT convert this back to an in-process call.
    _t = threading.Thread(target=_bg_model_prefetch, daemon=True, name="model-prefetch")
    _t.start()

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
        # A top-level navigation sends sec-fetch-site: none, which passes above.
        # No legitimate app request navigates to an API route (all data calls are
        # same-origin fetch/XHR → mode "cors"), but a phishing link like
        # <a href="http://127.0.0.1:8000/api/photo?path=…"> does. Reject it.
        if (request.headers.get("sec-fetch-mode") or "") == "navigate":
            return JSONResponse(status_code=404, content={"error": "Not found"})
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


# ── Routers (Milestone 4 split) ──────────────────────────────────────
# Mounted before the SPA catch-all so migrated routes register first.
from routers import mount_all
mount_all(app)

# SPA web root. Resolved once at startup; the first candidate containing a
# built index.html wins (source checkout vs PyInstaller onedir layout).
def _find_dist() -> Path:
    _candidates = [
        Path(__file__).parent / "frontend" / "dist",
        _DATA_DIR / "frontend" / "dist",
        _EXE_DIR / "frontend" / "dist",
        _EXE_DIR / "dist",
    ]
    for _c in _candidates:
        if (_c / "index.html").exists():
            return _c
    return _candidates[0]   # not built yet — serve_spa will 404 with guidance

DIST = _find_dist()

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
        # Only Vite's content-hashed build output is safe to cache forever.
        # The old heuristic ("stem contains - or _") also matched hand-authored
        # files like design-board.css and pinned them for a year.
        suffix = candidate.suffix.lower()
        _hashed = bool(re.search(r"-[0-9a-zA-Z_-]{8}$", candidate.stem))
        if suffix in (".js", ".css") and _hashed:
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
