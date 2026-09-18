import suppress_console  # patches subprocess/multiprocessing/asyncio/BLAS before anything else imports them
import os
# Personal taste blend RETIRED (2026-09-14, user decision): the user saw the
# taste head override legit keepers (TPE26-107 pulled from Strong to Mid) and
# asked for the personal calculation to be gone entirely. Grading now stands
# on the machine grader alone everywhere. The pipeline still honours an
# explicit FIRSTCUT_PERSONAL_TASTE=0/1 from the environment — this default is
# just no longer ON.
os.environ.setdefault("FIRSTCUT_PERSONAL_TASTE", "0")
# Boundary-band Deep Grade (2026-09-10): when Deep Grade runs, spend the VLM's
# seconds only on the ambiguous Mid/Strong boundary — unambiguous frames keep
# their instant CLIP verdicts. Full-folder VLM = 60–90 min; band = ~20 min.
os.environ.setdefault("FIRSTCUT_DEEP_BAND", "1")
# Vision engine A/B (2026-09-10): QWEN_VLM_DIR picks the grader weights. The
# HF stubs for Qwen3-VL-4B exist locally; set the var (or uncomment below) to
# promote the 2026-era judge once its weights are downloaded:
#   venv\Scripts\huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir models\qwen3_vl
# os.environ.setdefault("QWEN_VLM_DIR", "models/qwen3_vl")
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
from fastapi.middleware.gzip import GZipMiddleware
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
    from platform_compat import app_data_dir
    _DATA_DIR = Path(os.environ.get(
        'CURATOR_DATA_DIR', str(app_data_dir())))
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
    """HARDWARE CEILING for the thumbnail pools. Returns (on_demand, prewarm).

    History: this used to read FREE RAM and freeze the result for the process
    lifetime - a boot at 2-3 GB free pinned the grid to 2 workers for ever,
    even after Chrome closed and 5 GB freed up. Since 2026-09-14 the ceiling
    is hardware-only (installed RAM never changes mid-session) and the actual
    concurrency is gated by _thumb_od_permits / _thumb_pw_permits, which are
    retuned every 30 s from live free RAM (src/adaptive_permits.py). The old
    free-RAM table moved there - same numbers, now live instead of frozen."""
    try:
        import psutil as _ps
        total_gb = _ps.virtual_memory().total / 1e9
        by_total = 8 if total_gb >= 24 else 6 if total_gb >= 12 else 4 if total_gb >= 8 else 2
        return by_total, (4 if total_gb >= 16 else 2)
    except Exception:
        return 6, 2

_THUMB_OD_CEIL, _THUMB_PW_CEIL = _thumb_pool_sizes()
_THUMB_ONDEMAND = ThreadPoolExecutor(max_workers=_THUMB_OD_CEIL)  # high-priority, browser-facing
_THUMB_PREWARM  = ThreadPoolExecutor(max_workers=_THUMB_PW_CEIL)  # low-priority background warm-up

# ── Live concurrency gates (2026-09-14) ──────────────────────────────────────
# Each decode waits for a permit before decoding. Limits retune every 30 s
# from CURRENT free RAM, so the pool is wide when the machine is comfortable
# and politely narrow during a squeeze - without a restart. See
# src/adaptive_permits.py for the gate semantics.
from src.adaptive_permits import AdaptivePermits as _AdaptivePermits

def _thumb_permit_limits() -> tuple:
    try:
        import psutil as _ps
        free_gb = _ps.virtual_memory().available / 1e9
    except Exception:
        return _THUMB_OD_CEIL, _THUMB_PW_CEIL
    # On-demand ladder tuned for the card-reader case (2026-09-15): the grid's
    # first screen is 15+ tiles, each a ~40 MB RAW read. At 4-wide that screen
    # took tens of seconds to fill ("thumbnails taking forever"). Each decode
    # is a short ~200 MB transient; when NO grade is running the encoder is
    # idle, so we can run wider than the grade-time comfort numbers — the
    # hard pool ceiling still caps us, and the RAM gates in serve_thumb /
    # _gen_one_thumb_decode still refuse at the true starvation floor.
    _ga = globals().get("_grading_active")   # defined below — retune thread may run first
    _grade_squeezed = bool(_ga and _ga.is_set())
    if _grade_squeezed:
        od = 8 if free_gb >= 8 else 6 if free_gb >= 5 else 4 if free_gb >= 3 else 2
    else:
        od = 8 if free_gb >= 8 else 8 if free_gb >= 5 else 6 if free_gb >= 3 else 4
    pw = _THUMB_PW_CEIL if free_gb >= 4 else (2 if free_gb >= 2 else 1)
    return min(od, _THUMB_OD_CEIL), min(pw, _THUMB_PW_CEIL)

try:
    import psutil as _ps_boot_p
    _free_boot_p = _ps_boot_p.virtual_memory().available / 1e9
except Exception:
    _free_boot_p = None
_fb = _free_boot_p or 0.0
_od0, _pw0 = ((8, 4) if _fb >= 8 else (6, 4) if _fb >= 5 else (4, 2) if _fb >= 3 else (2, 1))
_thumb_od_permits = _AdaptivePermits(min(_od0, _THUMB_OD_CEIL), "thumb-ondemand")
_thumb_pw_permits = _AdaptivePermits(min(_pw0, _THUMB_PW_CEIL), "thumb-prewarm")

def _retune_thumb_permits() -> None:
    while True:
        try:
            _od, _pw = _thumb_permit_limits()
            _thumb_od_permits.set_limit(_od)
            _thumb_pw_permits.set_limit(_pw)
        except Exception:
            pass
        import time as _t_r
        _t_r.sleep(30)

threading.Thread(target=_retune_thumb_permits, daemon=True,
                 name="thumb-permit-retune").start()

# Set while a grade is streaming. Background thumbnail PREWARM (bulk RAW decodes)
# is skipped while this is set so it doesn't spike RAM next to the SigLIP-2 / Qwen
# load; on-demand thumbnails (what the user is actually looking at) still run.
_grading_active = threading.Event()


# ── RAM watchdog (2026-09-14) ────────────────────────────────────────────────
# The server process is the single biggest controllable consumer on the machine
# (~1.8 GB private measured on the 16 GB dev box): decode buffers, preview
# caches and allocator arenas from hours of thumbnail/preview work. When system
# RAM gets tight those idle pages still count against "available", which
# (a) 204'd the whole thumbnail grid at the 1.2 GB thumb gate, and (b) helped
# push the grader's judge-stage checkpoint below its floor ("grading stopped
# early"). The watchdog trims THIS process's working set back to the OS when
# free RAM drops — idle pages go to the pagefile, live objects are untouched —
# and the trim is also available to the thumb gate as a last resort before it
# refuses a decode.
_trim_lock = threading.Lock()
_last_trim_ts = 0.0

def _accurate_free_gb() -> "float | None":
    """THE one memory figure every endpoint and gate must agree on.

    The old sync bug: /api/system/ram reported psutil's physical-only
    'available', while the grade gates (memory_plan) gate on
    min(physical free, commit headroom) — so the UI could show "3.8 GB free"
    while the pagefile limit made the SAME machine refuse a grade. Numbers the
    UI and the gates quote must come from the same function, or they will
    disagree exactly when it matters (a squeeze).
    """
    try:
        import sys as _s_acc, os as _o_acc
        _src = _o_acc.path.join(_o_acc.path.dirname(__file__), "src")
        if _src not in _s_acc.path:
            _s_acc.path.insert(0, _src)
        from src.memory_plan import free_ram_gb as _frg
        return _frg()
    except Exception:
        try:
            import psutil as _ps_acc
            return _ps_acc.virtual_memory().available / 1e9
        except Exception:
            return None

def _accurate_commit_headroom_gb() -> "float | None":
    """Paging-file headroom (commit limit − commit charge), or None."""
    try:
        import sys as _s_ch, os as _o_ch
        _src = _o_ch.path.join(_o_ch.path.dirname(__file__), "src")
        if _src not in _s_ch.path:
            _s_ch.path.insert(0, _src)
        from src.memory_plan import commit_headroom_gb as _chg
        return _chg()
    except Exception:
        return None

def _trim_server_working_set() -> float:
    """gc + EmptyWorkingSet on the SERVER process; returns free GB after.

    Rate-limited to once per 30 s (a trim pages ~1 GB of idle pages out, which
    is wasted work if the memory was about to be used again anyway). Always
    returns the post-trim measurement so callers can retry a refused decode.
    """
    global _last_trim_ts
    import time as _t_trim
    with _trim_lock:
        if _t_trim.monotonic() - _last_trim_ts < 30.0:
            return _accurate_free_gb() or 0.0
        _last_trim_ts = _t_trim.monotonic()
    import gc as _gc_trim
    _gc_trim.collect()
    if os.name == "nt":
        try:
            import ctypes as _ct
            _k32 = _ct.windll.kernel32
            _handle = _k32.GetCurrentProcess()
            _ten = _ct.c_size_t(10)
            if not _k32.SetProcessWorkingSetSize(_handle, _ten, _ten):
                _ct.windll.psapi.EmptyWorkingSet(_handle)
        except Exception:
            pass
    _free_after = _accurate_free_gb()
    print(f"[server] RAM watchdog: trimmed server working set — "
          f"{_free_after if _free_after is not None else '?'} GB free", flush=True)
    return _free_after or 0.0

def _ram_watchdog() -> None:
    # Supervisor form (2026-09-14): the inner loop never ends, and if anything
    # escapes it (BaseException included) the outer loop logs and restarts — a
    # watchdog that dies silently is worse than no watchdog.
    import time as _t_w
    while True:
        try:
            _last_beat = 0.0
            while True:
                try:
                    _free = _accurate_free_gb()
                    _pressure = _free is not None and _free < 2.0
                    # Trim earlier than the old 1.0 GB trigger: 2.5 GB is just
                    # above the grade-admission floor, so the trim runs BEFORE
                    # a cull has to refuse instead of after it stalled.
                    if _free is not None and _free < 2.5:
                        _trim_server_working_set()
                    # Idle model unload — the idle cap shrinks under pressure:
                    #   comfortable: 5 min idle → unload
                    #   squeezed (<2 GB free): 60 s idle → unload now
                    #
                    # Unified residency reaper (2026-09-16): instead of the old
                    # two hardcoded module checks, EVERY model that registered
                    # itself with model_residency at load time (vision_vl,
                    # text_llm, siglip_text_cpu, ...) gets idle-evicted here.
                    # Skipped while a grade is active — the grade pipeline owns
                    # its models' lifecycles and the worker is a subprocess.
                    # Pressure-responsive sidecar evict (2026-09-16): when the
                    # machine is squeezed and no grade is running, the sidecar
                    # exits entirely — its RAM AND VRAM (the Qwen-VL can hold
                    # ~4 GB of VRAM) return to the OS at once. Next use respawns
                    # it; the warm-ahead path re-warms when headroom returns.
                    #
                    # BUSY/LOADING GUARD (the "stuck at 33%" fix): never evict
                    # while the sidecar is mid-inference, mid-preload, or still
                    # inside its first 120 s (cold-loading weights — which on a
                    # CPU-offloaded VL model transiently eats ~5 GB and would
                    # otherwise trip this very eviction, killing the job and
                    # forcing an endless kill→respawn→cold-load thrash). Also
                    # require genuine IDLENESS: an active story run calls the
                    # sidecar every few seconds, so its idle_s stays small and
                    # it survives the squeeze until the run finishes.
                    if _pressure and not _grading_active.is_set():
                        try:
                            import sidecar_client as _sc_press
                            _m_press = _sc_press.read_marker()
                            if _m_press:
                                _safe_to_evict = True
                                _h_press = _sc_press.probe_health(
                                    int(_m_press["port"]), timeout=1.5)
                                if _h_press:
                                    if _h_press.get("busy"):
                                        _safe_to_evict = False
                                    try:
                                        if float(_h_press.get("idle_s", 0)) < 120:
                                            _safe_to_evict = False
                                    except Exception:
                                        pass
                                try:
                                    _age = time.time() - float(
                                        _m_press.get("started", 0))
                                    if _age < 120:
                                        _safe_to_evict = False
                                except Exception:
                                    pass
                                if _safe_to_evict:
                                    print("[server] RAM pressure — evicting the "
                                          "model sidecar (RAM + VRAM back to the OS)",
                                          flush=True)
                                    _sc_press.evict()
                        except Exception:
                            pass
                    _idle_cap = 60.0 if _pressure else 300.0
                    _evicted = []
                    if not _grading_active.is_set():
                        try:
                            import sys as _s_iu, os as _o_iu
                            _src_iu = _o_iu.path.join(_o_iu.path.dirname(__file__), "src")
                            if _src_iu not in _s_iu.path:
                                _s_iu.path.insert(0, _src_iu)
                            import model_residency as _res_iu
                            _evicted = _res_iu.evict_idle(_idle_cap)
                            if _evicted:
                                print(f"[server] RAM: idle-evicted resident "
                                      f"model(s): {', '.join(_evicted)} "
                                      f"(idle > {_idle_cap:.0f} s)", flush=True)
                        except Exception:
                            pass
                    _loaded = []
                    try:
                        import fast_niche_detector as _fnd_iu
                        if _fnd_iu.is_ready():
                            _loaded.append("niche")
                            if _fnd_iu.idle_seconds() > _idle_cap:
                                print(f"[server] RAM: niche detector idle "
                                      f"{_fnd_iu.idle_seconds():.0f} s — unloading", flush=True)
                                _fnd_iu.release()
                                _loaded.remove("niche")
                    except Exception:
                        pass
                    # Heartbeat: every 10 min, prove the watchdog is alive and
                    # say what it sees — a dead watchdog must be OBSERVABLE.
                    _now = _t_w.monotonic()
                    if _now - _last_beat > 600:
                        _last_beat = _now
                        print(f"[server] watchdog: alive — "
                              f"{_free if _free is not None else '?'} GB free, "
                              f"models resident: {', '.join(_loaded) or 'none'}", flush=True)
                except Exception:
                    pass
                _t_w.sleep(20)
        except BaseException as _e_wd:
            try:
                print(f"[server] RAM watchdog crashed and restarted: {_e_wd!r}", flush=True)
            except Exception:
                pass
            _t_w.sleep(5)

threading.Thread(target=_ram_watchdog, daemon=True,
                 name="ram-watchdog").start()


# ── Warm-ahead on user intent (2026-09-16) ───────────────────────────────────
# Browsing a folder is the signal that a critique / Story / text-LLM session
# may follow. Warm the sidecar's models in the background NOW so the first
# real request doesn't pay the 60-120 s cold load. Guardrails:
#   never during a grade (the grade needs the RAM/VRAM itself),
#   text model always (small), vision only with ≥3.5 GB RAM headroom,
#   throttled to once per 10 minutes.
_sidecar_warm_lock = threading.Lock()
_sidecar_last_warm = 0.0

def _sidecar_warm_on_browse() -> None:
    global _sidecar_last_warm
    if _grading_active.is_set():
        return
    import time as _t_warm
    now = _t_warm.monotonic()
    with _sidecar_warm_lock:
        if now - _sidecar_last_warm < 600.0:
            return
        _sidecar_last_warm = now

    def _warm():
        try:
            import sys as _s, os as _o
            _src = _o.path.join(_o.path.dirname(__file__), "src")
            if _src not in _s.path:
                _s.path.insert(0, _src)
            import sidecar_client as _sc
            if not _sc.ensure():
                return
            print("[server] warm-ahead: sidecar up — preloading text model", flush=True)
            _sc.preload("text")
            # Vision is Ollama-first now (GPU, keep_alive self-unloads in 30 s).
            # Preloading the sidecar's CPU vision model here ballooned ~8 GB of
            # committed RAM — the "stuck at 33%" episode. The sidecar vision
            # slot stays cold unless Ollama is unavailable.
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True, name="sidecar-warm").start()


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

# Free-RAM floor reported to the UI so it can warn BEFORE the user starts.
#
# This is only the DISPLAY figure, for when no folder has been chosen yet. The
# real gate is per-cull and lives in run_profile.required_ram_gb(n_photos) —
# because what a cull needs depends on the decode path and the job size, and no
# constant can express that. The previous constant (1.8) was Balanced's ENCODER
# floor masquerading as a whole-cull budget; it admitted culls that then ran the
# machine to 0.10 GB free.
def _grade_min_ram_gb() -> float:
    try:
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
        import run_profile as _rp
        return _rp.required_ram_gb(0)
    except Exception:
        return 3.8          # matches run_profile's measured draft-on figure


_GRADE_MIN_RAM_GB = _grade_min_ram_gb()

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
          f"THUMB_POOLS={_thumb_od_permits.limit}/{_thumb_pw_permits.limit}")


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

    # Startup hygiene (2026-09-07): a hard-killed grade leaves cache/grading.lock
    # behind. Remove it iff provably dead (dead PID / PID reused by a non-Python
    # process) — BEFORE anything can read it — so the first grade or catalog
    # clear of this session starts from ground truth instead of a ghost marker.
    try:
        from src.grade_lock import sweep_stale
        if sweep_stale(_DATA_DIR):
            print("[server] startup: cleared stale grading.lock from a dead grade")
    except Exception:
        pass  # hygiene, not correctness — the 409 gates re-derive truth on use

    # Startup hygiene (2026-09-10): ghost warm encoders. A worker outlives its
    # parent when the parent dies without _warm_shutdown (window closed
    # mid-prewarm, hard-killed server, terminated runner). Each orphan sits on
    # ~430 MB until its 600 s idle timeout — and duplicates were observed
    # ALIVE together ("2.0 GB free with nothing open"). Sweep provably-orphan
    # encode_worker serve processes now; the machine-wide singleton gate in
    # encode_worker.serve() keeps the count at one from here on.
    try:
        from siglip2_encoder import _sweep_orphan_workers
        _n_swept = _sweep_orphan_workers()
        if _n_swept:
            print(f"[server] startup: swept {_n_swept} orphan encode_worker(s)")
    except Exception:
        pass  # hygiene, not correctness

    # Startup hygiene (2026-09-16): a dead server's warm worker can survive the
    # orphan sweep (pid markers lag behind the real interpreter) while still
    # holding encoder_warm.lock AND a loaded model — measured at multiple GB of
    # RAM that starves the next creative run's floor gate into an indefinite
    # pause. Force-release the lock at boot: the holder is killed only if it is
    # a python process, and a healthy server would have adopted/never started.
    try:
        from siglip2_encoder import _warm_force_release
        _warm_force_release()
    except Exception:
        pass  # hygiene, not correctness

    # Startup hygiene (2026-09-16): orphan VLM sidecar. The Qwen-VL GGUF now
    # lives in a disposable sidecar process (critique_engine's client). If the
    # owning server died without sweeping it, the sidecar still self-exits on
    # its own idle timer — but that can be minutes of GBs held for nothing.
    # Kill a provably orphaned sidecar now (its parent PID is checked, so a
    # live peer instance's sidecar is never touched).
    try:
        from src import critique_engine as _ce_sweep
        if _ce_sweep.sweep_orphan_sidecar():
            pass
    except Exception:
        pass  # hygiene, not correctness

    # Boot banner (2026-09-07): the incident ran a FULL-tier session on a
    # 16 GB machine for hours because a restart outside launch_hidden.vbs
    # silently lost the SIGLIP_TIER=low / FIRSTCUT_LITE=1 pins. One line,
    # first thing in every session — a tier mismatch is now visible instead
    # of discoverable by archaeology.
    #
    # VRAM is measured via nvidia-smi, NOT torch.cuda — this module's own
    # history (see the model-prefetch note below) proves initialising CUDA in
    # the server process wedges/kills it at C level before the grading path
    # ever runs. Every post-fix boot wedged exactly after a torch.cuda
    # banner query; nvidia-smi measures without touching the driver context.
    try:
        import psutil as _ps_boot_banner
        _ram_free = _ps_boot_banner.virtual_memory().available / (1 << 30)
        _vram_txt = ""
        try:
            import subprocess as _sp_banner
            _out = _sp_banner.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total",
                 "--format=csv,noheader,nounits"],
                creationflags=0x08000000 if os.name == "nt" else 0,
                text=True, timeout=5,
            )
            _free_mi, _total_mi = (float(v) for v in _out.strip().splitlines()[0].split(","))
            _vram_txt = (f", VRAM {_free_mi / 1024:.1f}/{_total_mi / 1024:.1f} GB free")
        except Exception:
            pass
        print(f"[server] BOOT: tier={os.environ.get('SIGLIP_TIER', '(auto)')} "
              f"LITE={os.environ.get('FIRSTCUT_LITE', '0')} "
              f"RAM {_ram_free:.1f} GB free{_vram_txt}", flush=True)
    except Exception:
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
            import lance_store as _ls
            try:
                _ls._open_table()
                print("[server] LanceDB table pre-opened OK")
            except MemoryError:
                # A bare MemoryError here is the MACHINE out of memory (measured
                # 2026-09-07: raised while allocating inside the lance import),
                # NOT a corrupt database. The old code rmtree'd the whole DB on
                # ANY exception — a memory storm destroyed every embedding and
                # the People index. Defer instead: the next successful open
                # finds the database untouched.
                print("[server] LanceDB pre-open deferred — out of memory (database NOT touched)")
            except Exception as _e_open:
                # Genuine suspected corruption: never rmtree. Move aside so a
                # human can inspect it and so the rebuild starts from nothing
                # without destroying the evidence.
                import shutil as _shutil_ldb
                _quarantine = _ls._DB_DIR + ".quarantine"
                try:
                    _shutil_ldb.rmtree(str(_quarantine), ignore_errors=True)
                    os.rename(str(_ls._DB_DIR), str(_quarantine))
                    print(f"[server] LanceDB open failed ({_e_open}) — database moved "
                          f"to {_quarantine} for inspection; a fresh one will be created")
                except Exception as _e_mv:
                    print(f"[server] LanceDB open failed ({_e_open}); quarantine move "
                          f"also failed ({_e_mv}) — leaving the database untouched")
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

# ── Identity marker ──────────────────────────────────────────────────────────
# A leftover server from the Sept-1 packaging experiment (curator-api.exe)
# once squatted on port 8000 and answered health checks — the launcher then
# "reused" it and the UI was served by the wrong program. Every FirstCut
# backend must answer /api/whoami with this marker; the launcher verifies it
# before reusing or trusting whatever is on the port.

@app.get("/api/whoami")
async def whoami():
    return {"app": "FirstCut", "pid": os.getpid()}

# ── MemoryError defense ──────────────────────────────────────────────────────
# A request that dies with MemoryError used to surface as "UNHANDLED …" and the
# UI as an unusable page — during a memory storm the app looked broken beyond
# repair. A clean 503 with the remedy keeps the failure honest AND recoverable
# (measured: the 2026-09-07 memory storm turned every GET / into a crash).
from fastapi.responses import JSONResponse as _JSONResp

@app.exception_handler(MemoryError)
async def _memory_error_handler(request, exc: MemoryError):
    print(f"[server] MemoryError serving {request.url.path} — returning clean 503", flush=True)
    return _JSONResp(
        status_code=503,
        content={"error": "Out of memory. Close some apps (browser tabs are the usual cause) and try again — if this persists, restart FirstCut."},
    )

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
]
if not getattr(sys, "frozen", False):
    # L2: the Vite dev server is trusted only in source builds. A frozen
    # (PyInstaller) install has no dev server, so allow-listing :5173 there
    # is attack surface without a purpose.
    _CORS_ORIGINS += ["http://127.0.0.1:5173", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
# GZip: the slim catalog alone is ~33 MB of JSON for a 60k library; the
# WebView fetches it on every cold start. Compressing the wire payload
# (~10× for JSON) turns a multi-second fetch into a fraction of one.
# minimum_size keeps tiny responses (health, thumbs) uncompressed.
#
# SSE exception: stock Starlette 0.36 GZipMiddleware never flushes the zlib
# buffer in streaming mode, so tiny text/event-stream chunks accumulate and
# are never sent — the client hangs forever and the request eventually dies
# with RuntimeError('No response returned.'). The responder below flags SSE
# responses as already-encoded so every body message flows through the
# passthrough branch untouched.
from starlette.datastructures import Headers as _GzipHeaders
from starlette.middleware.gzip import (
    GZipMiddleware as _BaseGZipMiddleware,
    GZipResponder as _BaseGZipResponder,
)


class _SSESafeResponder(_BaseGZipResponder):
    async def send_with_gzip(self, message):
        if message["type"] == "http.response.start":
            # Base class's own "start" handling (below) unconditionally sets
            # content_encoding_set = "content-encoding" in headers — which is
            # always False here (no such header exists yet) and would wipe
            # out an override made BEFORE this call. Let it run first, then
            # override the flag it just computed.
            await super().send_with_gzip(message)
            headers = _GzipHeaders(raw=message["headers"])
            if headers.get("content-type", "").startswith("text/event-stream"):
                # Mark as pre-encoded → every body chunk is forwarded as-is.
                self.content_encoding_set = True
            return
        await super().send_with_gzip(message)


class SSESafeGZipMiddleware(_BaseGZipMiddleware):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = _GzipHeaders(scope=scope)
            if "gzip" in headers.get("Accept-Encoding", ""):
                responder = _SSESafeResponder(
                    self.app, self.minimum_size, compresslevel=self.compresslevel
                )
                await responder(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(SSESafeGZipMiddleware, minimum_size=2048)

# ── I1: unhandled exceptions never reach the client as raw text ──────────────
# A handler crash used to bubble up as an opaque 500 with no trail (or, where
# handlers pre-format str(e), as raw exception text that can carry local file
# paths). Everything is now logged server-side with a crash id; the client
# gets a generic message plus the id to quote in a bug report.
@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    import uuid as _uuid, traceback as _tb
    _eid = _uuid.uuid4().hex[:12]
    print(f"[server] UNHANDLED {_eid} {request.method} {request.url.path}: {exc!r}", flush=True)
    _tb.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error — see crash.log", "id": _eid},
    )


# ── Security: isolate the local API from other origins ───────────────────────
# This server binds 127.0.0.1, but every website the user visits can still reach
# it. Without this guard a malicious page could read local photos (/api/photo?
# path=…), enumerate folders (/api/browse-folder) and CSRF the state-changing
# endpoints. Defense = Fetch-Metadata Resource Isolation + Host pinning:
#   1. Host header MUST be localhost, and MUST be present — an empty Host used
#      to skip the check entirely (fail-open). Blocks DNS-rebinding (a hostile
#      domain re-resolving to 127.0.0.1 to look same-origin).
#   2. Reject /api/* when Sec-Fetch-Site is cross-site/cross-origin. WebView2 is
#      Chromium and always sends Sec-Fetch-Site; it is a forbidden header, so
#      page JS cannot forge it. Same-origin (the app) and same-site (Vite dev,
#      different port) are allowed.
#   3. State-changing methods (POST/PUT/DELETE/PATCH) must additionally prove
#      browser-legitimate context: a well-formed Sec-Fetch-Site, OR the custom
#      X-Requested-With: FirstCut header the frontend sends. This closes the
#      stripped-header hole: a privacy extension or proxy that removes
#      Sec-Fetch-* no longer restores CSRF reach, because simple cross-origin
#      requests (form/img) cannot set custom headers — and setting one via
#      fetch() forces a CORS preflight this server rejects. Non-browser
#      clients (scripts, curl) pass by sending X-Requested-With: FirstCut.
#   4. A top-level navigation sends sec-fetch-mode: navigate. No legitimate app
#      request navigates to an API route (all data calls are same-origin
#      fetch/XHR → mode "cors"), but a phishing link like
#      <a href="http://127.0.0.1:8000/api/photo?path=…"> does. Reject it.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
_API_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_API_TOKEN = "FirstCut"

@app.middleware("http")
async def _security_isolation(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in _ALLOWED_HOSTS:
            return JSONResponse(status_code=403, content={"error": "Forbidden host"})
        sec_fetch_site = (request.headers.get("sec-fetch-site") or "")
        if sec_fetch_site in ("cross-site", "cross-origin"):
            return JSONResponse(
                status_code=403,
                content={"error": "Cross-origin access to the local API is blocked."},
            )
        # No legitimate app request navigates to an API route (all data calls
        # are same-origin fetch/XHR → mode "cors"), but a phishing link like
        # <a href="http://127.0.0.1:8000/api/photo?path=…"> does. Reject it.
        if (request.headers.get("sec-fetch-mode") or "") == "navigate":
            return JSONResponse(status_code=404, content={"error": "Not found"})
        # M1: fail closed on state-changing methods when the browser context
        # cannot be established — see note 3 above.
        if request.method not in _API_SAFE_METHODS:
            if sec_fetch_site in ("same-origin", "same-site", "none"):
                pass
            elif request.headers.get("x-requested-with") == _API_TOKEN:
                pass
            else:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Missing browser-context headers. "
                                      "Non-browser clients must send "
                                      "X-Requested-With: FirstCut."},
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
_RAW_EXTS = frozenset({".arw", ".cr2", ".cr3", ".nef", ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw"})

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
    # API paths must never fall through to the SPA shell: a wrong-method or
    # typo'd API GET used to "succeed" with 200 + index.html, masking client
    # bugs (GET /api/catalog/clear looked like it worked). A real 404 instead.
    if full_path == "api" or full_path.startswith("api/"):
        raise HTTPException(404, "Not found")
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
