import os
# Prevent any joblib/loky worker process from spawning (flashes a cmd window on Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
# Suppress the "unauthenticated requests" noise from HuggingFace hub without going full offline
# (HF_HUB_OFFLINE=1 breaks timm/open_clip local cache resolution).
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
try:
    import joblib.parallel as _jp
    _jp.DEFAULT_BACKEND = "threading"
except Exception:
    pass

import uvicorn, signal, sys, time, threading
# Force UTF-8 output so emoji in print() don't crash on cp1252 terminals/threads.
for _s in (sys.stdout, sys.stderr):
    try:
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel, field_validator
from typing import List

sys.path.insert(0, str(Path(__file__).parent / "src"))

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

# Background pre-computation
# Keyed by folder path so stale clusters from a previous grade never bleed through.
GLOBAL_CLUSTER_CACHE: dict = {}          # {"folder": str, "labels": ndarray, "paths": list}
_BG_EXECUTOR    = ThreadPoolExecutor(max_workers=1)
# Two separate executors so on-demand thumbnail requests (serve_thumb) are
# never queued behind background pre-warm jobs.
_THUMB_ONDEMAND = ThreadPoolExecutor(max_workers=8)   # high-priority, browser-facing
_THUMB_PREWARM  = ThreadPoolExecutor(max_workers=2)   # low-priority background warm-up

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
    """Run model auto-download in a daemon thread so server stays responsive."""
    try:
        from model_loader import ensure_all_models_downloaded
        ensure_all_models_downloaded()
    except Exception as exc:
        print(f"⚠️  Background model prefetch error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off model auto-download without blocking server startup.
    _t = threading.Thread(target=_bg_model_prefetch, daemon=True, name="model-prefetch")
    _t.start()
    yield

app = FastAPI(lifespan=lifespan)

class _LazyAnalyzer:
    """Proxy that forwards attribute access to the real analyzer once loaded."""
    def __getattr__(self, name):
        return getattr(get_analyzer(), name)

analyzer = _LazyAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thumbnail cache served as static files
THUMB_DIR = _DATA_DIR / "cache" / "thumbs"
THUMB_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/thumbs", StaticFiles(directory=str(THUMB_DIR)), name="thumbs")


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


@app.get("/api/models/status")
async def model_status():
    """Return current grader mode and model availability for the frontend indicator."""
    from pathlib import Path as _P

    draft_ok  = (_P("models/deepseek/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B") / "model.safetensors").exists()
    # 7B is available only when shard files are present, not just the index
    verify_dir = _P("models/deepseek/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B")
    verify_ok  = any(verify_dir.glob("model-*-of-*.safetensors")) if verify_dir.exists() else False
    judge_ok   = _P("models/deepseek-r1-8b-q5.gguf").exists()
    phi4_ok    = _P("models/phi4-mini-reasoning-q4.gguf").exists()

    try:
        import sys, os
        src_dir = os.path.join(os.path.dirname(__file__), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from grade_pipeline_v2 import _grader_status
        last = dict(_grader_status)
    except Exception:
        last = {"mode": "idle", "verify_used": False, "photos_last": 0, "error": None}

    return JSONResponse({
        "draft_available":  draft_ok,
        "verify_available": verify_ok,
        "judge_available":  judge_ok,
        "phi4_available":   phi4_ok,
        "last_mode":        last["mode"],
        "last_verify_used": last["verify_used"],
        "last_error":       last["error"],
    })


@app.get("/api/models/download-status")
async def model_download_status():
    """Return the current auto-download status for all SpecVLM model weights."""
    from model_loader import get_download_status
    return JSONResponse(get_download_status())


@app.get("/api/thumb")
async def serve_thumb(path: str = Query(...)):
    """Create or return a thumbnail (WEBP) for grid display. Fast path optimized."""
    import hashlib
    try:
        p = _safe_image_path(path)
    except HTTPException as e:
        raise
    src = Path(p).resolve()
    path_hash = hashlib.md5(str(src).encode()).hexdigest()[:10]
    safe_name = f"{src.stem.replace(' ', '_')}_{path_hash}.webp"
    thumb_path = THUMB_DIR / safe_name
    if not thumb_path.exists():
        try:
            from PIL import Image as _PILImg
            THUMB_SIZE = (200, 200)
            with _PILImg.open(str(src)) as img:
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)  # faster
                img.save(str(thumb_path), "WEBP", quality=60, method=3)  # skip optimize
        except Exception:
            # fallback to preview for RAW/HEIC
            if src.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
                import asyncio
                preview = await asyncio.get_event_loop().run_in_executor(None, _gen_preview, str(src))
                if preview:
                    return FileResponse(str(preview), media_type="image/jpeg")
            # otherwise return 404
            raise HTTPException(404, "Thumbnail could not be created")
    return FileResponse(str(thumb_path))


@app.get("/api/photo")
async def serve_photo(path: str = Query(...)):
    p = _safe_image_path(path)
    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        import asyncio
        preview = await asyncio.get_event_loop().run_in_executor(None, _gen_preview, str(p))
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
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}

            # ExifIFD sub-IFD holds most camera/exposure data; GPS IFD for location.
            exif_ifd = raw.get_ifd(0x8769)
            gps_ifd  = raw.get_ifd(0x8825)

            def _frac(v):
                try:
                    from fractions import Fraction
                    f = Fraction(v).limit_denominator(10000)
                    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
                except Exception:
                    return str(v)

            def _s(v):
                if isinstance(v, bytes):
                    return v.decode("utf-8", errors="replace").strip("\x00").strip()
                return str(v).strip() if v else None

            def _get(*tags, src=None):
                d = src if src is not None else raw
                for t in tags:
                    v = d.get(t)
                    if v is not None:
                        return v
                return None

            # Camera body — strip redundant make prefix from model string
            make  = _s(raw.get(271)) or ""
            model = _s(raw.get(272)) or ""
            if make and model.upper().startswith(make.split()[0].upper()):
                camera = model or None
            else:
                camera = f"{make} {model}".strip() or None

            # Lens model (ExifIFD 0xA434)
            lens = _s(_get(0xA434, src=exif_ifd) or _get(42036))

            # Focal length
            fl = _get(0x920A, src=exif_ifd) or _get(37386)
            focal = f"{round(float(fl))}mm" if fl else None

            # 35mm equivalent
            fl35 = _get(0xA405, src=exif_ifd) or _get(41989)
            focal_35mm = f"{int(fl35)}mm" if fl35 else None

            # Aperture (FNumber)
            fn = _get(0x829D, src=exif_ifd) or _get(33437)
            aperture = f"f/{float(fn):.1g}" if fn else None

            # Shutter speed
            ss = _get(0x829A, src=exif_ifd) or _get(33434)
            shutter = (_frac(ss) + "s") if ss else None

            # ISO
            iso = _get(0x8827, src=exif_ifd) or _get(34855)

            # Exposure bias
            ev_raw = _get(0x9204, src=exif_ifd) or _get(37380)
            ev = None
            if ev_raw is not None:
                ev_f = float(ev_raw)
                if ev_f != 0:
                    ev = f"{ev_f:+.1f} EV"

            # Exposure program
            _progs = {1:"Manual", 2:"Program", 3:"Aperture priority",
                      4:"Shutter priority", 5:"Creative", 6:"Action",
                      7:"Portrait", 8:"Landscape"}
            prog = _get(0x8822, src=exif_ifd) or _get(34850)
            program = _progs.get(int(prog)) if prog is not None else None

            # Metering mode
            _meters = {1:"Average", 2:"Center-weighted", 3:"Spot",
                       4:"Multi-spot", 5:"Multi-segment", 6:"Partial"}
            met = _get(0x9207, src=exif_ifd) or _get(37383)
            metering = _meters.get(int(met)) if met is not None else None

            # White balance
            wb = _get(0xA403, src=exif_ifd) or _get(41987)
            white_balance = ("Auto" if int(wb) == 0 else "Manual") if wb is not None else None

            # Flash
            fl_tag = _get(0x9209, src=exif_ifd) or _get(37385)
            flash = ("Fired" if (int(fl_tag) & 0x1) else "No flash") if fl_tag is not None else None

            # Date / time (DateTimeOriginal preferred)
            dt = _get(0x9003, src=exif_ifd) or _get(36867) or _get(306)
            date = time_s = None
            if dt:
                parts = _s(dt).split(" ")
                date   = parts[0].replace(":", "-") if parts else None
                time_s = parts[1][:8] if len(parts) > 1 else None

            # GPS
            gps = None
            if gps_ifd:
                try:
                    def _dms(v):
                        return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
                    lat = gps_ifd.get(2); lon = gps_ifd.get(4)
                    if lat and lon:
                        lat_d = _dms(lat) * (-1 if gps_ifd.get(1) in ("S", b"S") else 1)
                        lon_d = _dms(lon) * (-1 if gps_ifd.get(3) in ("W", b"W") else 1)
                        gps = f"{lat_d:.5f}, {lon_d:.5f}"
                except Exception:
                    pass

            return {k: v for k, v in {
                "camera":        camera,
                "lens":          lens,
                "focal":         focal,
                "focal_35mm":    focal_35mm,
                "aperture":      aperture,
                "shutter":       shutter,
                "iso":           str(iso) if iso else None,
                "ev":            ev,
                "program":       program,
                "metering":      metering,
                "white_balance": white_balance,
                "flash":         flash,
                "date":          date,
                "time":          time_s,
                "gps":           gps,
            }.items() if v is not None}
    except Exception:
        return {}


_RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw"}

def _gen_one_thumb(path: str) -> None:
    """Generate a single thumbnail into the cache directory (thread-safe). Optimized for speed."""
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
        
        if src.suffix.lower() in _RAW_EXTS:
            try:
                import rawpy, io, numpy as np
                with rawpy.imread(str(src)) as raw:
                    try:
                        # Fast path: embedded thumbnail is instant
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            img = _PILImg.open(io.BytesIO(thumb.data))
                        else:
                            img = _PILImg.fromarray(thumb.data)
                    except rawpy.LibRawNoThumbnailError:
                        # Fallback: half-size postprocess is faster than full
                        rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                        img = _PILImg.fromarray(rgb)
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)  # faster than LANCZOS
                img.save(str(dest), "WEBP", quality=60, method=3)  # skip optimize for speed
            except Exception:
                # Couldn't load RAW—skip it
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
                img.save(str(dest), "WEBP", quality=60, method=3)
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
                            img.save(str(dest), "WEBP", quality=60, method=3)
                        return
                except Exception:
                    pass
            # Draft-mode decode: PIL tells libjpeg to decode at 1/2, 1/4 or 1/8 scale
            # (4–8× faster for large JPEGs; no-op for PNG/WebP).
            with _PILImg.open(src) as img:
                img.draft("RGB", THUMB_SIZE)
                img = img.convert("RGB")
                img.thumbnail(THUMB_SIZE, _PILImg.Resampling.BILINEAR)
                img.save(str(dest), "WEBP", quality=60, method=3)
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

    loop = asyncio.get_event_loop()
    paths = await loop.run_in_executor(None, _scan)

    # Pre-warm ALL thumbnails in the background — no cap.
    # The low-priority executor (2 workers) processes them without blocking
    # on-demand requests from the browser.
    for p in paths:
        _THUMB_PREWARM.submit(_gen_one_thumb, p)

    # Return empty EXIF — frontend loads it lazily via /api/exif when needed.
    photos = [{"path": p, "exif": {}} for p in paths]
    return {"paths": paths, "photos": photos, "count": len(paths)}


@app.get("/api/exif")
async def get_exif(path: str = Query(...)):
    """Lazy EXIF loader — called by the frontend when a photo is selected."""
    import asyncio
    p = _safe_image_path(path)
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _read_exif, str(p))
    return data


@app.get("/api/thumb")
async def serve_thumb(path: str = Query(...)):
    import asyncio, hashlib
    src = _safe_image_path(path)                          # resolves symlinks
    safe = hashlib.md5(str(src).encode()).hexdigest()[:10] + ".webp"
    dest = THUMB_DIR / safe
    if not dest.exists():
        await asyncio.get_event_loop().run_in_executor(_THUMB_ONDEMAND, _gen_one_thumb, str(src))
        if not dest.exists():
            return FileResponse(str(src))                 # fallback: already resolved
    # Symlink-escape guard: ensure the cached thumb hasn't been tampered with
    if not str(dest.resolve()).startswith(str(THUMB_DIR.resolve())):
        raise HTTPException(403, "Forbidden")
    return FileResponse(
        str(dest),
        headers={"Cache-Control": "public, max-age=604800, immutable"},
    )


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

class GradeRequest(BaseModel):
    folder_path: str = ""
    folder_paths: list[str] = []   # multi-folder support; takes priority when non-empty
    preset: str = "Classic Street"
    deep_review: bool = False
    force_rescan: bool = False
    scan_mode: bool = False        # Low-Latency Scan: top 20% only get 7B verification

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
        loop = asyncio.get_event_loop()
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


@app.post("/api/grade/v2/stream")
async def grade_photos_v2_stream(req: GradeRequest):
    """
    V2 pipeline: SigLIP → Q-Align → PersonalHead → MOGCO-II.
    Same SSE format as /api/grade/stream for drop-in frontend compatibility.
    Supports multi-folder: grades each folder, then runs MOGCO-II once across all.
    """
    import asyncio as _aio, json as _json
    from fastapi.responses import StreamingResponse

    # Resolve all valid folders — folder_paths (multi) takes priority over folder_path
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders:
        if req.folder_path and os.path.isdir(req.folder_path):
            all_folders = [str(Path(req.folder_path).resolve())]
        else:
            raise HTTPException(400, "No valid folder path provided")

    aqueue: _aio.Queue = _aio.Queue()
    loop   = _aio.get_running_loop()

    def _progress(frac: float, desc: str = "") -> None:
        loop.call_soon_threadsafe(
            aqueue.put_nowait, {"progress": round(frac, 3), "desc": desc}
        )

    def _run() -> None:
        try:
            from grade_pipeline_v2 import run_v2
            n = len(all_folders)
            combined_gallery: list = []

            # Clear stale catalog before grading — Step 8b in grade_pipeline_v2
            # will rebuild it after LanceDB upsert.
            try:
                if _CATALOG_PATH.exists():
                    _CATALOG_PATH.unlink()
                    print("[grade_v2] Cleared stale catalog.json — will be rebuilt after grading")
            except Exception:
                pass

            for i, fp in enumerate(all_folders):
                # Slice the 0→1 progress bar across folders
                p_start = i / n
                p_end   = (i + 1) / n

                def _fp(frac: float, desc: str = "", _s=p_start, _e=p_end) -> None:
                    _progress(_s + frac * (_e - _s), desc)

                if n > 1:
                    _fp(0.0, f"Grading folder {i+1}/{n}: {Path(fp).name}")

                result = run_v2(
                    fp,
                    preset       = req.preset,
                    force_rescan = req.force_rescan,
                    progress     = _fp,
                    mogco_target = 0,   # skip per-folder MOGCO; run once at end
                    scan_mode    = req.scan_mode,
                )
                combined_gallery.extend(result.get("gallery", []))

            # Strip embeddings before sending over SSE
            gallery_slim = [
                {k: v for k, v in photo.items() if k != "embedding"}
                for photo in combined_gallery
            ]

            # NSGA-III across all photos (strict literal constraints)
            _progress(0.97, "Running NSGA-III (strict literal constraints)…")
            mogco_sequence: list = []
            mogco_error_msg: str = ""
            try:
                import numpy as _np
                from nsga3_sequencer import run_nsga3_sequence_with_vlm, SequencerConstraintError
                try:
                    from specvlm_pipeline import _CD_BRIEF as _brief
                except Exception:
                    _brief = ""
                seq_candidates = [
                    {
                        "path":          g["path"],
                        "score":         g.get("score", 0.5),
                        "embedding":     _np.array(
                            combined_gallery[idx].get("embedding", []),
                            dtype=_np.float32,
                        ),
                        "reasoning_log": g.get("reasoning_log", ""),
                        "breakdown":     g.get("breakdown", {}),
                    }
                    for idx, g in enumerate(gallery_slim)
                    if "Strong" in g.get("grade", "") or "Mid" in g.get("grade", "")
                ]
                selected = run_nsga3_sequence_with_vlm(
                    seq_candidates, target=5, brief=_brief
                )
                info_by_path = {g["path"]: g for g in gallery_slim}
                for rank, frame in enumerate(selected):
                    base = dict(info_by_path.get(frame["path"], {"path": frame["path"]}))
                    base.update({
                        "slot":             frame.get("slot", f"Slot {rank+1}"),
                        "slot_role":        frame.get("slot_role", ""),
                        "slot_score":       frame.get("slot_score", 0.0),
                        "mogco_objectives": frame.get("nsga3_objectives", {}),
                        "engine":           "nsga3",
                    })
                    mogco_sequence.append(base)
            except SequencerConstraintError as e:
                mogco_error_msg = str(e)
                print(f"[v2] NSGA-III constraint error: {e}")
            except Exception as e:
                print(f"[v2] NSGA-III multi-folder failed: {e}")

            strong = sum(1 for g in combined_gallery if "Strong" in g.get("grade", ""))
            mid    = sum(1 for g in combined_gallery if "Mid"    in g.get("grade", ""))
            weak   = sum(1 for g in combined_gallery if "Weak"   in g.get("grade", ""))

            # Write combined multi-folder catalog (overrides single-folder writes from
            # Step 8b in grade_pipeline_v2, which only cover one folder at a time).
            if len(all_folders) > 1:
                try:
                    import time as _cat_time
                    _cat_photos  = [{k: v for k, v in g.items() if k != "embedding"} for g in combined_gallery]
                    _cat_folders = list(dict.fromkeys(str(Path(g["path"]).parent) for g in combined_gallery))
                    _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _cat_tmp = _CATALOG_PATH.with_suffix(".json.tmp")
                    _cat_tmp.write_text(
                        json.dumps({
                            "photos":   _cat_photos,
                            "folders":  _cat_folders,
                            "saved_at": _cat_time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    _cat_tmp.replace(_CATALOG_PATH)
                    print(f"[grade_v2] catalog.json → {len(_cat_photos)} photos ({len(all_folders)} folders, atomic write)")
                except Exception as _e_cat_sv:
                    print(f"[grade_v2] catalog.json combined write failed: {_e_cat_sv}")

            loop.call_soon_threadsafe(aqueue.put_nowait, {
                "done":           True,
                "total":          len(combined_gallery),
                "strong":         strong,
                "mid":            mid,
                "weak":           weak,
                "data":           gallery_slim,
                "mogco_sequence": mogco_sequence,
                "mogco_error":    mogco_error_msg,
                "pipeline":       "v2",
            })
        except Exception as exc:
            import traceback as _tb
            _full_tb = _tb.format_exc()
            print(f"[grade_v2_stream] CRASH:\n{_full_tb}", file=sys.stderr, flush=True)
            try:
                _crash_path = _DATA_DIR / "crash.log"
                with open(_crash_path, "a", encoding="utf-8") as _cf:
                    import datetime as _dt
                    _cf.write(f"\n{'='*60}\n{_dt.datetime.now().isoformat()} grade_v2_stream crash:\n{_full_tb}\n")
            except Exception:
                pass
            loop.call_soon_threadsafe(aqueue.put_nowait, {"error": str(exc), "traceback": _full_tb})

    import threading as _th
    _th.Thread(target=_run, daemon=True).start()

    async def _stream():
        while True:
            msg = await aqueue.get()
            yield f"data: {_json.dumps(msg)}\n\n"
            if msg.get("done") or msg.get("error"):
                break

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/api/personal/update")
async def personal_update(payload: dict):
    """
    Update the PersonalHead MLP when the user moves a photo between grade buckets.

    Body: { path1, grade1, path2, grade2 }
    Fetches embeddings from LanceDB and runs a Margin Ranking Loss update step.
    """
    try:
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
        loss = ph.update(emb1, grade1, emb2, grade2)

        # Refresh personal scores for all stored photos
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = ph.score(all_embs)
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


@app.post("/api/update_preference")
async def update_preference(payload: dict):
    """
    Update preference by providing a winner and a loser image path.
    Body: { "winner_path": str, "loser_path": str }
    Runs a MarginRankingLoss update on the PersonalHead and refreshes stored scores.
    """
    try:
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

        loss = ph.update(emb_w, 1, emb_l, 0)

        # Refresh personal scores for all stored photos (LanceDB)
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = ph.score(all_embs)
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

    anchor_path  = (payload.get("anchor_path") or "").strip()
    folder_path  = (payload.get("folder_path") or "").strip()
    style_prompt = (payload.get("style_prompt") or "").strip()
    n_target     = int(payload.get("n_target", 7))
    n_target     = max(5, min(10, n_target))

    if not anchor_path:
        return JSONResponse({"error": "anchor_path is required"}, status_code=400)

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
            if not strong_paths and folder_path:
                from pathlib import Path as _Path
                fp = _Path(folder_path)
                if fp.exists():
                    all_imgs = sorted(
                        str(f) for f in fp.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )[:50]
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
                    Path(anchor_path).parent
                ),
                style_prompt      = style_prompt,
                n_target          = n_target,
                avoid_paths       = avoid_paths,
                progress          = _progress,
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

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
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
    loop = asyncio.get_event_loop()
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
        result = await asyncio.get_event_loop().run_in_executor(
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
            beam_result = await asyncio.get_event_loop().run_in_executor(
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
        result = await asyncio.get_event_loop().run_in_executor(
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
            input_data = await _aio.get_event_loop().run_in_executor(
                None, lambda: _clip_rank_by_brief(input_data, prompt)
            )

        loop = _aio.get_event_loop()
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
    loop = _aio.get_event_loop()
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

    # STREET - MAGNUM: balanced auth + comp + human, all above floor
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

    # LSPF (LONDON STREET): atmosphere + human, urban mood, between street and cinematic
    # Needs both human presence AND decent light — penalise if either is absent
    raw["LSPF (London Street)"] = (
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
        "LSPF (London Street)":
            "Urban atmosphere and human presence in soft, directional light. Between street photography and cinematic mood.",
    }

    # ── Per-niche actionable guidance ──────────────────────────────────────────
    GUIDANCE = {
        "Classic Street": {
            "submit":  ["World Street Photography Awards", "LSPF Annual Open", "Burn Magazine", "Magnum Photos Open Call", "6 Mois"],
            "market":  "Editorial agencies (Magnum, Panos, VII), documentary publishers, photobook imprints, festival circuits (Visa Pour l'Image).",
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
        "LSPF (London Street)": {
            "submit":  ["LSPF Annual Exhibition", "Street Foto San Francisco", "Sony World Photography (Street)", "Street Photo Prize"],
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
        # Subfolder per grade, strip emoji for safe dir name
        safe_grade = grade.replace("✅", "Strong").replace("⚠️", "Mid").replace("❌", "Weak").strip()
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
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, analyzer._ensure_sessions)
    loop = asyncio.get_event_loop()
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

@app.get("/api/catalog")
async def get_catalog():
    if not _CATALOG_PATH.exists():
        return {"exists": False}
    try:
        data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        return {"exists": True, **data}
    except Exception:
        return {"exists": False}

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


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    candidate = DIST / full_path
    if candidate.exists() and candidate.is_file():
        return FileResponse(str(candidate))
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(404, "Frontend not built. Run: cd frontend && npm run build")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
