import os
# Prevent any joblib/loky worker process from spawning (flashes a cmd window on Windows).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
try:
    import joblib.parallel as _jp
    _jp.DEFAULT_BACKEND = "threading"
except Exception:
    pass

import uvicorn, signal, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
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
    ".bmp", ".gif", ".heic", ".arw", ".cr2", ".nef", ".orf", ".rw2",
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

# Background pre-computation
# Keyed by folder path so stale clusters from a previous grade never bleed through.
GLOBAL_CLUSTER_CACHE: dict = {}          # {"folder": str, "labels": ndarray, "paths": list}
_BG_EXECUTOR    = ThreadPoolExecutor(max_workers=1)
# Two separate executors so on-demand thumbnail requests (serve_thumb) are
# never queued behind background pre-warm jobs.
_THUMB_ONDEMAND = ThreadPoolExecutor(max_workers=8)   # high-priority, browser-facing
_THUMB_PREWARM  = ThreadPoolExecutor(max_workers=2)   # low-priority background warm-up

# Defer heavy cv2/numpy/onnx imports so uvicorn can start accepting connections
# immediately — prevents the Tauri window from timing out on cold launch.
_analyzer_instance = None
_analyzer_lock = threading.Lock()

def get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        with _analyzer_lock:
            if _analyzer_instance is None:
                from lightweight_analyzer import LightweightStreetScorer
                _analyzer_instance = LightweightStreetScorer()
    return _analyzer_instance

def _get_editorial_fns():
    from editorial_renderer import generate_magazine_carousel, render_editorial_carousel
    return generate_magazine_carousel, render_editorial_carousel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # No eager warmup — _LazyAnalyzer handles init on first use.
    # Eager cv2/numpy import in a background thread was the startup bottleneck.
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

def _gen_preview(path: str) -> Path | None:
    """Return a JPEG preview for RAW files; return None for browser-renderable formats."""
    import hashlib
    src = Path(path).resolve()
    if src.suffix.lower() not in _RAW_EXTS:
        return None  # browser can render JPEG/PNG/WebP directly
    safe = hashlib.md5(str(src).encode()).hexdigest()[:10] + ".jpg"
    dest = _PREVIEW_DIR / safe
    if dest.exists():
        return dest
    try:
        from PIL import Image as _PILImg
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
        img.convert("RGB").save(str(dest), "JPEG", quality=90)
        return dest
    except Exception:
        return None

@app.get("/api/photo")
async def serve_photo(path: str = Query(...)):
    p = _safe_image_path(path)
    if p.suffix.lower() in _RAW_EXTS:
        import asyncio
        preview = await asyncio.get_event_loop().run_in_executor(None, _gen_preview, str(p))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    return FileResponse(str(p))


@app.post("/api/browse-folder")
async def browse_folder(body: dict):
    """Browse a folder — immediate, non-recursive scan of the current directory only."""
    folder = _safe_dir_path(body.get("folder_path", ""))

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif",
                  ".heic", ".arw", ".cr2", ".nef", ".orf", ".rw2"}
    folders, images = [], []
    try:
        for p in folder.iterdir():
            try:
                if p.is_dir():
                    folders.append(str(p))
                elif p.is_file() and p.suffix.lower() in image_exts:
                    images.append(str(p))
            except PermissionError:
                pass
    except PermissionError:
        pass

    folders.sort()
    images.sort()
    return {"folders": folders, "images": images, "files": []}


def _read_exif(path: str) -> dict:
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}
            def _frac(v):
                try:
                    from fractions import Fraction
                    f = Fraction(v).limit_denominator(10000)
                    if f.denominator == 1:
                        return str(f.numerator)
                    return f"{f.numerator}/{f.denominator}"
                except Exception:
                    return str(v)
            make  = raw.get(271, "")
            model = raw.get(272, "")
            camera = f"{make} {model}".strip() or None
            focal_raw = raw.get(37386)
            focal = f"{round(float(focal_raw))}mm" if focal_raw else None
            aperture_raw = raw.get(33437)
            aperture = f"f/{float(aperture_raw):.1g}" if aperture_raw else None
            shutter_raw = raw.get(33434)
            shutter = _frac(shutter_raw) + "s" if shutter_raw else None
            iso = raw.get(34855)
            lens = raw.get(42036) or raw.get(42037) or None
            dt_str = raw.get(36867) or raw.get(306)
            date = time = None
            if dt_str:
                parts = str(dt_str).split(" ")
                date = parts[0].replace(":", "-") if parts else None
                time = parts[1][:5] if len(parts) > 1 else None
            return {k: v for k, v in {
                "camera": camera, "lens": str(lens) if lens else None,
                "focal": focal, "aperture": aperture,
                "shutter": shutter, "iso": str(iso) if iso else None,
                "date": date, "time": time,
            }.items() if v is not None}
    except Exception:
        return {}


_RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw"}

def _gen_one_thumb(path: str) -> None:
    """Generate a single thumbnail into the cache directory (thread-safe)."""
    try:
        from PIL import Image as _PILImg
        import hashlib as _hl
        src = Path(path).resolve()                        # always work with real path
        if not src.exists() or src.suffix.lower() not in _IMAGE_EXTS:
            return
        safe = _hl.md5(str(src).encode()).hexdigest()[:10] + ".webp"
        dest = THUMB_DIR / safe
        if dest.exists():
            return
        if src.suffix.lower() in _RAW_EXTS:
            import rawpy, numpy as np
            with rawpy.imread(str(src)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        import io
                        img = _PILImg.open(io.BytesIO(thumb.data))
                    else:
                        img = _PILImg.fromarray(thumb.data)
                except rawpy.LibRawNoThumbnailError:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=False)
                    img = _PILImg.fromarray(rgb)
            img.thumbnail((600, 600), _PILImg.Resampling.LANCZOS)
            img.convert("RGB").save(str(dest), "WEBP", quality=75, optimize=True)
        else:
            with _PILImg.open(src) as img:
                img.thumbnail((600, 600), _PILImg.Resampling.LANCZOS)
                img.convert("RGB").save(str(dest), "WEBP", quality=75, optimize=True)
    except Exception:
        pass


@app.post("/api/list-folder")
async def list_folder(body: dict):
    """Return image paths instantly — no EXIF, no blocking I/O on the hot path."""
    import asyncio
    folder = _safe_dir_path(body.get("folder_path", ""))

    exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif",
            ".heic", ".arw", ".cr2", ".nef", ".orf", ".rw2"}

    def _scan():
        return sorted(
            str(p) for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

    loop = asyncio.get_event_loop()
    paths = await loop.run_in_executor(None, _scan)

    # Pre-warm thumbnails in the background using the low-priority executor so
    # on-demand requests via /api/thumb are never blocked by this batch.
    for p in paths[:300]:
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
    return FileResponse(str(dest))


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------

class GradeRequest(BaseModel):
    folder_path: str
    preset: str = "Classic Street"
    deep_review: bool = False

    @field_validator("folder_path")
    @classmethod
    def validate_folder_path(cls, v: str) -> str:
        # Resolve symlinks and normalise — user photos may live anywhere on disk
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


@app.post("/api/grade/stream")
async def grade_photos_stream(req: GradeRequest):
    """Streams grading progress as SSE, then emits the full result as the final event."""
    import asyncio, json
    from fastapi.responses import StreamingResponse

    global GLOBAL_CLUSTER_CACHE

    if not os.path.isdir(req.folder_path):
        raise HTTPException(400, "Invalid folder path")

    loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
    aqueue: asyncio.Queue = asyncio.Queue()

    def _progress(frac: float, desc: str = "") -> None:
        loop.call_soon_threadsafe(
            aqueue.put_nowait, {"progress": round(frac, 3), "desc": desc}
        )

    async def _run() -> None:
        global GLOBAL_CLUSTER_CACHE
        try:
            results = await loop.run_in_executor(
                None,
                lambda: analyzer.analyze_folder(
                    req.folder_path, preset=req.preset,
                    force_rescan=True, progress=_progress,
                ),
            )
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
            _BG_EXECUTOR.submit(_precompute_clusters, req.folder_path, results)
            if req.deep_review:
                _BG_EXECUTOR.submit(_run_vlm_deep_review, results)
            await aqueue.put({
                "done": True, "total": len(gallery),
                "strong": strong, "mid": mid, "weak": weak, "data": gallery,
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
