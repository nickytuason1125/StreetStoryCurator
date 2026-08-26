"""Library routes — moved verbatim from server_impl.py (Milestone 4 split).

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
    Path, THUMB_DIR, _HEIC_EXTS, _IMAGE_EXTS, _THUMB_ONDEMAND, _THUMB_PREWARM, _gen_preview, _grading_active, _safe_dir_path, _safe_image_path, asyncio, get_analyzer, os, threading,
)

router = APIRouter()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.get("/api/thumb")
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


@router.get("/api/photo")
async def serve_photo(path: str = Query(...)):
    p = _safe_image_path(path)
    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        import asyncio
        preview = await asyncio.get_running_loop().run_in_executor(None, _gen_preview, str(p))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    return FileResponse(str(p))


@router.post("/api/browse-folder")
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


@router.post("/api/list-folder")
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


@router.get("/api/exif")
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


