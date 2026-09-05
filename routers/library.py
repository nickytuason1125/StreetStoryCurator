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

# Thumbnail geometry — ONE definition, used by the generator AND the /api/thumb
# handler. They each built the cache filename independently, so the moment the
# size entered the key they disagreed and no thumbnail could ever be found.
#
# Sized for the LARGEST place a thumbnail is shown, not the smallest. This was
# 200px, commented "grid display only", but a contact-sheet cell is ~290 CSS px
# and each of those is 2 device pixels on a HiDPI screen — so every thumbnail
# was upscaled 1.5x to 3x. That is the "pictures are pixelated" report; the
# photographs were fine, the thumbnails were too small for where they land.
THUMB_PX = 448


def _thumb_cache_name(src) -> str:
    """Cache filename for a source path. The target size is part of a
    thumbnail's identity, so it belongs in the key: without it, raising the
    size left every already-cached image serving its old smaller file for
    ever, and the fix appeared not to work."""
    import hashlib as _h
    return f"{_h.md5(str(src).encode()).hexdigest()[:10]}_{THUMB_PX}.webp"



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
    safe_name = _thumb_cache_name(src)
    thumb_path = THUMB_DIR / safe_name
    if thumb_path.exists():
        # Explicit type: Windows' registry-backed mimetypes can mislabel .webp
        # as text/plain, and strict MIME clients would then refuse the image.
        return FileResponse(str(thumb_path), media_type="image/webp")
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
        return FileResponse(str(thumb_path), media_type="image/webp")
    # RAW/HEIC that produced no thumbnail (e.g. no embedded preview) → render a
    # full preview as a last resort.
    if src.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        preview = await loop.run_in_executor(None, _gen_preview, str(src))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    raise HTTPException(404, "Thumbnail could not be created")


@router.get("/api/photo-faces")
async def photo_faces(path: str = Query(...)):
    """Close-Ups payload for one photo: face verdicts + per-face crops.

    On demand (not grade-time) so the 64k rows graded before face geometry
    was persisted get the same panel as fresh grades — one ~0.4 s CPU pass
    per photo, run off the event loop. Never blocks grading: YuNet is a
    232 KB CPU model, not part of the VRAM/RAM budget.
    """
    p = _safe_image_path(path)

    def _compute():
        import face_signals
        return face_signals.faces_for_ui(str(p))

    data = await run_in_threadpool(_compute)
    return JSONResponse(data)


@router.get("/api/people-search")
async def people_search(path: str = Query(...), idx: int = Query(0)):
    """Find every photo containing a face similar to the clicked one.

    Reads the stored embedding for (path, face_idx) from the faces table and
    kNN-searches that table. MEASURED calibration (2026-09, 87-photo live
    index): same-appearance faces cluster at L2 0.70-0.80, everything else
    spreads 0.80-0.97 with no clean gap — SigLIP-2 is an appearance encoder,
    not a face-identity model, so this finds similar-looking people (strongest
    within a shoot/burst), not biometric identity. Default cutoff 0.80 keeps
    the close-match cluster; matches are sorted best-first so the gallery
    leads with the strongest. A dedicated face-recognition embedding
    (ArcFace-class) is the future upgrade for strict identity.
    {indexed: false} = People index hasn't been built for this photo yet
    (scripts/backfill_face_embeddings.py).
    """
    p = _safe_image_path(path)

    def _search() -> dict:
        import lance_store as ls
        emb = ls.face_embedding_for(str(p), idx)
        if emb is None:
            return {"indexed": False, "matches": []}
        rows = ls.search_faces(emb, top_k=600)
        best: dict = {}
        for r in rows:
            rp = r.get("path", "")
            d = float(r.get("_distance", 99.0))
            if rp not in best or d < best[rp]["distance"]:
                best[rp] = {"path": rp, "distance": round(d, 4)}
        matches = [m for m in best.values() if m["distance"] <= 0.80]
        matches.sort(key=lambda m: m["distance"])
        return {"indexed": True, "matches": matches[:500]}

    data = await run_in_threadpool(_search)
    return JSONResponse(data)


@router.get("/api/photo")
async def serve_photo(path: str = Query(...), max: int = Query(0)):
    """Serve a photograph for loupe display.

    `max` (optional, pixels on the long edge): serve a cached, resized
    preview instead of the original. A 7000px original can be 20-50 MB and
    its decode blocks the UI thread for seconds; a 2048px preview decodes
    in tens of milliseconds and is retina-sharp at loupe size. The cache
    lives beside the thumbnails, keyed like they are.
    """
    p = _safe_image_path(path)
    if max and max > 0:
        import asyncio
        loop = asyncio.get_running_loop()
        prev = await loop.run_in_executor(_THUMB_ONDEMAND, _gen_loupe_preview, str(p), int(max))
        if prev:
            return FileResponse(str(prev), media_type="image/jpeg")
    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        import asyncio
        preview = await asyncio.get_running_loop().run_in_executor(None, _gen_preview, str(p))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    return FileResponse(str(p))


def _gen_loupe_preview(src_str: str, max_px: int):
    """Cached long-edge resize for loupe display. Never raises — returns
    None on any failure so the caller can fall back to the original."""
    try:
        import hashlib
        src = Path(src_str).resolve()
        safe = f"{hashlib.md5(str(src).encode()).hexdigest()[:10]}_L{max_px}.jpg"
        cache = THUMB_DIR / safe
        if cache.exists():
            return cache
        # Source pixels: for RAW/HEIC reuse the decoded full preview; for
        # regular images decode the original directly.
        if src.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
            base = _gen_preview(str(src))
            if not base:
                return None
            src_img_path = Path(base)
        else:
            src_img_path = src
        from PIL import Image as _I
        im = _I.open(src_img_path)
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        im.thumbnail((max_px, max_px), _I.LANCZOS)
        THUMB_DIR.mkdir(parents=True, exist_ok=True)
        im.save(cache, "JPEG", quality=90)
        return cache
    except Exception:
        return None



@router.get("/api/places")
async def list_places():
    """Every root the user can start browsing from: drives + their own folders.

    Two bugs made this necessary. Nothing enumerated drives at all, so a
    library on D:/ or E:/ was unreachable unless you could already type the
    path — browse-folder only lists a directory you have already named. And
    the frontend's quick-access shortcuts were hardcoded to one developer's
    profile (a literal C-drive user path), which is a dead link on every
    other machine.

    Both answers belong on the server, which is the side that can actually see
    the filesystem.
    """
    import string
    from pathlib import Path as _P

    drives = []
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            root = f"{letter}:\\"
            if os.path.exists(root):
                label = f"{letter}:"
                try:
                    import ctypes
                    buf = ctypes.create_unicode_buffer(261)
                    if ctypes.windll.kernel32.GetVolumeInformationW(
                            ctypes.c_wchar_p(root), buf, 261,
                            None, None, None, None, 0):
                        if buf.value:
                            label = f"{buf.value} ({letter}:)"
                except Exception:
                    pass          # a label is a nicety; the drive still lists
                drives.append({"label": label, "path": root})
    else:
        drives.append({"label": "/", "path": "/"})

    home = _P.home()
    places = []
    for name in ("Desktop", "Pictures", "Downloads", "Documents"):
        candidate = home / name
        if candidate.is_dir():
            places.append({"label": name, "path": str(candidate)})

    return {"places": places, "drives": drives, "home": str(home)}


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
        # The TARGET SIZE is part of the identity of a thumbnail, so it belongs
        # in the key. Without it, raising THUMB_SIZE left every already-cached
        # image serving its old, smaller file forever — the library stayed
        # pixelated and only newly-seen photos got the better thumbnail, which
        # is the worst of both: a visible mix, and a fix that appears not to
        # work. Including the size means a change invalidates cleanly and the
        # old files simply fall out of use.
        safe = _thumb_cache_name(src)
        dest = THUMB_DIR / safe
        if dest.exists():
            return

        # Sized for the LARGEST place a thumbnail is shown, not the smallest.
        #
        # This was (200, 200) — "grid display only". But a contact-sheet cell is
        # ~290 CSS px wide at a 1500px window and wider on a bigger one, and on
        # a HiDPI screen each of those is 2 device pixels. A 200px thumbnail was
        # therefore being upscaled 1.5x to 3x on every cell, which is exactly
        # the "all the pictures are pixelated" report — the photographs were
        # never that soft, the thumbnails were too small for where they land.
        #
        # 448 covers a 290px cell at 2x DPR with a little headroom, and it is
        # the same short edge the vision models take, so nothing else in the
        # pipeline wants a different number. WebP keeps the file cost close to
        # the old 200px JPEG despite ~5x the pixels.
        THUMB_SIZE = (THUMB_PX, THUMB_PX)

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


