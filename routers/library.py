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
    Path, THUMB_DIR, _HEIC_EXTS, _IMAGE_EXTS, _THUMB_ONDEMAND, _gen_preview, _grading_active, _safe_dir_path, _safe_image_path, asyncio, get_analyzer, os, threading,
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
    """Cache filename for a source path, SHARDED into 256 two-hex-character
    subdirectories (`ea/eaa16746c2_448.webp`).

    Two properties are deliberate here:

    - The target size is part of a thumbnail's identity, so it belongs in the
      key: without it, raising the size left every already-cached image
      serving its old smaller file for ever, and the fix appeared not to work.
    - The shard prefix is the fix for the 2026-09-06 preview stall: THUMB_DIR
      had ~400k entries in ONE flat NTFS directory (a 380k-photo folder
      prewarmed everything), and every `exists()` stat in it became slow
      enough to be user-visible. The hash is already the key, so spending its
      first two characters on the directory costs nothing and keeps every
      shard in the low hundreds of files.
    """
    import hashlib as _h
    key = _h.md5(str(src).encode()).hexdigest()[:10]
    return f"{key[:2]}/{key}_{THUMB_PX}.webp"


def _legacy_thumb_name(src) -> str:
    """Pre-sharding flat cache name — looked up (and migrated) on shard miss,
    so thumbnails written before the sharding change heal themselves instead
    of forcing a full regeneration of a large library."""
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

    Rendering runs on the _THUMBS queue (off the asyncio event loop), ahead of
    every background prewarm job, and shares its cache with them.

    No RAM gate and no 204 (removed 2026-09-28): a render decodes an embedded
    preview at reduced scale — a few MB — so there is nothing to protect, and
    the browser holds at most ~6 requests open, so the queue never runs deep.
    The old gate existed because ARW previews were decoded at full size
    (~200 MB each for a 7008 px preview).
    """
    p = _safe_image_path(path)
    src = Path(p).resolve()
    cached = _find_cached_thumb(src)
    if cached:
        # Explicit type: Windows' registry-backed mimetypes can mislabel .webp
        # as text/plain, and strict MIME clients would then refuse the image.
        return FileResponse(str(cached), media_type="image/webp")
    # Pair key first, so a browser asking for X.ARW and X.JPG at the same
    # moment joins ONE render. If they turn out not to be a camera pair
    # (_pair_sibling refused the copy), this file still has no thumbnail —
    # render it under its own key.
    await asyncio.wrap_future(_THUMBS.request(_thumb_job_key(src), str(src)))
    out = _find_cached_thumb(src)
    if out is None:
        out = await asyncio.wrap_future(_THUMBS.request(str(src)))
    if out:
        return FileResponse(str(out), media_type="image/webp")
    # RAW/HEIC that produced no thumbnail (e.g. no embedded preview) → render a
    # full preview as a last resort.
    if src.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        loop = asyncio.get_running_loop()
        preview = await loop.run_in_executor(None, _gen_preview, str(src))
        if preview:
            return FileResponse(str(preview), media_type="image/jpeg")
    raise HTTPException(404, "Thumbnail could not be created")


@router.post("/api/encoder/warm")
async def warm_encoder():
    """Pre-load the vision encoder in the persistent warm worker.

    Called by the launcher right after the window opens: imports (torch/
    transformers/sklearn — the exact thing that OOM'd mid-session on
    2026-09-07) and the model load happen while the machine is at its
    freshest, and every cull afterwards reuses the loaded worker. Safe to
    call any time; no-op when FIRSTCUT_WARM_ENCODER=0.

    RAM pathline v2, Phase 1: the warm path imports torch/transformers and
    loads model weights — hundreds of MB of commit charge. During a grade
    those bytes are exactly the difference between the encode worker's
    per-chunk model load succeeding and dying with MemoryError / "The
    paging file is too small" (crash.log, 2026-09-08 ~02:00 — the 46% wall).
    _grading_active is the machine-wide arbiter, not a thumbnail-only
    detail: defer the warm, and the next boot (or the grade's own release)
    warms instead.
    """
    if _grading_active.is_set():
        return {"warmed": False, "deferred": "grade in progress — warm deferred"}
    try:
        from siglip2_encoder import warm_start
        return {"warmed": bool(warm_start())}
    except Exception as exc:
        return {"warmed": False, "error": str(exc)}


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


@router.get("/api/download")
async def download_original(path: str = Query(...)):
    """The ORIGINAL file as a download. /api/photo serves a display preview
    (a ~265 KB JPEG for a 40 MB ARW), so 'Download' used to hand back the
    preview under the raw file's name (2026-10-04). Same path-safety check as
    /api/photo: existing image files only, no traversal."""
    p = _safe_image_path(path)
    return FileResponse(str(p), filename=p.name, media_type="application/octet-stream")


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
        key = hashlib.md5(str(src).encode()).hexdigest()[:10]
        cache = THUMB_DIR / key[:2] / f"{key}_L{max_px}.jpg"
        if not cache.exists():
            # Flat-era loupe previews: migrate on first miss, same scheme as
            # the grid thumbnails.
            legacy = THUMB_DIR / f"{key}_L{max_px}.jpg"
            if legacy.exists():
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    legacy.replace(cache)
                except Exception:
                    cache = legacy
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
        cache.parent.mkdir(parents=True, exist_ok=True)
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


from src.raw_support import RAW_EXTS as _SHARED_RAW_EXTS   # the ONE raw list — copies drifted (.crw)
_RAW_EXTS = set(_SHARED_RAW_EXTS)

_JPEG_EXTS = {".jpg", ".jpeg"}


def _find_cached_thumb(src: Path):
    """The cached thumbnail for `src`, or None. A flat-era (pre-sharding) file
    is moved into its shard on first sight, so old caches heal themselves."""
    dest = THUMB_DIR / _thumb_cache_name(src)
    if dest.exists():
        return dest
    legacy = THUMB_DIR / _legacy_thumb_name(src)
    if legacy.exists():
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(dest)
            return dest
        except Exception:
            return legacy
    return None


def _open_preview(src: Path):
    """Open the cheapest source that is still sharp at THUMB_PX.

    Never the full image when the file carries a preview (measured cold off an
    external drive, 2026-09-28):
    - RAW: the embedded JPEG — 1-4 MB read of a ~40 MB ARW, ~0.1 s.
    - Camera JPEG: the MPF preview (Sony: 1616x1080, frame 1). The 160x120
      EXIF thumbnail is NOT used — it was fast but visibly soft in a 448 tile.
    - Anything else: the image itself; the caller's draft() makes JPEG decode
      at 1/2-1/8 scale in the DCT domain.
    """
    import io
    from PIL import Image as _PILImg
    suf = src.suffix.lower()
    if suf in _RAW_EXTS:
        import rawpy
        from src.raw_support import RAWPY_LOCK   # 6 queue workers: LibRaw is not thread-safe
        with RAWPY_LOCK, rawpy.imread(str(src)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            return _PILImg.open(io.BytesIO(thumb.data))
        return _PILImg.fromarray(thumb.data)
    if suf in _HEIC_EXTS:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    img = _PILImg.open(src)
    if suf in _JPEG_EXTS and getattr(img, "n_frames", 1) > 1:
        try:
            img.seek(1)
            if max(img.size) < THUMB_PX:
                img.seek(0)
        except Exception:
            img.seek(0)
    return img


def _pair_sibling(src: Path):
    """The other half of a RAW+JPEG pair, or None.

    Only a file written within 2 s of `src` counts: a camera writes both at the
    shutter press, while a JPEG exported next to its RAW later (Lightroom) is a
    different picture and must get its own thumbnail.
    """
    suf = src.suffix.lower()
    if suf in _RAW_EXTS:
        candidates = sorted(_JPEG_EXTS)
    elif suf in _JPEG_EXTS:
        candidates = sorted(_RAW_EXTS)
    else:
        return None
    try:
        mtime = src.stat().st_mtime
    except OSError:
        return None
    for ext in candidates:
        for variant in (ext, ext.upper()):
            sib = src.with_suffix(variant)
            try:
                if abs(sib.stat().st_mtime - mtime) <= 2.0:
                    return sib
            except OSError:
                continue
    return None


def _atomic_write(dest: Path, write) -> None:
    """Write via a per-thread temp file + os.replace, so a half-written WEBP
    is never served and concurrent writers never corrupt one file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}_{threading.get_ident()}.tmp.webp")
    try:
        write(tmp)
        os.replace(tmp, dest)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _render_thumb(path: str):
    """Render one thumbnail into the cache; returns its path, or None.
    Runs on a _THUMBS worker."""
    import shutil
    from PIL import Image as _PILImg
    try:
        src = Path(path).resolve()
        if src.suffix.lower() not in _IMAGE_EXTS or not src.exists():
            return None
        cached = _find_cached_thumb(src)
        if cached:
            return cached
        size = (THUMB_PX, THUMB_PX)
        with _open_preview(src) as img:
            # draft() BEFORE convert: a JPEG source decodes at reduced scale.
            # This is what keeps a worker at a few MB — the ARW previews of
            # newer bodies are full-size 7008x4672 JPEGs, ~200 MB decoded
            # without it (the real reason thumbnails used to need RAM gates).
            if img.format in ("JPEG", "MPO"):
                img.draft("RGB", size)
            out = img.convert("RGB")
        out.thumbnail(size, _PILImg.Resampling.BILINEAR)
        dest = THUMB_DIR / _thumb_cache_name(src)
        _atomic_write(dest, lambda tmp: out.save(str(tmp), "WEBP", quality=60, method=3))
        # One decode per RAW+JPEG pair: the camera's JPEG and the RAW's
        # embedded preview are the same in-camera render.
        sib = _pair_sibling(src)
        if sib is not None:
            sib_dest = THUMB_DIR / _thumb_cache_name(sib.resolve())
            if not sib_dest.exists():
                try:
                    _atomic_write(sib_dest, lambda tmp: shutil.copyfile(dest, tmp))
                except OSError:
                    pass
        return dest
    except Exception as exc:
        print(f"[thumb] could not render {Path(path).name}: {exc}")
        return None


def _skip_background_thumbs() -> bool:
    """Prewarm yields to a running grade and to a machine short on RAM."""
    if _grading_active.is_set():
        return True
    try:
        import psutil as _ps
        return _ps.virtual_memory().available / 1e9 < 1.5
    except Exception:
        return False


# The one thumbnail work queue — see src/thumb_queue.py. Six workers: the
# browser keeps at most ~6 image requests open per host, and cold throughput
# off an external HDD kept rising with concurrency (1: 16.6, 4: 25.9,
# 8: 32.6 img/s measured 2026-09-28), so throttling spinning disks only
# makes them slower.
from src.thumb_queue import ThumbQueue as _ThumbQueue
_THUMBS = _ThumbQueue(_render_thumb, workers=6,
                      skip_background=_skip_background_thumbs, name="thumb")


def _thumb_job_key(src: Path) -> str:
    """Queue key: RAW and JPEG files share one key per stem, so both halves of
    a pair requested together cost one render. Pure string work — no disk
    access, safe on the event loop."""
    if src.suffix.lower() in (_RAW_EXTS | _JPEG_EXTS):
        return str(src.with_suffix("")).lower() + "|pair"
    return str(src)


def prewarm_thumbs(paths) -> None:
    """Queue background thumbnails — always behind every on-screen request."""
    for p in paths:
        try:
            src = Path(p).resolve()
            _THUMBS.request(_thumb_job_key(src), str(src), urgent=False)
        except Exception:
            continue


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

    # Warm-ahead on user intent (2026-09-16): choosing a folder is the signal
    # that a critique / Story session may follow. Warm the model sidecar in
    # the background (never blocks this response; skipped during grades;
    # throttled inside the server helper).
    try:
        from server_impl import _sidecar_warm_on_browse
        _sidecar_warm_on_browse()
    except Exception:
        pass

    # Pre-warm thumbnails in the background — BOUNDED. These are background
    # jobs on the _THUMBS queue, so every on-screen request runs first.
    #
    # The cap is the fix for the 2026-09-06 stall: this loop used to submit
    # EVERY path ("no cap" was the old comment), so opening a 380k-photo
    # folder queued 380k decode jobs on a machine sitting at <1 GB free RAM
    # and the prewarm workers chewed on it for hours. The grid only renders a
    # few screens at a time, so pre-warming the first screens is what makes
    # it feel instant; everything past the cap is generated on demand by
    # /api/thumb when the user actually scrolls there, and cached like any
    # other thumbnail.
    #
    # No special case for card readers / USB drives any more: the old one
    # (2026-09-07) assumed each RAW thumbnail read the whole ~43 MB file. It
    # reads the 1-4 MB embedded preview, and because on-screen requests now
    # always jump the queue, prewarm can no longer slow the visible tiles.
    _prewarm_cap = int(os.environ.get("FIRSTCUT_PREWARM_CAP", "600") or 600)
    prewarm_thumbs(paths[:max(_prewarm_cap, 0)])

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
    # Per-folder file narrowing (2026-09-23): folder_path/folder_paths still name
    # every folder in scope, but a folder present here as a key is graded ONLY on
    # the listed files instead of every image run_v2 finds inside it — the
    # frontend's "check individual photos" picker. A folder absent from this dict
    # (the default: {}) grades in full, exactly as before this field existed.
    explicit_paths: dict[str, list[str]] = {}

    @field_validator("folder_path")
    @classmethod
    def validate_folder_path(cls, v: str) -> str:
        if not v:
            return v
        try:
            p = Path(v).resolve(strict=False)
        except (ValueError, OSError):
            raise ValueError("Invalid path")
        # Reachability (does the folder actually exist right now?) is NOT
        # checked here on purpose. This validator is a synchronous Pydantic
        # classmethod that runs inline while FastAPI parses the request body —
        # a blocking time.sleep() retry loop here (there used to be one, up to
        # ~4s absorbing a USB/SD-card wake-up nap) froze the WHOLE asyncio
        # event loop for that long, stalling every other client's SSE
        # progress, thumbnail requests, and health polling, not just this
        # request. The card-wake-up retry now lives only where it can be
        # async: grade_photos_v2_stream's own _dir_ok (routers/grading.py),
        # which every endpoint that grades a folder (including /api/regrade
        # and /api/scan) routes through.
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


