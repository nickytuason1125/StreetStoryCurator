"""Leaf cluster: cache clearing, pacing presets, saved sequences,
session catalog and photo flags. Moved verbatim from server.py
(Milestone 4 split) — decorators retargeted app -> router, shared
state imported lazily from server_impl inside each handler."""
import gzip
import json
import os
import threading
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

router = APIRouter()

# Serialises every read-modify-write on the small JSON state files below
# (photo_flags.json, saved_sequences.json). Without it, two concurrent
# toggles both load → mutate → write and the last writer silently drops
# the other's change.
_STATE_LOCK = threading.Lock()


def _impl():
    from server_impl import _DATA_DIR, _atomic_write_text, analyzer
    return _DATA_DIR, _atomic_write_text, analyzer

# Resolved at import time: mount_all() runs mid-server_impl (line ~3964),
# after _DATA_DIR / _atomic_write_text / analyzer are all defined.
_DATA_DIR, _atomic_write_text, analyzer = _impl()


@router.post("/api/clear_cache")
def clear_cache():
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    if os.path.exists(str(_DATA_DIR / "cache" / "light_scores.json")):
        os.remove(str(_DATA_DIR / "cache" / "light_scores.json"))
        analyzer.cache.clear()
        return {"status": "cleared"}
    analyzer.cache.clear()
    return {"status": "empty"}


# ---------------------------------------------------------------------------
# Pacing presets
# ---------------------------------------------------------------------------

@router.get("/api/presets")
def get_presets():
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    from sequence_engine import PacingManager
    return PacingManager().presets

@router.post("/api/presets/save")
def save_preset(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    from sequence_engine import PacingManager
    pm = PacingManager()
    pm.save_custom_weights(
        payload.get("name", "Custom"),
        payload.get("weights", {}),
    )
    return {"status": "saved"}

@router.get("/api/saved-sequences")
async def get_saved_sequences():
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Return list of saved sequences."""
    sequences_file = _DATA_DIR / "cache" / "saved_sequences.json"
    if not sequences_file.exists():
        return {"sequences": []}
    try:
        with open(sequences_file, "r") as f:
            return json.load(f)
    except Exception:
        return {"sequences": []}


@router.post("/api/save-sequence")
async def save_sequence(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Save a sequence to disk."""
    name = payload.get("name")
    sequence = payload.get("sequence", [])
    if not name or not sequence:
        raise HTTPException(400, "Name and sequence required")

    sequences_file = _DATA_DIR / "cache" / "saved_sequences.json"
    sequences_file.parent.mkdir(exist_ok=True)

    with _STATE_LOCK:
        try:
            with open(sequences_file, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"sequences": []}

        # Remove existing sequence with same name
        data["sequences"] = [s for s in data["sequences"] if s["name"] != name]
        data["sequences"].append({"name": name, "sequence": sequence})

        _atomic_write_text(sequences_file, json.dumps(data, indent=2))

    return {"success": True, "message": f"Sequence '{name}' saved"}


from server_impl import _CATALOG_PATH  # one definition, shared

_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}

# ── Catalog read cache ──────────────────────────────────────────────────────
# catalog.json is ~55 MB for a 60k library. Before this cache, EVERY
# /api/catalog, /api/photo-detail and /api/catalog/save re-read and
# re-parsed the whole file (~1-2 s each — measured by scripts/bench_api.py).
# The cache is keyed on (path, mtime, size): any external write (re-grade,
# manual edit) invalidates it automatically; saves prime it with the doc
# they just wrote.
_CATALOG_CACHE: dict = {"key": None, "data": None}
_CATALOG_CACHE_LOCK = threading.Lock()


def _load_catalog_cached():
    """Return the parsed catalog dict, re-reading only when the file changed."""
    if not _CATALOG_PATH.exists():
        return None
    st = _CATALOG_PATH.stat()
    key = (_CATALOG_PATH.name, st.st_mtime, st.st_size)
    with _CATALOG_CACHE_LOCK:
        if _CATALOG_CACHE["key"] != key or _CATALOG_CACHE["data"] is None:
            _CATALOG_CACHE["data"] = json.loads(
                _CATALOG_PATH.read_text(encoding="utf-8"))
            _CATALOG_CACHE["key"] = key
        return _CATALOG_CACHE["data"]


# ── Serialized slim-response cache ──────────────────────────────────────────
# Building the slim payload (64k row copies + JSON dump) costs ~2 s of CPU
# per call even with the parse cached, and the output only changes when
# catalog.json changes. So cache the SERIALIZED body and its gzip: an
# unchanged catalog serves pre-compressed bytes straight from memory.
_SLIM_CACHE: dict = {"key": None, "gz": b""}


@router.get("/api/catalog")
async def get_catalog(full: bool = False):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    # Fallback: a failed re-grade moves the live catalog to .pre-regrade.bak.
    # Serving the backup here keeps Resume working after a failed re-grade
    # instead of reporting an empty history.
    source = _CATALOG_PATH
    fallback = False
    if not _CATALOG_PATH.exists():
        bak = _CATALOG_PATH.with_name("catalog.json.pre-regrade.bak")
        if bak.exists():
            source = bak
            fallback = True
        else:
            return JSONResponse({"exists": False}, headers=_NO_CACHE_HEADERS)
    try:
        if not fallback and not full:
            st = _CATALOG_PATH.stat()
            key = (_CATALOG_PATH.name, st.st_mtime, st.st_size)
            with _CATALOG_CACHE_LOCK:
                if _SLIM_CACHE["key"] == key and _SLIM_CACHE["gz"]:
                    return Response(content=_SLIM_CACHE["gz"],
                                    media_type="application/json",
                                    headers={**_NO_CACHE_HEADERS,
                                             "Content-Encoding": "gzip"})
            data = _load_catalog_cached() or {"photos": []}
            # Slim payload (default): breakdown + reasoning_log are per-photo
            # heavyweights (aspects, bboxes, logs) the gallery never renders.
            # Including them made the payload — and the WebView's JSON parse —
            # scale with library size. They are fetched per-photo via
            # /api/photo-detail when a photo is selected, and survive saves via
            # the merge in /api/catalog/save. ?full=1 keeps the old behaviour
            # for consumers that genuinely need every field (XMP export).
            #
            # 'face' gets the same treatment (4.5 MB across 64k rows): the
            # gallery only needs the 4 filter scalars — per-face details
            # (boxes, crops, confidences) are fetched live via /api/photo-faces
            # when a photo is selected, and /api/catalog/save keeps the stored
            # details when the slim face comes back (see the save merge).
            _FACE_WIRE_KEYS = ("faces_detected", "subject_in_focus",
                               "focus_ratio", "largest_face_frac")

            def _slim_photo(p: dict) -> dict:
                q = {k: v for k, v in p.items()
                     if k not in ("breakdown", "reasoning_log", "face")}
                f = p.get("face")
                if isinstance(f, dict):
                    q["face"] = {k: f[k] for k in _FACE_WIRE_KEYS if k in f}
                return q

            slim = {
                **data,
                "photos": [_slim_photo(p) for p in data.get("photos", [])],
            }
            body = json.dumps({"exists": True, "fallback": False, **slim},
                              ensure_ascii=False).encode("utf-8")
            gz = gzip.compress(body, 5)
            with _CATALOG_CACHE_LOCK:
                _SLIM_CACHE["key"] = key
                _SLIM_CACHE["gz"] = gz
            return Response(content=gz, media_type="application/json",
                            headers={**_NO_CACHE_HEADERS,
                                     "Content-Encoding": "gzip"})
        if fallback:
            data = json.loads(source.read_text(encoding="utf-8"))
        else:
            # Cached parse (mtime/size key) — was a full 55 MB re-read per call.
            data = _load_catalog_cached() or {"photos": []}
        return JSONResponse({"exists": True, "fallback": fallback, **data},
                            headers=_NO_CACHE_HEADERS)
    except Exception:
        return JSONResponse({"exists": False}, headers=_NO_CACHE_HEADERS)


@router.get("/api/photo-detail")
async def get_photo_detail(path: str = ""):
    """One photo's full catalog entry (breakdown, reasoning_log) — the lazy
    counterpart of the slimmed /api/catalog. Serves the same .pre-regrade.bak
    fallback so a selected photo still resolves after a failed re-grade."""
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    source = _CATALOG_PATH
    if not _CATALOG_PATH.exists():
        bak = _CATALOG_PATH.with_name("catalog.json.pre-regrade.bak")
        if bak.exists():
            source = bak
        else:
            return JSONResponse({"exists": False, "photo": None})
    try:
        # Cached parse — was a full 55 MB re-read per selection (~1 s).
        data = _load_catalog_cached() or {}
        entry = next((p for p in data.get("photos", [])
                      if p.get("path") == path), None)
        return JSONResponse({"exists": entry is not None, "photo": entry},
                            headers=_NO_CACHE_HEADERS)
    except Exception:
        return JSONResponse({"exists": False, "photo": None})


@router.post("/api/catalog/save")
async def save_catalog(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    photos  = payload.get("photos", [])
    folders = payload.get("folders", [])
    # Merge against the stored catalog: the frontend works on a slimmed
    # payload (breakdown/reasoning_log excluded for render speed), so fields
    # absent from the incoming rows must survive from the stored entry —
    # otherwise every save would permanently erase them. Incoming fields
    # always win; only ABSENT keys fall back to the stored value.
    existing: dict = {}
    try:
        # Cached parse — the merge no longer re-reads 55 MB per save.
        old = _load_catalog_cached() or {}
        existing = {p.get("path"): p for p in old.get("photos", [])
                    if p.get("path")}
    except Exception:
        existing = {}
    # Face-slim guard: the wire payload carries only the 4 filter scalars of
    # 'face' (full per-face details live in /api/photo-faces). When a slim face
    # comes back from the frontend, fold it INTO the stored full face instead
    # of letting it overwrite the details — scalars update, boxes/crops stay.
    for p in photos:
        prev = existing.get(p.get("path"))
        if (isinstance(p.get("face"), dict) and "faces" not in p["face"]
                and isinstance(prev, dict) and isinstance(prev.get("face"), dict)
                and "faces" in prev["face"]):
            f = dict(prev["face"])
            f.update(p["face"])
            p["face"] = f
    merged = []
    for p in photos:
        prev = existing.get(p.get("path"))
        merged.append({**prev, **p} if prev else p)
    _atomic_write_text(
        _CATALOG_PATH,
        json.dumps({
            "photos":    merged,
            "folders":   folders,
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2),
    )
    # Prime the read cache with the doc just written — the next catalog or
    # photo-detail call must not re-parse the file we only just serialized.
    st = _CATALOG_PATH.stat()
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE["key"] = (_CATALOG_PATH.name, st.st_mtime, st.st_size)
        _CATALOG_CACHE["data"] = {"photos": merged, "folders": folders}
    return {"ok": True}

@router.post("/api/catalog/clear")
async def clear_catalog():
    """Clear the catalog — but keep one step of undo.

    This used to unlink() outright, so a misclick was unrecoverable: every
    grade in the library gone, with nothing on disk to fall back to. It is a
    deliberate action rather than a silent one (unlike the /api/scan bug), so
    it does not need a confirmation dialogue here — it needs the same recovery
    copy the rebuild paths already write, which /api/catalog then serves.
    """
    # M3 gate: a clear racing an in-flight grade's merge_write is
    # last-writer-wins — either the clear silently undoes itself, or a
    # just-finished grade's results vanish into the backup with no error
    # surfaced. Refuse while a cull is running, consistent with the
    # grade-start single-flight guard (the mirror race — a grade starting
    # during this sub-millisecond clear — remains theoretically open and is
    # accepted; the grade handler is the long operation worth gating).
    from server_impl import _grading_active
    if _grading_active.is_set():
        raise HTTPException(
            409,
            "A grade is running — wait for it to finish before clearing the catalog.",
        )
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    try:
        import catalog_store
        catalog_store.back_up("catalog/clear", path=_CATALOG_PATH)
    except Exception as _e:
        # Never let the safety net stop the action the user asked for.
        print(f"[catalog/clear] backup skipped: {_e}")
        if _CATALOG_PATH.exists():
            _CATALOG_PATH.unlink()
    return {"ok": True}


def _load_flags_file() -> dict:
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    f = _DATA_DIR / "cache" / "photo_flags.json"
    try:
        if f.exists():
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _write_flags_atomic(data: dict) -> None:
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Atomically replace photo_flags.json (temp file + os.replace).

    The old in-place open(...,'w') truncated the file before writing, so a
    crash or power loss mid-write destroyed every lock/used flag at once.
    """
    f = _DATA_DIR / "cache" / "photo_flags.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(str(tmp), str(f))


def _toggle_flag_key(key: str, path: str) -> dict:
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    with _STATE_LOCK:
        data = _load_flags_file()
        items = data.setdefault(key, [])
        present = path in items
        if present:
            items.remove(path)
        else:
            items.append(path)
        try:
            _write_flags_atomic(data)
        except Exception as e:
            return {"success": False, "message": str(e)}
    return {"success": True, key: not present}


@router.post("/api/flags/lock")
async def toggle_lock(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Toggle lock flag for a photo."""
    path = payload.get("path", "")
    if not path:
        return {"success": False, "message": "path required"}
    # File IO is offloaded so the event loop never blocks on disk.
    return JSONResponse(await run_in_threadpool(_toggle_flag_key, "locked", path))


@router.post("/api/flags/used")
async def toggle_used(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Toggle used flag for a photo."""
    path = payload.get("path", "")
    if not path:
        return {"success": False, "message": "path required"}
    return JSONResponse(await run_in_threadpool(_toggle_flag_key, "used", path))


@router.get("/api/flags/load")
async def load_flags():
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    """Load all photo flags."""
    flags_file = _DATA_DIR / "cache" / "photo_flags.json"
    try:
        if flags_file.exists():
            with open(flags_file, "r") as f:
                return json.load(f)
        return {"locked": [], "used": []}
    except Exception:
        return {"locked": [], "used": []}
