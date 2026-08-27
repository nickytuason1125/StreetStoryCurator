"""Leaf cluster: cache clearing, pacing presets, saved sequences,
session catalog and photo flags. Moved verbatim from server.py
(Milestone 4 split) — decorators retargeted app -> router, shared
state imported lazily from server_impl inside each handler."""
import json
import os
import threading
import time

from fastapi import APIRouter, HTTPException
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

@router.get("/api/catalog")
async def get_catalog():
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
        data = json.loads(source.read_text(encoding="utf-8"))
        return JSONResponse({"exists": True, "fallback": fallback, **data},
                            headers=_NO_CACHE_HEADERS)
    except Exception:
        return JSONResponse({"exists": False}, headers=_NO_CACHE_HEADERS)

@router.post("/api/catalog/save")
async def save_catalog(payload: dict):
    _DATA_DIR, _atomic_write_text, analyzer = _impl()
    photos  = payload.get("photos", [])
    folders = payload.get("folders", [])
    _atomic_write_text(
        _CATALOG_PATH,
        json.dumps({
            "photos":    photos,
            "folders":   folders,
            "saved_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=2),
    )
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
