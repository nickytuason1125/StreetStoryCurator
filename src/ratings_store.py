"""Durable user star ratings — the source of truth for the user's taste
baseline. Lives in cache/user_ratings.json, keyed by image path.

Why a separate store: the gallery/catalog is rebuilt on every cull (fresh
photos default to stars=0), so ratings kept only in the catalog get wiped on a
re-grade. This store is never rewritten by a cull — the pipeline READS it to
re-apply stars onto fresh gallery items, and the star endpoint WRITES each new
rating here. The PersonalHead can always be re-trained from it, so the baseline
is recoverable even if the model weights reset.
"""
import json
import os
import threading
import time
from pathlib import Path

_PATH   = Path(__file__).resolve().parent.parent / "cache" / "user_ratings.json"
_BACKUP = Path(__file__).resolve().parent.parent / "cache" / "user_ratings.backup.json"
_lock   = threading.Lock()


def _read_file(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(d, dict) and "ratings" in d and isinstance(d["ratings"], dict):
        return d["ratings"]
    return d if isinstance(d, dict) else {}


def _read_raw() -> dict:
    """Read the durable store, self-healing from the backup if the primary is
    missing/corrupt/empty. Always returns the LARGER of the two so a truncated
    write can never silently shrink the baseline."""
    primary = _read_file(_PATH)
    backup  = _read_file(_BACKUP)
    if len(backup) > len(primary):
        # Primary lost ratings (deleted/corrupt/partial) — recover from backup.
        try:
            _atomic_write(_PATH, backup)
            print(f"[ratings_store] recovered {len(backup)} ratings from backup")
        except Exception:
            pass
        return backup
    return primary


def _atomic_write(target: Path, ratings: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({
        "_doc": "Durable user star ratings — PersonalHead baseline; survives re-culls & restarts.",
        "_saved": time.strftime("%Y-%m-%d %H:%M"),
        "ratings": ratings,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def load() -> dict:
    """Return {path: stars(int)} for every rated photo."""
    return {k: int(v) for k, v in _read_raw().items()
            if isinstance(v, (int, float)) and int(v) > 0}


def get(path: str) -> int:
    """Stars for one path, 0 if unrated."""
    return int(_read_raw().get(path, 0) or 0)


def set_rating(path: str, stars: int) -> None:
    """Persist (or clear, when stars==0) one rating to BOTH the primary store and
    the mirror backup, atomically. Two synced copies = a deleted/corrupt primary
    self-recovers on the next read."""
    if not path:
        return
    with _lock:
        cur = _read_raw()
        if stars and int(stars) > 0:
            cur[path] = int(stars)
        else:
            cur.pop(path, None)
        _atomic_write(_PATH, cur)
        try:
            _atomic_write(_BACKUP, cur)   # keep the mirror current
        except Exception:
            pass
