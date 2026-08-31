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


def _stars_of(v) -> int:
    """A rating entry is either a bare int (legacy) or a dict with 'stars' —
    accept both so old cache/user_ratings.json files keep working."""
    if isinstance(v, dict):
        v = v.get("stars")
    return int(v) if isinstance(v, (int, float)) else 0


def load() -> dict:
    """Return {path: stars(int)} for every rated photo."""
    return {k: _stars_of(v) for k, v in _read_raw().items() if _stars_of(v) > 0}


def get(path: str) -> int:
    """Stars for one path, 0 if unrated."""
    return _stars_of(_read_raw().get(path))


def get_score_snapshot(path: str) -> dict | None:
    """The machine score(s) captured at the moment this photo was rated, or
    None if this rating predates snapshotting (legacy bare-int entry) or the
    path isn't rated. Lets accuracy measurement survive a re-grade/migration
    that wipes or reshuffles the live LanceDB/catalog rows for this photo."""
    v = _read_raw().get(path)
    if not isinstance(v, dict):
        return None
    return {"score": v.get("score"), "personal_score": v.get("personal_score")}


def get_source(path: str) -> str:
    """'tpe_master' for the permanently-authoritative baseline ratings, '' for
    an ordinary rating or an unrated path. See PersonalHead.fit()'s docstring
    for what this controls."""
    v = _read_raw().get(path)
    return v.get("source", "") if isinstance(v, dict) else ""


def set_rating(path: str, stars: int, score: float | None = None,
               personal_score: float | None = None, source: str | None = None) -> None:
    """Persist (or clear, when stars==0) one rating to BOTH the primary store and
    the mirror backup, atomically. Two synced copies = a deleted/corrupt primary
    self-recovers on the next read.

    When the caller has the photo's current machine score in hand (it does,
    right after a LanceDB lookup), pass it through — it's snapshotted onto the
    rating so agreement can still be measured after a later re-grade changes or
    removes the live row for this exact path.

    `source`, when passed, is stamped onto the entry (e.g. "tpe_master"); when
    omitted, any existing source tag on this path is preserved rather than
    wiped — the star endpoint calls this twice per rating (once immediately
    with just stars, once more after the score lookup) and the second call
    must not silently untag a master rating.
    """
    if not path:
        return
    with _lock:
        cur = _read_raw()
        if stars and int(stars) > 0:
            existing = cur.get(path)
            prior_source = existing.get("source") if isinstance(existing, dict) else None
            entry: dict = {"stars": int(stars)}
            if score is not None:
                entry["score"] = float(score)
            if personal_score is not None:
                entry["personal_score"] = float(personal_score)
            final_source = source if source is not None else prior_source
            if final_source:
                entry["source"] = final_source
            cur[path] = entry
        else:
            cur.pop(path, None)
        _atomic_write(_PATH, cur)
        try:
            _atomic_write(_BACKUP, cur)   # keep the mirror current
        except Exception:
            pass
