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
# Golden-grade runs (scripts/perf_guard.py) point this at an empty file so the
# taste loop — which is MEANT to move grades as you rate — cannot make the
# grading-machinery check flap. Unset in normal use.
import os as _os_rs
if _os_rs.environ.get("FIRSTCUT_RATINGS_PATH", "").strip():
    _PATH   = Path(_os_rs.environ["FIRSTCUT_RATINGS_PATH"])
    _BACKUP = _PATH.with_name(_PATH.stem + ".backup.json")
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


def load_records() -> list:
    """Return [{path, stars, score, rated_at}] for every rated photo that has a
    snapshotted machine score. This is the full cross-folder calibration table:
    ratings from LX3, tpe_master and every other folder together, each carrying
    the machine score as it stood when the rating was made. New ratings are
    stamped with `rated_at` (set_rating) so the calibration can weight recent
    taste more heavily; legacy entries carry no timestamp and count at full
    weight."""
    out = []
    for k, v in _read_raw().items():
        stars = _stars_of(v)
        if stars <= 0 or not isinstance(v, dict):
            continue
        out.append({
            "path": k,
            "stars": stars,
            "score": v.get("score"),
            "rated_at": v.get("rated_at"),
        })
    return out


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
            entry: dict = {"stars": int(stars), "rated_at": time.time()}
            if score is not None:
                entry["score"] = float(score)
            if personal_score is not None:
                entry["personal_score"] = float(personal_score)
            final_source = source if source is not None else prior_source
            if final_source:
                entry["source"] = final_source
            # Content fingerprint: the rating follows the photo when it is
            # copied off the card or renamed (stars_for_paths falls back to it).
            try:
                from photo_identity import fingerprint as _fp
                fp = _fp(path) or (existing.get("fp") if isinstance(existing, dict) else None)
            except Exception:
                fp = None
            if fp:
                entry["fp"] = fp
                try:
                    entry["size"] = os.path.getsize(path)
                except OSError:
                    if isinstance(existing, dict) and existing.get("size"):
                        entry["size"] = existing["size"]
            # Re-rating keeps the stored grade-time features (attach_features).
            if isinstance(existing, dict) and "features" in existing:
                entry["features"] = existing["features"]
            cur[path] = entry
        else:
            cur.pop(path, None)
        _atomic_write(_PATH, cur)
        try:
            _atomic_write(_BACKUP, cur)   # keep the mirror current
        except Exception:
            pass


def slim_features(bd: dict) -> dict:
    """The breakdown values the taste learner can use: numeric/bool scalars and
    the archetype weights. Strings, nested logs and the grade signature are
    dropped — they are not features and would bloat the store."""
    out: dict = {}
    for k, v in (bd or {}).items():
        if k == "_grade_sig":
            continue
        if isinstance(v, (bool, int, float)):
            out[k] = v
        elif k == "_arch_w" and isinstance(v, dict):
            out[k] = {a: float(x) for a, x in v.items() if isinstance(x, (int, float))}
    return out


def attach_features(path: str, features: dict, score: float | None = None) -> bool:
    """Store the grade-time features (and optionally the machine score) on an
    EXISTING rating without touching stars, rated_at or source — so the taste
    learner can retrain after the catalog is cleared or the file moves.
    Returns False when the path is not rated: features never create a rating."""
    with _lock:
        cur = _read_raw()
        e = cur.get(path)
        if e is None:
            return False
        if not isinstance(e, dict):
            e = {"stars": int(e)}
        e["features"] = slim_features(features)
        if score is not None:
            # NOT "score": that is the rating-time snapshot the grade
            # thresholds are fitted from (rating_calibration). Overwriting it
            # moved the Strong line 0.58 -> 0.52 (2026-10-04).
            e["score_current"] = float(score)
        cur[path] = e
        _atomic_write(_PATH, cur)
        try:
            _atomic_write(_BACKUP, cur)
        except Exception:
            pass
        return True


def stars_for_paths(paths: list) -> dict:
    """{path: stars} for every RATED photo among `paths` — exact path first,
    then content fingerprint, so a rating made on the SD card still applies
    after the photo is copied to the laptop or renamed."""
    raw = _read_raw()
    out, misses = {}, []
    for p in paths:
        s = _stars_of(raw.get(p))
        if s > 0:
            out[p] = s
        else:
            misses.append(p)
    if misses:
        fp_entries = [v for v in raw.values()
                      if isinstance(v, dict) and v.get("fp") and _stars_of(v) > 0]
        by_fp = {v["fp"]: _stars_of(v) for v in fp_entries}
        # Size pre-check: a copy has the same size, so only photos whose size
        # matches a rated photo need their first 64 KB read. Without it every
        # unrated photo in a cull was read (~320 MB on a 5,000-photo card).
        # Entries from before sizes were stored disable the shortcut.
        sizes = ({v["size"] for v in fp_entries}
                 if fp_entries and all(v.get("size") for v in fp_entries) else None)
        if by_fp:
            import photo_identity
            for p in misses:
                if sizes is not None:
                    try:
                        if os.path.getsize(p) not in sizes:
                            continue
                    except OSError:
                        continue
                fp = photo_identity.fingerprint(p)
                if fp and fp in by_fp:
                    out[p] = by_fp[fp]
    return out


def get_current_score(path: str) -> float | None:
    """The machine score that goes WITH the stored features (attach_features)."""
    v = _read_raw().get(path)
    s = v.get("score_current") if isinstance(v, dict) else None
    return float(s) if isinstance(s, (int, float)) else None


def get_features(path: str) -> dict | None:
    v = _read_raw().get(path)
    return v.get("features") if isinstance(v, dict) else None


# --- Star -> grade bucket -----------------------------------------------
# Product rule (2026-09-22): a rated photo shows the grade YOUR stars imply,
# not the algorithm's bucket for its raw score — no threshold math, and no
# exception for any rating's source tag (LX3/tpe_master included; see memory
# feedback_all_star_ratings_are_ground_truth). Centralised here so every
# place that finalises a displayed grade uses the identical mapping.
#
# Split (recalibrated 2026-09-22, photographer's own boundaries): Mid's
# borderline is 2★ and Strong's borderline is 4★ — i.e. 2★ is the LOWEST
# star count that still counts as Mid (not Weak), and 4★ is the LOWEST that
# counts as Strong. Only 1★ is Weak.
_STAR_GRADE = {
    5: "Strong ✅", 4: "Strong ✅",
    3: "Mid ⚠️", 2: "Mid ⚠️",
    1: "Weak ❌",
}


def grade_for_stars(stars) -> str | None:
    """The grade bucket the user's stars imply, or None when unrated
    (0/None/invalid) — callers fall back to the algorithm's grade in that
    case. This is meant to OVERRIDE the machine grade wherever a photo's
    grade is finalised for display, for any photo that has a rating."""
    try:
        return _STAR_GRADE.get(int(stars))
    except (TypeError, ValueError):
        return None
