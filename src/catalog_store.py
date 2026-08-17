"""
catalog_store.py — the gallery catalog, as a catalog rather than a snapshot.

The problem this fixes
----------------------
Every grade wrote cache/catalog.json with ONLY the photos from the folder it
just graded, replacing whatever was there. Grade folder B and folder A vanishes
from your gallery — even though its grades are still safe in LanceDB. During one
working session this clobbered the catalog five times.

That is the opposite of how a photo catalog should behave. Lightroom accumulates:
importing a shoot adds to the catalog, it never empties it. So writes here MERGE
BY PATH — new results update existing entries and append new ones, and every
other folder is left alone.

Two safety properties:
  * ATOMIC   — write to .tmp then os.replace, so a crash mid-write cannot leave a
               truncated catalog. A half-written catalog reads as "no photos",
               which looks exactly like data loss to the user.
  * REBUILDABLE — LanceDB is the source of truth; the catalog is a projection of
               it. rebuild_from_lance() regenerates the whole thing, so a lost or
               corrupt catalog is an inconvenience, not a loss.

Deliberately dependency-light (json + pathlib): this is imported on the grade
path, which is kept free of torch.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "cache" / "catalog.json"


def _np2py(o):
    return o.item() if hasattr(o, "item") else str(o)


def load(path: Optional[Path] = None) -> dict:
    """Read the catalog. A missing OR corrupt file yields an empty catalog
    rather than raising — callers must still be able to write a good one."""
    p = Path(path) if path else _default_path()
    try:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("photos"), list):
                return d
            print(f"[catalog] {p.name} has an unexpected shape — treating as empty")
    except Exception as exc:
        print(f"[catalog] {p.name} unreadable ({exc}) — treating as empty")
    return {"photos": [], "folders": []}


def merge_write(photos: Iterable[dict], path: Optional[Path] = None,
                tag: str = "") -> int:
    """Merge `photos` into the catalog by path and write atomically.

    Returns the total number of photos in the catalog afterwards. Entries for
    other folders are preserved; entries for these paths are replaced with the
    fresh result (a re-grade should update, not duplicate).
    """
    p = Path(path) if path else _default_path()
    incoming = [dict(ph) for ph in photos if ph.get("path")]
    existing = load(p)

    by_path: dict = {}
    for ph in existing.get("photos", []):
        if ph.get("path"):
            by_path[str(ph["path"])] = ph
    kept_before = len(by_path)
    for ph in incoming:
        ph.pop("embedding", None)          # never balloon the catalog with vectors
        by_path[str(ph["path"])] = ph

    merged = list(by_path.values())
    folders = list(dict.fromkeys(
        [str(Path(ph["path"]).parent) for ph in merged if ph.get("path")]))

    payload = json.dumps({"photos": merged, "folders": folders,
                          "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                         ensure_ascii=False, default=_np2py)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(p)

    added = len(merged) - kept_before
    print(f"[catalog] merged {len(incoming)} photos into catalog.json "
          f"({kept_before} existing -> {len(merged)} total, +{added} new, "
          f"{len(folders)} folders) {tag}".rstrip())
    return len(merged)


def mark_missing(path: Optional[Path] = None) -> int:
    """Flag catalog entries whose image file is gone, and unflag ones that are back.

    Photos move, drives get unplugged, folders get renamed. Silently dropping
    those entries would look like the app lost your grades; leaving them
    unmarked gives a gallery of broken thumbnails with no explanation. Lightroom
    flags them as missing and keeps the metadata, which is what this does — the
    grades stay, the entry is marked `missing: true`, and it clears itself the
    moment the file reappears.

    Returns the number of entries currently missing.
    """
    p = Path(path) if path else _default_path()
    d = load(p)
    photos = d.get("photos", [])
    if not photos:
        return 0
    n_missing, changed = 0, False
    for ph in photos:
        fp = ph.get("path")
        if not fp:
            continue
        gone = not os.path.exists(fp)
        if gone:
            n_missing += 1
        if bool(ph.get("missing")) != gone:
            ph["missing"] = gone
            changed = True
    if changed:
        payload = json.dumps({"photos": photos, "folders": d.get("folders", []),
                              "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                             ensure_ascii=False, default=_np2py)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(p)
    print(f"[catalog] {n_missing}/{len(photos)} photos are missing on disk"
          + (" (flags updated)" if changed else ""))
    return n_missing


def rebuild_from_lance(path: Optional[Path] = None,
                       min_score: float = 0.0) -> int:
    """Regenerate the catalog from LanceDB, the source of truth.

    For when the catalog is lost, truncated, or was overwritten by an older
    build. Grades live in the database; the catalog is only a view of them.
    Returns the number of photos written (0 if the store is unavailable).
    """
    try:
        import lance_store as ls
    except Exception as exc:
        print(f"[catalog] rebuild unavailable ({exc})")
        return 0
    try:
        rows = ls.query_all(min_score=min_score)
    except Exception as exc:
        print(f"[catalog] rebuild query failed ({exc})")
        return 0

    photos = []
    for r in rows:
        rp = r.get("path")
        if not rp:
            continue
        bd = r.get("breakdown", {})
        if isinstance(bd, str):
            try: bd = json.loads(bd)
            except Exception: bd = {}
        score = round(float(r.get("score", 0.0)), 3)
        photos.append({
            "id": rp, "path": rp, "filename": Path(rp).name,
            "grade": r.get("grade", ""), "score": score,
            "overall_score": score, "rating": score,
            "personal_score": round(float(r.get("personal_score", 0.5)), 3),
            "human_perception": round(float(r.get("personal_score", 0.5)), 3),
            "breakdown": bd, "critique": "", "reasoning_log": "",
            "is_verified": False, "exif_ts": float(r.get("exif_ts", 0.0) or 0.0),
            "stars": 0, "reject": False, "sim_flag": "", "cluster_id": -1,
            # Flag rather than drop: the grade is real even if the file moved.
            "missing": not os.path.exists(rp),
        })

    # Restore durable star ratings — they are the user's own input and must
    # survive any rebuild.
    try:
        import ratings_store
        stars = ratings_store.load()
        hit = 0
        for ph in photos:
            s = stars.get(ph["path"])
            if s:
                ph["stars"] = int(s); hit += 1
        if hit:
            print(f"[catalog] restored {hit} star ratings")
    except Exception as exc:
        print(f"[catalog] star restore skipped ({exc})")

    p = Path(path) if path else _default_path()
    folders = list(dict.fromkeys(str(Path(ph["path"]).parent) for ph in photos))
    payload = json.dumps({"photos": photos, "folders": folders,
                          "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                         ensure_ascii=False, default=_np2py)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(p)
    print(f"[catalog] rebuilt from LanceDB: {len(photos)} photos, {len(folders)} folders")
    return len(photos)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_from_lance()
    else:
        d = load()
        print(f"catalog: {len(d.get('photos', []))} photos across "
              f"{len(d.get('folders', []))} folders (saved {d.get('saved_at','?')})")
        for f in d.get("folders", []):
            n = sum(1 for ph in d["photos"] if str(Path(ph["path"]).parent) == f)
            print(f"   {n:5d}  {f}")
