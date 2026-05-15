"""
Script: convert_to_lance.py

Convert existing cache/light_scores.json and cache/cache.db (DuckDB) into a unified
LanceDB table called `unified_photos`.

Schema created in LanceDB:
- img_path : string
- vector   : fixed-size list<float32>[1152]
- aesthetic_level : int32   (Strong->2, Mid->1, Weak->0)
- personal_score   : float32 (defaults from DuckDB 'quality' or 0.5)
- metadata : string (JSON blob with breakdown, score, grade, critique, exif_ts, nima_score, faces)

Usage:
  python convert_to_lance.py [--lance-db cache/lance.db] [--force]

This script is safe to re-run; it uses merge_insert on img_path to upsert rows.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import numpy as np

DEFAULT_CACHE_JSON = Path("cache/light_scores.json")
DEFAULT_DUCKDB = Path("cache/cache.db")
DEFAULT_LANCE = Path("cache/lance.db")
UNIFIED_TABLE = "unified_photos"
VECTOR_DIM = 1152


def load_json_cache(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_vector(emb, target_dim=VECTOR_DIM):
    if emb is None:
        return np.zeros(target_dim, dtype=np.float32)
    a = np.asarray(emb, dtype=np.float32)
    if a.ndim > 1:
        a = a.flatten()
    ln = a.shape[0]
    if ln == target_dim:
        return a.astype(np.float32)
    if ln > target_dim:
        return a[:target_dim].astype(np.float32)
    # ln < target_dim: pad with zeros
    out = np.zeros(target_dim, dtype=np.float32)
    out[:ln] = a
    return out


def grade_to_level(grade_str: str) -> int:
    if not grade_str:
        return 1
    if "Strong" in grade_str:
        return 2
    if "Mid" in grade_str:
        return 1
    if "Weak" in grade_str:
        return 0
    # fallback: threshold by numeric if present
    try:
        v = float(grade_str)
        if v >= 0.7:
            return 2
        if v >= 0.4:
            return 1
        return 0
    except Exception:
        return 1


def main(lance_db: Path, force: bool = False):
    # Load JSON cache
    json_cache = load_json_cache(DEFAULT_CACHE_JSON)

    # Load DuckDB photo cache via existing module if possible
    rows = []
    try:
        from photo_cache import get_photo_cache
        pc = get_photo_cache(str(DEFAULT_DUCKDB))
        db_rows = pc.get_all()
    except Exception:
        db_rows = []

    # Build records list
    records = []
    for r in db_rows:
        path = r.get("path")
        if not path:
            continue
        # prefer full result from json_cache if available
        j = json_cache.get(path, {})
        grade = j.get("grade") or r.get("breakdown", {}).get("grade") or j.get("breakdown", {}).get("grade") or ""
        score = float(j.get("score", r.get("score", 0.0)))
        nima = j.get("nima_score") if j else None
        critique = j.get("critique") or ""
        breakdown = j.get("breakdown") or r.get("breakdown") or {}
        faces = int(j.get("faces", 0))
        exif_ts = float(r.get("exif_ts", 0.0) or 0.0)

        emb = r.get("embedding")
        vector = normalize_vector(emb, VECTOR_DIM)

        # aesthetic level mapping
        alevel = grade_to_level(grade or j.get("grade", ""))

        # personal_score: attempt to pull from DuckDB 'quality' if present, else default 0.5
        personal = float(r.get("quality") or r.get("personal_score") or 0.5)

        metadata = {
            "score": score,
            "grade": grade,
            "nima_score": nima,
            "critique": critique,
            "breakdown": breakdown,
            "faces": faces,
            "exif_ts": exif_ts,
        }

        records.append({
            "img_path": path,
            "vector": vector.astype(np.float32),
            "aesthetic_level": int(alevel),
            "personal_score": float(personal),
            "metadata": json.dumps(metadata, ensure_ascii=False),
        })

    # Connect to LanceDB and create/merge table
    try:
        import lancedb
        import pyarrow as pa
    except Exception as e:
        print("Missing lancedb or pyarrow. Install lancedb and pyarrow first.")
        raise

    db = lancedb.connect(str(lance_db))

    schema = pa.schema([
        pa.field("img_path", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
        pa.field("aesthetic_level", pa.int32()),
        pa.field("personal_score", pa.float32()),
        pa.field("metadata", pa.string()),
    ])

    if UNIFIED_TABLE in db.table_names():
        tbl = db.open_table(UNIFIED_TABLE)
    else:
        tbl = db.create_table(UNIFIED_TABLE, schema=schema)

    if not records:
        print("No records found in DuckDB cache to import.")
        return

    # Prepare pyarrow table
    pa_rows = {
        "img_path": [r["img_path"] for r in records],
        "vector": [r["vector"].tolist() for r in records],
        "aesthetic_level": [r["aesthetic_level"] for r in records],
        "personal_score": [r["personal_score"] for r in records],
        "metadata": [r["metadata"] for r in records],
    }
    table = pa.table(pa_rows)

    # Merge/Upsert
    with lancedb.utils.acquire_lance_lock(db._client._db_path() if hasattr(db, "_client") else str(lance_db)):
        # Use merge_insert pattern similar to other modules when available
        try:
            tbl.merge_insert("img_path").when_matched_update_all().when_not_matched_insert_all().execute(table)
            print(f"Upserted {len(records)} rows into {lance_db}:{UNIFIED_TABLE}")
        except Exception:
            # fallback to simple insert
            tbl.insert(table)
            print(f"Inserted {len(records)} rows into {lance_db}:{UNIFIED_TABLE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-db", type=Path, default=DEFAULT_LANCE, help="Path to LanceDB folder (default: cache/lance.db)")
    ap.add_argument("--force", action="store_true", help="Force recreate table (not implemented - future)")
    args = ap.parse_args()
    main(args.lance_db, force=args.force)
