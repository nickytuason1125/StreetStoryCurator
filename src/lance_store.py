"""
Step 1b — LanceDB vector store.

Replaces DuckDB for embedding storage.  LanceDB uses the Lance columnar
format for zero-copy Arrow reads and supports ANN vector search out of the
box — important once the library grows past a few thousand photos.

Schema
──────
    path          string          (primary key, absolute path)
    embedding     fixed_size_list<float32>[1152]   SigLIP So400M
    score         float32         Q-Align aesthetic score
    personal_score float32        PersonalHead preference score
    grade         string          "Strong ✅" | "Mid ⚠️" | "Weak ❌"
    breakdown     string          JSON blob
    exif_ts       float64         Unix timestamp from EXIF (0.0 if missing)
"""
from __future__ import annotations

import json
import threading
import numpy as np
from pathlib import Path
from typing import Optional

_DB_DIR    = "cache/lance.db"
_TBL_NAME  = "photos"
_EMBED_DIM = 1152

_lock = threading.Lock()
_tbl  = None   # cached lancedb Table reference


# ── Connection helpers ────────────────────────────────────────────────────────

def _open_table():
    global _tbl
    if _tbl is not None:
        return _tbl
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(_DB_DIR)
    schema = pa.schema([
        pa.field("path",           pa.string()),
        pa.field("embedding",      pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("score",          pa.float32()),
        pa.field("personal_score", pa.float32()),
        pa.field("grade",          pa.string()),
        pa.field("breakdown",      pa.string()),
        pa.field("exif_ts",        pa.float64()),
    ])
    if _TBL_NAME in db.table_names():
        _tbl = db.open_table(_TBL_NAME)
    else:
        _tbl = db.create_table(_TBL_NAME, schema=schema)
    return _tbl


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_batch(records: list[dict]) -> None:
    """
    Insert or replace rows.  Each record must have:
        path, embedding (list[float] len 1152), score, grade,
        personal_score (optional, default 0.5),
        breakdown (dict, optional), exif_ts (float, optional)
    """
    import pyarrow as pa

    if not records:
        return

    rows = {
        "path":           [r["path"]                       for r in records],
        "embedding":      [list(map(float, r["embedding"])) for r in records],
        "score":          [float(r.get("score", 0.0))       for r in records],
        "personal_score": [float(r.get("personal_score", 0.5)) for r in records],
        "grade":          [r.get("grade", "Mid ⚠️")         for r in records],
        "breakdown":      [json.dumps(r.get("breakdown", {})) for r in records],
        "exif_ts":        [float(r.get("exif_ts", 0.0))     for r in records],
    }
    tbl = _open_table()
    with _lock:
        tbl.merge_insert("path").when_matched_update_all().when_not_matched_insert_all().execute(
            pa.table(rows)
        )


def query_by_paths(paths: list[str]) -> list[dict]:
    """Fetch rows by path list. Missing paths are silently omitted."""
    if not paths:
        return []
    tbl = _open_table()
    quoted = ", ".join(f"'{p.replace(chr(39), chr(39)*2)}'" for p in paths)
    with _lock:
        rows = tbl.search().where(f"path IN ({quoted})", prefilter=True).to_list()
    return [_row_to_dict(r) for r in rows]


def query_all(min_score: float = 0.0) -> list[dict]:
    """Return all cached rows with score >= min_score."""
    tbl = _open_table()
    with _lock:
        if min_score > 0:
            rows = tbl.search().where(f"score >= {min_score}", prefilter=True).to_list()
        else:
            rows = tbl.to_pandas().to_dict("records")
    return [_row_to_dict(r) for r in rows]


def vector_search(query_emb: np.ndarray, top_k: int = 20, min_score: float = 0.0) -> list[dict]:
    """ANN vector search: return top_k most similar photos by SigLIP embedding."""
    tbl = _open_table()
    with _lock:
        results = (
            tbl.search(query_emb.tolist())
               .where(f"score >= {min_score}", prefilter=True)
               .limit(top_k)
               .to_list()
        )
    return [_row_to_dict(r) for r in results]


def update_personal_scores(path_score_map: dict[str, float]) -> None:
    """Bulk-update personal_score for a set of paths."""
    import pyarrow as pa

    rows = {
        "path":           list(path_score_map.keys()),
        "personal_score": [float(v) for v in path_score_map.values()],
    }
    tbl = _open_table()
    with _lock:
        tbl.merge_insert("path").when_matched_update_all().execute(pa.table(rows))


def count() -> int:
    tbl = _open_table()
    with _lock:
        return tbl.count_rows()


# ── Internal ──────────────────────────────────────────────────────────────────

def _row_to_dict(r: dict) -> dict:
    emb = r.get("embedding") or []
    try:
        bd = json.loads(r.get("breakdown") or "{}")
    except Exception:
        bd = {}
    return {
        "path":           r.get("path", ""),
        "embedding":      np.array(emb, dtype=np.float32),
        "score":          float(r.get("score", 0.0)),
        "personal_score": float(r.get("personal_score", 0.5)),
        "grade":          r.get("grade", "Mid ⚠️"),
        "breakdown":      bd,
        "exif_ts":        float(r.get("exif_ts", 0.0)),
    }
