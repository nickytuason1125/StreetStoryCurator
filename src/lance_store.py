"""
LanceDB vector store — SpecVLM edition.

Schema (1536-d SigLIP-2 embeddings)
────────────────────────────────────
    path           string          primary key
    embedding      fixed_size_list<float32>[1536]   SigLIP-2 NaFlex
    score          float32         SpecVLM / QAlign aesthetic score
    personal_score float32         PersonalHead preference score
    grade          string          "Strong ✅" | "Mid ⚠️" | "Weak ❌"
    reasoning_log  string          SpecVLM narrative reasoning (empty if fallback)
    breakdown      string          JSON blob
    exif_ts        float64         Unix timestamp from EXIF (0.0 if missing)

Migration: if an existing table has a different embedding dimension (e.g. 1152-d
from SigLIP-So400M), the table is dropped and recreated automatically — the data
is re-computable from re-grading.
"""
from __future__ import annotations

import json
import threading
import numpy as np
from pathlib import Path
from typing import Optional

# Absolute path anchored to this file — never affected by CWD changes in server threads.
_DB_DIR    = str(Path(__file__).resolve().parent.parent / "cache" / "lance.db")
_TBL_NAME  = "photos"
_EMBED_DIM = 1536   # SigLIP-2 ViT-g/14 NaFlex

print(f"[lance_store] DB path: {_DB_DIR}")

_lock = threading.Lock()
_tbl  = None   # cached lancedb Table reference


# ── Schema ────────────────────────────────────────────────────────────────────

def _make_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("path",           pa.string()),
        pa.field("embedding",      pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("score",          pa.float32()),
        pa.field("personal_score", pa.float32()),
        pa.field("grade",          pa.string()),
        pa.field("reasoning_log",  pa.string()),
        pa.field("breakdown",      pa.string()),
        pa.field("exif_ts",        pa.float64()),
    ])


# ── Connection helpers ────────────────────────────────────────────────────────

def _open_table():
    global _tbl
    if _tbl is not None:
        return _tbl

    import lancedb
    db     = lancedb.connect(_DB_DIR)
    schema = _make_schema()

    if _TBL_NAME in db.table_names():
        existing = db.open_table(_TBL_NAME)
        # Auto-migrate when embedding dimension changes (e.g. 1152→1536).
        try:
            existing_dim = None
            for field in existing.schema:
                if field.name == "embedding":
                    # PyArrow FixedSizeList stores size in field.type.list_size
                    existing_dim = getattr(field.type, "list_size", None)
                    break
            if existing_dim is not None and existing_dim != _EMBED_DIM:
                print("LEGACY 1152-D DATABASE DETECTED. PURGING AND RE-GRADING WITH 1536-D NAFLEX...")
                db.drop_table(_TBL_NAME)
                _tbl = db.create_table(_TBL_NAME, schema=schema)
            else:
                # Add missing columns (e.g. reasoning_log added later)
                _tbl = existing
                _ensure_columns(_tbl)
        except Exception as _me:
            print(f"[lance] Migration check failed ({_me}), using existing table as-is")
            _tbl = existing
    else:
        _tbl = db.create_table(_TBL_NAME, schema=schema)

    return _tbl


def _ensure_columns(tbl) -> None:
    """Add any schema columns that are missing from an older table.

    Each entry is (column_name, pyarrow_type, default_value).  Safe to call on
    every open — only issues ALTER TABLE when a column is actually absent.
    """
    import pyarrow as pa
    _REQUIRED = [
        ("reasoning_log", pa.string(),  ""),
        ("personal_score", pa.float32(), 0.5),
    ]
    try:
        col_names = {f.name for f in tbl.schema}
        n = None
        for col, dtype, default in _REQUIRED:
            if col not in col_names:
                if n is None:
                    n = tbl.count_rows()
                if dtype == pa.string():
                    arr = pa.array([default] * n, type=dtype)
                else:
                    arr = pa.array([float(default)] * n, type=dtype)
                tbl.add_columns({col: arr})
                print(f"[lance] Added missing column '{col}' (default={default!r})")
    except Exception as _e:
        print(f"[lance] Column migration warning: {_e}")


# ── Public API ────────────────────────────────────────────────────────────────

def upsert_batch(records: list[dict]) -> None:
    """
    Insert or replace rows.

    Each record must have:
        path        str
        embedding   list[float]  length _EMBED_DIM (1536 for SigLIP-2, 1152 for legacy)
        score       float
        grade       str
    Optional fields:
        personal_score float (default 0.5)
        reasoning_log  str   (default "")
        breakdown      dict  (default {})
        exif_ts        float (default 0.0)

    Embeddings shorter than _EMBED_DIM are zero-padded; longer are truncated.
    This lets legacy 1152-d batches co-exist until all photos are re-graded.
    """
    import pyarrow as pa

    if not records:
        return

    def _pad(emb: list) -> list:
        f = [float(x) for x in emb]
        if len(f) < _EMBED_DIM:
            f += [0.0] * (_EMBED_DIM - len(f))
        return f[:_EMBED_DIM]

    rows = {
        "path":           [r["path"]                            for r in records],
        "embedding":      [_pad(r.get("embedding", []))         for r in records],
        "score":          [float(r.get("score", 0.0))           for r in records],
        "personal_score": [float(r.get("personal_score", 0.5))  for r in records],
        "grade":          [r.get("grade", "Mid ⚠️")             for r in records],
        "reasoning_log":  [r.get("reasoning_log", "")           for r in records],
        "breakdown":      [json.dumps(r.get("breakdown", {}))   for r in records],
        "exif_ts":        [float(r.get("exif_ts", 0.0))         for r in records],
    }
    tbl = _open_table()
    with _lock:
        tbl.merge_insert("path").when_matched_update_all().when_not_matched_insert_all().execute(
            pa.table(rows)
        )

    # ── DISK WRITE VERIFICATION ───────────────────────────────────────────────
    # Reads back the first 3 written rows immediately after commit.
    # If score here differs from what the UI shows, the bug is in HTTP caching,
    # not in the write path.
    try:
        for _rec in records[:3]:
            _fp   = _rec["path"]
            _fn   = Path(_fp).name
            _safe = _fp.replace("'", "''")
            _vrows = tbl.search().where(f"path = '{_safe}'", prefilter=True).to_list()
            if _vrows:
                print(
                    f"[lance] VERIFIED DISK VALUE: {_fn}"
                    f"  score={float(_vrows[0].get('score', 0)):.3f}"
                    f"  grade={_vrows[0].get('grade', '?')!r}"
                )
            else:
                print(f"[lance] VERIFIED DISK VALUE: {_fn}  *** ROW NOT FOUND — upsert may have failed ***")
    except Exception as _ve:
        print(f"[lance] Write verification skipped: {_ve}")


def query_by_paths(paths: list[str]) -> list[dict]:
    """Fetch rows by path list. Missing paths are silently omitted."""
    if not paths:
        return []
    tbl    = _open_table()
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
    """ANN vector search: return top_k most similar photos by embedding."""
    tbl = _open_table()
    # Pad/truncate query to match stored dimension
    q = query_emb.flatten().tolist()
    if len(q) < _EMBED_DIM:
        q += [0.0] * (_EMBED_DIM - len(q))
    q = q[:_EMBED_DIM]
    with _lock:
        results = (
            tbl.search(q)
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


def compact_after_write() -> None:
    """
    Compact LanceDB fragments after a bulk write.

    LanceDB appends each upsert as a new fragment. Compacting merges them into
    a single file, eliminating scan overhead on the next query. Safe to skip —
    performance degrades gracefully without compaction.
    """
    try:
        tbl = _open_table()
        with _lock:
            tbl.compact_files()
        print("[lance] Compaction done")
    except Exception as e:
        print(f"[lance] Compaction skipped ({e})")


def count() -> int:
    tbl = _open_table()
    with _lock:
        return tbl.count_rows()


def close_table() -> None:
    """Release the cached table reference so file handles are freed.

    Call this after a bulk write to avoid holding the DB open between pipeline
    runs. The next read/write will re-open a fresh connection.
    """
    global _tbl
    with _lock:
        _tbl = None


def reset() -> None:
    """Drop and recreate the photos table. Used for testing or forced schema refresh."""
    global _tbl
    import lancedb
    db = lancedb.connect(_DB_DIR)
    if _TBL_NAME in db.table_names():
        db.drop_table(_TBL_NAME)
    _tbl = None


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
        "reasoning_log":  r.get("reasoning_log", ""),
        "breakdown":      bd,
        "exif_ts":        float(r.get("exif_ts", 0.0)),
    }
