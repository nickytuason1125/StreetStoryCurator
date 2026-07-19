"""
clip_vec_store.py -- persistent LanceDB cache for 512-d CLIP vectors.

Why
---
Story mode (vision_story_mode.py) re-encodes every image with CLIP on every run.
It doesn't need to: a CLIP image embedding is a pure function of (image bytes,
preprocessing). Caching the 512-d vector keyed by path lets Story mode skip the
encode on a warm run -- and skip loading the CLIP model onto the GPU entirely
when everything is already cached.

Isolation
---------
This is deliberately SEPARATE from lance_store.py (the 1536-d SigLIP-2 "photos"
table used by the grading pipeline). Same DB directory (cache/lance.db), different
table ("clip_vectors"). Nothing here touches the encoder or the photos table.

Source namespacing (critical)
------------------------------
The same image path can yield DIFFERENT 512-d vectors depending on how it was
encoded (e.g. Story encodes a GRAYSCALE 224 image — vision_story_mode: .convert("L")).
So every row carries a `source` tag and every lookup is scoped to one source. The
primary key is f"{source}::{path}", so distinct encoders never read each other's
vectors. (The legacy COLOR "competition" source is no longer written — the
standalone competition_scorer CLI was removed — but the scoping is retained so any
future consumer stays isolated.)

Public API
----------
  get_embeddings(paths, source)  -> dict[str, np.ndarray]   # hits only, (512,) float32
  upsert(records, source)        -> None                    # records: {"path", "embedding"}
  count(source=None)             -> int
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

# Absolute path anchored to this file -- CWD-independent, matches lance_store.py.
# Shares the DB directory with the photos table but uses its own table.
_DB_DIR    = str(Path(__file__).resolve().parent.parent / "cache" / "lance.db")
_TBL_NAME  = "clip_vectors"
_EMBED_DIM = 512   # CLIP ViT-B/32 -- both transformers and openai-clip variants

_lock = threading.Lock()
_tbl  = None   # cached lancedb Table reference


# -- Schema -------------------------------------------------------------------

def _make_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("key",       pa.string()),                       # f"{source}::{path}"
        pa.field("path",      pa.string()),                       # image path or text query
        pa.field("source",    pa.string()),                       # story_img | story_text | competition_img
        pa.field("embedding", pa.list_(pa.float32(), _EMBED_DIM)),
    ])


def _key(source: str, path: str) -> str:
    return f"{source}::{path}"


# -- Connection helpers -------------------------------------------------------

def _open_table():
    global _tbl
    if _tbl is not None:
        return _tbl
    with _lock:
        if _tbl is not None:   # double-checked locking
            return _tbl
        try:
            _tbl = _connect_or_create()
        except Exception as _e_conn:
            # DB corrupt -- delete directory and create fresh. The cache is
            # always re-computable by re-encoding, so this is safe.
            import shutil as _sh
            print(f"[clip_vec] DB error ({_e_conn}) -- recreating at {_DB_DIR}")
            _sh.rmtree(_DB_DIR, ignore_errors=True)
            _tbl = _connect_or_create()
        return _tbl


def _connect_or_create():
    """Open (or create) the clip_vectors table. Called only from _open_table()."""
    import lancedb
    db     = lancedb.connect(_DB_DIR)
    schema = _make_schema()
    if _TBL_NAME in db.table_names():
        return db.open_table(_TBL_NAME)
    return db.create_table(_TBL_NAME, schema=schema)


# -- Public API ---------------------------------------------------------------

def get_embeddings(paths: list[str], source: str) -> dict[str, np.ndarray]:
    """
    Return {path: (512,) float32} for every path already cached under `source`.
    Paths with no cached vector are simply absent from the result.
    """
    if not paths:
        return {}
    keys = [_key(source, p) for p in paths]
    escaped = [k.replace("'", "''") for k in keys]
    in_list = ", ".join(f"'{e}'" for e in escaped)
    try:
        tbl = _open_table()
        with _lock:
            rows = (
                tbl.search()
                   .where(f"key IN ({in_list})", prefilter=True)
                   .select(["path", "embedding"])
                   .limit(len(keys))
                   .to_list()
            )
        out: dict[str, np.ndarray] = {}
        for r in rows:
            emb = r.get("embedding")
            if emb is None:
                continue
            arr = np.asarray(emb, dtype=np.float32)
            if arr.shape == (_EMBED_DIM,):
                out[r["path"]] = arr
        return out
    except Exception as e:
        # Cache is an optimization -- a read failure must never break a grade run.
        print(f"[clip_vec] get_embeddings failed ({e}); treating as cache miss")
        return {}


def upsert(records: list[dict], source: str) -> None:
    """
    Insert or replace cached vectors for `source`.

    Each record needs:
        path       str            image path or text-query string
        embedding  list[float]    length 512 (CLIP)
    Rows whose embedding is not exactly 512-d are skipped (defensive).
    """
    import pyarrow as pa

    if not records:
        return

    keys, paths, srcs, embs = [], [], [], []
    for r in records:
        emb = [float(x) for x in r.get("embedding", [])]
        if len(emb) != _EMBED_DIM:
            print(f"[clip_vec] skip upsert for {r.get('path')!r}: "
                  f"got {len(emb)}-d, expected {_EMBED_DIM}-d")
            continue
        p = r["path"]
        keys.append(_key(source, p))
        paths.append(p)
        srcs.append(source)
        embs.append(emb)

    if not keys:
        return

    rows = {"key": keys, "path": paths, "source": srcs, "embedding": embs}
    try:
        tbl = _open_table()
        with _lock:
            tbl.merge_insert("key") \
               .when_matched_update_all() \
               .when_not_matched_insert_all() \
               .execute(pa.table(rows, schema=_make_schema()))
    except Exception as e:
        # Never let a cache write failure abort the caller's pipeline.
        print(f"[clip_vec] upsert failed ({e}); continuing without caching")


def count(source: str | None = None) -> int:
    """Row count for diagnostics. Scoped to `source` when provided."""
    try:
        tbl = _open_table()
        with _lock:
            if source is None:
                return tbl.count_rows()
            safe = source.replace("'", "''")
            return len(
                tbl.search().where(f"source = '{safe}'", prefilter=True)
                   .select(["key"]).limit(10_000_000).to_list()
            )
    except Exception as e:
        print(f"[clip_vec] count failed ({e})")
        return 0


def close_table() -> None:
    """Release the cached table handle so the next call re-opens fresh."""
    global _tbl
    with _lock:
        _tbl = None
