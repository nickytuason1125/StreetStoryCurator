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
    narrative_role     string      Story Mode: "opener"|"subject"|"closer"|"contrast"|"detail"|""
    sequence_position  int64       Story Mode: 0-based slot in the last sequence, -1 = never placed
    revision_history   string      Story Mode: JSON list[dict] of revision-loop iterations
    folder_key         string      Story Mode: sha1(output_dir)[:16]

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
#
# FRAMEGRADE_LANCE_DIR exists for ONE reason: tests and throwaway harnesses had
# no way to avoid the real store. `data_dir` in a grade request does not
# redirect it, so pytest wrote its fixtures straight into the photographer's
# vector store — rows from three separate runs were found sitting in a live
# library alongside real photographs. tests/conftest.py sets this for the whole
# session; production never sets it and behaviour there is unchanged.
#
# It is read once, at import, deliberately. Re-reading per call would let a
# stray os.environ edit mid-run point half a cull at a different database.
import os as _os_ls
_DB_DIR    = str(_os_ls.environ.get("FRAMEGRADE_LANCE_DIR")
                 or Path(__file__).resolve().parent.parent / "cache" / "lance.db")
# One table PER TIER. Each encoder tier produces a different embedding
# dimension, and _connect_or_create's dim-change path used to respond by
# DROPPING the table ("PURGING table, photos will re-encode") — so running a
# lighter encoder once destroyed every grade made with the heavier one, and
# switching back destroyed them again. With a table per tier the dimensions
# never collide, nothing is purged, and each tier's grades persist
# independently. "high" keeps the original name so existing work is untouched.
import sys as _sys_rp, os as _os_rp
_sys_rp.path.insert(0, _os_rp.path.dirname(_os_rp.path.abspath(__file__)))
import run_profile as _rp                                     # noqa: E402
# Table name and embedding width are tier state, and tier state is declared in
# exactly one place. Deriving them here independently is what let the table and
# the encoder disagree about width.
_TBL_NAME  = _rp.current().lance_table
# Embedding dim follows the active SIGLIP_TIER (Phase 2). Read from the env
# directly (not by importing the heavy encoder module) to keep lance_store light.
# The table auto-migrates (purge + rebuild) below when the stored dim differs, so
# switching tiers safely re-encodes everything. high=1536 mid=1024 low=768.
import os as _os_dim
_EMBED_DIM = _rp.current().embed_dim

print(f"[lance_store] DB path: {_DB_DIR}")

_lock = threading.Lock()
_tbl  = None   # cached lancedb Table reference


# ── Native-extension preload (ORDER-CRITICAL) ────────────────────────────────
# pyarrow and lancedb are C extensions. Loading their DLLs for the FIRST time in
# a process that has already spawned and reaped a CUDA subprocess faults with a
# Windows access violation (0xC0000005) — reproduced 6/6:
#
#     pyarrow/__init__.py -> create_module  ->  ACCESS VIOLATION
#     (called from upsert_batch's `import pyarrow`, after encode_worker exited)
#
# This module's imports used to be lazy (inside upsert_batch / _connect_or_
# create), so whether a grade survived depended on whether anything happened to
# touch the DB before the SigLIP encode. It usually did — query_embeddings_by_
# paths opens the table — which is why this stayed hidden. But on the first run
# after an encoder change, `_src_changed` skips that lookup entirely, so the
# first pyarrow import landed AFTER the encode and the grade died with no
# traceback. Importing eagerly here pins the natives into the process at
# module-import time, which for every pipeline entry point is long before any
# subprocess runs. Failure is non-fatal: the lazy imports below still raise a
# normal, catchable error if the package is genuinely broken.
def warm_native() -> bool:
    """Load the pyarrow/lancedb native extensions NOW. Idempotent.

    Call before spawning any GPU subprocess. Returns True when both are loaded.
    """
    try:
        import pyarrow   # noqa: F401
        import lancedb   # noqa: F401
        return True
    except Exception as _e:
        print(f"[lance_store] native preload deferred ({type(_e).__name__}: {_e})")
        return False


warm_native()


# ── Schema ────────────────────────────────────────────────────────────────────

def _make_schema():
    import pyarrow as pa
    return pa.schema([
        pa.field("path",            pa.string()),
        pa.field("embedding",       pa.list_(pa.float32(), _EMBED_DIM)),
        pa.field("score",           pa.float32()),
        pa.field("personal_score",  pa.float32()),
        pa.field("grade",           pa.string()),
        pa.field("reasoning_log",   pa.string()),
        pa.field("breakdown",       pa.string()),
        pa.field("exif_ts",         pa.float64()),
        pa.field("has_annotations", pa.string()),
        pa.field("score_factors",   pa.string()),
        pa.field("narrative_role",    pa.string()),
        pa.field("sequence_position", pa.int64()),
        pa.field("revision_history",  pa.string()),
        pa.field("folder_key",        pa.string()),
        # Which encoder produced `embedding`. See current_encoder_tag().
        pa.field("encoder_source",    pa.string()),
    ])


# ── Embedding vector-space tagging ───────────────────────────────────────────
# Two SigLIP-2 loaders (open_clip fp32 and HF fp16) hold the SAME weights but
# are NOT the same vector space (image cosine ~0.997), and mixing them corrupts
# dedup, the archetype projection and the PersonalHead.
#
# grade_pipeline_v2 guards the switch with cache/encoder_source.txt — but that
# marker is GLOBAL while the re-encode it triggers is per-run: the first grade
# after a loader change re-encodes only ITS OWN folder and then rewrites the
# marker, so every other folder's cached embeddings are silently reused from the
# old space forever after. Tagging each row with the space that produced it
# closes that hole precisely: a row is reusable only if its tag matches today's
# encoder, so each folder re-encodes exactly once, whenever it is next graded.
_ENC_TAG: "Optional[str]" = None


def current_encoder_tag() -> str:
    """Vector-space tag for rows written now ('' when undeterminable).

    Callers MUST treat '' as "do not reuse" — failing toward a re-encode costs
    time, whereas reusing a foreign-space vector silently corrupts grades.
    """
    global _ENC_TAG
    if _ENC_TAG is None:
        try:
            from siglip2_encoder import ENCODER_SOURCE as _src
            _ENC_TAG = str(_src or "")
        except Exception:
            _ENC_TAG = ""
    return _ENC_TAG


# ── Connection helpers ────────────────────────────────────────────────────────

def _open_table():
    global _tbl
    if _tbl is not None:
        return _tbl
    with _lock:
        if _tbl is not None:   # re-check inside lock (double-checked locking)
            return _tbl
        try:
            _tbl = _connect_or_create()
        except Exception as _e_conn:
            # ONLY genuine on-disk corruption justifies destroying the store.
            # Environmental failures — missing lancedb/pyarrow dependency, a
            # locked file, OOM, permissions — must NEVER wipe the user's data.
            # Re-raise those so the real error surfaces and the DB survives.
            if not _is_corruption_error(_e_conn):
                raise
            # Genuine corruption: move the bad DB aside (recoverable) rather
            # than rmtree it, then recreate fresh.
            _quarantine_db(_e_conn)
            _tbl = _connect_or_create()
        return _tbl


def _is_corruption_error(exc: Exception) -> bool:
    """True only when the on-disk DB looks genuinely corrupt — never for
    environmental failures (missing deps, file locks, permissions, OOM)."""
    # A missing/broken dependency is an environment problem, not corruption.
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return False
    # If the core deps don't even import, we cannot be looking at DB corruption.
    try:
        import lancedb   # noqa: F401
        import pyarrow   # noqa: F401
    except Exception:
        return False
    # Nothing on disk yet => the failure isn't corruption of existing data.
    return Path(_DB_DIR).exists()


def _quarantine_db(exc: Exception) -> None:
    """Move a corrupt DB aside instead of deleting it, so the original data is
    still recoverable for inspection. Keeps only the most recent quarantine."""
    import shutil as _sh
    src = Path(_DB_DIR)
    dst = src.with_name(src.name + ".corrupt")
    try:
        if dst.exists():
            _sh.rmtree(dst, ignore_errors=True)   # replace any stale quarantine
        if src.exists():
            _sh.move(str(src), str(dst))
            print(f"[lance] DB appears corrupt ({exc}) — moved aside to {dst}, "
                  f"recreating fresh. Original preserved for recovery.")
    except Exception as _e_q:
        # Last resort: if the move itself fails, fall back to delete so the app
        # can still come up. (Reaching here means the dir is already unusable.)
        print(f"[lance] quarantine failed ({_e_q}); recreating fresh at {src}")
        _sh.rmtree(src, ignore_errors=True)


def _connect_or_create():
    """Open (or create) the LanceDB table. Called only from _open_table()."""
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
                # Reached only by a genuine LEGACY dim change within one tier
                # (e.g. the old 1152-d SigLIP-So400M -> 1536-d migration), not
                # by a quality-tier switch: each tier now has its own table, so
                # a tier change lands on a table whose dim already matches.
                print(f"[lance] Legacy embedding dim ({existing_dim}-d -> {_EMBED_DIM}-d) "
                      f"in '{_TBL_NAME}' — rebuilding this table; other tiers are untouched.")
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
        ("reasoning_log",   pa.string(),  ""),
        ("personal_score",  pa.float32(), 0.5),
        ("has_annotations", pa.string(),  ""),
        ("score_factors",   pa.string(),  ""),
        # Story Mode narrative memory (creative_director.py Step 6) —
        # additive/informational: avoid_paths already handles "don't repeat
        # images on a re-run" end-to-end; these columns let a future pass
        # query what role/position an image last held without duplicating
        # that mechanism.
        ("narrative_role",     pa.string(), ""),   # "opener"|"subject"|"closer"|"contrast"|"detail"|""
        ("sequence_position",  pa.int64(),  -1),   # 0-based slot in the last sequence, -1 = never placed
        ("revision_history",   pa.string(), ""),   # JSON list[dict], one entry per revision-loop iteration that touched this image
        ("folder_key",         pa.string(), ""),   # sha1(output_dir)[:16] — same convention as grade_pipeline_v2's checkpoint key
        # "" on migration is deliberate: pre-existing rows have an UNKNOWN
        # vector space, so they must never be reused as an encode cache hit.
        ("encoder_source",     pa.string(), ""),
    ]
    try:
        col_names = {f.name for f in tbl.schema}
        # lancedb 0.30.2's Table.add_columns(transforms) takes a dict of
        # column_name -> SQL expression (evaluated per row), NOT a dict of
        # column_name -> pyarrow Array — passing arrays directly raises
        # "'StringArray' object cannot be converted to 'PyString'".
        transforms: dict[str, str] = {}
        for col, dtype, default in _REQUIRED:
            if col in col_names:
                continue
            if dtype == pa.string():
                escaped = str(default).replace("'", "''")
                transforms[col] = f"'{escaped}'"
            elif pa.types.is_integer(dtype):
                transforms[col] = str(int(default))
            else:
                transforms[col] = str(float(default))
        if transforms:
            tbl.add_columns(transforms)
            print(f"[lance] Added missing columns: {sorted(transforms)}")
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

    def _tag_for(rec: dict) -> str:
        """Vector-space tag to store for one record.

        An explicit encoder_source always wins. Otherwise the tag is the current
        encoder — EXCEPT for all-zero placeholder rows, which are not a real
        encode at all: server.py's upload stub and the disqualified-image
        pre-flush both write [0.0]*dim. Tagging those as current-space would
        make them a valid cache hit later, so the photo would skip encoding and
        then be discarded as a zero-norm row. They stay untagged ('') so they
        are always re-encoded when the image is actually graded.
        """
        explicit = str(rec.get("encoder_source") or "")
        if explicit:
            return explicit
        emb = rec.get("embedding") or []
        return current_encoder_tag() if any(emb) else ""

    def _pad(emb: list) -> list:
        # Fast path: a correctly-sized list of floats (what the grading pipeline
        # always passes) is handed straight to pyarrow. The old unconditional
        # `[float(x) for x in emb]` + `f[:_EMBED_DIM]` built TWO extra 1536-entry
        # boxed-float lists per record — on a 5 000-photo upsert that is ~500 MB
        # of pure copy on top of the caller's own list, for no change in value.
        if type(emb) is list and len(emb) == _EMBED_DIM:
            return emb
        f = [float(x) for x in emb]
        if len(f) < _EMBED_DIM:
            f += [0.0] * (_EMBED_DIM - len(f))
            return f
        return f[:_EMBED_DIM] if len(f) > _EMBED_DIM else f

    rows = {
        "path":            [r["path"]                            for r in records],
        "embedding":       [_pad(r.get("embedding", []))         for r in records],
        "score":           [float(r.get("score", 0.0))           for r in records],
        "personal_score":  [float(r.get("personal_score", 0.5))  for r in records],
        "grade":           [r.get("grade", "Mid ⚠️")             for r in records],
        "reasoning_log":   [r.get("reasoning_log", "")           for r in records],
        "breakdown":       [json.dumps(r.get("breakdown", {}))   for r in records],
        "exif_ts":         [float(r.get("exif_ts", 0.0))         for r in records],
        "has_annotations": [r.get("has_annotations", "")         for r in records],
        "score_factors":   [r.get("score_factors", "")           for r in records],
        # Story-Mode columns — regular grading writes never set these, but they
        # must still be present in every batch or merge_insert rejects the whole
        # write with a schema mismatch (the table has them once migrated).
        "narrative_role":    [str(r.get("narrative_role") or "")                          for r in records],
        "sequence_position": [int(r.get("sequence_position")) if r.get("sequence_position") is not None else -1
                               for r in records],
        "revision_history":  [str(r.get("revision_history") or "")                        for r in records],
        "folder_key":        [str(r.get("folder_key") or "")                              for r in records],
        # Stamp the vector space these embeddings came from (see _tag_for).
        "encoder_source":    [_tag_for(r)                                                 for r in records],
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


# Columns safe to fetch without the 1536-dim embedding blob. Any metadata-only
# lookup should request these — pulling embeddings for a whole-library scan is
# what made the per-click annotations endpoint copy hundreds of MB into RAM.
_LIGHT_COLUMNS = ["path", "score", "grade", "breakdown", "reasoning_log",
                  "has_annotations", "score_factors", "is_verified",
                  "personal_score", "exif_ts"]


def query_by_path_fragment(fragment: str,
                           columns: "list[str] | None" = None) -> list[dict]:
    """Fetch light rows whose path contains ``fragment`` (SQL LIKE).

    Metadata-only alternative to query_all(): the embedding column is excluded
    unless explicitly requested, so a single-record lookup costs kilobytes
    instead of copying every graded photo's vectors into RAM. Callers that need
    exact-path semantics should re-check the returned rows (a directory name
    could theoretically contain the fragment).
    """
    frag = (fragment or "").strip()
    if not frag:
        return []
    tbl = _open_table()
    esc   = frag.replace("'", "''")
    cols  = columns if columns is not None else _LIGHT_COLUMNS
    try:
        with _lock:
            rows = (
                tbl.search()
                   .where(f"path LIKE '%{esc}%'", prefilter=True)
                   .select(cols)
                   .to_list()
            )
        return rows
    except Exception as _e:
        print(f"[lance_store] query_by_path_fragment failed: {_e}")
        return []



def query_embeddings_by_paths(paths: list[str]) -> dict[str, np.ndarray]:
    """
    Return {path: embedding (float32, shape 1536)} for every path that already
    exists in LanceDB.  Paths not found are absent from the result dict.

    Used by grade_pipeline_v2 to skip SigLIP-2 re-encoding for images whose
    embeddings were computed in a previous session.
    """
    if not paths:
        return {}
    # Only rows from TODAY's vector space are a valid encode-cache hit. A row
    # written by the other loader (or by a build that predates tagging, tag "")
    # is skipped so the caller re-encodes it — see current_encoder_tag().
    cur_tag = current_encoder_tag()
    if not cur_tag:
        print("[lance_store] encoder tag unknown — not reusing cached embeddings")
        return {}
    tbl = _open_table()
    # Build an IN-clause; escape single quotes in paths.
    escaped = [p.replace("'", "''") for p in paths]
    in_list = ", ".join(f"'{e}'" for e in escaped)
    try:
        with _lock:
            rows = (
                tbl.search()
                   .where(f"path IN ({in_list})", prefilter=True)
                   .select(["path", "embedding", "encoder_source"])
                   .to_list()
            )
        result: dict[str, np.ndarray] = {}
        _skipped = 0
        for r in rows:
            emb = r.get("embedding")
            if emb is None:
                continue
            if str(r.get("encoder_source") or "") != cur_tag:
                _skipped += 1           # foreign / untagged space → re-encode
                continue
            arr = np.array(emb, dtype=np.float32)
            if arr.shape == (_EMBED_DIM,):
                result[r["path"]] = arr
        if _skipped:
            print(f"[lance_store] {_skipped} cached embeddings are from another "
                  f"encoder space — they will be re-encoded (current: {cur_tag})")
        return result
    except Exception as _e:
        print(f"[lance_store] query_embeddings_by_paths failed: {_e}")
        return {}


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


def update_annotations(path: str, score_factors: list) -> None:
    """Write has_annotations='true' and score_factors JSON for a single row.

    Safe to call from the background queue manager — only touches these two
    columns; never reads or writes score/grade/embedding.
    """
    safe = path.replace("'", "''")
    tbl  = _open_table()
    with _lock:
        tbl.update(
            where=f"path = '{safe}'",
            values={
                "has_annotations": "true",
                "score_factors":   json.dumps(score_factors),
            },
        )
    print(f"[lance] update_annotations: {Path(path).name}  {len(score_factors)} factors")


def update_narrative_metadata(
    path: str,
    role: str,
    seq_pos: int,
    revision_history: list[dict],
    folder_key: str,
) -> None:
    """Write Story Mode narrative metadata for a single row.

    Safe to call from run_creative_direction's Step 6 output loop — only
    touches these four columns; never reads or writes score/embedding/grade,
    same discipline as update_annotations() above. Additive/informational:
    the existing avoid_paths mechanism already handles "don't repeat images
    on a re-run" end-to-end, so this doesn't change selection behavior —
    it just makes the history queryable.
    """
    safe = path.replace("'", "''")
    tbl  = _open_table()
    with _lock:
        tbl.update(
            where=f"path = '{safe}'",
            values={
                "narrative_role":    role,
                "sequence_position": int(seq_pos),
                "revision_history":  json.dumps(revision_history),
                "folder_key":        folder_key,
            },
        )


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
    Compact fragments AND reap old versions after a bulk write.

    LanceDB appends each upsert as a new fragment and keeps every prior version.
    compact_files() merged the fragments but left the history, so photos.lance
    reached 859 MB holding ~10 MB of vectors across 409 versions — growth
    unbounded in the number of CULLS, not the number of photos.

    optimize() does both in one call. Two deliberate choices:

      * delete_unverified=False — all three tier tables share one database, and
        a reader on another tier may hold references to fragments this call
        cannot prove are orphaned.
      * A retention WINDOW rather than "keep only current", so a cull that
        writes bad grades can still be rolled back.

    Safe to skip: this runs after grades are durably committed, so a failure
    here must never fail the cull.
    """
    from datetime import timedelta
    try:
        import run_profile as _rp
        days = max(0, int(_rp.setting("FRAMEGRADE_LANCE_RETENTION_DAYS")))
    except Exception:
        days = 7
    try:
        tbl = _open_table()
        with _lock:
            tbl.optimize(cleanup_older_than=timedelta(days=days),
                         delete_unverified=False)
        print(f"[lance] Compaction + version cleanup done (retention {days}d)")
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
    # Use explicit None-check instead of `or` — numpy arrays are falsy-ambiguous
    # when they contain more than one element, which is always true for embeddings.
    _emb = r.get("embedding")
    emb  = _emb if _emb is not None else []

    _bd_raw = r.get("breakdown")
    _bd_str = "" if _bd_raw is None else str(_bd_raw)
    try:
        bd = json.loads(_bd_str) if _bd_str else {}
    except Exception:
        bd = {}

    # String fields: pandas may return None, numpy.nan, or numpy.str_ for nulls
    def _safe_str(v) -> str:
        if v is None:
            return ""
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return ""
        except Exception:
            pass
        return str(v)

    def _safe_int(v, default: int) -> int:
        if v is None:
            return default
        try:
            import math
            if isinstance(v, float) and math.isnan(v):
                return default
        except Exception:
            pass
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    return {
        "path":            r.get("path", ""),
        "embedding":       np.array(emb, dtype=np.float32),
        "score":           float(r.get("score") or 0.0),
        "personal_score":  float(r.get("personal_score") or 0.5),
        "grade":           _safe_str(r.get("grade")) or "Mid ⚠️",
        "reasoning_log":   _safe_str(r.get("reasoning_log")),
        "breakdown":       bd,
        "exif_ts":         float(r.get("exif_ts") or 0.0),
        "has_annotations": _safe_str(r.get("has_annotations")),
        "score_factors":   _safe_str(r.get("score_factors")),
        "narrative_role":     _safe_str(r.get("narrative_role")),
        "sequence_position":  _safe_int(r.get("sequence_position"), -1),
        "revision_history":   _safe_str(r.get("revision_history")),
        "folder_key":         _safe_str(r.get("folder_key")),
    }
