"""
Thread-safe DuckDB-backed photo cache.

Sits alongside the existing JSON cache (cache/light_scores.json) and gives
MOGCO fast vector retrieval without altering the existing JSON-based pipeline.
Writes are advisory — any failure is silently suppressed so grading is never blocked.
"""
import threading
import json
import numpy as np
from pathlib import Path


class PhotoCache:
    """DuckDB-backed store for photo embeddings and scores."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS photos (
            path      TEXT    PRIMARY KEY,
            score     DOUBLE,
            quality   DOUBLE,
            breakdown TEXT,
            embedding DOUBLE[],
            exif_ts   DOUBLE
        )
    """

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "cache" / "cache.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._conn = None
        self._init()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        import duckdb
        if self._conn is None:
            self._conn = duckdb.connect(self._path)
        return self._conn

    def _init(self):
        with self._lock:
            try:
                self._connect().execute(self._SCHEMA)
            except Exception:
                pass

    def _row_to_dict(self, row: tuple) -> dict:
        path, score, quality, bd_str, emb, exif_ts = row
        try:
            bd = json.loads(bd_str) if bd_str else {}
        except Exception:
            bd = {}
        return {
            "path":      path,
            "score":     float(score or 0.0),
            "quality":   float(quality or 0.0),
            "breakdown": bd,
            "embedding": np.array(emb if emb else [], dtype=np.float64),
            "exif_ts":   float(exif_ts or 0.0),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(self, path: str, photo_data: dict, embedding: np.ndarray) -> None:
        """Insert or replace one photo row (called per-photo after grading)."""
        try:
            score   = float(photo_data.get("score", 0.0))
            quality = float(photo_data.get("human_perception", score))
            exif_ts = float(photo_data.get("exif_ts") or 0.0)
            bd_str  = json.dumps(photo_data.get("breakdown", {}))
            emb     = np.asarray(embedding, dtype=np.float64).tolist()
            with self._lock:
                self._connect().execute(
                    """INSERT OR REPLACE INTO photos
                       (path, score, quality, breakdown, embedding, exif_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [path, score, quality, bd_str, emb, exif_ts],
                )
        except Exception:
            pass

    def get_by_paths(self, paths: list[str]) -> list[dict]:
        """Bulk-fetch rows by path list. Missing paths are silently omitted."""
        if not paths:
            return []
        try:
            placeholders = ", ".join("?" * len(paths))
            with self._lock:
                rows = self._connect().execute(
                    f"SELECT path, score, quality, breakdown, embedding, exif_ts "
                    f"FROM photos WHERE path IN ({placeholders})",
                    paths,
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            return []

    def get_all(self) -> list[dict]:
        """Return every cached row."""
        try:
            with self._lock:
                rows = self._connect().execute(
                    "SELECT path, score, quality, breakdown, embedding, exif_ts FROM photos"
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]
        except Exception:
            return []

    def count(self) -> int:
        try:
            with self._lock:
                return int(self._connect().execute("SELECT COUNT(*) FROM photos").fetchone()[0])
        except Exception:
            return 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


# Module-level singleton (imported by server.py and lightweight_analyzer.py)
_default_cache: PhotoCache | None = None
_cache_lock = threading.Lock()


def get_photo_cache(db_path: str | None = None) -> PhotoCache:
    """Return the shared PhotoCache singleton, initialising it on first call."""
    global _default_cache
    if _default_cache is None:
        with _cache_lock:
            if _default_cache is None:
                _default_cache = PhotoCache(db_path)
    return _default_cache
