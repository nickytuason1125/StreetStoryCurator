"""
engine_utils.py — Production utilities for Street Story Curator.

Four capabilities (all optional-import-safe; degrade gracefully):
  1. pHash deduplication  — hybrid perceptual + cosine similarity
  2. LocalVectorDB        — sqlite-vec ANN search with cosine fallback
  3. FolderWatcher        — watchdog-based incremental file event listener
  4. XMP/JSON sidecar export — pyexiv2 if available, else JSON sidecar

IMPORTANT: None of these change any existing API contract or grading logic.
They are additive utilities consumed by server.py endpoints only.
"""

from __future__ import annotations
import json, os, sqlite3
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. pHash deduplication
# ─────────────────────────────────────────────────────────────────────────────

def get_dedup_score(path1: str, path2: str,
                    emb1: list[float], emb2: list[float]) -> float:
    """
    Hybrid duplicate confidence: 0.4 × pHash + 0.6 × cosine.
    Returns float in [0, 1].  > 0.85 → likely duplicate.
    pHash is robust to JPEG re-saves; cosine catches content similarity.
    """
    phash_sim = 0.0
    try:
        import imagehash
        from PIL import Image
        h1 = imagehash.phash(Image.open(path1))
        h2 = imagehash.phash(Image.open(path2))
        phash_sim = 1.0 - (h1 - h2) / 64.0
    except Exception:
        pass

    cos_sim = 0.0
    try:
        a = np.array(emb1, dtype=np.float32)
        b = np.array(emb2, dtype=np.float32)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos_sim = float(np.dot(a, b) / (na * nb))
    except Exception:
        pass

    return 0.4 * phash_sim + 0.6 * cos_sim


def phash_of(path: str) -> str | None:
    """Return hex pHash string for a single image, or None on failure."""
    try:
        import imagehash
        from PIL import Image
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fast file hashing
# ─────────────────────────────────────────────────────────────────────────────

def get_file_hash(path: str) -> str:
    """
    xxHash-64 of file contents, chunked to handle 50 MB+ RAW files without
    loading the whole file into RAM.
    Falls back to a quick mtime+size fingerprint if the file doesn't exist
    or xxhash is missing.
    """
    try:
        import xxhash
        h = xxhash.xxh64()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()
    except ImportError:
        try:
            st = os.stat(path)
            return f"{st.st_size}-{int(st.st_mtime)}"
        except OSError:
            return path
    except OSError:
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 3. LocalVectorDB — sqlite-vec ANN search
# ─────────────────────────────────────────────────────────────────────────────

class LocalVectorDB:
    """
    SQLite-backed image library with optional sqlite-vec ANN indexing.

    Schema
    ------
    images (path PK, hash, emb_dim, embedding BLOB, metadata JSON)
    vec_images  (rowid, embedding)  ← sqlite-vec virtual table (if available)

    Falls back to exact cosine search in Python when sqlite-vec is absent.
    Thread-safe: uses check_same_thread=False + explicit commit after writes.
    """

    def __init__(self, db_path: str = "cache/library.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._vec_ok = False
        self._dim: int | None = None

        try:
            import sqlite_vec as _sv
            self.conn.enable_load_extension(True)
            _sv.load(self.conn)
            self.conn.enable_load_extension(False)
            self._vec_ok = True
        except Exception:
            pass

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                path      TEXT PRIMARY KEY,
                hash      TEXT,
                emb_dim   INTEGER,
                embedding BLOB,
                metadata  TEXT
            )
        """)
        self.conn.commit()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _ensure_vec_table(self, dim: int) -> None:
        if not self._vec_ok or self._dim == dim:
            return
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_images "
            f"USING vec0(embedding float[{dim}])"
        )
        self.conn.commit()
        self._dim = dim

    @staticmethod
    def _to_blob(emb: Any) -> bytes:
        return np.array(emb, dtype=np.float32).tobytes()

    # ── public API ────────────────────────────────────────────────────────────

    def upsert(self, path: str, emb: Any, meta: dict) -> None:
        """Store or replace a photo's embedding + metadata."""
        arr  = np.array(emb, dtype=np.float32)
        dim  = arr.shape[0]
        blob = arr.tobytes()
        h    = get_file_hash(path)

        self.conn.execute(
            "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
            (path, h, dim, blob, json.dumps(meta))
        )

        if self._vec_ok:
            self._ensure_vec_table(dim)
            row = self.conn.execute(
                "SELECT rowid FROM images WHERE path = ?", (path,)
            ).fetchone()
            if row:
                self.conn.execute(
                    "INSERT OR REPLACE INTO vec_images(rowid, embedding) VALUES (?,?)",
                    (row[0], blob)
                )

        self.conn.commit()

    def search(self, query_emb: Any, limit: int = 20) -> list[dict]:
        """
        Return up to `limit` nearest neighbours sorted by distance (ascending).
        Uses sqlite-vec ANN when available, exact cosine otherwise.
        """
        arr  = np.array(query_emb, dtype=np.float32)
        blob = arr.tobytes()

        if self._vec_ok and self._dim is not None:
            rows = self.conn.execute("""
                SELECT i.path, v.distance
                FROM vec_images v
                JOIN images i ON i.rowid = v.rowid
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """, (blob, limit)).fetchall()
        else:
            # Exact cosine fallback — O(n) but fine for <10k images
            raw = self.conn.execute(
                "SELECT path, embedding FROM images"
            ).fetchall()
            norm_q = np.linalg.norm(arr) + 1e-9
            results: list[tuple[str, float]] = []
            for p, eblob in raw:
                e = np.frombuffer(eblob, dtype=np.float32)
                sim = float(np.dot(arr, e) / (norm_q * (np.linalg.norm(e) + 1e-9)))
                results.append((p, 1.0 - sim))          # distance = 1 - cosine
            results.sort(key=lambda x: x[1])
            rows = results[:limit]

        return [{"path": r[0], "distance": float(r[1])} for r in rows]

    def remove(self, path: str) -> None:
        row = self.conn.execute(
            "SELECT rowid FROM images WHERE path = ?", (path,)
        ).fetchone()
        if row and self._vec_ok:
            self.conn.execute("DELETE FROM vec_images WHERE rowid = ?", (row[0],))
        self.conn.execute("DELETE FROM images WHERE path = ?", (path,))
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Incremental folder watcher
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}

from concurrent.futures import ThreadPoolExecutor as _TPE


class IncrementalHandler:
    """
    Watchdog FileSystemEventHandler that:
      - Detects new / modified images (full extension set, case-insensitive)
      - Runs `analyzer._analyze(path, preset)` in a bounded thread pool
        (synchronous call — no asyncio, no new event-loop per file)
      - Persists the result to analyzer.cache + optional LocalVectorDB

    Fixes vs. the naive asyncio.run() pattern:
      • _analyze is sync — no coroutine wrapper needed
      • ThreadPoolExecutor(max_workers=2) prevents thread explosion on big drops
      • Extension check uses .lower() so .JPG / .JPEG are handled correctly
    """

    def __init__(self, analyzer, db: "LocalVectorDB | None" = None,
                 preset: str = "Street - Magnum") -> None:
        self._analyzer = analyzer
        self._db       = db
        self._preset   = preset
        self._pool     = _TPE(max_workers=2, thread_name_prefix="incr_analyze")

    # ── watchdog entry points ─────────────────────────────────────────────────

    def on_created(self, event) -> None:
        self._submit(event)

    def on_modified(self, event) -> None:
        self._submit(event)

    def _submit(self, event) -> None:
        if event.is_directory:
            return
        if Path(event.src_path).suffix.lower() in _IMAGE_EXTS:
            self._pool.submit(self._process, event.src_path)

    # ── background worker ─────────────────────────────────────────────────────

    def _process(self, path: str) -> None:
        try:
            result = self._analyzer._analyze(path, self._preset)
            self._analyzer.cache[path] = result
            self._analyzer._save_cache()
            if self._db is not None:
                emb = result.get("embedding")
                if emb:
                    meta = {k: result.get(k)
                            for k in ("grade", "score", "critique", "breakdown")}
                    self._db.upsert(path, emb, meta)
        except Exception:
            pass   # never crash the background thread

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)


class FolderWatcher:
    """
    High-level watcher: creates an IncrementalHandler per folder and manages
    the watchdog Observer lifecycle.

    Usage:
        watcher = FolderWatcher(analyzer, db)
        watcher.watch("/path/to/photos")
        # ... later ...
        watcher.stop()
    """

    def __init__(self, analyzer=None, db: "LocalVectorDB | None" = None,
                 preset: str = "Street - Magnum",
                 callback=None) -> None:
        """
        If `analyzer` is given, uses IncrementalHandler for full auto-processing.
        If only `callback` is given, fires callback(path, event_kind) instead.
        """
        self._analyzer  = analyzer
        self._db        = db
        self._preset    = preset
        self._callback  = callback
        self._observer  = None
        self._handlers: list[IncrementalHandler] = []
        self._watching: set[str] = set()

    def watch(self, folder: str) -> None:
        folder = str(Path(folder).resolve())
        if folder in self._watching:
            return
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            if self._observer is None:
                self._observer = Observer()
                self._observer.start()

            if self._analyzer is not None:
                handler = IncrementalHandler(self._analyzer, self._db, self._preset)
                self._handlers.append(handler)
                # Wrap in a FileSystemEventHandler bridge so watchdog accepts it
                bridge = _WatchdogBridge(handler)
            else:
                cb = self._callback

                class _SimpleHandler(FileSystemEventHandler):
                    def _fire(self, event, kind):
                        if not event.is_directory and \
                                Path(event.src_path).suffix.lower() in _IMAGE_EXTS:
                            cb(event.src_path, kind)
                    def on_created(self, e):  self._fire(e, "created")
                    def on_modified(self, e): self._fire(e, "modified")

                bridge = _SimpleHandler()

            self._observer.schedule(bridge, folder, recursive=True)
            self._watching.add(folder)
        except ImportError:
            pass

    def stop(self) -> None:
        for h in self._handlers:
            h.shutdown()
        self._handlers.clear()
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        self._watching.clear()


class _WatchdogBridge:
    """Thin adapter so IncrementalHandler works as a watchdog event handler."""
    def __init__(self, handler: IncrementalHandler) -> None:
        self._h = handler
    def dispatch(self, event) -> None:
        self._h.on_created(event) if event.event_type == "created" \
            else self._h.on_modified(event)


# ─────────────────────────────────────────────────────────────────────────────
# 5. XMP / JSON sidecar export
# ─────────────────────────────────────────────────────────────────────────────

def export_metadata(path: str, meta: dict, out_dir: str | None = None) -> str:
    """
    Write grade/score/critique to XMP sidecar (via pyexiv2) if available,
    otherwise write a JSON sidecar next to the original file.

    `meta` keys used:  score, grade, critique, breakdown, nima_score

    Returns the path of the sidecar file written.
    """
    src = Path(path)
    dest_dir = Path(out_dir) if out_dir else src.parent

    # ── Try pyexiv2 first (embeds XMP into a copy / sidecar) ─────────────────
    try:
        import pyexiv2  # type: ignore
        sidecar = dest_dir / (src.stem + ".xmp")
        img = pyexiv2.Image(path)
        rating = max(0, min(5, round(meta.get("score", 0) * 5)))
        xmp_data = {
            "Xmp.xmp.Rating":     str(rating),
            "Xmp.dc.description": meta.get("critique", ""),
            "Xmp.xmp.Label":      meta.get("grade", ""),
            "Xmp.xmp.Nickname":   json.dumps(meta.get("breakdown", {})),
        }
        img.modify_xmp(xmp_data)
        img.close()
        # Write companion .xmp sidecar so Lightroom/Capture One can read it
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(_minimal_xmp(path, xmp_data))
        return str(sidecar)
    except Exception:
        pass

    # ── JSON sidecar fallback ─────────────────────────────────────────────────
    sidecar = dest_dir / (src.stem + ".json")
    payload = {
        "path":       str(src),
        "score":      meta.get("score"),
        "grade":      meta.get("grade"),
        "critique":   meta.get("critique"),
        "breakdown":  meta.get("breakdown", {}),
        "nima_score": meta.get("nima_score"),
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(sidecar)


def _minimal_xmp(image_path: str, xmp: dict) -> str:
    """Minimal XMP packet readable by Lightroom / Capture One."""
    desc_parts = "\n    ".join(
        f'<{k}>{v}</{k}>' for k, v in xmp.items()
    )
    return f"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="{image_path}"
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
    {desc_parts}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""
