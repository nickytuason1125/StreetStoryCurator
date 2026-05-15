"""
LanceDB Migration Utilities

Migrates embeddings from legacy storage (SQLite/FAISS) to LanceDB
with IVF-PQ indexing for efficient vector search.

Schema:
    path          string          (primary key, absolute path)
    embedding     fixed_size_list<float32>[1536]   SigLIP-2 ViT-g/14
    score         float32         SpecVLM quality score
    reasoning_log string          VLM reasoning log
    grade         string          "Strong ✅" | "Mid ⚠️" | "Weak ❌"
    confidence    float32         Draft confidence score
    exif_ts       float64         Unix timestamp from EXIF
"""

from __future__ import annotations

import json
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np

# LanceDB imports
try:
    import lancedb
    import pyarrow as pa
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False


# ── Configuration ──────────────────────────────────────────────────────────────


DB_DIR = Path("cache/lancedb_v2")
DB_DIR.mkdir(parents=True, exist_ok=True)

TBL_NAME = "photos_v2"
EMBED_DIM = 1536  # SigLIP-2 ViT-g/14 dimension


# ── LanceDB Connection ─────────────────────────────────────────────────────────


def open_table() -> Optional[Any]:
    """Open or create LanceDB table."""
    if not HAS_LANCEDB:
        return None
    
    db = lancedb.connect(str(DB_DIR))
    
    schema = pa.schema([
        pa.field("path", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), EMBED_DIM)),
        pa.field("score", pa.float32()),
        pa.field("reasoning_log", pa.string()),
        pa.field("grade", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("exif_ts", pa.float64()),
        pa.field("is_verified", pa.bool_()),
        pa.field("breakdown", pa.string()),  # JSON blob
    ])
    
    if TBL_NAME in db.table_names():
        return db.open_table(TBL_NAME)
    else:
        return db.create_table(TBL_NAME, schema=schema)


# ── Migration Utilities ────────────────────────────────────────────────────────


def migrate_from_sqlite(
    sqlite_path: str,
    image_dir: Optional[str] = None,
    batch_size: int = 1000,
    progress=None,
) -> int:
    """
    Migrate embeddings from SQLite to LanceDB.
    
    Args:
        sqlite_path: Path to SQLite database
        image_dir: Optional directory to scan for images
        batch_size: Batch size for migration
        progress: Optional progress callback
    
    Returns:
        Number of records migrated
    """
    if not HAS_LANCEDB:
        print("LanceDB not available - migration skipped")
        return 0
    
    # Import SQLite connector
    import sqlite3
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    # Query existing embeddings
    cursor.execute("""
        SELECT path, embedding, score, grade, exif_ts
        FROM embeddings
        ORDER BY path
    """)
    
    records = []
    total = 0
    
    for row in cursor.fetchall():
        path, embedding_blob, score, grade, exif_ts = row
        
        # Convert embedding blob to numpy array
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        # Create record
        record = {
            "path": path,
            "embedding": embedding.tolist(),
            "score": float(score) if score else 0.5,
            "reasoning_log": "",
            "grade": grade or "Mid ⚠️",
            "confidence": 0.5,
            "exif_ts": float(exif_ts) if exif_ts else 0.0,
            "is_verified": False,
            "breakdown": json.dumps({}),
        }
        
        records.append(record)
        
        if len(records) >= batch_size:
            _insert_batch(records)
            total += len(records)
            records = []
            
            if progress:
                progress(total, "Migrating embeddings...")
    
    # Insert remaining records
    if records:
        _insert_batch(records)
        total += len(records)
    
    conn.close()
    
    # Create IVF-PQ index
    create_ivf_pq_index()
    
    if progress:
        progress(1.0, f"Migrated {total} records")
    
    return total


def migrate_from_faiss(
    faiss_index_path: str,
    metadata_path: str,
    progress=None,
) -> int:
    """
    Migrate embeddings from FAISS to LanceDB.
    
    Args:
        faiss_index_path: Path to FAISS index (.index file)
        metadata_path: Path to metadata JSON
        progress: Optional progress callback
    
    Returns:
        Number of records migrated
    """
    if not HAS_LANCEDB:
        print("LanceDB not available - migration skipped")
        return 0
    
    import faiss
    
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    embeddings = index.reconstruct_n(0, index.ntotal)
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Create records
    records = []
    for i, path in enumerate(metadata.get("paths", [])):
        record = {
            "path": path,
            "embedding": embeddings[i].tolist(),
            "score": metadata.get("scores", [0.5] * len(metadata["paths"]))[i],
            "reasoning_log": "",
            "grade": metadata.get("grades", ["Mid ⚠️"] * len(metadata["paths"]))[i],
            "confidence": 0.5,
            "exif_ts": metadata.get("exif_ts", [0.0] * len(metadata["paths"]))[i],
            "is_verified": False,
            "breakdown": json.dumps({}),
        }
        records.append(record)
    
    # Insert records
    _insert_batch(records)
    
    # Create IVF-PQ index
    create_ivf_pq_index()
    
    if progress:
        progress(1.0, f"Migrated {len(records)} records")
    
    return len(records)


def _insert_batch(records: List[Dict[str, Any]]) -> None:
    """Insert a batch of records into LanceDB."""
    if not HAS_LANCEDB or not records:
        return
    
    table = open_table()
    if table is None:
        return
    
    # Convert to PyArrow table
    data = {
        "path": [r["path"] for r in records],
        "embedding": [r["embedding"] for r in records],
        "score": [r["score"] for r in records],
        "reasoning_log": [r["reasoning_log"] for r in records],
        "grade": [r["grade"] for r in records],
        "confidence": [r["confidence"] for r in records],
        "exif_ts": [r["exif_ts"] for r in records],
        "is_verified": [r["is_verified"] for r in records],
        "breakdown": [r["breakdown"] for r in records],
    }
    
    table.add(pa.table(data))


# ── Index Management ───────────────────────────────────────────────────────────


def create_ivf_pq_index(
    num_partitions: int = 16,
    num_sub_vectors: int = 96,
    metric: str = "cosine",
) -> None:
    """
    Create IVF-PQ index for efficient vector search.
    
    Args:
        num_partitions: Number of IVF partitions
        num_sub_vectors: Number of PQ sub-vectors
        metric: Distance metric ("cosine" or "l2")
    """
    if not HAS_LANCEDB:
        return
    
    table = open_table()
    if table is None:
        return
    
    try:
        table.create_index(
            column="embedding",
            index_type="IVF_PQ",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            metric=metric,
            replace=True,
        )
        print(f"Created IVF-PQ index: {num_partitions} partitions, "
              f"{num_sub_vectors} sub-vectors, {metric} metric")
    except Exception as e:
        print(f"Failed to create index: {e}")


def drop_index() -> None:
    """Drop existing index."""
    if not HAS_LANCEDB:
        return
    
    table = open_table()
    if table is None:
        return
    
    try:
        table.drop_index("embedding")
        print("Dropped existing index")
    except Exception as e:
        print(f"Failed to drop index: {e}")


# ── Search Functions ───────────────────────────────────────────────────────────


def search_by_embedding(
    embedding: np.ndarray,
    k: int = 10,
    nprobes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for similar embeddings using IVF-PQ index.
    
    Args:
        embedding: Query embedding (1536-d)
        k: Number of results
        nprobes: Number of IVF partitions to probe
    
    Returns:
        List of matching records
    """
    if not HAS_LANCEDB:
        return []
    
    table = open_table()
    if table is None:
        return []
    
    # Convert to list
    embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    # Search with IVF-PQ
    results = (
        table.search(embedding_list)
        .metric("cosine")
        .nprobes(nprobes)
        .limit(k)
        .to_list()
    )
    
    return results


def search_by_text(
    query: str,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search by text query (future: with text encoder).
    
    Args:
        query: Text query
        k: Number of results
    
    Returns:
        List of matching records
    """
    # Placeholder - would use text encoder to convert query to embedding
    return []


# ── High-Level API ─────────────────────────────────────────────────────────────


def migrate_all(
    sqlite_path: Optional[str] = None,
    faiss_index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    progress=None,
) -> Dict[str, int]:
    """
    Migrate all legacy embeddings to LanceDB.
    
    Args:
        sqlite_path: Path to SQLite database
        faiss_index_path: Path to FAISS index
        metadata_path: Path to metadata JSON
        progress: Optional progress callback
    
    Returns:
        Dict with migration counts
    """
    results = {}
    
    if sqlite_path:
        results["sqlite"] = migrate_from_sqlite(sqlite_path, progress=progress)
    
    if faiss_index_path and metadata_path:
        results["faiss"] = migrate_from_faiss(
            faiss_index_path, metadata_path, progress=progress
        )
    
    return results


def get_table_stats() -> Dict[str, Any]:
    """Get table statistics."""
    if not HAS_LANCEDB:
        return {"error": "LanceDB not available"}
    
    table = open_table()
    if table is None:
        return {"error": "Table not found"}
    
    return {
        "num_rows": table.count_rows(),
        "num_vectors": table.count_rows(),
        "embedding_dim": EMBED_DIM,
    }
