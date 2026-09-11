"""One-time Unsplash exemplar-bank builder.

Pulls a curated Strong/Weak split from Street Photography / Documentary
collections via the Unsplash API (https://unsplash.com/developers) — an
access key is required; set UNSPLASH_ACCESS_KEY, and the collection(s) to
pull from via UNSPLASH_COLLECTION_IDS (comma-separated Unsplash collection
IDs — find these by browsing unsplash.com/collections and reading the ID
out of the collection's URL). Only derived embeddings are ever written to
this repo (models/exemplar_bank.npz); raw images are downloaded to a scratch
folder outside the repo and never committed, matching the same boundary
already drawn for AADB and the RAG PDFs.

Usage:
    set UNSPLASH_ACCESS_KEY=your_key_here
    set UNSPLASH_COLLECTION_IDS=317099,1114848
    venv\\Scripts\\python.exe unsplash_setup.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

_BANK_OUT = _ROOT / "models" / "exemplar_bank.npz"
_SCRATCH_DIR = _ROOT / "cache" / "unsplash_scratch"


def fetch_collection_photos(collection_id: str, access_key: str,
                            scratch_dir: Path, max_photos: int = 500) -> list:
    """Page through an Unsplash collection's photos (30/page, the API max),
    download each original to scratch_dir, and return
    [{"path": local_jpg_path, "likes": int}, ...]. Stops at max_photos or
    when the collection is exhausted, whichever comes first."""
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    photos = []
    page = 1
    while len(photos) < max_photos:
        url = (f"https://api.unsplash.com/collections/{collection_id}/photos"
               f"?page={page}&per_page=30&client_id={access_key}")
        req = urllib.request.Request(url, headers={"Accept-Version": "v1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        for item in batch:
            if len(photos) >= max_photos:
                break
            img_url = item["urls"]["regular"]
            local_path = scratch_dir / f"{item['id']}.jpg"
            if not local_path.exists():
                urllib.request.urlretrieve(img_url, str(local_path))
            photos.append({"path": str(local_path), "likes": int(item.get("likes", 0))})
        page += 1
    return photos


def split_by_engagement(photos: list, frac: float = 0.25) -> tuple:
    """photos: [{"path": str, "likes": int}, ...] all drawn from the SAME
    curated street/documentary collections (never mix in an unrelated pool —
    a weak exemplar must still be on-topic, just worse, see the spec).
    Returns (strong_paths, weak_paths): top/bottom frac by "likes"."""
    ordered = sorted(photos, key=lambda p: p["likes"], reverse=True)
    n = max(1, int(round(len(ordered) * frac)))
    strong = [p["path"] for p in ordered[:n]]
    weak = [p["path"] for p in ordered[-n:]]
    return strong, weak


if __name__ == "__main__":
    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    collection_ids = os.environ.get("UNSPLASH_COLLECTION_IDS", "")
    if not key or not collection_ids:
        print("Set UNSPLASH_ACCESS_KEY and UNSPLASH_COLLECTION_IDS first. "
              "See this file's docstring.")
        sys.exit(1)

    all_photos = []
    for cid in collection_ids.split(","):
        all_photos.extend(fetch_collection_photos(cid.strip(), key, _SCRATCH_DIR))
    print(f"[unsplash_setup] fetched {len(all_photos)} photos across "
          f"{len(collection_ids.split(','))} collection(s)")

    strong_paths, weak_paths = split_by_engagement(all_photos)
    print(f"[unsplash_setup] {len(strong_paths)} strong / {len(weak_paths)} weak "
          f"(top/bottom quartile by engagement)")
    # Task 9b continues this script's __main__ block with encode + save + validate.
