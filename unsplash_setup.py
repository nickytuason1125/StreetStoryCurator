"""One-time Unsplash exemplar-bank builder.

Pulls a curated Strong/Weak split from Street Photography / Documentary
collections via the Unsplash API (https://unsplash.com/developers) — an
access key is required; set UNSPLASH_ACCESS_KEY, and the collection(s) to
pull from via UNSPLASH_COLLECTION_IDS (comma-separated Unsplash collection
IDs — find these by browsing unsplash.com/collections and reading the ID
out of the collection's URL). Only derived embeddings are ever written to
this repo (models/exemplar_bank.npz); raw images are downloaded to a scratch
folder (cache/unsplash_scratch/) that is gitignored and never committed,
matching the same boundary already drawn for AADB and the RAG PDFs.

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

import dataset_embed  # noqa: E402  (after sys.path insert above)

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
               f"?page={page}&per_page=30")
        req = urllib.request.Request(url, headers={
            "Accept-Version": "v1",
            "Authorization": f"Client-ID {access_key}",
        })
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
    Returns (strong_paths, weak_paths): top/bottom frac by "likes". Strong and
    Weak are always disjoint, regardless of pool size or frac — n is clamped
    to at most len(ordered) // 2 so the top-n and bottom-n slices can never
    share an index. On a pool too small to split at all (0 or 1 photos) both
    sides come back empty rather than overlapping."""
    ordered = sorted(photos, key=lambda p: p["likes"], reverse=True)
    half = len(ordered) // 2
    n = max(1, min(int(round(len(ordered) * frac)), half)) if half > 0 else 0
    strong = [p["path"] for p in ordered[:n]]
    weak = [p["path"] for p in ordered[-n:]] if n > 0 else []
    return strong, weak


def build_exemplar_bank(strong_paths: list, weak_paths: list, out_npz: "Path | str") -> dict:
    """Encode the Strong pool, save it as the exemplar bank, and validate
    that Strong self-similarity beats Weak self-similarity against that
    bank — the sanity check the spec requires before trusting this feature.
    Does NOT save the Weak pool; it exists only for this validation."""
    import numpy as np
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    _SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # encode_folder always writes to the path it's given; that write is
    # throwaway (an implementation detail of that helper), so point it at
    # _SCRATCH_DIR — the directory already used for this exact purpose
    # elsewhere in this file — rather than models/, and do the one real save
    # (with path provenance preserved) ourselves below.
    strong_scratch = _SCRATCH_DIR / "unsplash_strong_scratch.npz"
    strong_embs = dataset_embed.encode_folder(strong_paths, strong_scratch)
    weak_scratch = _SCRATCH_DIR / "unsplash_weak_scratch.npz"
    if weak_paths:
        weak_embs = dataset_embed.encode_folder(weak_paths, weak_scratch)
    else:
        weak_embs = np.empty((0, strong_embs.shape[1]), dtype=strong_embs.dtype)

    # Embedding dimensionality alone doesn't guarantee two embeddings come
    # from the same encoder/checkpoint (see specvlm_pipeline.probe_
    # fingerprint's docstring) — stamp the bank with the same cheap encoder
    # identity string grade_pipeline_v2's source-change guard already uses,
    # so exemplar_scorer.score() can refuse a same-dimension-different-space
    # bank instead of silently scoring against the wrong space.
    from siglip2_encoder import ENCODER_SOURCE, EMBED_DIM
    np.savez(out_npz, paths=np.array(strong_paths), embeddings=strong_embs,
              encoder_source=ENCODER_SOURCE, embed_dim=EMBED_DIM)

    def _mean_self_sim(query_embs, bank_embs):
        if len(query_embs) == 0:
            return float("nan")
        # +1e-9 epsilon guards against a zero-norm row (e.g. a corrupt source
        # image producing a degenerate embedding) — matches the convention
        # used elsewhere in this codebase (see exemplar_scorer.py).
        bank_norm = bank_embs / (np.linalg.norm(bank_embs, axis=1, keepdims=True) + 1e-9)
        q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-9)
        return float((q_norm @ bank_norm.T).mean())

    return {
        "strong_self_sim": _mean_self_sim(strong_embs, strong_embs),
        "weak_self_sim": _mean_self_sim(weak_embs, strong_embs),
    }


def _bank_is_valid(result: dict) -> bool:
    """True iff Strong scores strictly higher than Weak against the Strong
    bank. Written as `strong > weak` (not `not (strong <= weak)`) so that a
    NaN on either side — e.g. an empty Weak pool, which split_by_engagement
    can return for pathologically tiny collections — is never mistaken for
    a pass: any comparison with NaN is False in IEEE 754, so `strong > weak`
    is already False when either side is NaN, and this function correctly
    reports "not valid" instead of silently treating undefined as valid."""
    return result["strong_self_sim"] > result["weak_self_sim"]


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

    result = build_exemplar_bank(strong_paths, weak_paths, _BANK_OUT)
    print(f"[unsplash_setup] strong self-sim={result['strong_self_sim']:.4f} "
          f"weak self-sim={result['weak_self_sim']:.4f}")
    if not _bank_is_valid(result):
        print("[unsplash_setup] REFUSING to keep this bank — the Strong pool "
              "does not score higher than the Weak pool against its own "
              "bank, which means the engagement split isn't capturing a real "
              "quality signal for this collection.")
        _BANK_OUT.unlink(missing_ok=True)
        sys.exit(1)
    print(f"[unsplash_setup] saved {_BANK_OUT}")
