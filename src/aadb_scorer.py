"""AADB-trained aesthetic head scorer.

Loads the tiny ridge model produced by aadb_setup.py and scores SigLIP-2
embeddings already resident in the grade worker's memory — no extra encode
pass, unlike NIMA which needs its own image decode/forward pass.

None is returned whenever models/aadb_head.npz is absent or unreadable,
or when the embedding dimensionality does not match the head's training
dimensionality (e.g., tier mismatch: head trained on 512-d but invoked
with 1024-d embeddings). Callers must degrade to leaving the "AADB"
breakdown key unset, never to a fabricated 0.0 (see master_judge.
feature_vector's NaN-not-zero contract).
"""
from pathlib import Path
import numpy as np

_HEAD_PATH = Path(__file__).resolve().parent.parent / "models" / "aadb_head.npz"


def score(embeddings: np.ndarray) -> "np.ndarray | None":
    if not _HEAD_PATH.exists():
        return None
    try:
        d = np.load(_HEAD_PATH, allow_pickle=False)
        coef, intercept, mean, std = d["coef"], float(d["intercept"]), d["mean"], d["std"]
    except Exception as e:
        print(f"[aadb_scorer] model unreadable ({e}) — keeping the AADB feature unset")
        return None
    embeddings = np.asarray(embeddings, dtype=np.float64)

    # Check embedding dimensionality matches the head's training dimensionality
    # (e.g., head trained on 512-d but invoked with 1024-d embeddings on a different tier)
    if embeddings.shape[-1] != coef.shape[0]:
        print(f"[aadb_scorer] trained on {coef.shape[0]}d embeddings but this tier emits {embeddings.shape[-1]}d — skipping")
        return None

    return (embeddings - mean) / std @ coef + intercept
