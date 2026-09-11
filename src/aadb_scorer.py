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

Embedding dimensionality alone does not guarantee two embeddings come from
the same encoder/checkpoint — same-dimension-different-space is a real,
silent-corruption risk this project has hit before (see
specvlm_pipeline.probe_fingerprint's docstring). aadb_head.npz therefore
also carries "encoder_source" (siglip2_encoder.ENCODER_SOURCE at train
time); score() refuses a mismatch the same way it refuses a dimension
mismatch. Heads saved before this field existed have no "encoder_source"
key — that's treated as "unknown", not a mismatch, so old test fixtures
and any head trained before this fix still score.
"""
from pathlib import Path
import numpy as np

_HEAD_PATH = Path(__file__).resolve().parent.parent / "models" / "aadb_head.npz"


def score(embeddings: np.ndarray) -> "np.ndarray | None":
    if not _HEAD_PATH.exists():
        return None
    try:
        with np.load(_HEAD_PATH, allow_pickle=False) as d:
            coef, intercept, mean, std = d["coef"], float(d["intercept"]), d["mean"], d["std"]
            trained_source = str(d["encoder_source"]) if "encoder_source" in d.files else None
    except Exception as e:
        print(f"[aadb_scorer] model unreadable ({e}) — keeping the AADB feature unset")
        return None
    embeddings = np.asarray(embeddings, dtype=np.float64)

    # Check embedding dimensionality matches the head's training dimensionality
    # (e.g., head trained on 512-d but invoked with 1024-d embeddings on a different tier)
    if embeddings.shape[-1] != coef.shape[0]:
        print(f"[aadb_scorer] trained on {coef.shape[0]}d embeddings but this tier emits {embeddings.shape[-1]}d — skipping")
        return None

    if trained_source is not None:
        try:
            from siglip2_encoder import ENCODER_SOURCE as _current_source
        except Exception:
            _current_source = None
        if _current_source is not None and trained_source != _current_source:
            print(f"[aadb_scorer] head trained against encoder '{trained_source}' but the "
                  f"active encoder is '{_current_source}' — same dimensionality can still be a "
                  f"different embedding space, skipping")
            return None

    return (embeddings - mean) / std @ coef + intercept
