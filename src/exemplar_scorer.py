"""Strong-exemplar kNN similarity scorer.

Scores SigLIP-2 embeddings by mean cosine similarity to the k nearest
neighbors in a curated Strong-pool bank (Unsplash-derived, see
unsplash_setup.py). Plays to SigLIP's actual strength — image-image
similarity — rather than the project's existing image-text zero-shot probes.

None is returned whenever models/exemplar_bank.npz is absent, unreadable,
or when the embedding dimensionality does not match the bank's stored
dimensionality (e.g., tier mismatch: bank built from 512-d embeddings but
invoked with 1024-d embeddings), matching the same degrade-gracefully
contract as nima_scorer.py and aadb_scorer.py.

Embedding dimensionality alone does not guarantee two embeddings come from
the same encoder/checkpoint — same-dimension-different-space is a real,
silent-corruption risk this project has hit before (see
specvlm_pipeline.probe_fingerprint's docstring). exemplar_bank.npz therefore
also carries "encoder_source" (siglip2_encoder.ENCODER_SOURCE at build
time); score() refuses a mismatch the same way it refuses a dimension
mismatch. Banks saved before this field existed have no "encoder_source"
key — that's treated as "unknown", not a mismatch, so old test fixtures
and any bank built before this fix still score.
"""
from pathlib import Path
import numpy as np

_BANK_PATH = Path(__file__).resolve().parent.parent / "models" / "exemplar_bank.npz"


def score(embeddings: np.ndarray, k: int = 8) -> "np.ndarray | None":
    if not _BANK_PATH.exists():
        return None
    try:
        with np.load(_BANK_PATH, allow_pickle=False) as d:
            bank = d["embeddings"].astype(np.float64)
            trained_source = str(d["encoder_source"]) if "encoder_source" in d.files else None
    except Exception as e:
        print(f"[exemplar_scorer] bank unreadable ({e}) — keeping the Exemplar feature unset")
        return None

    # Empty/malformed bank: bail out before any computation. Without this,
    # k_eff = min(k, 0) = 0 and np.sort(...)[:, -0:] hits Python's -0 == 0
    # slicing pitfall (returns the FULL array instead of an empty one),
    # producing garbage instead of a clean None.
    if bank.ndim != 2 or bank.size == 0:
        print(f"[exemplar_scorer] bank is empty or malformed (shape={bank.shape}) — keeping the Exemplar feature unset")
        return None

    embeddings = np.asarray(embeddings, dtype=np.float64)

    # Check embedding dimensionality matches the bank's stored dimensionality
    # (e.g., bank built from 512-d embeddings but invoked with 1024-d
    # embeddings on a different RAM/quality tier).
    if embeddings.shape[-1] != bank.shape[-1]:
        print(f"[exemplar_scorer] bank holds {bank.shape[-1]}d embeddings but this tier emits {embeddings.shape[-1]}d — skipping")
        return None

    if trained_source is not None:
        try:
            from siglip2_encoder import ENCODER_SOURCE as _current_source
        except Exception:
            _current_source = None
        if _current_source is not None and trained_source != _current_source:
            print(f"[exemplar_scorer] bank built against encoder '{trained_source}' but the "
                  f"active encoder is '{_current_source}' — same dimensionality can still be a "
                  f"different embedding space, skipping")
            return None

    # +1e-9 epsilon guards against a zero-norm row (a plausible degenerate
    # embedding from an unreadable/corrupt source image — see nima_scorer.py
    # and siglip2_encoder.py). Without it, a zero-norm row produces a NaN in
    # its normalized vector; since np.sort places NaN last, a NaN column
    # always survives into the top-k slice, poisoning .mean(axis=1) for
    # EVERY query row, not just the one that touched the degenerate row.
    bank_norms = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-9)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    sims = emb_norms @ bank_norms.T   # (N, bank_size) cosine similarity
    k_eff = min(k, sims.shape[1])
    top_k = np.sort(sims, axis=1)[:, -k_eff:]
    return top_k.mean(axis=1)
