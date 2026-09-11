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
"""
from pathlib import Path
import numpy as np

_BANK_PATH = Path(__file__).resolve().parent.parent / "models" / "exemplar_bank.npz"


def score(embeddings: np.ndarray, k: int = 8) -> "np.ndarray | None":
    if not _BANK_PATH.exists():
        return None
    try:
        bank = np.load(_BANK_PATH, allow_pickle=False)["embeddings"].astype(np.float64)
    except Exception as e:
        print(f"[exemplar_scorer] bank unreadable ({e}) — keeping the Exemplar feature unset")
        return None

    embeddings = np.asarray(embeddings, dtype=np.float64)

    # Check embedding dimensionality matches the bank's stored dimensionality
    # (e.g., bank built from 512-d embeddings but invoked with 1024-d
    # embeddings on a different RAM/quality tier).
    if embeddings.shape[-1] != bank.shape[-1]:
        print(f"[exemplar_scorer] bank holds {bank.shape[-1]}d embeddings but this tier emits {embeddings.shape[-1]}d — skipping")
        return None

    bank_norms = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = emb_norms @ bank_norms.T   # (N, bank_size) cosine similarity
    k_eff = min(k, sims.shape[1])
    top_k = np.sort(sims, axis=1)[:, -k_eff:]
    return top_k.mean(axis=1)
