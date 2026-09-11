import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_score_returns_none_when_bank_absent(tmp_path, monkeypatch):
    import exemplar_scorer
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", tmp_path / "missing.npz")
    assert exemplar_scorer.score(np.zeros((2, 4))) is None


def test_score_ranks_closer_vector_higher(tmp_path, monkeypatch):
    import exemplar_scorer

    bank = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bank_path = tmp_path / "exemplar_bank.npz"
    np.savez(bank_path, embeddings=bank)
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", bank_path)

    close_to_bank = np.array([[0.9, 0.1]], dtype=np.float32)   # near [1,0] cluster
    far_from_bank = np.array([[-1.0, -1.0]], dtype=np.float32)  # near neither

    close_score = exemplar_scorer.score(close_to_bank, k=2)[0]
    far_score = exemplar_scorer.score(far_from_bank, k=2)[0]

    assert close_score > far_score


def test_strong_pool_scores_higher_than_weak_pool_against_own_bank(tmp_path, monkeypatch):
    """Sanity check from the spec: the Strong pool must score higher
    self-similarity against the Strong bank than the Weak pool does."""
    import exemplar_scorer

    strong_bank = np.array([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]], dtype=np.float32)
    bank_path = tmp_path / "exemplar_bank.npz"
    np.savez(bank_path, embeddings=strong_bank)
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", bank_path)

    weak_pool = np.array([[0.0, 1.0], [-0.1, 0.9]], dtype=np.float32)
    strong_pool_holdout = np.array([[0.92, 0.08], [0.97, 0.03]], dtype=np.float32)

    weak_scores = exemplar_scorer.score(weak_pool, k=2)
    strong_scores = exemplar_scorer.score(strong_pool_holdout, k=2)

    assert strong_scores.mean() > weak_scores.mean()


def test_score_returns_none_on_embedding_dim_mismatch(tmp_path, monkeypatch):
    """Bank saved with 2-d embeddings, called with 4-d embeddings (tier mismatch)."""
    import exemplar_scorer

    bank = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    bank_path = tmp_path / "exemplar_bank.npz"
    np.savez(bank_path, embeddings=bank)
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", bank_path)

    embs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    result = exemplar_scorer.score(embs)

    assert result is None
