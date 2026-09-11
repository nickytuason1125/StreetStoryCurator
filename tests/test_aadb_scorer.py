import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_score_returns_none_when_model_absent(tmp_path, monkeypatch):
    import aadb_scorer
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", tmp_path / "missing.npz")
    result = aadb_scorer.score(np.zeros((3, 4), dtype=np.float32))
    assert result is None


def test_score_matches_manual_ridge_predict(tmp_path, monkeypatch):
    import aadb_scorer

    coef = np.array([0.5, -0.25], dtype=np.float64)
    intercept = 0.1
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    head_path = tmp_path / "aadb_head.npz"
    np.savez(head_path, coef=coef, intercept=intercept, mean=mean, std=std)
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", head_path)

    embs = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    result = aadb_scorer.score(embs)

    expected = (embs - mean) / std @ coef + intercept
    np.testing.assert_allclose(result, expected)


def test_score_returns_none_on_embedding_dim_mismatch(tmp_path, monkeypatch):
    """Head trained on 8-d embeddings, called with 4-d embeddings (tier mismatch)."""
    import aadb_scorer

    # Head trained on 8-dimensional embeddings
    coef = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    intercept = 0.05
    mean = np.zeros(8)
    std = np.ones(8)
    head_path = tmp_path / "aadb_head.npz"
    np.savez(head_path, coef=coef, intercept=intercept, mean=mean, std=std)
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", head_path)

    # Called with 4-dimensional embeddings (different tier)
    embs = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    result = aadb_scorer.score(embs)

    # Should return None, not raise ValueError
    assert result is None


def test_score_returns_none_on_encoder_source_mismatch(tmp_path, monkeypatch):
    """Same dimensionality can still be a different embedding space — the
    head must refuse to score against an encoder it wasn't trained on."""
    import aadb_scorer

    coef = np.array([0.5, -0.25], dtype=np.float64)
    intercept = 0.1
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    head_path = tmp_path / "aadb_head.npz"
    np.savez(head_path, coef=coef, intercept=intercept, mean=mean, std=std,
             encoder_source="openclip-high-ViT-gopt-16-SigLIP2-384", embed_dim=2)
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", head_path)
    monkeypatch.setattr("siglip2_encoder.ENCODER_SOURCE", "hf-onnx-mid-ViT-L-16-SigLIP2-384")

    embs = np.array([[1.0, 2.0]], dtype=np.float64)
    result = aadb_scorer.score(embs)

    assert result is None


def test_score_still_works_when_encoder_source_matches(tmp_path, monkeypatch):
    import aadb_scorer

    coef = np.array([0.5, -0.25], dtype=np.float64)
    intercept = 0.1
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    head_path = tmp_path / "aadb_head.npz"
    np.savez(head_path, coef=coef, intercept=intercept, mean=mean, std=std,
             encoder_source="openclip-high-ViT-gopt-16-SigLIP2-384", embed_dim=2)
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", head_path)
    monkeypatch.setattr("siglip2_encoder.ENCODER_SOURCE", "openclip-high-ViT-gopt-16-SigLIP2-384")

    embs = np.array([[1.0, 2.0]], dtype=np.float64)
    result = aadb_scorer.score(embs)

    expected = (embs - mean) / std @ coef + intercept
    np.testing.assert_allclose(result, expected)
