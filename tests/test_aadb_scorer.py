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
