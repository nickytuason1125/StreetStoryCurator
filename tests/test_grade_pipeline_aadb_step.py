import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_apply_aadb_step_writes_breakdown_key(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    to_rate_indices = [0, 1]
    breakdowns = [{}, {}]

    monkeypatch.setattr(
        "aadb_scorer.score",
        lambda e: np.array([0.7, 0.3], dtype=np.float32),
    )

    gp._apply_aadb_step(embs, to_rate_indices, breakdowns)

    assert breakdowns[0]["AADB"] == 0.7
    assert breakdowns[1]["AADB"] == 0.3


def test_apply_aadb_step_no_op_when_model_absent(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.zeros((2, 2), dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("aadb_scorer.score", lambda e: None)

    gp._apply_aadb_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert "AADB" not in breakdowns[1]
