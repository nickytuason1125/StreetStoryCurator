# tests/test_grade_pipeline_exemplar_step.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_apply_exemplar_step_writes_breakdown_key(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: np.array([0.6, 0.2], dtype=np.float32))

    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert breakdowns[0]["Exemplar"] == 0.6
    assert breakdowns[1]["Exemplar"] == 0.2


def test_apply_exemplar_step_no_op_when_bank_absent(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.zeros((2, 2), dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: None)

    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "Exemplar" not in breakdowns[0]
    assert "Exemplar" not in breakdowns[1]
