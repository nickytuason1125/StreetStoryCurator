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


def test_apply_exemplar_step_skips_non_finite_value(monkeypatch):
    """Defense in depth: even if the scorer's own guards are ever bypassed
    and it returns a NaN/Inf, the breakdown key must stay unset for that
    photo rather than writing a non-finite value into it."""
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]

    monkeypatch.setattr(
        "exemplar_scorer.score",
        lambda e, k=8: np.array([float("nan"), 0.4], dtype=np.float32),
    )

    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "Exemplar" not in breakdowns[0]
    assert breakdowns[1]["Exemplar"] == 0.4


def test_safe_apply_exemplar_step_swallows_exception(monkeypatch, capsys):
    """The call-site wrapper used by run_v2 must degrade to 'Exemplar stays
    unset' instead of propagating an unexpected exception from the scorer."""
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]

    def _boom(e, k=8):
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr("exemplar_scorer.score", _boom)

    gp._safe_apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "Exemplar" not in breakdowns[0]
    assert "Exemplar" not in breakdowns[1]
    assert "Exemplar step failed" in capsys.readouterr().out
