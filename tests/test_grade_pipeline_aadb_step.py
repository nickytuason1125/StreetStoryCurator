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


def test_apply_aadb_step_skips_non_finite_value(monkeypatch):
    """Defense in depth: even if the scorer's own guards are ever bypassed
    and it returns a NaN/Inf, the breakdown key must stay unset for that
    photo rather than writing a non-finite value into it."""
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]

    monkeypatch.setattr(
        "aadb_scorer.score",
        lambda e: np.array([float("nan"), 0.4], dtype=np.float32),
    )

    gp._apply_aadb_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert breakdowns[1]["AADB"] == 0.4


def test_safe_apply_aadb_step_swallows_exception(monkeypatch, capsys):
    """The call-site wrapper used by run_v2 must degrade to 'AADB stays
    unset' instead of propagating an unexpected exception from the scorer."""
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]

    def _boom(e):
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr("aadb_scorer.score", _boom)

    gp._safe_apply_aadb_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert "AADB" not in breakdowns[1]
    assert "AADB step failed" in capsys.readouterr().out
