import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gate_fails_when_no_aadb_metrics(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics", lambda: None)

    passes, reason = pmj.check_aadb_gate()
    assert passes is False
    assert "no trained aadb head" in reason.lower()


def test_gate_fails_below_chance_floor(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics",
                         lambda: {"rho_holdout": 0.05, "n_train": 8000, "n_holdout": 2000})

    passes, reason = pmj.check_aadb_gate(min_rho=0.149)
    assert passes is False
    assert "0.05" in reason


def test_gate_passes_above_chance_floor(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics",
                         lambda: {"rho_holdout": 0.42, "n_train": 8000, "n_holdout": 2000})

    passes, reason = pmj.check_aadb_gate(min_rho=0.149)
    assert passes is True
