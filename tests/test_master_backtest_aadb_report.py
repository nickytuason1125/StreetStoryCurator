import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_read_aadb_metrics_returns_none_when_absent(tmp_path, monkeypatch):
    import importlib
    mb = importlib.import_module("scripts.master_backtest")
    monkeypatch.setattr(mb, "_AADB_METRICS_PATH", tmp_path / "missing.json")
    assert mb.read_aadb_metrics() is None


def test_aadb_metrics_path_is_under_models_not_cache():
    """cache/ is this project's ephemeral/derived-data convention and gets
    cleared routinely — the durable AADB metrics file must live in models/,
    next to the durable aadb_head.npz it describes, or clearing cache/ makes
    promote_master_judge's gate falsely report 'no trained head found'."""
    import importlib
    mb = importlib.import_module("scripts.master_backtest")
    assert mb._AADB_METRICS_PATH.parent.name == "models"


def test_read_aadb_metrics_parses_existing_file(tmp_path, monkeypatch):
    import importlib
    mb = importlib.import_module("scripts.master_backtest")
    metrics_path = tmp_path / "aadb_head_metrics.json"
    metrics_path.write_text(json.dumps({"rho_holdout": 0.42, "n_train": 8000, "n_holdout": 2000}), encoding="utf-8")
    monkeypatch.setattr(mb, "_AADB_METRICS_PATH", metrics_path)

    result = mb.read_aadb_metrics()
    assert result["rho_holdout"] == 0.42
    assert result["n_holdout"] == 2000
