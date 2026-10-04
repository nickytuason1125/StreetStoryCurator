"""The learning loop: a local judge that PROVED an agreement gain on held-out
shoots grades without any flag; a lost challenge never changes grading."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import master_judge as mj  # noqa: E402


def _record(promoted, ci_lo=0.03):
    feats = list(mj.DESIGN)
    return {"promoted": promoted, "features": feats, "coef": [0.0] * len(feats),
            "intercept": 0.5, "mean": [0.0] * len(feats), "std": [1.0] * len(feats),
            "feature_fingerprint": mj._feature_fingerprint(feats), "rho_holdout": 0.5,
            "rho_baseline": 0.4, "agree_judge": 0.7, "agree_base": 0.56,
            "agree_ci": [ci_lo, 0.2], "n": 200}


def test_promoted_local_judge_is_used_without_any_flag(tmp_path, monkeypatch):
    p = tmp_path / "j.json"
    p.write_text(json.dumps(_record(True)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE_OFF", raising=False)
    judge, w = mj.active()
    assert judge is not None and judge.get("_source") == "cache" and w > 0


def test_lost_challenge_never_changes_grading(tmp_path, monkeypatch):
    p = tmp_path / "j.json"
    p.write_text(json.dumps(_record(False)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    shipped = mj._load_shipped()
    judge, _ = mj.active()
    assert (judge or {}).get("coef") == (shipped or {}).get("coef")


def test_old_record_without_agreement_proof_still_needs_opt_in(tmp_path, monkeypatch):
    rec = _record(True)
    rec.pop("agree_ci")
    p = tmp_path / "j.json"
    p.write_text(json.dumps(rec))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    judge, _ = mj.active()
    assert (judge or {}).get("_source") != "cache"


def test_kill_switch_still_wins(tmp_path, monkeypatch):
    p = tmp_path / "j.json"
    p.write_text(json.dumps(_record(True)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE_OFF", "1")
    assert mj.active() == (None, 0.0)


def test_autofit_runs_without_opt_in_flag(monkeypatch, tmp_path):
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", tmp_path / "none.json")
    calls = []
    monkeypatch.setattr(mj, "fit", lambda: calls.append(1) or {"promoted": False})
    out = mj.maybe_autofit(n_now=1000)
    if mj._autofit_thread is not None:
        mj._autofit_thread.join(5)
    assert out["triggered"] is True and calls == [1]


def _grading(monkeypatch, tmp_path):
    import ratings_store as rs
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server_impl  # noqa: F401  (circular with routers.grading — load first)
    from routers import grading
    import lance_store
    monkeypatch.setattr(grading, "_CATALOG_PATH", tmp_path / "catalog.json")
    monkeypatch.setattr(lance_store, "query_by_paths", lambda paths: [])
    return grading


def test_rating_a_photo_checks_for_a_refit(tmp_path, monkeypatch):
    import asyncio
    grading = _grading(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(mj, "maybe_autofit", lambda n_now=None: calls.append(1) or {"triggered": False})
    asyncio.run(grading.personal_star({"path": "C:/x/DSC1.ARW", "stars": 4}))
    assert calls == [1]


def test_accuracy_endpoint_reports_the_proven_number(tmp_path, monkeypatch):
    import asyncio
    grading = _grading(monkeypatch, tmp_path)
    p = tmp_path / "j.json"
    p.write_text(json.dumps(_record(True)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    out = asyncio.run(grading.accuracy_status())
    assert out["agree"] == 0.7 and out["agree_base"] == 0.56 and out["promoted"] is True
    p.write_text(json.dumps(_record(False)))
    out = asyncio.run(grading.accuracy_status())
    assert out["agree"] == 0.56 and out["promoted"] is False


def test_a_losing_refit_keeps_the_promoted_judge(tmp_path):
    """Review finding: an unpromoted refit used to OVERWRITE the promoted
    record, so grading silently fell back to the shipped judge."""
    p = tmp_path / "j.json"
    p.write_text(json.dumps(_record(True)))
    rng = __import__("numpy").random.default_rng(0)
    rows = []
    for i in range(150):
        bd = {k: 0.5 for k in mj.FEATURES}
        bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}
        bd.update({"_subject_sharp": 0.7, "_subject_streak": 0.0, "_living": 1.0})
        rows.append({"path": f"C:/s{i % 6}/{i}.ARW", "stars": int(rng.integers(1, 6)),
                     "score": float(rng.random()), "breakdown": bd})
    out = mj.fit_from_rows(rows, weights_path=p)
    assert out["promoted"] is False
    kept = mj.load(p)
    assert kept is not None and kept["agree_judge"] == 0.7      # the old winner still grades
    assert json.loads(p.read_text())["history"][-1]["promoted"] is False
