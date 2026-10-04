"""Ratings carry their grade-time features, so retraining survives cleared caches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ratings_store as rs  # noqa: E402


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")


def test_attach_features_keeps_stars_and_source(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    rs.set_rating("C:/a.jpg", 2, source="tpe_master")
    before = rs._read_raw()["C:/a.jpg"]["rated_at"]
    assert rs.attach_features("C:/a.jpg", {"Technical": 0.5, "_grade_sig": "x",
                                           "_arch_w": {"geo": 1.0}, "label": "boat"}, score=0.44)
    e = rs._read_raw()["C:/a.jpg"]
    assert e["stars"] == 2 and e["source"] == "tpe_master" and e["rated_at"] == before
    # The rating-time score is a historical record the grade thresholds are
    # fitted from — a re-grade must NOT overwrite it (doing so moved the
    # Strong line from 0.58 to 0.52 on 2026-10-04). Fresh scores go beside it.
    assert "score" not in e and e["score_current"] == 0.44
    assert rs.get_features("C:/a.jpg") == {"Technical": 0.5, "_arch_w": {"geo": 1.0}}


def test_attach_features_refuses_unrated(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert rs.attach_features("C:/none.jpg", {"Technical": 0.5}) is False
    assert "C:/none.jpg" not in rs._read_raw()


def test_rows_without_features_are_skipped_not_zeroed(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    import master_judge as mj
    import lance_store
    rs.set_rating("C:/old.jpg", 4)                       # legacy: no score, no features
    rs.set_rating("C:/new.jpg", 2, score=0.5)
    rs.attach_features("C:/new.jpg", {k: 0.5 for k in mj.FEATURES} | {"_arch_w": {a: 0.2 for a in mj.ARCHES}})
    monkeypatch.setattr(lance_store, "query_all", lambda min_score=0.0: [])
    rows = mj.collect_rows()
    assert [r["path"] for r in rows] == ["C:/new.jpg"]
    assert rows[0]["breakdown"]["Technical"] == 0.5


def test_star_endpoint_snapshots_features(tmp_path, monkeypatch):
    """Rating a photo stores its grade-time features on the rating itself."""
    import asyncio
    _isolate(tmp_path, monkeypatch)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server_impl  # noqa: F401  (circular with routers.grading — load first)
    from routers import grading
    import lance_store
    monkeypatch.setattr(grading, "_CATALOG_PATH", tmp_path / "catalog.json")
    bd = {"Technical": 0.61, "_subject_sharp": 0.7, "_pan": True, "_grade_sig": "v|full"}
    monkeypatch.setattr(lance_store, "query_by_paths",
                        lambda paths: [{"path": paths[0], "score": 0.55, "personal_score": None,
                                        "breakdown": bd}])
    asyncio.run(grading.personal_star({"path": "C:/x/DSC1.ARW", "stars": 3}))
    e = rs._read_raw()["C:/x/DSC1.ARW"]
    assert e["stars"] == 3 and e["score"] == 0.55
    assert e["features"] == {"Technical": 0.61, "_subject_sharp": 0.7, "_pan": True}


def test_training_uses_the_current_score_with_current_features(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    import master_judge as mj
    import lance_store
    rs.set_rating("C:/a.jpg", 3, score=0.30)                   # old grader, at rating time
    rs.attach_features("C:/a.jpg", {"Technical": 0.5}, score=0.55)   # new grader
    monkeypatch.setattr(lance_store, "query_all", lambda min_score=0.0: [])
    rows = mj.collect_rows()
    assert rows[0]["score"] == 0.55
    assert rs.get_score_snapshot("C:/a.jpg")["score"] == 0.30
