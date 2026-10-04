"""Teach session: unrated photos spread across the whole machine-score range."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import teach_sampler as ts  # noqa: E402


def _p(i, score, cluster=-1, flag=""):
    return {"path": f"C:/c/{i}.ARW", "score": score, "cluster_id": cluster, "sim_flag": flag}


def test_spans_the_score_range(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {})
    photos = [_p(i, i / 100) for i in range(100)]
    got = ts.pick(photos, n=10)
    scores = sorted(float(p.split("/")[-1][:-4]) / 100 for p in got)
    assert len(got) == 10 and scores[0] <= 0.05 and scores[-1] >= 0.94


def test_skips_rated_and_hidden_duplicates(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {"C:/c/1.ARW": 3})
    photos = [_p(1, .5), _p(2, .5, cluster=7, flag="🔁 Duplicate"), _p(3, .6, cluster=7, flag="★ Best of 2")]
    assert ts.pick(photos, n=5) == ["C:/c/3.ARW"]


def test_sample_small_or_fully_rated_folder(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {p: 3 for p in paths})
    assert ts.pick([_p(1, .5)], n=50) == []
    monkeypatch.setattr(ts, "_rated", lambda paths: {})
    assert ts.pick([], n=50) == []


def test_endpoint_reads_catalog(tmp_path, monkeypatch):
    import asyncio, json
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import server_impl  # noqa: F401  (circular with routers.grading — load first)
    from routers import grading
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"photos": [_p(i, i / 10) for i in range(10)]}), encoding="utf-8")
    monkeypatch.setattr(server_impl, "_CATALOG_PATH", cat)
    monkeypatch.setattr(ts, "_rated", lambda paths: {})
    out = asyncio.run(grading.teach_sample(n=4))
    assert out["available"] == 10 and len(out["paths"]) == 4
