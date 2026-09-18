"""Rating-aware vision critique context (2026-09-15).

The critique prompt now carries the photographer's own data (stars, verdict
margin, taste profile, similar rated relatives). These tests pin the builder's
contract: personal context only for RATED photos, graceful '' on any failure,
and the token cap."""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import critique_engine as ce
import preference_model as pm


class _FakeRS:
    """Stand-in for ratings_store inside _taste_context. By default every path
    is rated `stars`; pass `only_for` to rate just those paths (anything else
    reads as unrated)."""
    def __init__(self, stars=4, records=None, only_for=None):
        self._stars = stars
        self._records = records or []
        self._only = only_for
    def get(self, path):
        if self._only is not None:
            return self._only.get(path, 0)
        return self._stars
    def load_records(self):
        return self._records
    def get_score_snapshot(self, path):
        return {"score": 0.5}


class _FakeLS:
    """Stand-in for lance_store (taste_summary + embedding neighbours)."""
    def __init__(self, rows=None, hits=None):
        self._rows = rows or []
        self._hits = hits or []
    def query_light_all(self):
        return self._rows
    def query_embeddings_for_paths_bulk(self, paths):
        return {}
    def vector_search(self, emb, top_k=20, min_score=0.0):
        return self._hits


def _install(monkeypatch, rs=None, ls=None, gate=None, taste=None):
    monkeypatch.setitem(sys.modules, "ratings_store", rs or _FakeRS())
    monkeypatch.setitem(sys.modules, "lance_store", ls or _FakeLS())
    pm_mod = types.SimpleNamespace(
        load_gate=lambda: gate or {"active": False},
        taste_summary=lambda: taste or {},
    )
    monkeypatch.setitem(sys.modules, "preference_model", pm_mod)


def test_taste_context_present_for_rated_photo(monkeypatch):
    _install(monkeypatch, rs=_FakeRS(stars=4, records=[]))
    rec = {"path": "C:/x/a.jpg", "score": 0.62, "grade": "Strong ✅"}
    ctx = ce._taste_context("C:/x/a.jpg", rec)
    assert "Your rating: 4 stars" in ctx
    assert "Verdict: Strong" in ctx
    assert "PHOTOGRAPHER CONTEXT" in ctx


def test_taste_context_empty_for_unrated(monkeypatch):
    _install(monkeypatch, rs=_FakeRS(stars=0))
    rec = {"path": "C:/x/b.jpg", "score": 0.5, "grade": "Mid ⚠️"}
    assert ce._taste_context("C:/x/b.jpg", rec) == ""


def test_taste_context_includes_taste_and_relatives(monkeypatch):
    ls = _FakeLS(
        hits=[{"path": "C:/x/keeper.jpg", "score": 0.8},
              {"path": "C:/x/other.jpg", "score": 0.4}],   # unrated → skipped
    )
    _install(monkeypatch,
             rs=_FakeRS(stars=5, records=[],
                        only_for={"C:/x/keeper.jpg": 5, "C:/x/a.jpg": 5}),
             ls=ls,
             taste={"keeper_leans": [("Documentary", 0.12), ("Layered", 0.08)],
                    "reject_leans": [("Static", -0.06)],
                    "n_keepers": 40, "n_rejects": 200})
    rec = {"path": "C:/x/a.jpg", "score": 0.62, "grade": "Strong ✅",
           "embedding": [0.1, 0.2, 0.3]}
    ctx = ce._taste_context("C:/x/a.jpg", rec)
    assert "Documentary +0.12" in ctx
    assert "Static -0.06" in ctx
    assert "keeper.jpg — you rated it 5 stars" in ctx
    assert "other.jpg" not in ctx          # unrated relative never surfaces


def test_taste_context_borderline_flagged(monkeypatch):
    _install(monkeypatch, rs=_FakeRS(stars=3, records=[]))
    # 0.41 sits exactly on the default Maybe line → borderline
    rec = {"path": "C:/x/a.jpg", "score": 0.41, "grade": "Mid ⚠️"}
    ctx = ce._taste_context("C:/x/a.jpg", rec)
    assert "borderline" in ctx


def test_taste_context_capped(monkeypatch):
    long_taste = {"keeper_leans": [("A" * 200, 0.5), ("B" * 200, 0.4),
                                   ("C" * 200, 0.3)],
                  "reject_leans": [("D" * 200, -0.5)],
                  "n_keepers": 1, "n_rejects": 1}
    _install(monkeypatch, rs=_FakeRS(stars=4, records=[]), taste=long_taste)
    rec = {"path": "C:/x/a.jpg", "score": 0.7, "grade": "Strong ✅"}
    ctx = ce._taste_context("C:/x/a.jpg", rec)
    assert len(ctx) <= 1600


def test_taste_context_never_raises(monkeypatch):
    # a store that explodes must degrade to '' — the critique then runs generic
    class _Boom(_FakeRS):
        def get(self, path):
            raise RuntimeError("boom")
    _install(monkeypatch, rs=_Boom())
    rec = {"path": "C:/x/a.jpg", "score": 0.6, "grade": "Strong ✅"}
    assert ce._taste_context("C:/x/a.jpg", rec) == ""


# ── taste_summary ────────────────────────────────────────────────────────────

def test_taste_summary_tallies(monkeypatch):
    rows = [
        {"path": "C:/x/k1.jpg", "breakdown": {"Composition": 0.9, "Light": 0.4}},
        {"path": "C:/x/k2.jpg", "breakdown": {"Composition": 0.8, "Light": 0.6}},
        {"path": "C:/x/r1.jpg", "breakdown": {"Static": 0.9, "Light": 0.1}},
    ]
    monkeypatch.setitem(sys.modules, "lance_store", _FakeLS(rows=rows))
    recs = [{"path": "C:/x/k1.jpg", "stars": 5},
            {"path": "C:/x/k2.jpg", "stars": 4},
            {"path": "C:/x/r1.jpg", "stars": 1}]
    ts = pm.taste_summary(recs)
    assert ts["n_keepers"] == 2 and ts["n_rejects"] == 1
    # gaps: keepers−rejects. Composition: 0.85−0 = +0.85 (rejects lack it);
    # Light: 0.5−0.1 = +0.40; Static: 0−0.9 = −0.90
    assert ts["keeper_leans"][0] == ("Composition", 0.85)
    assert ts["reject_leans"][0] == ("Static", -0.9)


def test_taste_summary_ignores_flag_keys(monkeypatch):
    rows = [{"path": "C:/x/k1.jpg",
             "breakdown": {"Composition": 0.6, "_nima": True, "_grader": 1.0}},
            {"path": "C:/x/r1.jpg",
             "breakdown": {"Composition": 0.2, "_nima": True}}]
    monkeypatch.setitem(sys.modules, "lance_store", _FakeLS(rows=rows))
    ts = pm.taste_summary([{"path": "C:/x/k1.jpg", "stars": 5},
                           {"path": "C:/x/r1.jpg", "stars": 1}])
    assert ts["keeper_leans"] == [("Composition", 0.4)]
    assert ts["reject_leans"] == []


def test_taste_summary_empty_when_no_anchors(monkeypatch):
    monkeypatch.setitem(sys.modules, "lance_store", _FakeLS(rows=[]))
    assert pm.taste_summary([{"path": "C:/x/a.jpg", "stars": 4}]) == {}
