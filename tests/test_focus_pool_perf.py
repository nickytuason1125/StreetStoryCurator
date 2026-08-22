r"""
Pool focusing must not pay for the whole library twice.

Profiled on 5,634 Strong photos:

    select(k=10)           0.04s   <- the greedy loop, essentially free
    query_by_paths(5634)   4.28s   <- taste lookup for every photo
    embed_text_query       4.67s   <- SigLIP text tower, to embed one sentence
    stack+normalise        0.03s

An earlier commit message blamed the greedy loop at "0.9s per pick". That was
wrong; it was never measured. The cost is the two lookups around it.

Taste is only fetched for a shortlist now. That is not the old anchor funnel
returning: the shortlist is by BRIEF MATCH AND QUALITY across the whole library,
not by resemblance to one frame, and taste can only reorder within it.

Run:  venv\Scripts\python.exe -m pytest tests/test_focus_pool_perf.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director as cd  # noqa: E402


def _big_pool(n=3000):
    rs = np.random.RandomState(1)
    paths = ["/p%04d.jpg" % i for i in range(n)]
    embs = []
    for i in range(n):
        v = rs.randn(8).astype(np.float32)
        embs.append(v / np.linalg.norm(v))
    scores = list(rs.uniform(0.6, 0.99, size=n).astype(float))
    return paths, embs, scores


def test_taste_is_looked_up_for_a_shortlist_not_the_whole_library(monkeypatch):
    seen = {}

    def _fake_query(paths):
        seen["n"] = len(paths)
        return [{"path": p, "personal_score": 0.5} for p in paths]

    import lance_store
    monkeypatch.setattr(lance_store, "query_by_paths", _fake_query)
    monkeypatch.setattr(cd, "embed_text_query", lambda q: np.eye(8)[0])

    paths, embs, scores = _big_pool(3000)
    cd._focus_pool(paths, embs, scores, None, style_prompt="a brief", k=7)

    assert seen.get("n") is not None, "taste lookup never happened"
    assert seen["n"] <= cd._TASTE_LOOKUP_N, (
        "looked up %d paths; the whole point is not to" % seen["n"])
    assert seen["n"] < 3000


def test_the_shortlist_still_spans_the_library_not_one_neighbourhood(monkeypatch):
    """The shortlist is by brief match and quality. A strong, relevant frame
    far from the top scorer must survive it."""
    monkeypatch.setattr(cd, "embed_text_query", lambda q: np.eye(8)[7])
    paths, embs, scores = [], [], []
    for i in range(200):
        v = np.zeros(8, dtype=np.float32); v[0] = 1.0; v[1] = 0.001 * i
        paths.append("/cluster%03d.jpg" % i); embs.append(v / np.linalg.norm(v))
        scores.append(0.95)
    paths.append("/outlier.jpg"); embs.append(np.eye(8)[7].astype(np.float32))
    scores.append(0.99)

    kept, _, _, _ = cd._focus_pool(paths, embs, scores, None,
                                   style_prompt="the outlier", k=5)
    assert "/outlier.jpg" in kept


def test_brief_vectors_are_cached_between_calls(monkeypatch):
    calls = {"n": 0}

    def _counting(q):
        calls["n"] += 1
        return np.eye(8)[0]

    monkeypatch.setattr(cd, "embed_text_query", _counting)
    cd._brief_cache_clear()
    paths, embs, scores = _big_pool(50)
    for _ in range(3):
        cd._focus_pool(paths, embs, scores, None, style_prompt="same brief", k=4)
    assert calls["n"] == 1, "embedded the same brief %d times" % calls["n"]
