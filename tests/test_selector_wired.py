r"""
run_creative_direction must select over the whole pool, not one neighbourhood.

The funnel it replaces took the single highest-scoring photo and kept its 40
nearest neighbours (creative_director.py:1148-1161). Everything downstream --
the manifest, the LLM, the sequencer -- then worked inside whatever corner that
one frame happened to occupy.

Run:  venv\Scripts\python.exe -m pytest tests/test_selector_wired.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director as cd  # noqa: E402


def _pool(n_cluster=60, with_outlier=True):
    """A dominant cluster plus one strong, distant frame."""
    paths, embs, scores = [], [], []
    for i in range(n_cluster):
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        v[1] = 0.001 * i
        paths.append("/cluster%02d.jpg" % i)
        embs.append(v / np.linalg.norm(v))
        scores.append(0.95)
    if with_outlier:
        v = np.zeros(8, dtype=np.float32)
        v[7] = 1.0
        paths.append("/outlier.jpg")
        embs.append(v)
        scores.append(0.99)
    return paths, embs, scores


def test_focus_pool_keeps_a_distant_strong_frame(monkeypatch):
    """The cul-de-sac test: with a brief pointing at the outlier, it must be
    reachable. Under the old top-40-nearest-neighbours funnel it was not."""
    paths, embs, scores = _pool()
    brief_vec = np.zeros(8, dtype=np.float32)
    brief_vec[7] = 1.0
    monkeypatch.setattr(cd, "embed_text_query", lambda q: brief_vec)

    kept_paths, kept_embs, kept_scores, _ = cd._focus_pool(
        paths, embs, scores, None, style_prompt="the outlier", k=8)
    assert "/outlier.jpg" in kept_paths, "a relevant distant frame was unreachable"


def _varied(n):
    """n genuinely distinct frames: spread around the sphere, not 0.001 apart."""
    rs = np.random.RandomState(0)
    embs, paths, scores = [], [], []
    for i in range(n):
        v = rs.randn(8).astype(np.float32)
        v /= np.linalg.norm(v)
        embs.append(v)
        paths.append("/v%02d.jpg" % i)
        scores.append(0.7 + 0.001 * i)
    return paths, embs, scores


def test_focus_pool_honours_k_when_the_pool_is_varied():
    paths, embs, scores = _varied(60)
    for k in (4, 7, 10):
        kept, _, _, _ = cd._focus_pool(paths, embs, scores, None,
                                       style_prompt="", k=k)
        assert len(kept) == k, "asked for %d, got %d" % (k, len(kept))


def test_a_pool_of_near_identical_frames_returns_fewer_not_duplicates():
    """Documented behaviour, not a bug. Sixty frames 0.001 apart are one
    photograph sixty times; returning k of them would be a padded set, so the
    selector returns what genuinely differs and the caller reports why."""
    paths, embs, scores = _pool(n_cluster=60, with_outlier=False)
    kept, _, _, _ = cd._focus_pool(paths, embs, scores, None,
                                   style_prompt="", k=10)
    assert 0 < len(kept) < 10


def test_focus_pool_survives_an_empty_pool():
    kept, embs, scores, aspects = cd._focus_pool([], [], [], None,
                                                 style_prompt="x", k=7)
    assert kept == [] and embs == [] and scores == []


def test_focus_pool_keeps_aspects_aligned():
    """Aspect dicts are positional: misaligning them silently attaches one
    photo's breakdown to another."""
    paths, embs, scores = _pool(n_cluster=20, with_outlier=False)
    aspects = [{"Composition": i / 100.0} for i in range(len(paths))]
    kept, _, kept_scores, kept_aspects = cd._focus_pool(
        paths, embs, scores, aspects, style_prompt="", k=5)
    assert len(kept_aspects) == len(kept)
    for p, a in zip(kept, kept_aspects):
        i = int(p.split("cluster")[1].split(".")[0])
        assert a == aspects[i], "aspect dict no longer matches its photo"
