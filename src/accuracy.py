"""Agreement between machine grades and the photographer's stars.

The scorecard every taste change must beat (2026-10-03 baseline: 0.561
3-class agreement on 157 on-disk rated photos). Pure numpy — importable from
the CUDA-free grade worker. Groups are SHOOTS (parent folders): resampling
photos instead would let near-identical burst frames inflate confidence.
"""
from __future__ import annotations

import os

import numpy as np

_STAR_CLASS = {5: 2, 4: 2, 3: 1, 2: 1, 1: 0}   # must equal ratings_store._STAR_GRADE


def stars_to_class(stars: int) -> int:
    return _STAR_CLASS[int(stars)]


def score_to_class(score: float, strong_t: float, mid_t: float) -> int:
    return 2 if score >= strong_t else 1 if score >= mid_t else 0


def agreement(pred, true) -> float:
    p, t = np.asarray(pred), np.asarray(true)
    return float((p == t).mean()) if t.size else float("nan")


def confusion(pred, true) -> list:
    """3x3 counts indexed [true][pred] (0 Weak, 1 Mid, 2 Strong)."""
    cm = [[0, 0, 0] for _ in range(3)]
    for p, t in zip(pred, true):
        cm[int(t)][int(p)] += 1
    return cm


def shoot_of(path: str) -> str:
    return os.path.dirname(path.replace("/", "\\")).lower()


def paired_bootstrap(true, pred_a, pred_b, groups, n: int = 2000, seed: int = 0) -> dict:
    """Agreement(b) − agreement(a), resampling whole shoots. lo/hi = 95% CI."""
    t, a, b = map(np.asarray, (true, pred_a, pred_b))
    g = np.asarray(groups)
    ids = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ids}
    hit_a, hit_b = (a == t).astype(float), (b == t).astype(float)
    delta = float(hit_b.mean() - hit_a.mean())
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(ids, size=len(ids), replace=True)])
        ds.append(hit_b[pick].mean() - hit_a[pick].mean())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"delta": delta, "lo": float(lo), "hi": float(hi)}
