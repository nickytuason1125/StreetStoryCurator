"""Duplicate groups are complete-linkage: no chaining (2026-10-03).

Single-linkage union-find put a whole harbour walk into one 207-photo group
(least-similar pair 0.75) and the gallery hid every non-best member.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline_stages import cluster_similar, duplicate_groups  # noqa: E402


def _chain(n=30, step=0.15, dim=64, seed=0):
    """Each vector ~0.97 similar to its neighbour, the ends nearly unrelated."""
    rng = np.random.default_rng(seed)
    v = [rng.standard_normal(dim)]
    for _ in range(n - 1):
        nxt = v[-1] + step * rng.standard_normal(dim)
        v.append(nxt)
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_a_chain_of_neighbours_does_not_become_one_group():
    e = _chain()
    s = e @ e.T
    assert s[0, -1] < 0.9                        # ends are different photos
    assert (np.diag(s, 1) > 0.96).all()          # ...yet every neighbour pair is a "duplicate"
    groups = duplicate_groups(e, 0.96)
    assert max(len(g) for g in groups) < len(e)
    for g in groups:                             # every pair inside a group is a near-twin
        assert (s[np.ix_(g, g)] > 0.96 - 1e-6).all()


def test_a_real_burst_stays_together():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(64)
    burst = np.array([base + 0.05 * rng.standard_normal(64) for _ in range(6)], np.float32)
    other = rng.standard_normal((4, 64)).astype(np.float32)
    e = np.vstack([burst, other])
    groups = duplicate_groups(e, 0.96)
    assert sorted(map(sorted, groups)) == [list(range(6))]


def test_deterministic_and_cluster_ids_match():
    e = _chain(seed=3)
    assert duplicate_groups(e, 0.96) == duplicate_groups(e.copy(), 0.96)
    ids = cluster_similar(e, 0.96)
    for g in duplicate_groups(e, 0.96):
        assert len({ids[i] for i in g}) == 1
