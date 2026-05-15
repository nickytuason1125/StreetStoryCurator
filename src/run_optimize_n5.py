"""
Run NSGA-III optimize_top_n(N=5) and compare against Top-5-by-Aesthetic baseline.

Usage: from the repo root run:
  python street-story-curator/src/run_optimize_n5.py

Ensure dependencies are installed: pymoo, lancedb, pyarrow, numpy
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from portfolio_optimizer import optimize_top_n

import lance_store as ls
import numpy as np


def mean_pairwise_cosine_distance(embs):
    if len(embs) <= 1:
        return 0.0
    arr = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embs])
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    nrm = arr / norms
    sim = nrm @ nrm.T
    iu = np.triu_indices(sim.shape[0], k=1)
    if iu[0].size == 0:
        return 0.0
    mean_sim = float(sim[iu].mean())
    return float(max(0.0, 1.0 - mean_sim))


def run():
    print('\nRunning NSGA-III optimization (N=5) ... this may take a while')
    out = optimize_top_n(5, pop_size=40, generations=30, max_candidates=200)

    sel = out.get('selected_paths', [])
    objs = out.get('objectives', {})
    print('\nOptimization result:')
    print(json.dumps({'selected_count': len(sel), 'selected_paths': sel, 'objectives': objs, 'full_candidates_count': out.get('full_candidates_count')}, indent=2))

    # Constraint satisfaction
    print('\nConstraint check: expected 5, got', len(sel))

    # Compute diversity for optimized selection
    rows = ls.query_all()
    by_path = {r['path']: r for r in rows}
    sel_embs = [by_path[p]['embedding'] for p in sel if p in by_path]
    sel_div = mean_pairwise_cosine_distance(sel_embs)
    print(f"\nOptimized selection diversity: {sel_div:.4f}")

    # Baseline: Top-5 by aesthetic_level then personal_score
    cand = rows
    if not cand:
        print('No candidates in LanceDB. Aborting baseline comparison.')
        return
    sorted_by_aest = sorted(cand, key=lambda r: (int(r.get('aesthetic_level',0)), float(r.get('personal_score',0.0))), reverse=True)
    top5 = sorted_by_aest[:5]
    top5_paths = [r['path'] for r in top5]
    top5_embs = [r['embedding'] for r in top5]
    top5_div = mean_pairwise_cosine_distance(top5_embs)
    print('\nBaseline Top-5 Aesthetic paths:')
    for p in top5_paths:
        print(p)
    print(f"\nBaseline diversity: {top5_div:.4f}")

    print('\nDiversity delta (optimized - baseline):', round(sel_div - top5_div, 6))

    # Pareto summary
    try:
        res = out.get('pymoo_result')
        if res is not None:
            print('\nPymoo result summary:')
            # Try to print basic info
            print('  algorithm:', type(res.algorithm).__name__ if hasattr(res, 'algorithm') else str(res.get('algorithm', '')))
            F = getattr(res, 'F', None)
            X = getattr(res, 'X', None)
            if F is not None:
                print('  Pareto front size:', F.shape[0] if hasattr(F, 'shape') else len(F))
            if X is not None:
                print('  Population shape:', X.shape)
    except Exception:
        pass

    print('\nDone.')


if __name__ == '__main__':
    run()
