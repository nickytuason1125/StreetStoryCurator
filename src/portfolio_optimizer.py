"""
Portfolio optimizer using NSGA-III (pymoo) to select Top N images from LanceDB.

Objectives (maximized):
 - Aesthetic level (discrete, normalized)
 - Personal score (0..1)
 - Diversity (mean pairwise cosine distance among selected SigLIP vectors)

Because pymoo minimizes objectives, the optimizer minimizes negative of each metric.

API:
 - optimize_top_n(n_select: int, pop_size=100, generations=120, max_candidates=None) -> dict
     returns { 'selected_paths': [...], 'objectives': [...], 'history': algorithm }

Notes:
 - Requires pymoo installed: pip install pymoo
 - Requires lance_store to expose query_all() returning rows with keys:
     path, embedding (np.ndarray), aesthetic_level (int), personal_score (float)
 - For performance, if candidates exceed max_candidates, the module keeps a filtered
   subset by aesthetic_level+personal_score ranking to reduce optimization size.
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from pathlib import Path


def _safe_stack(arrs):
    return np.vstack([np.asarray(a, dtype=np.float32).reshape(1, -1) for a in arrs]) if arrs else np.zeros((0, 0), dtype=np.float32)


def optimize_top_n(n_select: int, pop_size: int = 100, generations: int = 80, max_candidates: Optional[int] = 400, seed: int = 42):
    """Run NSGA-III to select n_select items.

    Returns dict with keys: selected_paths, objectives (aesthetic, personal, diversity), full_candidates_count
    """
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.factory import get_reference_directions, get_sampling, get_crossover, get_mutation
        from pymoo.optimize import minimize
        from pymoo.operators.sampling.rnd import BinaryRandomSampling
        from pymoo.operators.crossover.bin import UniformCrossover
        from pymoo.operators.mutation.bitflip import BitflipMutation
    except Exception as e:
        raise RuntimeError("pymoo is required for portfolio optimization") from e

    # Load candidates from LanceDB
    try:
        import lance_store as ls
        rows = ls.query_all()
    except Exception:
        raise RuntimeError("Failed to load candidates from LanceDB (lance_store.query_all())")

    if not rows:
        return {"selected_paths": [], "objectives": [], "full_candidates_count": 0}

    paths = [r.get("path") for r in rows]
    aest = np.array([int(r.get("aesthetic_level", 1)) for r in rows], dtype=np.float32)  # 0..2
    pers = np.array([float(r.get("personal_score", 0.5)) for r in rows], dtype=np.float32)
    embs_list = [r.get("embedding") for r in rows]

    # Convert embeddings to array (N, D)
    try:
        embs = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embs_list])
    except Exception:
        # fallback to zeros
        embs = np.zeros((len(paths), 1152), dtype=np.float32)

    N = len(paths)

    # Optionally reduce candidate set to keep optimization tractable
    if max_candidates is not None and N > max_candidates:
        # rank by simple combined metric (aesthetic normalized + personal)
        aest_norm = aest / (aest.max() + 1e-9)
        score_rank = 0.6 * aest_norm + 0.4 * pers
        idx = np.argsort(-score_rank)[:max_candidates]
        paths = [paths[i] for i in idx]
        aest = aest[idx]
        pers = pers[idx]
        embs = embs[idx]
        N = len(paths)

    # Precompute pairwise cosine similarity for diversity metric
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    normed = embs / norms
    sim_matrix = normed @ normed.T  # cosine similarity

    # Problem definition: binary selection vector of length N; constrain sum == n_select
    class SelectionProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=N, n_obj=3, n_constr=2, xl=0, xu=1, elementwise=True, type_var=np.int8)

        def _evaluate(self, x, out, *args, **kwargs):
            # x is a 0/1 vector
            sel = np.asarray(x, dtype=bool)
            k = int(sel.sum())
            # constraints: sum - n_select <= 0 and n_select - sum <= 0 -> equality
            out["G"] = np.array([k - n_select, n_select - k], dtype=np.float64)

            if k == 0:
                # worst objectives
                out["F"] = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                return

            # Aesthetic objective: maximize mean(aest)/2 -> minimize negative
            aest_mean = float(aest[sel].mean()) / max(1.0, float(aest.max()))
            personal_mean = float(pers[sel].mean())

            # Diversity: mean pairwise cosine distance = 1 - mean(similarity over upper triangle)
            if k == 1:
                diversity = 0.0
            else:
                sub = sim_matrix[np.ix_(sel, sel)]
                # take upper triangle excluding diagonal
                iu = np.triu_indices(k, k=1)
                if iu[0].size == 0:
                    diversity = 0.0
                else:
                    mean_sim = float(sub[iu].mean())
                    diversity = max(0.0, 1.0 - mean_sim)

            # minimize negatives (pymoo minimizes)
            f1 = -aest_mean
            f2 = -personal_mean
            f3 = -diversity
            out["F"] = np.array([f1, f2, f3], dtype=np.float64)

    # Reference directions for 3 objectives
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    algorithm = NSGA3(pop_size=pop_size, ref_dirs=ref_dirs,
                      sampling=BinaryRandomSampling(),
                      crossover=UniformCrossover(),
                      mutation=BitflipMutation())

    problem = SelectionProblem()

    res = minimize(problem,
                   algorithm,
                   ('n_gen', generations),
                   seed=seed,
                   verbose=False)

    # Extract best feasible solution(s) from the result.
    # res.X is (n_pop, n_var) or (n_var,) — find first feasible with sum == n_select
    X = res.X
    F = res.F
    if X.ndim == 1:
        X = X.reshape(1, -1)
        F = F.reshape(1, -1)

    best_idx = None
    for i in range(X.shape[0]):
        if int(X[i].sum()) == n_select:
            best_idx = i
            break
    if best_idx is None:
        # try best by constraint violation
        ci = np.argmin(np.abs(X.sum(axis=1) - n_select))
        best_idx = ci

    sel_vec = X[best_idx].astype(int)
    selected_paths = [paths[i] for i in range(N) if sel_vec[i] == 1]

    # Recompute objectives for the selected set for reporting
    sel = np.asarray(sel_vec, dtype=bool)
    aest_mean = float(aest[sel].mean()) if sel.sum() > 0 else 0.0
    personal_mean = float(pers[sel].mean()) if sel.sum() > 0 else 0.0
    if sel.sum() <= 1:
        diversity = 0.0
    else:
        sub = sim_matrix[np.ix_(sel, sel)]
        iu = np.triu_indices(sel.sum(), k=1)
        mean_sim = float(sub[iu].mean())
        diversity = max(0.0, 1.0 - mean_sim)

    return {
        "selected_paths": selected_paths,
        "objectives": {
            "aesthetic_mean": aest_mean,
            "personal_mean": personal_mean,
            "diversity": diversity,
        },
        "full_candidates_count": len(rows),
        "pymoo_result": res,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=80)
    ap.add_argument("--maxc", type=int, default=400)
    args = ap.parse_args()
    out = optimize_top_n(args.n, pop_size=args.pop, generations=args.gens, max_candidates=args.maxc)
    print(f"Selected {len(out['selected_paths'])} paths (candidates: {out['full_candidates_count']})")
    for p in out['selected_paths']:
        print(p)
