"""
NSGA-III Sequencer — Purist Edition

Selects a 5-image Story Sequence using only Original Pixel Metadata.
No stylization scores or generated-image quality signals are used.

Objectives (all maximised internally, negated for pymoo minimisation):
    1. Semantic_Brief_Match      — VLM score weighted by brief-relevant keywords
    2. Visual_Diversity          — pairwise visual dissimilarity across selected set
    3. Original_Lighting_Consistency — consistency of original pixel luminance

API
───
    run_nsga3_sequence_with_vlm(candidates, target, progress) → list[dict]
    run_nsga3_sequence(candidates, target, progress) → list[dict]

Fallback: greedy Pareto approach when pymoo is absent.

Candidates dict keys expected:
    path          str
    score         float
    embedding     np.ndarray (1536-d)
    reasoning_log str   (optional)
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

try:
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.repair import Repair
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    try:
        from pymoo.util.ref_dirs import get_reference_directions
    except ImportError:
        from pymoo.util.reference_direction import ReferenceDirectionFactory as _RDF
        def get_reference_directions(name, n_dim, n_points, **kw):
            return _RDF(name, n_dim, n_points).do()
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False


# ── Objective 1: Semantic Brief Match ─────────────────────────────────────────

def _semantic_brief_match(candidate: dict) -> float:
    """
    Combine VLM score with brief-relevant keyword depth in the reasoning log.
    Measures how well this original capture satisfies the curation brief.
    """
    base = float(candidate.get("score", 0.5))
    log  = candidate.get("reasoning_log", "") or ""
    if not log:
        return base
    length_score = min(1.0, len(log) / 400.0)
    kw = ["composition", "lighting", "contrast", "narrative", "balance",
          "decisive", "layer", "shadow", "texture", "depth",
          "empty", "liminal", "geometry", "atmosphere", "original"]
    kw_score = min(1.0, sum(1 for w in kw if w in log.lower()) / 4.0)
    return 0.6 * base + 0.25 * length_score + 0.15 * kw_score


# ── Objective 3: Original Lighting Consistency ────────────────────────────────

def _mean_luminance(path: str) -> float:
    """Mean luminance in [0, 1] via 64×64 PIL grayscale thumbnail."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        img.thumbnail((64, 64), Image.LANCZOS)
        return float(np.asarray(img, dtype=np.float32).mean() / 255.0)
    except Exception:
        return 0.5


# ── Per-candidate objective matrix ────────────────────────────────────────────

def _pre_compute_objectives(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """
    Return (N, 3) float32 matrix.
    Columns: [semantic_brief_match, luminance, zero_pad]
    Visual_Diversity and Original_Lighting_Consistency are set-level;
    per-image luminance is stored for fast set-level lookup.
    """
    n   = len(candidates)
    obj = np.zeros((n, 3), dtype=np.float32)
    for i, c in enumerate(candidates):
        obj[i, 0] = _semantic_brief_match(c)
        obj[i, 1] = _mean_luminance(c.get("path", ""))
    return obj


def _pairwise_sim_matrix(candidates: List[Dict[str, Any]]) -> np.ndarray:
    n    = len(candidates)
    embs = np.zeros((n, 1536), dtype=np.float32)
    for i, c in enumerate(candidates):
        e = c.get("embedding")
        if e is not None:
            arr    = np.asarray(e, dtype=np.float32).flatten()
            length = min(len(arr), 1536)
            embs[i, :length] = arr[:length]
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / (norms + 1e-9)
    return (normed @ normed.T).astype(np.float32)


def _eval_set_objectives(
    indices: np.ndarray,
    obj_matrix: np.ndarray,
    sim_matrix: np.ndarray,
) -> np.ndarray:
    """
    Return 3-element objective vector for a subset of candidates.
    All objectives in [0, 1] (higher = better).

    F1  Semantic_Brief_Match          — mean per-image brief-match score
    F2  Visual_Diversity              — mean pairwise dissimilarity (1 - cosine sim)
    F3  Original_Lighting_Consistency — 1 - std(luminances) × 3, clamped [0, 1]
    """
    idx = indices.astype(int)
    idx = idx[idx < len(obj_matrix)]
    if len(idx) == 0:
        return np.array([0.0, 0.5, 0.5], dtype=np.float32)

    # F1: mean Semantic Brief Match
    f1 = float(np.mean(obj_matrix[idx, 0]))

    # F2: Visual Diversity — mean pairwise dissimilarity
    if len(idx) > 1:
        sims  = sim_matrix[np.ix_(idx, idx)]
        upper = sims[np.triu_indices(len(idx), k=1)]
        f2    = float(1.0 - np.mean(upper))
    else:
        f2 = 0.5

    # F3: Original Lighting Consistency — lower std = more consistent = higher score
    lum  = obj_matrix[idx, 1]
    std  = float(np.std(lum)) if len(lum) > 1 else 0.0
    f3   = float(np.clip(1.0 - std * 3.0, 0.0, 1.0))

    return np.array([f1, f2, f3], dtype=np.float32)


# ── pymoo Problem ──────────────────────────────────────────────────────────────

if HAS_PYMOO:
    class _PhotoSelectionProblem(ElementwiseProblem):
        def __init__(self, n_candidates: int, target_size: int,
                     obj_matrix: np.ndarray, sim_matrix: np.ndarray):
            self.target_size = target_size
            self.obj_matrix  = obj_matrix
            self.sim_matrix  = sim_matrix
            super().__init__(
                n_var=target_size,
                n_obj=3,
                xl=np.zeros(target_size, dtype=float),
                xu=np.full(target_size, float(n_candidates - 1)),
                vtype=float,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            indices = np.unique(np.round(x).astype(int))
            while len(indices) < self.target_size:
                candidate = np.random.randint(0, len(self.obj_matrix))
                if candidate not in indices:
                    indices = np.append(indices, candidate)
            objs    = _eval_set_objectives(indices[:self.target_size], self.obj_matrix, self.sim_matrix)
            out["F"] = -objs

    class _UniqueRepair(Repair):
        def _do(self, problem, pop, **kwargs):
            for individual in pop:
                x         = np.round(individual.X).astype(int)
                seen      = set()
                available = [i for i in range(len(problem.obj_matrix)) if i not in set(x)]
                for j in range(len(x)):
                    if x[j] in seen or x[j] >= len(problem.obj_matrix):
                        if available:
                            x[j] = available.pop(0)
                    seen.add(x[j])
                individual.X = x.astype(float)
            return pop


# ── Greedy Pareto fallback ─────────────────────────────────────────────────────

def _greedy_pareto_select(
    candidates: List[Dict[str, Any]],
    target: int,
    obj_matrix: np.ndarray,
    sim_matrix: np.ndarray,
) -> List[int]:
    """
    Greedy selection matching the 3-objective NSGA-III intent without
    the evolutionary overhead when pymoo is absent.
    """
    n      = len(candidates)
    target = min(target, n)
    selected: list[int] = []
    remaining = list(range(n))

    best_seed = int(np.argmax(obj_matrix[:, 0]))
    selected.append(best_seed)
    remaining.remove(best_seed)

    while len(selected) < target and remaining:
        scores = np.zeros(len(remaining))
        for j, idx in enumerate(remaining):
            trial = np.array(selected + [idx])
            objs  = _eval_set_objectives(trial, obj_matrix, sim_matrix)
            scores[j] = (
                0.40 * objs[0]   # Semantic Brief Match
                + 0.40 * objs[1] # Visual Diversity
                + 0.20 * objs[2] # Original Lighting Consistency
            )
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)

    return selected


# ── Public API ────────────────────────────────────────────────────────────────

def run_nsga3_sequence_with_vlm(
    candidates: List[Dict[str, Any]],
    target: int = 5,
    progress=None,
) -> List[Dict[str, Any]]:
    """
    Run NSGA-III (or greedy Pareto fallback) on candidates using original pixel metadata.

    Args:
        candidates: dicts with 'path', 'score', 'embedding', 'reasoning_log'
        target:     number of photos to select
        progress:   optional progress callback(frac, desc)

    Returns:
        Ordered list of selected candidate dicts, each with 'nsga3_objectives'
        containing: semantic_brief_match, visual_diversity,
        original_lighting_consistency.
    """
    if not candidates:
        return []
    target = min(target, len(candidates))

    obj_matrix = _pre_compute_objectives(candidates)
    sim_matrix = _pairwise_sim_matrix(candidates)

    if HAS_PYMOO and len(candidates) >= target * 2:
        try:
            n_candidates = len(candidates)
            problem      = _PhotoSelectionProblem(n_candidates, target, obj_matrix, sim_matrix)
            ref_dirs     = get_reference_directions("energy", 3, min(100, n_candidates * 2), seed=1)

            algorithm = NSGA3(
                pop_size=max(len(ref_dirs), 20),
                ref_dirs=ref_dirs,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15, vtype=float),
                mutation=PM(eta=20, vtype=float),
                repair=_UniqueRepair(),
                eliminate_duplicates=True,
            )

            n_gen = max(30, min(80, n_candidates))
            res   = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=False)

            if res.X is not None:
                mean_obj     = (-res.F).mean(axis=1)
                best_sol     = res.X[np.argmax(mean_obj)]
                best_idx     = np.unique(np.round(best_sol).astype(int))
                if len(best_idx) < target:
                    remaining = [i for i in range(n_candidates) if i not in set(best_idx)]
                    np.random.shuffle(remaining)
                    best_idx = np.concatenate([best_idx, remaining[:target - len(best_idx)]])
                selected_indices = best_idx[:target].tolist()
            else:
                raise RuntimeError("NSGA-III returned no solutions")

        except Exception as e:
            print(f"[nsga3] pymoo run failed ({e}), using greedy fallback")
            selected_indices = _greedy_pareto_select(candidates, target, obj_matrix, sim_matrix)
    else:
        selected_indices = _greedy_pareto_select(candidates, target, obj_matrix, sim_matrix)

    result = []
    for rank, idx in enumerate(selected_indices):
        c    = dict(candidates[idx])
        objs = _eval_set_objectives(
            np.array(selected_indices[:rank + 1]), obj_matrix, sim_matrix
        )
        c["nsga3_objectives"] = {
            "semantic_brief_match":          round(float(objs[0]), 3),
            "visual_diversity":              round(float(objs[1]), 3),
            "original_lighting_consistency": round(float(objs[2]), 3),
        }
        result.append(c)

    return result


def run_nsga3_sequence(
    candidates: List[Dict[str, Any]],
    target: int = 5,
    progress=None,
) -> List[Dict[str, Any]]:
    """Alias for run_nsga3_sequence_with_vlm."""
    return run_nsga3_sequence_with_vlm(candidates, target=target, progress=progress)
