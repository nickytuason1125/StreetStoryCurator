"""
Step 4 — MOGCO-II: 3-objective Pareto optimizer.

Objectives (all higher-is-better)
──────────────────────────────────
1. semantic_score   — cosine similarity to an optional SigLIP text query
                      (uniform 0.5 when no query is provided)
2. personal_score   — PersonalHead MLP output, learned from user re-bucketing
3. visual_diversity — 1 − max cosine-sim to any already-selected embedding
                      (0.5 for the first frame; rewards selecting visually
                      distinct shots)

Algorithm
─────────
For each narrative slot (opener → subject → detail → contrast → closer):
  1. Score all remaining candidates on all 3 objectives.
  2. Compute the Pareto front (non-dominated rows).
  3. Apply role-weighted linear scalarisation over the Pareto front.
  4. Add small reproducible jitter to break deterministic ties.

This is identical to the MOGCO Pareto logic in mogco_engine.py but
operates on the richer (1152-d SigLIP, personal, diversity) signal set.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Optional


# ── Shot role definitions ─────────────────────────────────────────────────────

SHOT_ROLES_II = {
    "opener":   {"semantic": 0.30, "personal": 0.30, "diversity": 0.40},
    "subject":  {"semantic": 0.40, "personal": 0.40, "diversity": 0.20},
    "detail":   {"semantic": 0.35, "personal": 0.25, "diversity": 0.40},
    "contrast": {"semantic": 0.25, "personal": 0.25, "diversity": 0.50},
    "closer":   {"semantic": 0.35, "personal": 0.35, "diversity": 0.30},
}


# ── Pareto front (same as mogco_engine.py — duplicated to keep module self-contained)

def _pareto_front(obj: np.ndarray) -> np.ndarray:
    n = len(obj)
    if n == 0:
        return np.zeros(0, dtype=bool)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        diff          = obj - obj[i]
        at_least_good = np.all(diff >= -1e-9, axis=1)
        strictly_good = np.any(diff >   1e-9, axis=1)
        mask          = at_least_good & strictly_good
        mask[i]       = False
        if mask.any():
            dominated[i] = True
    return ~dominated


# ── Main entry point ──────────────────────────────────────────────────────────

def mogco2_sequence(
    candidates: list[dict[str, Any]],
    target: int = 5,
    query_emb: Optional[np.ndarray] = None,
    jitter_scale: float = 0.015,
    rng_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select `target` frames using MOGCO-II.

    Parameters
    ──────────
    candidates : list of dicts with keys:
        path           str
        embedding      np.ndarray (1152,) — SigLIP embedding, L2-normalised
        score          float              — Q-Align aesthetic score
        personal_score float              — PersonalHead preference
        grade          str
        breakdown      dict
        exif_ts        float
    target    : number of frames to select
    query_emb : (1152,) SigLIP text embedding of the user's brief (optional)
    """
    if not candidates or target == 0:
        return []

    rng   = np.random.default_rng(rng_seed)
    roles = list(SHOT_ROLES_II.items())
    n     = len(candidates)

    embs  = np.stack([np.asarray(c["embedding"], dtype=np.float32) for c in candidates])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-9)                     # (N, 1152) normalised

    # ── Semantic scores ───────────────────────────────────────────────────────
    if query_emb is not None:
        q = np.asarray(query_emb, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        sem_scores = np.clip((embs_n @ q + 1.0) / 2.0, 0.0, 1.0)    # [0,1]
    else:
        sem_scores = np.full(n, 0.5, dtype=np.float32)

    # ── Personal preference scores ────────────────────────────────────────────
    pers_scores = np.array(
        [float(c.get("personal_score", 0.5)) for c in candidates],
        dtype=np.float32,
    )

    remaining_ids = set(range(n))
    selected_embs: list[np.ndarray] = []
    sequence: list[dict] = []

    for slot_i in range(min(target, len(roles))):
        if not remaining_ids:
            break

        role_name, rw = roles[slot_i]
        cand_ids = list(remaining_ids)
        m = len(cand_ids)

        # ── Objective 1: semantic ─────────────────────────────────────────────
        obj_sem  = sem_scores[cand_ids]

        # ── Objective 2: personal preference ─────────────────────────────────
        obj_pers = pers_scores[cand_ids]

        # ── Objective 3: visual diversity from selected frames ────────────────
        if selected_embs:
            sel_mat = np.stack(selected_embs)           # (k, 1152) normalised
            sims    = embs_n[cand_ids] @ sel_mat.T      # (m, k)
            max_sim = sims.max(axis=1)                  # (m,)
            obj_div = np.clip(1.0 - max_sim, 0.0, 1.0)
        else:
            obj_div = np.full(m, 0.5, dtype=np.float32)

        # ── Pareto front ──────────────────────────────────────────────────────
        obj_matrix = np.stack([obj_sem, obj_pers, obj_div], axis=1)    # (m, 3)
        front_mask = _pareto_front(obj_matrix)
        front_local = np.where(front_mask)[0]

        # ── Role-weighted scalarisation ───────────────────────────────────────
        W = np.array([rw["semantic"], rw["personal"], rw["diversity"]], dtype=np.float64)
        W /= W.sum()
        front_scores = obj_matrix[front_local] @ W
        jitter       = rng.uniform(0.0, jitter_scale, size=len(front_local))
        best_local   = int(np.argmax(front_scores + jitter))
        best_cand_id = cand_ids[front_local[best_local]]

        raw = obj_matrix[front_local[best_local]]
        sel = dict(candidates[best_cand_id])
        sel["slot"]        = role_name
        sel["mogco2_objectives"] = {
            "semantic":   round(float(raw[0]), 3),
            "personal":   round(float(raw[1]), 3),
            "diversity":  round(float(raw[2]), 3),
        }
        sequence.append(sel)
        remaining_ids.discard(best_cand_id)
        selected_embs.append(embs_n[best_cand_id])

    return sequence


# ── Convenience wrapper using LanceDB ─────────────────────────────────────────

def run_mogco2(
    query: Optional[str] = None,
    target: int = 5,
    min_score: float = 0.55,
    min_personal: float = 0.0,
) -> dict:
    """
    Pull candidates from LanceDB and run MOGCO-II.

    If `query` is provided, encodes it with SigLIPEncoder for semantic scoring.
    Returns a dict compatible with the existing MOGCO response format.

    min_score is intentionally set to 0.55 (mid-high) to exclude low-quality
    mid photos from the sequence.  Callers may override.
    """
    import lance_store as ls

    candidates = ls.query_all(min_score=min_score)
    if min_personal > 0:
        candidates = [c for c in candidates if c["personal_score"] >= min_personal]

    # If PersonalHead has not been trained, all personal_scores will be the
    # same constant (0.5).  Substitute the aesthetic score so the personal
    # objective still differentiates candidates.
    pers_vals = [round(c.get("personal_score", 0.5), 2) for c in candidates]
    if candidates and len(set(pers_vals)) <= 1:
        for c in candidates:
            c["personal_score"] = float(c.get("score", 0.5))

    if len(candidates) < target:
        # Relax threshold so we can still build a sequence
        candidates = ls.query_all(min_score=0.0)
        if pers_vals and len(set(pers_vals)) <= 1:
            for c in candidates:
                c["personal_score"] = float(c.get("score", 0.5))
        if len(candidates) < target:
            return {
                "error": f"Only {len(candidates)} photos available (need {target}).",
                "paths": [], "slots": [], "sequence": [], "global_score": 0.0,
            }

    query_emb = None
    if query:
        try:
            from siglip_encoder import SigLIPEncoder
            enc = SigLIPEncoder()
            query_emb = enc.encode_text([query])[0]
            enc.unload()
        except Exception:
            pass

    seq = mogco2_sequence(candidates, target=target, query_emb=query_emb)

    return {
        "paths":        [f["path"]  for f in seq],
        "slots":        [f["slot"]  for f in seq],
        "global_score": round(float(np.mean([
            np.mean(list(f["mogco2_objectives"].values())) for f in seq
        ])), 4),
        "sequence":     seq,
        "engine":       "mogco2",
    }
