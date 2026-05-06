"""
MOGCO Sequencer — Multi-Objective Greedy Constrained Optimization.

Architecture: pure NumPy computation running inside FastAPI workers.
DuckDB supplies pre-fetched embeddings; this module does no I/O.

For each narrative slot MOGCO:
  1. Scores all remaining candidates on 5 objectives:
       role_fit     – how well the photo matches the slot's SHOT_ROLES weights
       quality      – composite grader score
       visual_div   – cosine distance from the previous frame (1 − similarity)
       temporal     – penalty for huge timestamp jumps vs. smooth progression
       event_div    – bonus when the photo comes from a different temporal event
  2. Finds the Pareto front (non-dominated candidates across all 5 objectives).
  3. Applies a role-weighted linear scalarization over the Pareto front to
     pick the single best photo for the slot.

The Pareto filtering prevents a single dominant objective from collapsing the
selection into a pure quality sort, giving the algorithm its multi-objective
character while staying deterministic and O(n² · m) for small pools.
"""

from __future__ import annotations
import json
import numpy as np
from typing import Any

from sequence_engine import (
    SHOT_ROLES,
    _COMP_KEYS, _HUMAN_KEYS, _LIGHT_KEYS, _TECH_KEYS, _AUTH_KEYS,
    segment_events,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dv(b: dict, keys: frozenset) -> float:
    return next((v for k, v in b.items() if k in keys), 0.0)


def _role_score(breakdown: dict, role_weights: dict) -> float:
    comp  = _dv(breakdown, _COMP_KEYS)
    human = _dv(breakdown, _HUMAN_KEYS)
    mood  = _dv(breakdown, _LIGHT_KEYS)
    return (comp  * role_weights["comp_weight"] +
            human * role_weights["human_weight"] +
            mood  * role_weights["mood_weight"])


# ── Pareto front ──────────────────────────────────────────────────────────────

def _pareto_front(obj: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of non-dominated rows.

    obj : [n × m] float64, all objectives are higher-is-better.
    A row i is dominated if any other row j satisfies:
        obj[j, k] >= obj[i, k]  for all k   AND
        obj[j, k] >  obj[i, k]  for at least one k.
    O(n² · m) — acceptable for pool sizes < 500.
    """
    n = len(obj)
    if n == 0:
        return np.zeros(0, dtype=bool)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        diff          = obj - obj[i]                        # [n × m]
        at_least_good = np.all(diff >= -1e-9, axis=1)      # j ≥ i in all dims
        strictly_good = np.any(diff >   1e-9, axis=1)      # j > i in ≥ 1 dim
        mask          = at_least_good & strictly_good
        mask[i]       = False                               # skip self
        if mask.any():
            dominated[i] = True
    return ~dominated


# ── Main sequencer ────────────────────────────────────────────────────────────

def mogco_sequence(
    candidates: list[dict[str, Any]],
    target: int = 5,
    stype: str = "street",
    custom_shot_roles: dict | None = None,
    jitter_scale: float = 0.015,
    pareto_budget: int = 300,
    rng_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Fill `target` narrative slots greedily using Pareto multi-objective selection.

    Parameters
    ----------
    candidates : list of dicts, each with:
        path      str        — absolute file path
        score     float      — composite quality score ∈ [0, 1]
        breakdown dict       — per-dimension scores from the grader
        embedding np.ndarray — CLIP/DINOv2 feature vector
        exif_ts   float      — Unix timestamp (0.0 if unavailable)
    target : number of frames to select (default 5)
    stype  : subject type for role labels (unused internally — roles come from SHOT_ROLES)
    custom_shot_roles : OrderedDict in the same format as SHOT_ROLES to override defaults
    jitter_scale : small random nudge applied to Pareto scores to break ties (default 0.015)
    pareto_budget : max candidates sent into the Pareto computation per slot (default 300)
    rng_seed : reproducibility seed

    Returns
    -------
    Ordered list of dicts (≤ target), each being the original candidate dict with
    an added 'slot' key (role name) and 'mogco_objectives' key (raw objective scores).
    """
    if not candidates or target == 0:
        return []

    rng   = np.random.default_rng(rng_seed)
    roles = list((custom_shot_roles or SHOT_ROLES).items())

    # ── Stable indexing ───────────────────────────────────────────────────────
    # Assign each candidate a permanent integer id; remaining is a set of ids.
    n_total       = len(candidates)
    remaining_ids = set(range(n_total))

    # ── Event map: candidate_id → event bucket index ─────────────────────────
    photo_recs = [{"exif_ts": c.get("exif_ts", 0.0), "_vi": i}
                  for i, c in enumerate(candidates)]
    events     = segment_events(photo_recs, gap_threshold=900)
    event_of: dict[int, int] = {}
    for ev_i, ev_group in enumerate(events):
        for rec in ev_group:
            event_of[rec["_vi"]] = ev_i

    sequence: list[dict] = []
    prev_emb: np.ndarray | None = None
    prev_event: int = -1

    for slot_i in range(min(target, len(roles))):
        if not remaining_ids:
            break

        role_name, rw = roles[slot_i]

        # ── Candidate list for this slot ──────────────────────────────────────
        cand_ids = list(remaining_ids)

        # Trim to Pareto budget (keep highest-quality candidates when oversized)
        if len(cand_ids) > pareto_budget:
            cand_ids = sorted(cand_ids,
                              key=lambda i: candidates[i].get("score", 0.0),
                              reverse=True)[:pareto_budget]

        m = len(cand_ids)

        # ── Objective 1: role fit ─────────────────────────────────────────────
        obj_role = np.array(
            [_role_score(candidates[i].get("breakdown", {}), rw) for i in cand_ids],
            dtype=np.float64,
        )

        # ── Objective 2: quality ──────────────────────────────────────────────
        obj_qual = np.array(
            [float(candidates[i].get("score", 0.0)) for i in cand_ids],
            dtype=np.float64,
        )

        # ── Objective 3: visual diversity from previous frame ─────────────────
        if prev_emb is not None:
            pe     = prev_emb / (np.linalg.norm(prev_emb) + 1e-9)
            embs   = np.array([candidates[i]["embedding"] for i in cand_ids],
                              dtype=np.float64)
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            embs_n = embs / (norms + 1e-9)
            sims   = embs_n @ pe                                 # [m]
            obj_div = np.clip(0.5 - sims * 0.5, 0.0, 1.0)       # remap → [0,1]
        else:
            obj_div = np.full(m, 0.5, dtype=np.float64)

        # ── Objective 4: temporal coherence ───────────────────────────────────
        if sequence:
            prev_ts = float(sequence[-1].get("exif_ts", 0.0))
            this_ts = np.array([float(candidates[i].get("exif_ts", 0.0))
                                for i in cand_ids], dtype=np.float64)
            gaps    = np.abs(this_ts - prev_ts)
            max_gap = float(gaps.max()) if gaps.max() > 0 else 1.0
            obj_time = 1.0 - gaps / max_gap        # penalise huge jumps
        else:
            obj_time = np.full(m, 0.5, dtype=np.float64)

        # ── Objective 5: event diversity ──────────────────────────────────────
        obj_event = np.array(
            [0.3 if event_of.get(i, -2) != prev_event else 0.0 for i in cand_ids],
            dtype=np.float64,
        )

        # ── Objective matrix [m × 5] ──────────────────────────────────────────
        obj_matrix = np.stack(
            [obj_role, obj_qual, obj_div, obj_time, obj_event], axis=1
        )

        # ── Pareto front ──────────────────────────────────────────────────────
        front_mask = _pareto_front(obj_matrix)
        front_local = np.where(front_mask)[0]        # indices into cand_ids

        # ── Role-weighted scalarization over Pareto front ─────────────────────
        W = np.array([
            rw["comp_weight"] * 0.8 + 0.10,          # role fit
            0.25,                                     # quality
            rw["diversity_penalty"],                  # visual diversity
            0.10,                                     # temporal coherence
            0.10,                                     # event diversity
        ], dtype=np.float64)
        W /= W.sum()

        front_scores = obj_matrix[front_local] @ W
        jitter       = rng.uniform(0.0, jitter_scale, size=len(front_local))
        best_local   = int(np.argmax(front_scores + jitter))
        best_cand_id = cand_ids[front_local[best_local]]

        # ── Record selection ──────────────────────────────────────────────────
        raw_obj = obj_matrix[front_local[best_local]].tolist()
        selected = dict(candidates[best_cand_id])
        selected["slot"]             = role_name
        selected["mogco_objectives"] = {
            "role_fit":    round(raw_obj[0], 3),
            "quality":     round(raw_obj[1], 3),
            "visual_div":  round(raw_obj[2], 3),
            "temporal":    round(raw_obj[3], 3),
            "event_div":   round(raw_obj[4], 3),
        }

        sequence.append(selected)
        remaining_ids.discard(best_cand_id)
        prev_emb  = candidates[best_cand_id]["embedding"]
        prev_event = event_of.get(best_cand_id, -2)

    return sequence


# ── Beam-search variant (DuckDB-native) ───────────────────────────────────────

def run_mogco_beam(
    db_path: str,
    vibe_vec: np.ndarray | None = None,
    vibe_thresh: float = 0.60,
    target: int = 5,
    beam_width: int = 4,
    min_score: float = 0.45,
    _rows: list | None = None,
) -> dict:
    """
    Beam-search MOGCO that queries DuckDB directly for candidates.

    Unlike ``mogco_sequence`` (which accepts pre-fetched dicts), this function
    owns its own DuckDB connection and pulls embeddings in one vectorised query,
    making it ideal for large libraries where fetching everything up front would
    be wasteful.

    Parameters
    ----------
    db_path     : absolute path to the DuckDB file (cache/cache.db)
    vibe_vec    : optional reference embedding — candidates with cosine similarity
                  below ``vibe_thresh`` are dropped before sequencing.
                  Pass the embedding of a "mood reference" photo for style filtering.
    vibe_thresh : minimum cosine similarity to the vibe vector (default 0.60)
    target      : number of frames to select (default 5)
    beam_width  : parallel paths kept alive during beam search (default 4)
    min_score   : hard quality floor applied in the DB query (default 0.45)
    _rows       : pre-fetched rows [(path, score, embedding, breakdown, exif_ts), ...]
                  skips the DuckDB connection entirely when provided (avoids Windows
                  file-lock conflict when PhotoCache already holds a write connection)

    Returns
    -------
    dict with keys:
        paths        list[str]   – selected file paths in sequence order
        slots        list[str]   – narrative role labels (opener, subject, …)
        global_score float       – cumulative beam score (lower = tighter edit)
        beam_objectives list     – per-frame scores for transparency
    """

    # ── 1. Pull candidates from DuckDB ────────────────────────────────────────
    if _rows is not None:
        rows = _rows
    else:
        import duckdb
        conn  = duckdb.connect(db_path, read_only=True)
        rows  = conn.execute(
            "SELECT path, score, embedding, breakdown, exif_ts FROM photos WHERE score >= ?",
            [min_score],
        ).fetchall()
        conn.close()

    if len(rows) < target:
        return {"error": f"Only {len(rows)} photos meet score ≥ {min_score} (need {target})."}

    raw_paths      = [r[0] for r in rows]
    raw_scores     = np.array([float(r[1] or 0.0) for r in rows], dtype=np.float64)
    raw_embs       = np.array(
        [list(r[2]) if r[2] is not None else [] for r in rows], dtype=np.float64
    )
    raw_breakdowns = []
    for r in rows:
        try:
            raw_breakdowns.append(json.loads(r[3]) if r[3] else {})
        except Exception:
            raw_breakdowns.append({})
    raw_ts = np.array([float(r[4] or 0.0) for r in rows], dtype=np.float64)

    # Drop rows with zero-norm embeddings
    valid_norms = np.linalg.norm(raw_embs, axis=1)
    valid_mask  = valid_norms > 1e-6
    if not np.any(valid_mask):
        return {"error": "No photos have valid embeddings in DuckDB."}

    paths      = [p for p, v in zip(raw_paths,      valid_mask) if v]
    scores     = raw_scores[valid_mask]
    embs       = raw_embs[valid_mask]
    breakdowns = [b for b, v in zip(raw_breakdowns, valid_mask) if v]
    timestamps = raw_ts[valid_mask]

    # ── 2. Vibe filter — style-similarity gate ────────────────────────────────
    if vibe_vec is not None:
        v_norm = np.asarray(vibe_vec, dtype=np.float64)
        vnorm  = np.linalg.norm(v_norm)
        if vnorm < 1e-9:
            vibe_vec = None              # guard against zero vector
        else:
            v_norm      = v_norm / vnorm
            embs_norm   = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
            sims        = embs_norm @ v_norm          # [N]
            vibe_mask   = sims >= vibe_thresh
            if vibe_mask.sum() < target:
                # not enough matching style — relax to score fallback
                vibe_mask = scores >= 0.30
            paths      = [p for p, m in zip(paths, vibe_mask) if m]
            embs       = embs[vibe_mask]
            scores     = scores[vibe_mask]
            breakdowns = [b for b, m in zip(breakdowns, vibe_mask) if m]
            timestamps = timestamps[vibe_mask]

    if len(paths) < target:
        return {"paths": paths, "slots": [], "global_score": 0.0, "beam_objectives": [],
                "error": f"Only {len(paths)} candidates after vibe filter."}

    # ── 3. Similarity matrix (vectorised cosine) ──────────────────────────────
    e_norm    = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim_matrix = (e_norm @ e_norm.T + 1.0) / 2.0      # remap [-1, 1] → [0, 1]

    # ── 4. Beam search ────────────────────────────────────────────────────────
    role_names = list(SHOT_ROLES.keys())

    # Initialise beam with top-score candidates
    init_idxs = np.argsort(-scores)[:beam_width]
    beam: list[dict] = [
        {
            "indices": [int(i)],
            "score":   float(scores[i]),
            "log":     [{"path": paths[i], "slot": role_names[0], "step_score": float(scores[i])}],
        }
        for i in init_idxs
    ]

    for step in range(1, target):
        slot_name  = role_names[step] if step < len(role_names) else role_names[-1]
        rw         = SHOT_ROLES.get(slot_name, list(SHOT_ROLES.values())[-1])
        next_beam: list[dict] = []

        for state in beam:
            last_idx = state["indices"][-1]
            used     = set(state["indices"])

            for cand in range(len(paths)):
                if cand in used:
                    continue

                # Flow: reward transitions near 0.55 cosine (not too close, not too far)
                flow = 1.0 - abs(sim_matrix[last_idx, cand] - 0.55)

                # Quality from grader score
                quality = float(scores[cand]) * 0.40

                # Role fit from breakdown dimensions (preset-agnostic key lookup)
                bd = breakdowns[cand]
                comp  = _dv(bd, _COMP_KEYS)
                human = _dv(bd, _HUMAN_KEYS)
                mood  = _dv(bd, _LIGHT_KEYS)
                role_fit = (
                    comp  * rw["comp_weight"] +
                    human * rw["human_weight"] +
                    mood  * rw["mood_weight"]
                ) * 0.30

                step_score = quality + role_fit + flow * 0.30
                total      = state["score"] + step_score

                next_beam.append({
                    "indices": state["indices"] + [cand],
                    "score":   total,
                    "log": state["log"] + [{
                        "path":       paths[cand],
                        "slot":       slot_name,
                        "step_score": round(step_score, 4),
                        "flow":       round(float(flow), 4),
                        "quality":    round(quality, 4),
                        "role_fit":   round(role_fit, 4),
                    }],
                })

        next_beam.sort(key=lambda x: -x["score"])
        beam = next_beam[:beam_width]

    best = beam[0]
    return {
        "paths":            [paths[i]      for i in best["indices"]],
        "slots":            [role_names[s] for s in range(len(best["indices"]))],
        "global_score":     round(best["score"], 4),
        "beam_objectives":  best["log"],
    }
