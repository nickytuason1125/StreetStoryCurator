"""
Creative Director — MOGCO-II parameter optimizer for Flux 2 [klein] stylization.

For each Strong image MOGCO-II finds the Pareto-optimal
(img2img strength, CFG guidance, ControlNet weight) triple across 3 objectives:

  1. style_fidelity    — how strongly the anchor aesthetic is applied
                         Proxy: (guidance - min_g) / (max_g - min_g)  → [0, 1]
  2. struct_integrity  — how faithfully the original geometry is preserved
                         Proxy: 1 − strength          (lower noise → more structure)
  3. set_cohesion      — how naturally this image fits the Strong portfolio set
                         Signal: mean SigLIP cosine sim to the bucket centroid

All three objectives are in [0, 1] and higher is better.
The Pareto front over the 48-point parameter grid is identical in logic to
mogco2.py but operates in parameter space rather than image space.

Per-image narrative roles are assigned by image content (not list position):
  subject  → highest aesthetic score (hero shot)
  opener   → most representative of the set (closest to centroid)
  closer   → second-highest score
  contrast → most visually distinct from the centroid (counterpoint)
  detail   → third-highest score (texture / decisive gesture)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional

# ── Parameter search space (4 × 4 × 3 = 48 combinations) ────────────────────

_STRENGTHS    = [0.25, 0.40, 0.55, 0.70]   # img2img denoising strength
_GUIDANCES    = [2.0,  3.5,  5.0,  7.5]    # CFG guidance scale
_CTRL_WEIGHTS = [0.40, 0.60, 0.80]          # ControlNet conditioning scale

_G_MIN = _GUIDANCES[0]   # 2.0
_G_MAX = _GUIDANCES[-1]  # 7.5

_PARAM_GRID: list[dict] = [
    {"strength": s, "guidance": g, "ctrl_weight": cw}
    for s  in _STRENGTHS
    for g  in _GUIDANCES
    for cw in _CTRL_WEIGHTS
]

# ── Shot role weights for creative direction ──────────────────────────────────

_CD_ROLES: dict[str, dict[str, float]] = {
    "opener":   {"style": 0.20, "structure": 0.30, "cohesion": 0.50},
    "subject":  {"style": 0.55, "structure": 0.15, "cohesion": 0.30},
    "detail":   {"style": 0.15, "structure": 0.60, "cohesion": 0.25},
    "contrast": {"style": 0.25, "structure": 0.25, "cohesion": 0.50},
    "closer":   {"style": 0.35, "structure": 0.30, "cohesion": 0.35},
}

_ROLE_ORDER = ["subject", "opener", "closer", "contrast", "detail"]


# ── Pareto front ──────────────────────────────────────────────────────────────

def _pareto_front(obj: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows (higher-is-better on all axes)."""
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
        mask = at_least_good & strictly_good
        mask[i] = False
        if mask.any():
            dominated[i] = True
    return ~dominated


# ── Objective proxy functions ─────────────────────────────────────────────────

def _style_fidelity_proxy(guidance: float) -> float:
    """Map actual guidance range [2.0, 7.5] → [0.0, 1.0]."""
    return float(np.clip((guidance - _G_MIN) / (_G_MAX - _G_MIN), 0.0, 1.0))


def _struct_integrity_proxy(strength: float) -> float:
    return float(np.clip(1.0 - strength, 0.0, 1.0))


def _set_cohesion_signal(
    image_emb: np.ndarray,
    bucket_embs: np.ndarray,
) -> float:
    """Mean cosine similarity of this image to every other Strong image."""
    if len(bucket_embs) == 0:
        return 0.0   # no peers → no cohesion signal
    img_n    = image_emb  / (np.linalg.norm(image_emb)  + 1e-9)
    bucket_n = bucket_embs / (np.linalg.norm(bucket_embs, axis=1, keepdims=True) + 1e-9)
    sims     = bucket_n @ img_n
    return float(np.clip(np.mean(sims), 0.0, 1.0))


# ── Content-aware role assignment ─────────────────────────────────────────────

def _assign_roles_by_content(
    embeddings: list[np.ndarray],
    scores: Optional[list[float]] = None,
    paths: Optional[list[str]] = None,
) -> list[str]:
    """
    Assign narrative roles based on image content, not list order.

    subject  → highest aesthetic score (hero shot)
    opener   → most representative of the set (closest to centroid)
    closer   → second-highest score
    contrast → most visually distinct (furthest from centroid)
    detail   → third-highest score (texture / gesture)

    Remaining images beyond 5 cycle through _ROLE_ORDER.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return ["subject"]

    embs   = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-9)

    centroid = embs_n.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim_to_centroid = embs_n @ centroid   # (N,)

    sc = np.array(scores if scores and len(scores) == n else [0.5] * n,
                  dtype=np.float32)

    used        = set()
    assignments: dict[int, str] = {}

    def _pick(rank_arr: np.ndarray) -> int:
        for idx in rank_arr:
            if int(idx) not in used:
                used.add(int(idx))
                return int(idx)
        return -1

    score_desc    = np.argsort(-sc)
    centroid_desc = np.argsort(-sim_to_centroid)
    centroid_asc  = np.argsort(sim_to_centroid)

    for role, rank in [
        ("subject",  score_desc),
        ("opener",   centroid_desc),
        ("closer",   score_desc),
        ("contrast", centroid_asc),
        ("detail",   score_desc),
    ]:
        if len(used) >= n:
            break
        idx = _pick(rank)
        if idx >= 0:
            assignments[idx] = role

    # Remaining images cycle through roles
    ri = 0
    for i in range(n):
        if i not in assignments:
            assignments[i] = _ROLE_ORDER[ri % len(_ROLE_ORDER)]
            ri += 1

    result_roles = [assignments[i] for i in range(n)]
    labels = paths or [f"img_{i}" for i in range(n)]
    for i, (role, lbl) in enumerate(zip(result_roles, labels)):
        print(
            f"[cd] role={role:8s}  score={sc[i]:.3f}  "
            f"centroid_sim={sim_to_centroid[i]:.3f}  {Path(lbl).name}"
        )
    return result_roles


# ── Per-image parameter selection via MOGCO-II ───────────────────────────────

def pick_params_for_image(
    image_emb:   np.ndarray,
    bucket_embs: np.ndarray,
    role:        str  = "subject",
    jitter_scale: float = 0.01,
    rng_seed:    int  = 42,
) -> dict:
    """
    Run MOGCO-II over the 48-point parameter grid for a single image.
    Returns the winning param dict: {strength, guidance, ctrl_weight}.
    """
    rng      = np.random.default_rng(rng_seed)
    rw       = _CD_ROLES.get(role, _CD_ROLES["subject"])
    cohesion = _set_cohesion_signal(image_emb, bucket_embs)

    # Build (N_params, 3) objective matrix
    obj = np.array(
        [
            [
                _style_fidelity_proxy(p["guidance"]),
                _struct_integrity_proxy(p["strength"]),
                cohesion,   # same for all params — image-level fit to set
            ]
            for p in _PARAM_GRID
        ],
        dtype=np.float32,
    )

    front_mask  = _pareto_front(obj)
    front_local = np.where(front_mask)[0]

    W = np.array([rw["style"], rw["structure"], rw["cohesion"]], dtype=np.float64)
    W /= W.sum()
    front_scores = obj[front_local] @ W
    jitter       = rng.uniform(0.0, jitter_scale, size=len(front_local))
    best_local   = int(np.argmax(front_scores + jitter))
    best_idx     = front_local[best_local]

    result = dict(_PARAM_GRID[best_idx])
    result["mogco_objectives"] = {
        "style_fidelity":   round(float(obj[best_idx, 0]), 3),
        "struct_integrity": round(float(obj[best_idx, 1]), 3),
        "set_cohesion":     round(float(cohesion),         3),
    }
    print(
        f"[cd]   params({role}): "
        f"strength={result['strength']:.2f}  guidance={result['guidance']:.1f}  "
        f"ctrl={result['ctrl_weight']:.2f}  pareto_front={len(front_local)}/{len(obj)}  "
        f"style={result['mogco_objectives']['style_fidelity']:.3f}  "
        f"struct={result['mogco_objectives']['struct_integrity']:.3f}  "
        f"cohesion={cohesion:.3f}  weights=({W[0]:.2f},{W[1]:.2f},{W[2]:.2f})"
    )
    return result


# ── Batch parameter planning ──────────────────────────────────────────────────

def plan_batch(
    strong_paths: list[str],
    embeddings:   list[np.ndarray],
    scores:       Optional[list[float]] = None,
    style_prompt: str = "",
    num_steps:    int = 4,
) -> list[dict]:
    """
    For each Strong image run MOGCO-II to select the optimal stylization
    parameters and assign a narrative role.  Returns one param dict per image.

    Roles are assigned by image content (score + embedding), not list position.
    """
    if not embeddings:
        return [{"strength": 0.50, "guidance": 3.5, "ctrl_weight": 0.60,
                 "prompt": style_prompt, "num_steps": num_steps, "role": "subject"}
                for _ in strong_paths]

    bucket_embs = np.stack(embeddings).astype(np.float32)

    print(f"[cd] plan_batch: {len(strong_paths)} images, scores={[round(s,3) for s in (scores or [])]}")
    # Content-aware role assignment
    roles = _assign_roles_by_content(embeddings, scores=scores, paths=strong_paths)

    params_list = []
    for i, (path, emb) in enumerate(zip(strong_paths, embeddings)):
        role  = roles[i]
        param = pick_params_for_image(
            image_emb   = np.asarray(emb, dtype=np.float32),
            bucket_embs = bucket_embs,
            role        = role,
            rng_seed    = i * 31 + 7,
        )
        param["prompt"]    = style_prompt
        param["num_steps"] = num_steps
        param["role"]      = role

        print(
            f"[cd] {Path(path).name} ({role}): "
            f"strength={param['strength']:.2f}  "
            f"guidance={param['guidance']:.1f}  "
            f"ctrl={param['ctrl_weight']:.2f}  "
            f"cohesion={param['mogco_objectives']['set_cohesion']:.3f}"
        )
        params_list.append(param)

    return params_list


# ── Top-level orchestrator ────────────────────────────────────────────────────

def run_creative_direction(
    strong_paths:   list[str],
    embeddings:     list[np.ndarray],
    anchor_path:    str,
    output_dir:     str,
    scores:         Optional[list[float]] = None,
    style_prompt:   str = "",
    structure_mode: str = "canny",
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Full Creative Direction pipeline:

    1. MOGCO-II plans optimal (strength, guidance, ctrl_weight) per image,
       assigning roles by content (score + embedding similarity).
    2. Flux 2 [klein] stylizes each image sequentially with VRAM flushing.
    3. Saves stylized images to output_dir/Final_Portfolio/.
    4. Returns a results dict compatible with the SSE stream handler.
    """
    from flux_stylizer import FluxStylizer

    _p = progress or (lambda f, d: None)

    n = len(strong_paths)
    if n == 0:
        return {"error": "No Strong images to stylize.", "outputs": [], "total": 0}

    out_dir = Path(output_dir) / "Final_Portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: MOGCO-II parameter planning ──────────────────────────────────
    _p(0.02, f"MOGCO-II planning parameters for {n} images…")
    params_per_image = plan_batch(
        strong_paths = strong_paths,
        embeddings   = embeddings,
        scores       = scores,
        style_prompt = style_prompt,
    )
    _p(0.08, "Parameters planned")

    # ── Steps 2–4: Flux batch stylization ────────────────────────────────────
    stylizer = FluxStylizer()
    outputs  = stylizer.process_batch(
        strong_paths     = strong_paths,
        anchor_path      = anchor_path,
        output_dir       = out_dir,
        params_per_image = params_per_image,
        structure_mode   = structure_mode,
        style_prompt     = style_prompt,
        progress         = lambda f, d: _p(0.10 + f * 0.90, d),
    )

    success = [r for r in outputs if r["success"]]
    return {
        "outputs":     outputs,
        "output_dir":  str(out_dir),
        "total":       n,
        "success":     len(success),
        "failed":      n - len(success),
        "anchor_path": anchor_path,
    }
