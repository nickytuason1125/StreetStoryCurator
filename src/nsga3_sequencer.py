"""
NSGA-III Sequencer — Strict Literal Constraint Edition

Two-phase architecture:
  Phase 1: Hard literal pre-filter gate
    - Extract structural constraints from the creative brief (rule-based;
      Phi-4 Mini Judge used if phi4_mini_judge module is installed)
    - Drop any candidate that violates even one constraint before the
      evolutionary algorithm sees it
    - Raise SequencerConstraintError if the remaining pool is too small

  Phase 2: Slot-locked NSGA-III optimisation
    - 5 fixed roles: Opener · Subject · Contrast · Detail · Closer
    - pymoo ElementwiseProblem with n_ieq_constr=5 slot-fitness penalties
    - A candidate with g[slot] > 0 is infeasible and never enters the
      Pareto front regardless of diversity or aesthetic scores
    - Greedy fallback honours the same strict slot constraints

  No soft compensation: a literal mismatch cannot be offset by higher
  visual diversity or aesthetic score.

API
───
    run_nsga3_sequence_with_vlm(candidates, target, progress, brief) → list[dict]
    run_nsga3_sequence(candidates, target, progress)                  → list[dict]
    SequencerConstraintError                                          (exception)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

# ── Slot definitions ──────────────────────────────────────────────────────────

SLOT_NAMES = ["Opener", "Subject", "Contrast", "Detail", "Closer"]
SLOT_DESC  = {
    "Opener":   "Wide/Scale — establishing shot, spatial context, cityscape or landscape",
    "Subject":  "Focal Point — decisive moment, human element, or strong visual anchor",
    "Contrast": "Luminance/Color shift — dramatic tonal jump from the sequence midpoint",
    "Detail":   "Macro/Texture — close-up surface, pattern, or intimate detail",
    "Closer":   "Vanishing/Finality — receding perspective, horizon, or resolution",
}
SLOT_THRESHOLD  = 0.25   # global fallback — overridden per-run by DirectorBrief boundary params
_SIM_HARD_LIMIT_DEFAULT = 0.82   # global fallback — overridden per-run by DirectorBrief


# ── Constraint model ──────────────────────────────────────────────────────────

@dataclass
class LiteralConstraints:
    people_allowed:  bool      = True
    required_tags:   Set[str]  = field(default_factory=set)
    forbidden_tags:  Set[str]  = field(default_factory=set)
    min_score:       float     = 0.0
    monochrome_only: bool      = False

    def is_empty_brief(self) -> bool:
        return not self.people_allowed


class SequencerConstraintError(RuntimeError):
    """Raised when literal constraints eliminate all valid candidate combinations."""


# ── Step 1: Literal constraint extraction ────────────────────────────────────

_EMPTY_KWS    = {"empty", "no people", "nobody", "deserted", "abandoned",
                 "liminal", "void", "desolate", "uninhabited", "evacuated"}
_INTERIOR_KWS = {"interior", "indoor", "inside", "room", "corridor", "hallway"}
_EXTERIOR_KWS = {"exterior", "outdoor", "outside", "street", "alley", "cityscape"}
_MONO_KWS     = {"black and white", "monochrome", "b&w", "greyscale", "grayscale"}


def _extract_literal_constraints(brief: str) -> LiteralConstraints:
    """
    Extract hard structural constraints from the creative direction brief.

    Priority: Phi-4 Mini Judge (if phi4_mini_judge is installed) →
              rule-based parsing of common keyword patterns.
    """
    if not brief:
        return LiteralConstraints()

    try:
        from phi4_mini_judge import extract_sequence_constraints
        return extract_sequence_constraints(brief)
    except ImportError:
        pass

    lo  = brief.lower()
    c   = LiteralConstraints()

    if any(kw in lo for kw in _EMPTY_KWS):
        c.people_allowed = False
    if any(kw in lo for kw in _INTERIOR_KWS):
        c.required_tags.add("interior")
    if any(kw in lo for kw in _EXTERIOR_KWS):
        c.required_tags.add("exterior")
    if any(kw in lo for kw in _MONO_KWS):
        c.monochrome_only = True

    return c


# ── Step 2: Literal pre-filter gate ──────────────────────────────────────────

def _passes_literal_constraints(
    candidate: dict,
    c: LiteralConstraints,
    yolo_detections: Optional[Set[str]] = None,
) -> bool:
    """
    Return True only if the candidate satisfies ALL literal constraints.
    A single False short-circuits — no compensation allowed.
    """
    path = candidate.get("path", "")

    # People constraint — prefer YOLO ground-truth; fall back to log heuristic
    if not c.people_allowed:
        if yolo_detections is not None:
            if path in yolo_detections:
                return False
        else:
            log = (candidate.get("reasoning_log", "") or "").lower()
            if any(w in log for w in ("person", "people", "pedestrian",
                                      "figure", "crowd", "human")):
                return False

    # Score floor
    if float(candidate.get("score", 0.0)) < c.min_score:
        return False

    # Tag constraints
    log_lower = (candidate.get("reasoning_log", "") or "").lower()
    for tag in c.required_tags:
        if tag not in log_lower:
            return False
    for tag in c.forbidden_tags:
        if tag in log_lower:
            return False

    return True


def _apply_literal_prefilter(
    candidates: List[Dict[str, Any]],
    constraints: LiteralConstraints,
    yolo_detections: Optional[Set[str]] = None,
    min_pool: int = 5,
) -> List[Dict[str, Any]]:
    """
    Hard-filter candidates.  Raises SequencerConstraintError if the pool
    shrinks below min_pool — never falls back to a compromised selection.
    """
    filtered  = [
        c for c in candidates
        if _passes_literal_constraints(c, constraints, yolo_detections)
    ]
    n_dropped = len(candidates) - len(filtered)
    if n_dropped:
        print(f"[nsga3] Literal pre-filter: dropped {n_dropped}/{len(candidates)} candidates")
    if len(filtered) < min_pool:
        raise SequencerConstraintError(
            f"Disqualified: No combination matches 100% of the literal constraints. "
            f"{len(filtered)}/{len(candidates)} candidates survive pre-filter "
            f"(need at least {min_pool})."
        )
    return filtered


def _isolate_cluster_bests(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Upstream finalist isolation (Requirement 2).

    For every cluster_id >= 0, keep only the single candidate flagged as
    'Best of N'.  Unclassed photos (cluster_id == -1 or missing) pass
    through unchanged — each is treated as visually unique.

    Hard-excludes any 'Duplicate' sim_flag before optimisation begins so
    the NSGA-III never evaluates visually redundant candidates.
    """
    from collections import defaultdict as _dd
    clusters: dict = _dd(list)
    no_cluster: list = []

    for c in candidates:
        cid = c.get("cluster_id", -1)
        if cid is None or int(cid) < 0:
            no_cluster.append(c)
        else:
            clusters[int(cid)].append(c)

    isolated: list = list(no_cluster)
    for members in clusters.values():
        best = next(
            (m for m in members if "Best" in (m.get("sim_flag") or "")),
            None,
        )
        if best is None:
            best = max(members, key=lambda m: float(m.get("score", 0.0)))
        isolated.append(best)

    n_dropped = len(candidates) - len(isolated)
    if n_dropped:
        print(
            f"[nsga3] Cluster isolation: removed {n_dropped} duplicate-cluster candidates "
            f"({len(isolated)} unique finalists remain)"
        )
    return isolated


# ── Step 3: Slot role classification ─────────────────────────────────────────

_SLOT_KEYWORDS: Dict[str, List[str]] = {
    "Opener":   ["wide", "establishing", "scale", "cityscape", "landscape",
                 "architecture", "overview", "exterior", "skyline", "panoramic"],
    "Subject":  ["subject", "face", "person", "portrait", "decisive",
                 "emotion", "candid", "human", "figure", "expression"],
    "Contrast": ["contrast", "shadow", "highlight", "dramatic", "silhouette",
                 "chiaroscuro", "dark", "bright", "high-key", "low-key", "tonal"],
    "Detail":   ["texture", "pattern", "detail", "close", "macro",
                 "surface", "grain", "material", "intimate", "abstract"],
    "Closer":   ["vanishing", "perspective", "depth", "horizon",
                 "receding", "resolution", "closure", "finality", "fade"],
}

_SLOT_ASPECTS: Dict[str, Dict[str, float]] = {
    "Opener":   {"Composition": 0.40, "Technical": 0.30, "Lighting": 0.30},
    "Subject":  {"Human/Culture": 0.50, "Narrative": 0.35, "Technical": 0.15},
    "Contrast": {"Lighting": 0.60, "Technical": 0.25, "Composition": 0.15},
    "Detail":   {"Technical": 0.50, "Composition": 0.30, "Lighting": 0.20},
    "Closer":   {"Composition": 0.45, "Lighting": 0.35, "Narrative": 0.20},
}


def _luminance(path: str) -> float:
    try:
        from PIL import Image
        with Image.open(path) as _raw:
            img = _raw.convert("L")
        img.thumbnail((64, 64), Image.LANCZOS)
        return float(np.asarray(img, dtype=np.float32).mean() / 255.0)
    except Exception:
        return 0.5


def _slot_score_for(
    candidate: dict,
    slot: str,
    lum: float,
    all_lums: np.ndarray,
) -> float:
    """Compute fitness of `candidate` for `slot` in [0, 1]."""
    log  = (candidate.get("reasoning_log", "") or "").lower()
    bd   = candidate.get("breakdown", {}) or {}
    kws  = _SLOT_KEYWORDS.get(slot, [])

    kw_hits = sum(1 for kw in kws if kw in log)
    kw_sc   = min(1.0, kw_hits / max(2, len(kws) * 0.3))

    asp_weights = _SLOT_ASPECTS.get(slot, {})
    asp_sc = sum(float(bd.get(asp, 0.5)) * w for asp, w in asp_weights.items())

    # Luminance bonus: Contrast slot rewards extreme brightness deviation
    lum_sc = 0.0
    if slot == "Contrast" and len(all_lums) > 1:
        lum_sc = float(np.clip(abs(lum - float(all_lums.mean())) * 3.0, 0.0, 1.0))
    elif slot == "Opener":
        lum_sc = float(np.clip(lum * 1.5, 0.0, 1.0))

    if slot == "Contrast":
        return round(float(0.30 * kw_sc + 0.30 * asp_sc + 0.40 * lum_sc), 3)
    elif slot == "Opener":
        return round(float(0.30 * kw_sc + 0.55 * asp_sc + 0.15 * lum_sc), 3)
    else:
        return round(float(0.35 * kw_sc + 0.65 * asp_sc), 3)


def _classify_slot_roles(candidates: List[Dict[str, Any]]) -> np.ndarray:
    """Return (n, 5) float32 matrix: row=candidate, col=slot fitness."""
    n    = len(candidates)
    lums = np.array([_luminance(c.get("path", "")) for c in candidates], dtype=np.float32)
    role_scores = np.zeros((n, 5), dtype=np.float32)
    for i, c in enumerate(candidates):
        for j, slot in enumerate(SLOT_NAMES):
            role_scores[i, j] = _slot_score_for(c, slot, float(lums[i]), lums)
    return role_scores


# ── Objective helpers ─────────────────────────────────────────────────────────

def _semantic_brief_match(candidate: dict) -> float:
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


def _pre_compute_objectives(candidates: List[Dict[str, Any]]) -> np.ndarray:
    n   = len(candidates)
    obj = np.zeros((n, 3), dtype=np.float32)
    for i, c in enumerate(candidates):
        obj[i, 0] = _semantic_brief_match(c)
        obj[i, 1] = _luminance(c.get("path", ""))
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
    idx = indices.astype(int)
    idx = idx[(idx >= 0) & (idx < len(obj_matrix))]
    if len(idx) == 0:
        return np.array([0.0, 0.5, 0.5], dtype=np.float32)

    f1 = float(np.mean(obj_matrix[idx, 0]))

    if len(idx) > 1:
        sims  = sim_matrix[np.ix_(idx, idx)]
        upper = sims[np.triu_indices(len(idx), k=1)]
        # Use MAX (worst-pair) similarity so one near-duplicate pair tanks diversity,
        # regardless of how diverse the other pairs are. Removes fuzzy compensation.
        f2    = float(1.0 - np.max(upper))
    else:
        f2 = 0.5

    lum = obj_matrix[idx, 1]
    std = float(np.std(lum)) if len(lum) > 1 else 0.0
    f3  = float(np.clip(1.0 - std * 3.0, 0.0, 1.0))

    return np.array([f1, f2, f3], dtype=np.float32)


# ── pymoo problem ─────────────────────────────────────────────────────────────

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


if HAS_PYMOO:
    class _SequenceProblem(ElementwiseProblem):
        """
        Slot-locked 5-variable selection problem.

        x = [i_Opener, i_Subject, i_Contrast, i_Detail, i_Closer]

        Inequality constraints g <= 0 = satisfied:
            g[0..4]  slot fitness  (slot_thresholds[j] − role_scores[x[j], j])
            g[5]     cluster uniqueness  (duplicate cluster_id count among 5)
            g[6]     pairwise similarity  (max off-diagonal cosine − sim_limit)

        Per-slot thresholds and sim_limit are injected from DirectorBrief
        boundary params — not global constants — so brief-specific constraints
        tighten or relax slot gates without touching global state.

        Any g > 0 renders the solution infeasible — no objective score can
        compensate a violated constraint.
        """

        def __init__(
            self,
            n_candidates: int,
            obj_matrix: np.ndarray,
            sim_matrix: np.ndarray,
            role_scores: np.ndarray,
            cluster_ids: Optional[List[int]] = None,
            slot_thresholds: Optional[List[float]] = None,
            sim_limit: float = _SIM_HARD_LIMIT_DEFAULT,
        ):
            self.n_candidates    = n_candidates
            self.obj_matrix      = obj_matrix
            self.sim_matrix      = sim_matrix
            self.role_scores     = role_scores
            self.cluster_ids     = cluster_ids
            self.slot_thresholds = slot_thresholds if slot_thresholds else [SLOT_THRESHOLD] * 5
            self._sim_limit      = sim_limit
            super().__init__(
                n_var=5,
                n_obj=3,
                n_ieq_constr=7,              # 5 slot + 1 cluster + 1 similarity
                xl=np.zeros(5),
                xu=np.full(5, float(n_candidates - 1)),
                vtype=float,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            indices = np.round(x).astype(int).clip(0, self.n_candidates - 1)

            # g[0..4]: per-slot fitness — threshold from DirectorBrief boundary params
            g: List[float] = [
                self.slot_thresholds[j] - float(self.role_scores[int(indices[j]), j])
                for j in range(5)
            ]

            # g[5]: cluster uniqueness — count duplicate cluster memberships
            if self.cluster_ids is not None:
                cids     = [self.cluster_ids[int(indices[j])] for j in range(5)]
                pos_cids = [c for c in cids if c >= 0]
                g_cluster = float(len(pos_cids) - len(set(pos_cids)))
            else:
                g_cluster = 0.0

            # g[6]: pairwise similarity hard limit (per-run, from DirectorBrief)
            sub_sim = self.sim_matrix[np.ix_(indices, indices)]
            upper   = sub_sim[np.triu_indices(5, k=1)]
            max_sim = float(np.max(upper)) if len(upper) > 0 else 0.0
            g_sim   = max(0.0, max_sim - self._sim_limit)

            out["G"] = np.array(g + [g_cluster, g_sim], dtype=float)

            # Objectives (negated for minimisation)
            objs = _eval_set_objectives(indices, self.obj_matrix, self.sim_matrix)
            # Exponential penalty degrades objective score for near-duplicate pairs
            if max_sim > self._sim_limit:
                penalty = float(np.exp(10.0 * (max_sim - self._sim_limit)))
                objs    = objs / penalty
            out["F"] = -objs

    class _UniqueRepair(Repair):
        def _do(self, problem, pop, **kwargs):
            for individual in pop:
                x         = np.round(individual.X).astype(int)
                seen      = set()
                available = [i for i in range(problem.n_candidates) if i not in set(x)]
                for j in range(len(x)):
                    v = int(x[j])
                    if v in seen or v >= problem.n_candidates:
                        x[j] = available.pop(0) if available else np.random.randint(0, problem.n_candidates)
                    seen.add(int(x[j]))
                individual.X = x.astype(float)
            return pop


# ── Greedy slot-respecting fallback ──────────────────────────────────────────

def _greedy_slot_select(
    candidates: List[Dict[str, Any]],
    obj_matrix: np.ndarray,
    sim_matrix: np.ndarray,
    role_scores: np.ndarray,
    cluster_ids: Optional[List[int]] = None,
    slot_thresholds: Optional[List[float]] = None,
    sim_limit: float = _SIM_HARD_LIMIT_DEFAULT,
) -> List[int]:
    """
    Greedy slot-assignment enforcing slot fitness, cluster uniqueness, and
    pairwise similarity below sim_limit.  Per-slot thresholds injected from
    DirectorBrief boundary params — not global SLOT_THRESHOLD.

    Two-pass: first pass applies all constraints; if a slot finds no eligible
    candidate, second pass relaxes only the similarity constraint (cluster
    uniqueness is always hard).
    Raises SequencerConstraintError if any slot still has no eligible candidate.
    """
    _SIM_LIM      = sim_limit
    _SLOT_THRESH  = slot_thresholds if slot_thresholds else [SLOT_THRESHOLD] * 5
    n             = len(candidates)
    used          = set()
    used_clusters: Set[int] = set()
    result: List[int] = []

    for slot_i, slot in enumerate(SLOT_NAMES):
        def _build_eligible(relax_sim: bool) -> List[int]:
            out: List[int] = []
            for i in range(n):
                if i in used:
                    continue
                if float(role_scores[i, slot_i]) < _SLOT_THRESH[slot_i]:
                    continue
                if cluster_ids is not None:
                    cid = cluster_ids[i]
                    if cid >= 0 and cid in used_clusters:
                        continue
                if not relax_sim and result:
                    trial   = result + [i]
                    sims    = sim_matrix[np.ix_(trial, trial)]
                    upper   = sims[np.triu_indices(len(trial), k=1)]
                    if float(np.max(upper)) > _SIM_LIM:
                        continue
                out.append(i)
            return out

        eligible = _build_eligible(relax_sim=False)
        if not eligible:
            eligible = _build_eligible(relax_sim=True)   # relax similarity only
        if not eligible:
            raise SequencerConstraintError(
                "Sequencing Failed: Insufficient visual diversity in candidate pool. "
                f"No unique-cluster candidate qualifies for slot '{slot}'."
            )

        best_idx = eligible[0]
        best_val = -np.inf
        for cand_i in eligible:
            trial_objs = _eval_set_objectives(
                np.array(result + [cand_i], dtype=int),
                obj_matrix, sim_matrix,
            )
            score = float(0.40 * trial_objs[0] + 0.40 * trial_objs[1] + 0.20 * trial_objs[2])
            if score > best_val:
                best_val = score
                best_idx = cand_i

        used.add(best_idx)
        if cluster_ids is not None:
            cid = cluster_ids[best_idx]
            if cid >= 0:
                used_clusters.add(cid)
        result.append(best_idx)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def run_nsga3_sequence_with_vlm(
    candidates: List[Dict[str, Any]],
    target: int = 5,
    progress=None,
    brief: str = "",
    yolo_detections: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Select the 5-slot story sequence using strict literal constraints + NSGA-III.

    Args:
        candidates:      dicts with 'path', 'score', 'embedding',
                         'reasoning_log', 'breakdown'
        target:          ignored — always produces exactly 5 slots
        progress:        optional progress(frac, desc) callback
        brief:           creative direction brief text; falls back to _CD_BRIEF
        yolo_detections: set of paths where YOLO detected people

    Returns:
        List of 5 dicts, each enriched with 'slot', 'slot_role', 'slot_score',
        and 'nsga3_objectives'.

    Raises:
        SequencerConstraintError — if literal constraints leave no valid sequence.
        Never falls back to a compromised selection.
    """
    if not candidates:
        return []

    _p = progress or (lambda f, d: None)

    # Attempt to pull brief from global CD context when not passed
    if not brief:
        try:
            from specvlm_pipeline import _CD_BRIEF
            brief = _CD_BRIEF
        except Exception:
            pass

    _p(0.960, "Extracting literal constraints from brief…")
    constraints = _extract_literal_constraints(brief)
    print(
        f"[nsga3] Constraints: people_allowed={constraints.people_allowed}  "
        f"required={constraints.required_tags}  forbidden={constraints.forbidden_tags}  "
        f"mono={constraints.monochrome_only}"
    )

    # Phase 1: Hard literal pre-filter — raises on failure
    _p(0.962, "Applying literal pre-filter…")
    filtered = _apply_literal_prefilter(
        candidates, constraints, yolo_detections, min_pool=5
    )

    # Phase 1b: Upstream finalist isolation — one candidate per cluster only
    _p(0.963, "Isolating cluster bests…")
    filtered = _isolate_cluster_bests(filtered)
    if len(filtered) < 5:
        raise SequencerConstraintError(
            "Sequencing Failed: Insufficient visual diversity in candidate pool. "
            f"Only {len(filtered)} unique-cluster candidates remain after isolation."
        )

    # Phase 1c: DirectorBrief → per-slot boundary parameters (Python only, no LLM)
    # Tokenize candidates to IMG_NN tokens — strips all paths/filenames/reasoning_logs
    # so no PII ever reaches any future LLM context in this pipeline.
    _p(0.964, "Generating DirectorBrief boundary parameters…")
    slot_thresholds: List[float] = [SLOT_THRESHOLD] * 5
    sim_limit: float = _SIM_HARD_LIMIT_DEFAULT
    try:
        from creative_director_agent import (
            generate_director_brief,
            extract_nsga3_boundary_params,
            tokenize_candidates,
        )
        db = generate_director_brief(brief)
        bp = extract_nsga3_boundary_params(db)
        slot_thresholds = bp["slot_thresholds"]
        sim_limit       = bp["sim_limit"]
        # Tokenize for PII isolation — token_map held in memory, never sent to LLM
        _token_map, _tokenized = tokenize_candidates(filtered)
        print(
            f"[nsga3] DirectorBrief boundary: "
            f"slots={[round(t, 2) for t in slot_thresholds]}  "
            f"sim_limit={sim_limit:.2f}  "
            f"color={bp['color_emphasis']}  "
            f"require_people={bp['require_people']}"
        )
    except SequencerConstraintError:
        raise
    except Exception as e:
        print(f"[nsga3] DirectorBrief extraction failed ({e}) — global thresholds")

    # Phase 2: Pre-compute matrices
    _p(0.965, f"Pool: {len(filtered)} candidates — computing objectives…")
    cluster_ids_list: List[int] = [int(c.get("cluster_id", -1) or -1) for c in filtered]
    obj_matrix  = _pre_compute_objectives(filtered)
    sim_matrix  = _pairwise_sim_matrix(filtered)
    role_scores = _classify_slot_roles(filtered)

    print(
        "[nsga3] Mean slot fitness: "
        + "  ".join(f"{SLOT_NAMES[j]}={role_scores[:, j].mean():.2f}" for j in range(5))
    )

    n = len(filtered)
    selected_indices: List[int] = []

    # Phase 3: NSGA-III with per-slot thresholds + per-run sim_limit from DirectorBrief
    if HAS_PYMOO and n >= 5:
        try:
            _p(0.966, "Running slot-locked NSGA-III (DirectorBrief constraints)…")
            problem  = _SequenceProblem(
                n, obj_matrix, sim_matrix, role_scores, cluster_ids_list,
                slot_thresholds=slot_thresholds,
                sim_limit=sim_limit,
            )
            ref_dirs = get_reference_directions("energy", 3, min(100, n * 2), seed=1)
            algorithm = NSGA3(
                pop_size=max(len(ref_dirs), 30),
                ref_dirs=ref_dirs,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=0.9, eta=15, vtype=float),
                mutation=PM(eta=20, vtype=float),
                repair=_UniqueRepair(),
                eliminate_duplicates=True,
            )
            n_gen = max(40, min(100, n * 2))
            res   = minimize(problem, algorithm, ("n_gen", n_gen), seed=42, verbose=False)

            if res.X is not None and res.F is not None:
                G = res.G if res.G is not None else np.zeros((len(res.F), 7))
                feasible_mask = (G <= 0).all(axis=1)
                if feasible_mask.any():
                    feasible_F = -res.F[feasible_mask]
                    feasible_X = res.X[feasible_mask]
                    best_sol   = feasible_X[np.argmax(feasible_F.mean(axis=1))]
                    selected_indices = np.round(best_sol).astype(int).clip(0, n - 1).tolist()
                    print(f"[nsga3] Feasible solutions found: {feasible_mask.sum()}")
                else:
                    print("[nsga3] No feasible NSGA-III solutions — falling to greedy")
                    selected_indices = _greedy_slot_select(
                        filtered, obj_matrix, sim_matrix, role_scores, cluster_ids_list,
                        slot_thresholds=slot_thresholds, sim_limit=sim_limit,
                    )
            else:
                raise RuntimeError("NSGA-III returned no result")

        except SequencerConstraintError:
            raise
        except Exception as e:
            print(f"[nsga3] pymoo error ({e}), using greedy slot-select")
            selected_indices = _greedy_slot_select(
                filtered, obj_matrix, sim_matrix, role_scores, cluster_ids_list,
                slot_thresholds=slot_thresholds, sim_limit=sim_limit,
            )
    else:
        _p(0.966, "Running greedy slot-assignment (pymoo unavailable)…")
        selected_indices = _greedy_slot_select(
            filtered, obj_matrix, sim_matrix, role_scores, cluster_ids_list,
            slot_thresholds=slot_thresholds, sim_limit=sim_limit,
        )

    # Assemble result
    result: List[Dict[str, Any]] = []
    for slot_i, cand_idx in enumerate(selected_indices):
        c    = dict(filtered[cand_idx])
        objs = _eval_set_objectives(
            np.array(selected_indices[:slot_i + 1], dtype=int),
            obj_matrix, sim_matrix,
        )
        slot_name = SLOT_NAMES[slot_i]
        c["slot"]       = slot_name
        c["slot_role"]  = SLOT_DESC[slot_name]
        c["slot_score"] = round(float(role_scores[cand_idx, slot_i]), 3)
        c["nsga3_objectives"] = {
            "semantic_brief_match":          round(float(objs[0]), 3),
            "visual_diversity":              round(float(objs[1]), 3),
            "original_lighting_consistency": round(float(objs[2]), 3),
        }
        result.append(c)

    _p(0.980, f"Sequence locked: {' → '.join(SLOT_NAMES)}")
    return result


def run_nsga3_sequence(
    candidates: List[Dict[str, Any]],
    target: int = 5,
    progress=None,
) -> List[Dict[str, Any]]:
    """Alias for run_nsga3_sequence_with_vlm."""
    return run_nsga3_sequence_with_vlm(candidates, target=target, progress=progress)
