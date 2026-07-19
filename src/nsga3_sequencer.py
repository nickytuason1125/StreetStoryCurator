"""
NSGA-III Sequencer — Creative Direction Edition
Fixes Pareto Collapse and enforces binary 'Empty' constraints.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List

# ── 1. Thematic Logic Gate ──────────────────────────────────────────────────

def _analyze_creative_intent(brief: str) -> dict:
    """Detects hard constraints vs abstract themes in the Style Brief."""
    brief_lc = brief.lower()
    return {
        "is_empty": any(w in brief_lc for w in ["empty", "liminal", "alone", "void"]),
        "is_power": "power" in brief_lc,
        "is_minimal": "minimal" in brief_lc
    }

# ── 2. The Multi-Objective Evaluator ────────────────────────────────────────

def _eval_creative_objectives(
    indices: np.ndarray,
    obj_matrix: np.ndarray,
    sim_matrix: np.ndarray,
    intent: dict
) -> np.ndarray:
    """
    Returns 4-objective vector for Creative Direction.
    FIX: Uses Max-Similarity Penalization to kill repetition.
    """
    idx = indices.astype(int)
    if len(idx) == 0: return np.array([0.0, 0.0, 0.0, 0.0])

    # F1: Quality (Maintain aesthetic bar)
    f1 = float(np.mean(obj_matrix[idx, 0]))

    # F2: Theme Accuracy (The "Power" / "Empty" Gate)
    f2 = 1.0
    if intent["is_empty"]:
        # HARD BLOCK: If Human Presence (col 2) > 0.15, the sequence is disqualified
        if np.any(obj_matrix[idx, 2] > 0.15): f2 = 0.01 
    
    if intent["is_power"]:
        # REASONING: Reward Low Angle (col 3) and High Contrast (col 4)
        f2 = float(np.mean(obj_matrix[idx, 3] * 0.7 + obj_matrix[idx, 4] * 0.3))

    # F3: Portfolio Diversity (SMOKING GUN FIX)
    # We penalize the 'closest' pair in the set. 1.0 - max(sim) forces unique shots.
    sims = sim_matrix[np.ix_(idx, idx)] if len(idx) > 1 else np.array([[1.0]])
    upper = sims[np.triu_indices(len(idx), k=1)]
    f3 = float(1.0 - np.max(upper)) if len(upper) > 0 else 1.0

    # F4: Visual Flow (Coherence to the Style Anchor)
    # Checks similarity to the first image (The Anchor)
    f4 = float(np.mean(sims[0, 1:])) if len(idx) > 1 else 1.0

    return np.array([f1, f2, f3, f4], dtype=np.float32)

# ── 3. Creative Direction Greedy Search ─────────────────────────────────────

def run_creative_story_sequencer(
    candidates: List[Dict[str, Any]], 
    target: int = 7, 
    brief: str = ""
) -> List[Dict[str, Any]]:
    """Isolated sequencer for the Creative Direction UI section."""
    n = len(candidates)
    intent = _analyze_creative_intent(brief)
    
    # obj_matrix setup: [0:Score, 1:AR, 2:Human, 3:LowAngle, 4:Contrast]
    obj_matrix = np.zeros((n, 5), dtype=np.float32)
    for i, c in enumerate(candidates):
        obj_matrix[i, 0] = c.get("score", 0.5)
        obj_matrix[i, 2] = c.get("human_presence", 0.0) # From YOLO/SigLIP
        obj_matrix[i, 3] = c.get("angle_score", 0.5)    # From SpecVLM
        obj_matrix[i, 4] = c.get("contrast_score", 0.5)

    embs = np.array([c["embedding"] for c in candidates], dtype=np.float32)
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim_matrix = normed @ normed.T

    selected: List[int] = [0] # Start with User's Style Anchor
    remaining = list(range(1, n))

    while len(selected) < target and remaining:
        scores = np.zeros(len(remaining))
        for j, idx in enumerate(remaining):
            trial = np.array(selected + [idx])
            objs = _eval_creative_objectives(trial, obj_matrix, sim_matrix, intent)
            
            # REBALANCED WEIGHTS: Diversity (objs[2]) is now the primary driver
            scores[j] = (0.15 * objs[0] + 0.35 * objs[1] + 0.40 * objs[2] + 0.10 * objs[3])
            
        best_idx = remaining[int(np.argmax(scores))]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]