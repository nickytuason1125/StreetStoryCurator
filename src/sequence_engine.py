"""
Shared constants and utilities for sequence_story in lightweight_analyzer.py.
"""

import json
import os

_WEIGHT_KEYS = ("visual_flow", "visual_diversity", "time_coherence")


class PacingManager:
    """
    Loads pacing_presets.json and returns normalised weight dicts.

    Safe guarantees:
      • description field is ignored (never included in the sum)
      • custom_overrides never mutate the loaded preset (copy-on-write)
      • zero-sum edge case falls back to equal thirds
      • save_custom_weights only writes to the "Custom" slot
    """

    def __init__(self, config_path: str = "pacing_presets.json") -> None:
        # config_path is relative to this file's directory, not the caller's cwd
        self.config_path = os.path.join(os.path.dirname(__file__), config_path)
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.presets = json.load(f)
        except Exception:
            self.presets = {}

    def get_normalized_weights(
        self,
        preset_name: str,
        custom_overrides: dict | None = None,
    ) -> dict[str, float]:
        """
        Returns {"visual_flow": float, "visual_diversity": float, "time_coherence": float}
        where all three values sum to 1.0.
        """
        # Copy so overrides never mutate the cached preset
        base = dict(self.presets.get(preset_name) or self.presets.get("Street - Magnum") or {})
        if custom_overrides:
            base.update(custom_overrides)

        # Only sum the three numeric weight keys — ignore "description" etc.
        raw = {k: max(0, float(base.get(k, 0))) for k in _WEIGHT_KEYS}
        total = sum(raw.values())
        if total == 0:
            return {k: 1 / 3 for k in _WEIGHT_KEYS}
        return {k: v / total for k, v in raw.items()}

    def save_custom_weights(self, preset_name: str, weights: dict) -> None:
        """
        Persist weights to pacing_presets.json.
        Only the "Custom" slot can be overwritten — built-in presets are protected.
        """
        if preset_name != "Custom":
            return
        if not weights:
            return
        self.presets["Custom"] = {
            **{k: max(0, int(weights.get(k, 0))) for k in _WEIGHT_KEYS},
            "description": str(weights.get("description", "User-defined tuning.")),
        }
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2)
        except Exception:
            pass

# ── Slot weights per subject type ─────────────────────────────────────────────
_SLOT_WEIGHTS: dict[str, list[dict]] = {
    "street": [
        {"comp": 0.50, "tech": 0.25, "auth": 0.25},
        {"human": 0.45, "auth": 0.35, "light": 0.20},
        {"tech": 0.45, "comp": 0.35, "auth": 0.20},
        {"light": 0.50, "comp": 0.30, "human": 0.20},
        {"auth": 0.40, "light": 0.35, "comp": 0.25},
    ],
    "nature": [
        {"light": 0.65, "comp": 0.20, "tech": 0.15},
        {"light": 0.55, "comp": 0.30, "tech": 0.15},
        {"tech": 0.45, "comp": 0.35, "auth": 0.20},
        {"light": 0.65, "comp": 0.20, "auth": 0.15},
        {"light": 0.50, "comp": 0.35, "auth": 0.15},
    ],
    "portrait": [
        {"human": 0.50, "light": 0.30, "comp": 0.20},
        {"human": 0.45, "auth": 0.35, "light": 0.20},
        {"comp": 0.40, "human": 0.35, "light": 0.25},
        {"human": 0.40, "auth": 0.40, "light": 0.20},
        {"light": 0.40, "human": 0.35, "auth": 0.25},
    ],
    "architecture": [
        {"comp": 0.55, "tech": 0.30, "light": 0.15},
        {"tech": 0.45, "comp": 0.40, "auth": 0.15},
        {"light": 0.50, "comp": 0.30, "tech": 0.20},
        {"comp": 0.40, "tech": 0.30, "auth": 0.30},
        {"comp": 0.50, "light": 0.35, "auth": 0.15},
    ],
}

# ── Role requirements per subject type ────────────────────────────────────────
_ROLE_REQUIREMENTS: dict = {
    "street": [
        {"trait": lambda hv, lv, cv, tv, av: cv >= 0.40,               "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: hv >= 0.35 or av >= 0.50, "penalty": 0.20},
        {"trait": lambda hv, lv, cv, tv, av: tv >= 0.38,               "penalty": 0.10},
        {"trait": None,                                                  "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.40,               "penalty": 0.10},
    ],
    "nature": [
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.45,                "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.42 or cv >= 0.50, "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: tv >= 0.40 or av >= 0.45, "penalty": 0.10},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.42,                "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.36 or cv >= 0.44, "penalty": 0.10},
    ],
    "portrait": [
        {"trait": lambda hv, lv, cv, tv, av: hv >= 0.42,               "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: hv >= 0.52 or av >= 0.55, "penalty": 0.20},
        {"trait": lambda hv, lv, cv, tv, av: cv >= 0.40,               "penalty": 0.10},
        {"trait": lambda hv, lv, cv, tv, av: hv >= 0.38 or av >= 0.48, "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.40,               "penalty": 0.10},
    ],
    "architecture": [
        {"trait": lambda hv, lv, cv, tv, av: cv >= 0.50,               "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: tv >= 0.45 or cv >= 0.52, "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: lv >= 0.46 or av >= 0.50, "penalty": 0.15},
        {"trait": lambda hv, lv, cv, tv, av: hv >= 0.15 or av >= 0.30, "penalty": 0.10},
        {"trait": lambda hv, lv, cv, tv, av: cv >= 0.45,               "penalty": 0.10},
    ],
}
_ROLE_REQUIREMENTS["auto"] = _ROLE_REQUIREMENTS["street"]

# ── Breakdown key-sets (used inside _mdp_sequence) ────────────────────────────
_TECH_KEYS  = frozenset({
    "Technical", "News Sharpness", "Cleanliness", "Execution",
    "Detail Retention", "Exposure", "Sharpness & Detail",
})
_COMP_KEYS  = frozenset({
    "Composition", "Framing", "Context", "Geometry & Balance",
    "Negative Space", "Framing Instinct", "Layered Depth",
})
_AUTH_KEYS  = frozenset({
    "Decisive Moment", "Cultural Depth", "Journalistic Integrity",
    "Narrative Suggestion", "Conceptual Weight", "Reduction",
    "Authenticity", "Immediacy", "Environmental Truth",
})
_HUMAN_KEYS = frozenset({
    "Human/Culture", "Sense of Place", "Human Impact",
    "Character Presence", "Emotional Resonance", "Scale Element",
    "Presence", "Scale & Life", "Subject Isolation",
})
_LIGHT_KEYS = frozenset({
    "Lighting", "Atmosphere", "Natural Light", "Mood & Tone",
    "Tonal Purity", "Contrast Purity", "Available Light",
    "Natural Light Quality",
})

# ── Narrative role labels per subject type ────────────────────────────────────
_ROLE_LABELS: dict[str, list[str]] = {
    "street":       ["Establishing Context", "Decisive Moment", "Detail / Texture", "Visual Contrast", "Atmospheric Close"],
    "nature":       ["Scene Opener", "Landscape Anchor", "Detail / Wildlife", "Mood & Atmosphere", "Quiet Close"],
    "portrait":     ["Subject Introduction", "Eye Contact / Peak", "Environmental Context", "Unguarded Moment", "Defining Frame"],
    "architecture": ["Facade & Scale", "Geometric Detail", "Light & Shadow", "Human Scale", "Abstract Close"],
}


def _mdp_sequence(
    pool:       list,
    breakdowns: list,
    scores,             # np.ndarray, indexed by valid-list position
    sim,                # (n_valid × n_valid) cosine similarity matrix
    stype:      str,
    target:     int,
    seq_scores,         # list[float] | None
    W_FLOW:     float,
    W_DIV:      float,
    W_TIME:     float,
    n_valid:    int,
) -> list:
    """
    Viterbi MDP editorial sequencer.

    Solves image-to-slot assignment as a shortest-path DP:
      state  = narrative slot (0 … target-1)
      action = assign an image from pool to that slot
      reward = role_fit + quality + flow + diversity + time_coherence + rhythm

    Post-hoc deduplication replaces any repeated pool entry with the
    next-best unused image — avoids exponential state tracking while
    still enforcing uniqueness.

    Returns list[int] of valid-list indices (same format as beam search).
    """
    import numpy as np

    _ROLE_W    = 0.25
    _QUALITY_W = 0.10

    n_pool = len(pool)
    if n_pool <= target:
        return list(pool[:target])

    sw   = _SLOT_WEIGHTS.get(stype, _SLOT_WEIGHTS["street"])
    reqs = _ROLE_REQUIREMENTS.get(stype, _ROLE_REQUIREMENTS["street"])

    def _dv(b, keys):
        return next((v for k, v in b.items() if k in keys), 0.0)

    def role_fit(idx: int, slot: int) -> float:
        b   = breakdowns[idx]
        hv  = _dv(b, _HUMAN_KEYS)
        lv  = _dv(b, _LIGHT_KEYS)
        cv  = _dv(b, _COMP_KEYS)
        tv  = _dv(b, _TECH_KEYS)
        av  = _dv(b, _AUTH_KEYS)
        dim     = {"human": hv, "light": lv, "comp": cv, "tech": tv, "auth": av}
        weights = sw[slot] if slot < len(sw) else sw[-1]
        fit     = sum(dim[k] * w for k, w in weights.items())
        req     = reqs[slot] if slot < len(reqs) else reqs[-1]
        if req["trait"] is not None and not req["trait"](hv, lv, cv, tv, av):
            fit = max(0.0, fit - req["penalty"])
        return fit

    def q(idx: int) -> float:
        if seq_scores is not None:
            return 0.33 * float(scores[idx]) + 0.67 * float(seq_scores[idx])
        return float(scores[idx])

    NEG_INF = float("-inf")
    dp = [[NEG_INF] * n_pool for _ in range(target)]
    bp = [[-1]      * n_pool for _ in range(target)]

    # Slot 0 — no predecessor
    for pi, idx in enumerate(pool):
        dp[0][pi] = role_fit(idx, 0) * _ROLE_W + q(idx) * _QUALITY_W

    # Forward pass
    for slot in range(1, target):
        for pi, idx in enumerate(pool):
            best_val = NEG_INF
            best_pi  = -1
            lv_cur   = _dv(breakdowns[idx], _LIGHT_KEYS)
            for prev_pi, prev_idx in enumerate(pool):
                prev_val = dp[slot - 1][prev_pi]
                if prev_val == NEG_INF or prev_idx == idx:
                    continue
                flow     = float(sim[prev_idx, idx])
                div      = 1.0 - flow
                time_p   = 1.0 - abs(prev_idx - idx) / max(n_valid - 1, 1)
                # Light variation between consecutive slots rewards editorial rhythm
                lv_prev  = _dv(breakdowns[prev_idx], _LIGHT_KEYS)
                rhythm   = abs(lv_cur - lv_prev)
                step = (prev_val
                        + role_fit(idx, slot) * _ROLE_W
                        + q(idx)              * _QUALITY_W
                        + flow                * W_FLOW
                        + div                 * W_DIV
                        + time_p              * W_TIME
                        + rhythm              * 0.05)
                if step > best_val:
                    best_val = step
                    best_pi  = prev_pi
            dp[slot][pi] = best_val
            bp[slot][pi] = best_pi

    # Backtrack from best terminal state
    best_end = max(range(n_pool), key=lambda pi: dp[target - 1][pi])
    seq_pi   = [best_end]
    for slot in range(target - 1, 0, -1):
        seq_pi.append(bp[slot][seq_pi[-1]])
    seq_pi.reverse()

    result = [pool[pi] for pi in seq_pi]

    # Post-hoc deduplication: replace any duplicate with best unused from pool
    seen     = set()
    used_set = set(result)
    unused   = sorted(
        [i for i in pool if i not in used_set],
        key=lambda i: float(scores[i]),
        reverse=True,
    )
    deduped: list = []
    for idx in result:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
        elif unused:
            sub = unused.pop(0)
            seen.add(sub)
            deduped.append(sub)

    return deduped[:target]
