"""
Step 3 — Personal Head: a 2-layer MLP that learns user aesthetic taste.

Architecture
────────────
    SigLIP-2 embedding (1536-d)  [or 1152-d for legacy SigLIP So400M]
         │
    Linear(embed_dim → 256) + ReLU
         │
    Linear(256 → 64) + ReLU
         │
    Linear(64 → 1)  → preference score ∈ ℝ
         │
    Sigmoid          → normalised to [0, 1]

The embed_dim is detected from the saved weights on load, so the same
checkpoint logic works regardless of which encoder produced the embeddings.
Saved weights from a 1152-d run are silently discarded when the embedding
space changes to 1536-d — the head retrains from scratch.

Learning
────────
Whenever the user moves a photo between grade buckets the model receives a
Margin Ranking Loss update:

    L = max(0, -y · (s₁ - s₂) + margin)

Weights are persisted to cache/personal_head.pt after every update.
"""
from __future__ import annotations

import json
import threading
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

_WEIGHTS_PATH = Path("cache/personal_head.pt")
_DEFAULT_EMBED_DIM = 1536   # SigLIP-2; 1152 for legacy SigLIP So400M


class PersonalHead(nn.Module):
    def __init__(self, embed_dim: int = _DEFAULT_EMBED_DIM) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Singleton ─────────────────────────────────────────────────────────────────

# H1: every entry point below (lazy init, update()'s backward/step, fit()'s
# wholesale _head/_opt reassignment, score()'s forward read) runs on real
# threadpool threads — star-ratings from the UI, the every-25-ratings
# auto-retrain, manual retrain — with NO serialization between them. Two
# racing update() calls corrupt gradients; a fit() mid-update orphans the
# optimizer outright. RLock, not Lock: fit() → _save() → _get_head() nests.
_LOCK = threading.RLock()

_head: Optional[PersonalHead] = None
_opt:  Optional[torch.optim.Adam] = None

_GRADE_RANK = {"Strong ✅": 2, "Mid ⚠️": 1, "Weak ❌": 0}
_MARGIN     = 0.10
_LR         = 3e-4
_STEPS      = 5        # gradient steps per update call


def _margin_for_rank_gap(gap: int) -> float:
    """Margin grows with how far apart the two grades sit.

    A 5★ vs 1★ disagreement is a stronger statement about taste than 3★ vs 2★,
    so it trains with a proportionally wider margin (0.10 per band gap + the
    base band, capped at 0.20): Strong↔Mid and Mid↔Weak keep the historical
    0.10, Strong↔Weak gets 0.15. Previously every pair pulled with the same
    0.10 force, so adjacent-grade noise and chasm-grade signal taught the head
    at identical volume."""
    gap = max(0, int(gap))
    return min(_MARGIN * (1.0 + 0.5 * max(0, gap - 1)), 2.0 * _MARGIN)


def _hinge(sa, sb, margin: float):
    """MarginRankingLoss for y=+1 (sa must outrank sb) with a per-pair margin.
    Equivalent to nn.MarginRankingLoss(margin)(sa, sb, +1) — inlined because
    the margin now varies per pair and the module fixes one at construction."""
    return torch.clamp(margin - (sa - sb), min=0.0).mean()


def _get_head(embed_dim: int = _DEFAULT_EMBED_DIM) -> tuple[PersonalHead, torch.optim.Adam]:
    global _head, _opt
    with _LOCK:
        if _head is None:
            _head = PersonalHead(embed_dim=embed_dim)
            if _WEIGHTS_PATH.exists():
                try:
                    # weights_only=True: _save() writes a plain state_dict, so there
                    # is nothing here that needs the pickle machinery. Without it a
                    # .pt on disk is an arbitrary-code-execution vector at startup —
                    # the file lives in a user-writable cache dir and this app is
                    # meant to be distributed, so "we wrote it ourselves" is not a
                    # property that survives shipping.
                    saved = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
                    # Infer saved embed_dim from first Linear weight shape
                    saved_dim = saved.get("net.0.weight", torch.zeros(1, embed_dim)).shape[1]
                    if saved_dim == embed_dim:
                        _head.load_state_dict(saved)
                    else:
                        print(
                            f"[PersonalHead] Saved weights dim={saved_dim} ≠ current dim={embed_dim}. "
                            "Discarding old weights — head will retrain from scratch."
                        )
                except Exception:
                    pass
            _opt = torch.optim.Adam(_head.parameters(), lr=_LR)
        return _head, _opt   # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def score(embeddings: np.ndarray) -> np.ndarray:
    """
    Return personal preference scores in [0, 1] for an (N, D) embedding array.
    D is inferred from the input shape — head is (re-)initialised on first call.

    NOTE: grading does NOT call this — it uses personal_head_np.score(), which
    runs the same forward pass in numpy so the CUDA-free grade worker never
    imports torch. This remains the reference implementation and the training
    path; personal_head_np mirrors its weights and is verified against it.
    """
    embed_dim = embeddings.shape[1] if embeddings.ndim == 2 else _DEFAULT_EMBED_DIM
    head, _ = _get_head(embed_dim=embed_dim)
    with _LOCK:
        head.eval()
        with torch.no_grad():
            t    = torch.tensor(embeddings, dtype=torch.float32)
            pref = head(t).numpy()
    return pref.astype(np.float32)


def update(
    emb1: np.ndarray,
    grade1: str,
    emb2: np.ndarray,
    grade2: str,
) -> float:
    """
    Run `_STEPS` Margin Ranking Loss gradient steps given two embeddings
    and their human-assigned grades.  Returns the final loss value.
    """
    with _LOCK:
        embed_dim = int(np.asarray(emb1).flatten().shape[0])
        head, opt = _get_head(embed_dim=embed_dim)
        head.train()

        r1  = _GRADE_RANK.get(grade1, 1)
        r2  = _GRADE_RANK.get(grade2, 1)
        # y = +1 if emb1 should score higher, -1 if lower, 0 if equal
        y   = float(np.sign(r1 - r2))
        if y == 0.0:
            return 0.0
        margin = _margin_for_rank_gap(abs(r1 - r2))

        t1  = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
        t2  = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
        # forward() squeezes to shape (1,), so the target must be (1,) too —
        # a (1,1) target raises "All input tensors should have same dimension"
        # and (because the frontend swallows the 500) silently no-ops training.
        _y = torch.tensor([y], dtype=torch.float32)

        last_loss = 0.0
        for _ in range(_STEPS):
            opt.zero_grad()
            s1, s2 = head(t1), head(t2)
            # y=±1 folds into the hinge directly: max(0, margin − y·(s1−s2))
            loss = torch.clamp(margin - _y * (s1 - s2), min=0.0).mean()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        _save()
        return last_loss


def update_batch(pairs: list[dict]) -> float:
    """
    Convenience wrapper for a list of
    {"emb1": np.ndarray, "grade1": str, "emb2": np.ndarray, "grade2": str}.
    """
    total = 0.0
    for p in pairs:
        total += update(p["emb1"], p["grade1"], p["emb2"], p["grade2"])
    return total / max(len(pairs), 1)


_MASTER_SOURCE = "tpe_master"
_MASTER_PAIR_FLOOR = 0.30  # >=30% of every combo's pairs always involve a master-tagged sample


def fit(samples: list[tuple], epochs: int = 3, pairs_per_combo: int = 120) -> dict:
    """
    Full retrain from scratch on the entire labelled baseline.

    samples: [(embedding(np.ndarray), grade_str), ...] or
             [(embedding(np.ndarray), grade_str, source_str), ...] for every
             rated photo. A sample tagged source=="tpe_master" is the
             permanently-authoritative baseline (currently: the TPE ratings) —
             pure random pair-sampling would let it dilute away as the rest of
             the rating history grows into the thousands, so at least
             _MASTER_PAIR_FLOOR of each combo's pairs are forced to involve a
             master-tagged sample on at least one side, regardless of how
             large the general pool gets. Everything else still trains
             normally and still moves the score — this only puts a floor under
             the master set's influence, it doesn't cap anyone else's.

    Resets the head and trains on balanced ranking pairs across grade tiers, so
    the fit reflects the WHOLE baseline rather than the drift of incremental
    per-rating updates. Use this when the baseline grows (e.g. every N new
    ratings) — far more stable than update() at 100s of ratings.

    Returns {"n": rated, "pairs": npairs, "loss": last, "tiers": {...}}.
    """
    with _LOCK:
        global _head, _opt
        import random

        by:      dict[str, list] = {"Strong ✅": [], "Mid ⚠️": [], "Weak ❌": []}
        by_master: dict[str, list] = {"Strong ✅": [], "Mid ⚠️": [], "Weak ❌": []}
        for s in samples:
            emb, gr = s[0], s[1]
            src = s[2] if len(s) > 2 else ""
            if gr not in by:
                continue
            e = np.asarray(emb, dtype=np.float32)
            by[gr].append(e)
            if src == _MASTER_SOURCE:
                by_master[gr].append(e)
        n = sum(len(v) for v in by.values())
        if n == 0:
            return {"n": 0, "pairs": 0, "loss": 0.0, "tiers": {}}

        dim = int(np.asarray(samples[0][0]).flatten().shape[0])
        _head = PersonalHead(embed_dim=dim)
        _opt  = torch.optim.Adam(_head.parameters(), lr=_LR)

        pairs = []
        for hi, lo in (("Strong ✅", "Weak ❌"), ("Strong ✅", "Mid ⚠️"), ("Mid ⚠️", "Weak ❌")):
            if by[hi] and by[lo]:
                has_master = by_master[hi] or by_master[lo]
                n_floor = int(pairs_per_combo * _MASTER_PAIR_FLOOR) if has_master else 0
                # Margin scales with how far apart the two tiers are: Strong↔Weak
                # is a stronger taste statement than Mid↔Weak (see
                # _margin_for_rank_gap) and now trains with proportionally more force.
                margin = _margin_for_rank_gap(_GRADE_RANK[hi] - _GRADE_RANK[lo])
                for i in range(pairs_per_combo):
                    if i < n_floor:
                        a = random.choice(by_master[hi]) if by_master[hi] else random.choice(by[hi])
                        b = random.choice(by_master[lo]) if by_master[lo] else random.choice(by[lo])
                    else:
                        a = random.choice(by[hi])
                        b = random.choice(by[lo])
                    pairs.append((a, b, margin))
        if not pairs:
            return {"n": n, "pairs": 0, "loss": 0.0,
                    "tiers": {k: len(v) for k, v in by.items()}}

        _head.train()
        last = 0.0
        for _ in range(max(1, epochs)):
            random.shuffle(pairs)
            for a, b, margin in pairs:
                _opt.zero_grad()
                sa = _head(torch.tensor(a).unsqueeze(0))
                sb = _head(torch.tensor(b).unsqueeze(0))
                loss = _hinge(sa, sb, margin)
                loss.backward()
                _opt.step()
                last = float(loss.item())
        _save()
        return {"n": n, "pairs": len(pairs), "loss": round(last, 5),
                "tiers": {k: len(v) for k, v in by.items()},
                "master_tiers": {k: len(v) for k, v in by_master.items()}}


def _save() -> None:
    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_get_head()[0].state_dict(), _WEIGHTS_PATH)
