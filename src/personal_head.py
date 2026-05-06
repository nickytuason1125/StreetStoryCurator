"""
Step 3 — Personal Head: a 2-layer MLP that learns user aesthetic taste.

Architecture
────────────
    SigLIP embedding (1152-d)
         │
    Linear(1152 → 256) + ReLU
         │
    Linear(256 → 64) + ReLU
         │
    Linear(64 → 1)  → preference score ∈ ℝ
         │
    Sigmoid          → normalised to [0, 1]

Total parameters: ~311 K  ≈ 1.2 MB in FP32.

Learning
────────
Whenever the user moves a photo between grade buckets the model receives a
Margin Ranking Loss update:

    L = max(0, -y · (s₁ - s₂) + margin)

where
    y  = +1  if photo 1 should rank higher than photo 2
    s₁ = personal_head(emb₁),  s₂ = personal_head(emb₂)
    margin = 0.1   (guards against ties)

Weights are persisted to cache/personal_head.pt after every update.
"""
from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

_WEIGHTS_PATH = Path("cache/personal_head.pt")
_EMBED_DIM    = 1152


class PersonalHead(nn.Module):
    def __init__(self, embed_dim: int = _EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Singleton ─────────────────────────────────────────────────────────────────

_head: Optional[PersonalHead] = None
_opt:  Optional[torch.optim.Adam] = None

_GRADE_RANK = {"Strong ✅": 2, "Mid ⚠️": 1, "Weak ❌": 0}
_MARGIN     = 0.10
_LR         = 3e-4
_STEPS      = 5        # gradient steps per update call


def _get_head() -> tuple[PersonalHead, torch.optim.Adam]:
    global _head, _opt
    if _head is None:
        _head = PersonalHead()
        if _WEIGHTS_PATH.exists():
            try:
                _head.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu"))
            except Exception:
                pass
        _opt = torch.optim.Adam(_head.parameters(), lr=_LR)
    return _head, _opt   # type: ignore[return-value]


# ── Public API ────────────────────────────────────────────────────────────────

def score(embeddings: np.ndarray) -> np.ndarray:
    """
    Return personal preference scores in [0, 1] for an (N, 1152) embedding array.
    Higher = more aligned with the user's historical taste.
    """
    head, _ = _get_head()
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
    Run `_STEPS` Margin Ranking Loss gradient steps given two (1152,) embeddings
    and their human-assigned grades.  Returns the final loss value.

    Call this whenever a user moves a photo from one bucket to another.
    """
    head, opt = _get_head()
    head.train()

    r1  = _GRADE_RANK.get(grade1, 1)
    r2  = _GRADE_RANK.get(grade2, 1)
    # y = +1 if emb1 should score higher, -1 if lower, 0 if equal
    y   = float(np.sign(r1 - r2))
    if y == 0.0:
        return 0.0

    t1  = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0)
    t2  = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0)
    criterion = nn.MarginRankingLoss(margin=_MARGIN)

    last_loss = 0.0
    for _ in range(_STEPS):
        opt.zero_grad()
        s1, s2 = head(t1), head(t2)
        loss   = criterion(s1, s2, torch.tensor([[y]]))
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


def _save() -> None:
    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_get_head()[0].state_dict(), _WEIGHTS_PATH)
