"""
story_selector.py — choose a cohesive set of k photographs from the whole library.

What this replaces
------------------
`creative_director.py:1123` took the single highest-scoring photo and kept only
its 40 nearest neighbours; the manifest was then 12 of those. Of 5,634 graded
Strong photos the model saw 12, and the 12 were chosen for RESEMBLING ONE FRAME
— immediately before the sequencer was asked to find contrast. If that top
scorer happened to be an outlier, the entire story was built inside a cul-de-sac
and the rest of the library never competed.

That step was not pure damage: 40 look-alikes bought visual coherence, which a
photo story genuinely needs. So coherence survives here as an explicit term
rather than an accident of the funnel.

Why cohesion is a term and not a threshold
------------------------------------------
Growing a set while cohesion stayed above a floor was measured across four
briefs on the real library:

    floor 0.55 -> 10, 10, 10, 10
    floor 0.80 -> 10, 10, 10, 10
    floor 0.85 -> 10,  1,  2,  1
    floor 0.88 ->  1,  1,  1,  1

A cliff, not a curve — cohesion is measured against a centroid that moves as
frames are added, so one photo scores 1.0 trivially and the second falls off the
edge. No floor produces lengths in the 4–10 range. Hence: k is the user's
choice, and cohesion is weighted, never gated.

Why taste is confidence-weighted
--------------------------------
Over 5,634 Strong photos, `personal_score` correlates with the aesthetic score
at 0.010. It is not a tinted copy of quality — it is the only signal carrying
information the others lack. But it is trained on 124 ratings into a
1536-256-64-1 network, with alignment recorded at 0.52 against a 0.50 coin flip,
mean 0.568 and std 0.075. For most of the library the head holds no opinion.

So it leads where it is confident and steps aside where it is not, reusing the
mechanism already proven in grade_pipeline_v2 Step 5.

No model is loaded here. Everything is numpy over embeddings LanceDB already
holds, which is why the whole library can be considered rather than 40 frames.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Above this, two frames are the same photograph. Matches
# creative_director._DUP_SIM_THRESH — not a taste judgement, so it stays hard.
DUP_SIM_THRESH = 0.88

# What an unmeasured frame scores on tone. Mid-scale on purpose: it competes on
# its other merits instead of being penalised for a missing thumbnail.
NEUTRAL_TONE = 0.5

# Taste weight rides its own confidence. Floor: a head sitting at 0.5 knows
# nothing about this photo and must not steer it. Ceiling: a head at 0.0 or 1.0
# is as sure as it gets, and taste leads.
TASTE_FLOOR = 0.10
TASTE_CEIL = 0.50

# Objective weights. NOT measured — a starting point, to be tuned against sets
# the user judges. The cohesion weight is the consequential one: it is the dial
# between a repetitive set that scores well and a jumble of strong frames.
W_BRIEF = 0.35
W_MERIT = 0.35
W_COHESION = 0.15
# Tonality: brightness and saturation agreement with the frames already chosen.
# Added after a real run returned warm night colour, black-and-white and a
# bright daytime frame in one six-image sequence, all at 0.891 cohesion --
# every frame matched the SUBJECT, because that is what SigLIP encodes. Tone is
# most of what makes photographs read as one body of work, and none of it was
# being measured.
W_TONE = 0.15

# Not the user's photographs. Two sources were found in the live library:
#   100 rows under the repo's own dataset_images/ (bundled demo assets)
#     8 rows under Temp/claude/.../scratchpad/demo_images (a previous
#       development session's scratch files, graded in by accident)
# One of the latter reached a story with score 1.00 because the first version
# of this filter only matched dataset_images. Separators are normalised because
# the same path arrives both ways on Windows.
_EXCLUDE_MARKERS = ("dataset_images", "/scratchpad/", "/temp/claude/",
                    "/demo_images/")


def taste_weight(personal: Any) -> float:
    """How much this photo's taste score should count, from its confidence.

    conf = |p - 0.5| / 0.5, so a neutral head collapses to TASTE_FLOOR and a
    decisive one rises to TASTE_CEIL. A head that has never seen this genre
    cannot make the choice worse.
    """
    try:
        p = float(personal)
    except (TypeError, ValueError):
        return TASTE_FLOOR
    conf = min(1.0, abs(p - 0.5) / 0.5)
    return TASTE_FLOOR + (TASTE_CEIL - TASTE_FLOOR) * conf


def merit(score: Any, personal: Any) -> float:
    """Quality blended with taste, taste weighted by its own confidence."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0.5
    try:
        p = float(personal)
    except (TypeError, ValueError):
        return s
    w = taste_weight(p)
    return (1.0 - w) * s + w * p


def _normalise(x: np.ndarray) -> np.ndarray:
    """Min-max to 0..1 so terms measured on different scales can be summed."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def _eligible(rows: "list[dict]") -> np.ndarray:
    keep = np.ones(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        path = str(r.get("path", "")).lower().replace("\\", "/")
        if any(m in path for m in _EXCLUDE_MARKERS):
            keep[i] = False
    return keep


def cohesion_stats(M: np.ndarray, indices: "list[int]") -> dict:
    """Mean/min similarity of the chosen frames to their own centre, and the
    tightest pair. Reported so the user can judge whether the set holds."""
    if len(indices) < 2:
        return {"cohesion_mean": 1.0, "cohesion_min": 1.0, "max_pair": 0.0}
    sub = M[indices]
    centre = sub.mean(axis=0)
    centre /= np.linalg.norm(centre) + 1e-9
    coh = sub @ centre
    pair = sub @ sub.T
    np.fill_diagonal(pair, 0.0)
    return {"cohesion_mean": float(coh.mean()),
            "cohesion_min": float(coh.min()),
            "max_pair": float(pair.max())}


def select(brief_vec: np.ndarray, rows: "list[dict]", M: np.ndarray,
           k: int = 7, tone: "Optional[np.ndarray]" = None) -> "tuple[list[int], dict]":
    """Choose k photographs. Returns (indices into rows, diagnostics).

    Greedy: each step adds the frame with the highest value given what is
    already chosen. Cohesion is recomputed against the current set, so it is a
    preference that adapts rather than a fixed threshold.

    Returns fewer than k, with a reason, rather than relaxing the duplicate
    constraint. Four photographs and an explanation beat seven and a lie.
    """
    diag: dict = {"k_requested": int(k)}
    n = len(rows)
    if n == 0 or M.size == 0:
        diag["reason"] = "no graded photographs available"
        return [], diag

    keep = _eligible(rows)
    n_excluded = int((~keep).sum())
    if n_excluded:
        diag["excluded_bundled"] = n_excluded

    q = np.asarray(brief_vec, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    brief = _normalise((M @ q).clip(-1.0, 1.0))

    mer = np.array([merit(r.get("score"), r.get("personal_score")) for r in rows],
                   dtype=np.float32)

    base = W_BRIEF * brief + W_MERIT * mer
    base = np.where(keep, base, -np.inf)

    T = None
    if tone is not None:
        T = np.asarray(tone, dtype=np.float32)
        if T.ndim != 2 or T.shape[0] != n or T.shape[1] < 2:
            T = None          # wrong shape is not worth guessing about

    chosen: "list[int]" = []
    while len(chosen) < k:
        if not chosen:
            value = base
        else:
            sub = M[chosen]
            centre = sub.mean(axis=0)
            centre /= np.linalg.norm(centre) + 1e-9
            coh = (M @ centre).clip(-1.0, 1.0)
            value = base + W_COHESION * coh
            if T is not None:
                ref = T[chosen]
                ref = ref[np.all(np.isfinite(ref), axis=1)]
                if ref.shape[0]:
                    centre = ref.mean(axis=0)
                    fit = 1.0 - np.abs(T - centre).mean(axis=1)
                    # Unmeasured frames score neutral: no thumbnail is not a
                    # reason to drop a photograph from a set.
                    fit = np.where(np.all(np.isfinite(T), axis=1),
                                   np.clip(fit, 0.0, 1.0), NEUTRAL_TONE)
                    value = value + W_TONE * fit
            # hard: never the same photograph twice
            too_close = (sub @ M.T).max(axis=0) > DUP_SIM_THRESH
            value = np.where(too_close, -np.inf, value)
        value = value.copy()
        value[chosen] = -np.inf
        if not np.isfinite(value).any():
            diag["reason"] = ("only %d of %d requested: everything else is a "
                              "near-duplicate of what was already chosen"
                              % (len(chosen), k))
            break
        chosen.append(int(np.argmax(value)))

    if len(chosen) < k and "reason" not in diag:
        diag["reason"] = "only %d photographs available" % len(chosen)

    diag.update(cohesion_stats(M, chosen))
    diag["k_returned"] = len(chosen)
    return chosen, diag
