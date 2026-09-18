"""Rating-anchored verdict thresholds (2026-09-14).

Product rule from the photographer: a re-grade must not diverge from their
star ratings — the ratings ARE the accuracy standard. The retired personal
taste head tried to enforce that per-photo with a neural blend and produced
mush (legit keepers dragged toward Mid). This module does the honest,
global version: it fits WHERE the Strong/Mid/Weak cut-points sit on the
machine's own score scale from the photographer's rated photos.

    strong_thresh = 10th percentile of machine scores the photographer gave
                    4–5★   (≈90% of their keepers land Strong)
    mid_thresh    = 90th percentile of machine scores the photographer gave
                    1–2★   (≈90% of their rejects land Weak)

Guards, all of which fall back to the shipped defaults rather than
manufacturing agreement with noise:
  • both star groups must have >= min_group rated photos;
  • the keepers' low edge must sit ABOVE the rejects' high edge — if the
    photographer's stars do not order the machine's scores at all, the
    disagreement is reported, not papered over;
  • neither threshold may move more than max_shift from the shipped default
    (a pathological rating batch cannot slide the ruler onto the floor).

Pure numpy; no model imports, so tests can exercise it standalone.
"""
from __future__ import annotations

import time

import numpy as np

DEFAULT_STRONG = 0.60
DEFAULT_MID = 0.41


def _weighted_percentile(values, weights, q: float) -> float:
    """The value at cumulative weight fraction `q` (0–100). Standard weighted
    quantile: sort by value, walk the cumulative weight, take the first value
    whose cumulative weight reaches the cut."""
    order = np.argsort(np.asarray(values, dtype=np.float64))
    v = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cw = np.cumsum(w)
    cut = (q / 100.0) * cw[-1]
    return float(v[int(np.searchsorted(cw, cut, side="left"))])


def thresholds_from_records(
    records,
    strong_default: float = DEFAULT_STRONG,
    mid_default: float = DEFAULT_MID,
    k: float = 10.0,
    min_eff: float = 6.0,
    max_shift: float = 0.12,
    half_life_days: float = 365.0,
    now: float | None = None,
) -> tuple:
    """Global, shrinkage-stabilised fit over ALL rated photos across folders.

    `records` is ratings_store.load_records() output — {path, stars, score,
    rated_at?} — optionally with `score` overridden by the current run's live
    machine score for paths in the folder being graded (live beats snapshot:
    it reflects the scorer that is about to produce today's grades).

    Three upgrades over the folder-only percentile fit:
      • cross-folder — a folder with few rated photos still calibrates off the
        full rating history;
      • recency weighting — taste evolves; a rating's weight decays with a
        `half_life_days` exponential (legacy entries without a timestamp count
        at full weight);
      • Bayesian shrinkage — instead of jumping to the fitted value the moment
        a minimum count is reached, the estimate is
            t = (n_eff · fitted + k · default) / (n_eff + k)
        so thin data stays near the shipped defaults and rich data dominates.
        `min_eff` on the effective weight is still required before moving at
        all (below it, the data cannot outvote the prior).

    Returns (strong_thresh, mid_thresh, info) with a refusal reason in `info`
    whenever the data cannot support a ruler."""
    hi_v, hi_w, lo_v, lo_w = [], [], [], []
    now = time.time() if now is None else now
    for r in records:
        st = int(r.get("stars") or 0)
        sc = r.get("score")
        if st <= 0 or not isinstance(sc, (int, float)) or not np.isfinite(sc):
            continue
        sc = float(sc)
        w = 1.0
        ra = r.get("rated_at")
        if half_life_days > 0 and isinstance(ra, (int, float)) and ra > 0:
            age_days = max(0.0, (now - float(ra)) / 86400.0)
            w = 0.5 ** (age_days / half_life_days)
        if st >= 4:
            hi_v.append(sc); hi_w.append(w)
        elif st <= 2:
            lo_v.append(sc); lo_w.append(w)

    n_hi, n_lo = float(sum(hi_w)), float(sum(lo_w))
    info = {"n_keepers": len(hi_v), "n_rejects": len(lo_v),
            "w_keepers": round(n_hi, 2), "w_rejects": round(n_lo, 2)}

    if n_hi < min_eff or n_lo < min_eff:
        info["reason"] = (f"anchor weight too small: {n_lo:.1f} reject / "
                          f"{n_hi:.1f} keeper effective (need >= {min_eff})")
        return strong_default, mid_default, info

    s_hi = _weighted_percentile(hi_v, hi_w, 10)   # low edge of the keepers
    s_lo = _weighted_percentile(lo_v, lo_w, 90)   # high edge of the rejects
    info.update({"keeper_edge": round(s_hi, 4), "reject_edge": round(s_lo, 4)})

    # Shrink toward the shipped defaults by effective sample size.
    strong = (n_hi * s_hi + k * strong_default) / (n_hi + k)
    mid = (n_lo * s_lo + k * mid_default) / (n_lo + k)

    if not strong >= mid + 0.05:
        info["reason"] = (f"shrunk lines collide (strong {strong:.3f} vs mid "
                          f"{mid:.3f}) — these stars do not order the "
                          f"machine's scores; keeping defaults")
        return strong_default, mid_default, info

    # Trust region kept as a final safety even under shrinkage.
    strong = float(np.clip(strong, strong_default - max_shift,
                           strong_default + max_shift))
    mid = float(np.clip(mid, mid_default - max_shift,
                        mid_default + max_shift))
    if mid > strong - 0.05:
        mid = strong - 0.05                # keep a real Mid band
    info.update({"strong": round(strong, 4), "mid": round(mid, 4)})
    return strong, mid, info


def thresholds_from_ratings(
    stars_by_path: dict,
    paths,
    scores,
    strong_default: float = DEFAULT_STRONG,
    mid_default: float = DEFAULT_MID,
    min_group: int = 8,
    max_shift: float = 0.12,
) -> tuple:
    """Return (strong_thresh, mid_thresh, info) fitted to the photographer's
    stars. `stars_by_path` maps path → 1–5; `paths`/`scores` are the run's
    machine outputs. `info` is a dict describing what happened (or a refusal
    reason) so callers can log the decision."""
    scores = np.asarray(scores, dtype=np.float64)
    hi, lo = [], []
    for p, s in zip(paths, scores):
        st = int(stars_by_path.get(str(p), 0) or 0)
        if not np.isfinite(s):
            continue
        if st >= 4:
            hi.append(float(s))
        elif 0 < st <= 2:
            lo.append(float(s))

    info = {"n_keepers": len(hi), "n_rejects": len(lo)}

    if len(hi) < min_group or len(lo) < min_group:
        info["reason"] = (f"anchor groups too small: {len(lo)} rejects / "
                          f"{len(hi)} keepers, need >= {min_group} each")
        return strong_default, mid_default, info

    s_hi = float(np.percentile(hi, 10))   # low edge of the keepers
    s_lo = float(np.percentile(lo, 90))   # high edge of the rejects
    info.update({"keeper_edge": round(s_hi, 4), "reject_edge": round(s_lo, 4)})

    if not s_hi > s_lo:
        info["reason"] = (f"keepers' low edge ({s_hi:.3f}) does not sit above "
                          f"rejects' high edge ({s_lo:.3f}) — these stars do "
                          f"not order the machine's scores; keeping defaults")
        return strong_default, mid_default, info

    strong = float(np.clip(s_hi, strong_default - max_shift,
                           strong_default + max_shift))
    mid = float(np.clip(s_lo, mid_default - max_shift,
                        mid_default + max_shift))
    if mid > strong - 0.05:
        mid = strong - 0.05                # keep a real Mid band
    info.update({"strong": round(strong, 4), "mid": round(mid, 4)})
    return strong, mid, info
