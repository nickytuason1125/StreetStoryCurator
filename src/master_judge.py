"""
MasterJudge — the learned judge head trained on the photographer's own
master ratings (cache/user_ratings.json, mostly tagged ``tpe_master``).

Why this exists
───────────────
Every weighting in the grading pipeline (aspect blend, archetype formulas,
penalty gates) is hand-tuned and none of it has ever seen the one source of
ground truth this app has: 790+ photos the photographer personally rated
1-5 stars. The incumbent grader measures only ρ ≈ +0.23 against those stars
(2026-08-29 rank-agreement investigation). A head trained ON that baseline,
champion/challenger-gated so it can only ship when it measurably beats the
hand-tuned grader on held-out master photos, is the cheapest large ρ lever
there is.

What it is
──────────
A ridge regression (closed-form numpy — this module is imported by the
CUDA-free grade worker, so NO torch / sklearn at module scope) from the
per-photo aspect breakdown (Technical, Composition, Lighting, Narrative,
Human/Culture, Aesthetic — the same numbers the Breakdown tab shows) to the
photographer's star rating.

Champion/challenger
───────────────────
fit() holds out a stratified 25% of master photos, fits on the rest, and
promotes the weights to cache/master_judge.json ONLY when the held-out
Spearman against the photographer's stars beats the incumbent machine
score's Spearman on the exact same held-out photos. The blend weight used
at grade time scales with that measured advantage (0.10 floor, 0.40 cap) —
the hand-tuned grader always keeps a vote. Disable entirely with
FIRSTCUT_MASTER_JUDGE_OFF=1.

It also hosts human_anchor_lo_hi(): the derivation of the photographer-
anchored calibration ruler used by scripts/derive_master_anchors.py, which
anchors the absolute score scale at the human 1-2★ / 4-5★ discriminant
quartiles instead of library-volume percentiles.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

# Canonical feature order — the aspect keys exactly as they appear in the
# per-photo breakdown dict at GRADE time (grade_pipeline_v2.per_photo_breakdowns)
# and in the stored LanceDB blob. "Aesthetic" is deliberately NOT a feature:
# it is the base grader score itself (added to the stored blob only at
# Lance-write time, and never present at grade time). Order is part of the
# saved fingerprint: reordering the list invalidates saved weights rather
# than silently permuting the regression.
FEATURES = ["Technical", "Composition", "Lighting",
            "Narrative", "Human/Culture", "AADB", "Exemplar"]

# Per-photo archetype weights (the `_arch_w` blob the hand formula uses to
# modulate the aspect blend). Without these the student sees strictly LESS
# than the incumbent sees — it cannot relearn a blend that depends on inputs
# it never receives. With them, the exam is fair: same inputs, learned vs
# hand-tuned combination.
ARCHES = ["geo", "night", "layer", "messy", "maxdoc"]

# The stacked design matrix is [machine score, *FEATURES, *arch weights]: the
# incumbent's own vote is feature 0, so the ridge learns RESIDUAL corrections
# on top of it — e.g. Human/Culture measures ρ = −0.15 against this
# photographer's stars while the hand formula weights it 0.30–0.35 in several
# archetypes. Without the score term the judge only sees the aspects, which
# measured WEAKER than the incumbent (0.78 vs 0.87) and correctly failed
# promotion.
DESIGN = (["(machine score)"] + FEATURES
          + [f"arch:{a}" for a in ARCHES])

_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "cache" / "master_judge.json"
_SHIPPED_PATH = Path(__file__).resolve().parent.parent / "data" / "master_judge_defaults.json"
_LOCK = threading.Lock()

# ── Fit hyperparameters ───────────────────────────────────────────────────────
_HOLDOUT_FRAC   = 0.25
_SEED           = 20260830
_RIDGE_LAMBDAS  = (0.1, 1.0, 10.0, 100.0)   # chosen by CV on TRAIN only
_CV_FOLDS       = 5
_MIN_PROMOTE_N  = 60    # refuse to promote off a tiny baseline
_MIN_HOLDOUT_N  = 12    # a ρ from fewer held-out photos is noise, not evidence

# ── Blend policy at grade time ────────────────────────────────────────────────
_BLEND_FLOOR    = 0.10
_BLEND_CAP      = 0.40
_BLEND_ADV_GAIN = 3.0   # weight = floor + gain × (ρ_judge − ρ_baseline), capped

# ── Statistics ────────────────────────────────────────────────────────────────

def _rankdata(xs: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean), 1-based — matches
    scripts_accuracy_report._rankdata so both harnesses agree."""
    xs = np.asarray(xs, dtype=np.float64)
    order = np.argsort(xs, kind="mergesort")
    ranks = np.empty(len(xs), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman ρ; NaN when undefined (n<3, or either side constant)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or len(a) != len(b):
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


# ── Features ──────────────────────────────────────────────────────────────────

def feature_vector(bd: dict) -> np.ndarray:
    """Aspect values + archetype weights in canonical DESIGN order (the
    machine-score column is prepended separately by the fit/predict code);
    NaN where an input is missing so callers can drop or flag the row
    instead of silently imputing zero (a missing Composition is not a
    zero-Composition)."""
    vals = [bd.get(k) for k in FEATURES]
    arch = bd.get("_arch_w") or {}
    vals += [arch.get(a) for a in ARCHES]
    return np.array([float(v) if isinstance(v, (int, float)) else np.nan
                     for v in vals], dtype=np.float64)


# ── Dataset collection ────────────────────────────────────────────────────────

def collect_rows() -> list:
    """Join the durable star ratings against the LanceDB store.

    Prefers the live store row (breakdown + current score); falls back to the
    machine score snapshotted at rating time (survives re-grades that moved the
    live row). Rows with no score at all are dropped — there is nothing to
    train against or compare with. Returns:
        [{path, stars, source, score, breakdown, from_snapshot}, ...]
    """
    import ratings_store as rs
    import lance_store as ls

    ratings = rs.load()
    if not ratings:
        return []
    try:
        store = {r["path"]: r for r in ls.query_all(min_score=0.0)}
    except Exception as err:
        print(f"[master_judge] lance store unavailable ({err}) — "
              f"falling back to rating-time score snapshots")
        store = {}

    out = []
    for path, stars in ratings.items():
        row = store.get(path)
        bd = {}
        score = None
        from_snapshot = False
        if row is not None:
            score = row.get("score")
            raw_bd = row.get("breakdown")
            # _row_to_dict() already json.loads()s the breakdown into a dict;
            # accept a raw JSON string too (other callers hand us both shapes).
            if isinstance(raw_bd, str) and raw_bd:
                try:
                    bd = json.loads(raw_bd)
                except Exception:
                    bd = {}
            elif isinstance(raw_bd, dict):
                bd = raw_bd
        if not isinstance(score, (int, float)):
            snap = rs.get_score_snapshot(path) or {}
            score = snap.get("score")
            from_snapshot = score is not None
        if not isinstance(score, (int, float)):
            continue
        out.append({"path": path, "stars": int(stars),
                    "source": rs.get_source(path), "score": float(score),
                    "grade": (row or {}).get("grade", "") if row is not None else "",
                    "breakdown": bd, "from_snapshot": from_snapshot})
    return out


# ── Stratified holdout ────────────────────────────────────────────────────────

def _stratified_split(y: np.ndarray, frac: float, seed: int):
    """(train_idx, holdout_idx) with every star bucket represented in the
    holdout proportionally. A bucket with a single member stays in train —
    a lone 1★ must not end up in the holdout (nothing left to learn it from)
    nor alone define the whole holdout's 1★ share."""
    rng = np.random.default_rng(seed)
    train = []
    hold = []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        rng.shuffle(idx)
        n_hold = int(round(frac * len(idx))) if len(idx) >= 2 else 0
        hold.extend(idx[:n_hold].tolist())
        train.extend(idx[n_hold:].tolist())
    return np.asarray(train, dtype=int), np.asarray(hold, dtype=int)


# ── Ridge (closed form, numpy) ────────────────────────────────────────────────

def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    """Standardised-input ridge via the normal equations. Returns
    (coef, intercept, mean, std) with zero-variance columns left unscaled."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xs = (X - mean) / std
    d = X.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    coef = np.linalg.solve(A, Xs.T @ y)
    intercept = float(y.mean() - Xs.mean(axis=0) @ coef)
    return coef, intercept, mean, std


def _ridge_predict(model: tuple, X: np.ndarray) -> np.ndarray:
    coef, intercept, mean, std = model
    return (X - mean) / std @ coef + intercept


def _cv_rho(X: np.ndarray, y: np.ndarray, lam: float, k: int, seed: int) -> float:
    """Mean held-out Spearman across k stratified folds — used to pick λ on
    TRAIN only, so the true holdout never selects a model."""
    folds = [[] for _ in range(k)]
    for val in np.unique(y):
        b_idx = np.where(y == val)[0]
        rng = np.random.default_rng(seed + int(val * 1000))
        rng.shuffle(b_idx)
        for j, i in enumerate(b_idx):
            folds[j % k].append(int(i))
    rhos = []
    for f in range(k):
        te = np.asarray(folds[f], dtype=int)
        if len(te) < 3:
            continue
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 10:
            continue
        r = spearman(_ridge_predict(_ridge_fit(X[tr], y[tr], lam), X[te]), y[te])
        if not np.isnan(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


# ── Fit + promote ─────────────────────────────────────────────────────────────

def _feature_fingerprint() -> str:
    import hashlib
    return hashlib.sha1(json.dumps(DESIGN).encode()).hexdigest()[:16]


def _write_weights(out: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with _LOCK:
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def fit_from_rows(rows: list, weights_path: "Path | None" = None,
                  holdout_frac: float = _HOLDOUT_FRAC,
                  seed: int = _SEED,
                  lambdas: tuple = _RIDGE_LAMBDAS) -> dict:
    """Fit the ridge judge on `rows` (as produced by collect_rows()) and write
    champion/challenger metadata. Promotion rule: held-out ρ against the
    photographer's stars must BEAT the incumbent machine score's ρ on the
    exact same held-out photos, with enough data for either ρ to mean
    anything. Returns the full stats dict (promoted True/False either way)."""
    rows = [r for r in rows
            if r.get("breakdown") and np.isfinite(feature_vector(r["breakdown"])).all()]
    n = len(rows)
    if n < _MIN_PROMOTE_N:
        return {"n": n, "promoted": False,
                "reason": f"only {n} rows with a complete aspect breakdown; "
                          f"need >= {_MIN_PROMOTE_N}"}

    X_asp = np.stack([feature_vector(r["breakdown"]) for r in rows])
    y = np.array([r["stars"] for r in rows], dtype=np.float64)
    machine = np.array([r["score"] for r in rows], dtype=np.float64)
    # Stacked design: the incumbent's own score is feature 0 (see DESIGN).
    X = np.column_stack([machine, X_asp])

    tr, ho = _stratified_split(y, holdout_frac, seed)
    if len(ho) < _MIN_HOLDOUT_N:
        return {"n": n, "promoted": False,
                "reason": f"holdout only {len(ho)} photos; need >= {_MIN_HOLDOUT_N}"}

    # λ chosen by CV on the TRAIN split only — the holdout never selects a model.
    best_lam, best_cv = lambdas[0], -2.0
    cv_by_lam = {}
    for lam in lambdas:
        r = _cv_rho(X[tr], y[tr], lam, _CV_FOLDS, seed)
        cv_by_lam[lam] = None if np.isnan(r) else round(r, 4)
        if not np.isnan(r) and r > best_cv:
            best_lam, best_cv = lam, r

    model = _ridge_fit(X[tr], y[tr] / 5.0, best_lam)
    pred_ho = _ridge_predict(model, X[ho])
    rho_judge = spearman(pred_ho, y[ho] / 5.0)
    rho_base = spearman(machine[ho], y[ho])

    promoted = bool(not np.isnan(rho_judge) and not np.isnan(rho_base)
                    and rho_judge > rho_base)

    out = {
        "n": n,
        "n_holdout": int(len(ho)),
        "n_train": int(len(tr)),
        "lambda": best_lam,
        "cv_by_lambda": cv_by_lam,
        "rho_holdout": None if np.isnan(rho_judge) else round(rho_judge, 4),
        "rho_baseline": None if np.isnan(rho_base) else round(rho_base, 4),
        "promoted": promoted,
        "features": list(DESIGN),
        "coef": [round(float(c), 6) for c in model[0]],
        "intercept": round(float(model[1]), 6),
        "mean": [round(float(m), 6) for m in model[2]],
        "std": [round(float(s), 6) for s in model[3]],
        "feature_fingerprint": _feature_fingerprint(),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not promoted:
        out["reason"] = (
            f"challenger lost: judge ρ={out['rho_holdout']} vs incumbent "
            f"ρ={out['rho_baseline']} on the same {len(ho)} held-out master photos")

    # ρ trend: keep the last 20 challenge results so improvement over time is
    # a curve, not a single number. A lost challenge still records itself —
    # the history IS the continuous-improvement log.
    history = []
    try:
        wp = weights_path or _WEIGHTS_PATH
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            history = list(prev.get("history") or [])[-19:]
    except Exception:
        history = []
    history.append({"generated": out["generated"], "n": n,
                    "rho_holdout": out["rho_holdout"],
                    "rho_baseline": out["rho_baseline"],
                    "promoted": promoted})
    out["history"] = history

    _write_weights(out, weights_path or _WEIGHTS_PATH)
    return out


def fit() -> dict:
    """collect_rows() + fit_from_rows() — the one-call retrain."""
    return fit_from_rows(collect_rows())


# ── Grade-time inference ──────────────────────────────────────────────────────

def load(weights_path: "Path | None" = None) -> "dict | None":
    """Promoted weights only. A file whose fingerprint no longer matches the
    canonical feature order is treated as absent — a permuted regression would
    silently score every photo against the wrong coefficients."""
    if os.environ.get("FIRSTCUT_MASTER_JUDGE_OFF", "").strip():
        return None
    p = weights_path or _WEIGHTS_PATH
    try:
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("promoted"):
            return None
        if d.get("feature_fingerprint") != _feature_fingerprint():
            print("[master_judge] saved weights carry a stale feature "
                  "fingerprint — treating as absent")
            return None
        d["_source"] = "cache"
        return d
    except Exception as err:
        print(f"[master_judge] could not read weights ({err})")
        return None


def blend_weight(weights_path: "Path | None" = None) -> float:
    """The share of the final score the judge earns, scaled by how decisively
    it beat the incumbent on the holdout. 0.0 = no vote."""
    d = load(weights_path)
    if not d:
        return 0.0
    rho_j, rho_b = d.get("rho_holdout"), d.get("rho_baseline")
    if rho_j is None or rho_b is None:
        return 0.0
    adv = rho_j - rho_b
    return float(min(_BLEND_CAP, max(_BLEND_FLOOR,
                                     _BLEND_FLOOR + _BLEND_ADV_GAIN * adv)))


def predict_many(breakdowns: list, base_scores=None,
                 weights_path: "Path | None" = None,
                 weights: "dict | None" = None) -> np.ndarray:
    """Judge scores in [0, 1] for a list of breakdown dicts. `base_scores` are
    the incumbent's own per-photo scores (the stacked model's feature 0 —
    without them the stacked head cannot predict). Pass `weights` (a judge
    dict, e.g. from active()) to skip the cache lookup. Photos with an
    incomplete breakdown OR a missing base score get NaN — the caller keeps
    the incumbent score for those rather than guessing."""
    w = weights if weights is not None else load(weights_path)
    out = np.full(len(breakdowns), np.nan, dtype=np.float64)
    if not w or not breakdowns:
        return out
    if base_scores is None:
        base = np.full(len(breakdowns), np.nan, dtype=np.float64)
    else:
        base = np.asarray(base_scores, dtype=np.float64)
        if base.shape[0] != len(breakdowns):
            raise ValueError("base_scores must align with breakdowns")
    X_asp = np.stack([feature_vector(bd or {}) for bd in breakdowns])
    X = np.column_stack([base, X_asp])
    coef = np.asarray(w["coef"], dtype=np.float64)
    mean = np.asarray(w["mean"], dtype=np.float64)
    std = np.where(np.asarray(w["std"], dtype=np.float64) < 1e-9, 1.0,
                   np.asarray(w["std"], dtype=np.float64))
    intercept = float(w["intercept"])
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        Xs = (X[ok] - mean) / std
        out[ok] = np.clip(Xs @ coef + intercept, 0.0, 1.0)
    return out


# ── The human-anchored ruler (scripts/derive_master_anchors.py) ───────────────

def human_anchor_lo_hi(disc_by_star: dict, hi_stars=(4, 5), lo_stars=(1, 2),
                       min_group: int = 8) -> tuple:
    """Derive (lo, hi) calibration anchors from the photographer's own stars.

    The shipped ruler anchors at corpus percentiles: the scale measures how
    MANY photos exist, not how GOOD they are. This anchors the same stretch at
    the human quartiles instead:

        hi = median(4-5★ discriminant) + ¼·IQR   → your strong work tops the scale
        lo = median(1-2★ discriminant) − ¼·IQR   → your rejects bottom it

    and REFUSES unless the 3★ median sits between them — a scale where your
    mids don't land between your hits and your rejects is measuring noise.

    Returns (lo, hi, info); raises ValueError with the reason when the data
    cannot support a scale.
    """
    def _group(stars):
        vals = [v for s, vals in disc_by_star.items() if s in stars for v in vals]
        return np.asarray(vals, dtype=np.float64)

    hi_g, lo_g, mid_g = _group(hi_stars), _group(lo_stars), _group((3,))
    if len(hi_g) < min_group or len(lo_g) < min_group:
        raise ValueError(f"anchor groups too small: {len(lo_g)} low / {len(hi_g)} "
                         f"high; need >= {min_group} each")

    def _q(v, p):
        return float(np.percentile(v, p))

    hi_a = _q(hi_g, 50) + 0.25 * (_q(hi_g, 75) - _q(hi_g, 25))
    lo_a = _q(lo_g, 50) - 0.25 * (_q(lo_g, 75) - _q(lo_g, 25))
    if hi_a - lo_a < 1e-5:
        raise ValueError(f"degenerate span: lo={lo_a:.5f} hi={hi_a:.5f}")
    if len(mid_g) >= min_group:
        mid_med = _q(mid_g, 50)
        if not (lo_a < mid_med < hi_a):
            raise ValueError(f"3-star median {mid_med:.5f} does not sit between "
                             f"the anchored scale ({lo_a:.5f}, {hi_a:.5f}) — "
                             f"these stars are not a monotone ruler")

    info = {"n_low": int(len(lo_g)), "n_high": int(len(hi_g)),
            "n_mid": int(len(mid_g)), "lo": lo_a, "hi": hi_a,
            "median_low": round(float(np.median(lo_g)), 5),
            "median_mid": round(float(np.median(mid_g)), 5) if len(mid_g) else None,
            "median_high": round(float(np.median(hi_g)), 5)}
    return lo_a, hi_a, info


# ── The master algo: shipped defaults + promotion ────────────────────────────
# Two-phase design (2026-08-30):
#   BUILD (offline): ratings train challengers; a challenger that beats the
#     incumbent on held-out photos is "promoted" into the cache record —
#     and ONLY THEN may an operator deliberately bake it into
#     data/master_judge_defaults.json via promote_to_shipped().
#   RUN (the grader every user gets): the SHIPPED judge is part of the
#     algorithm itself — it grades by default for every install, no flags,
#     no ratings, exactly like the shipped encoder weights. A LOCAL cache
#     challenger, by contrast, is trained on a user's own ratings and stays
#     opt-in (FIRSTCUT_MASTER_JUDGE=1): user ratings must never grade.

def _valid_judge_dict(d: dict) -> bool:
    """A judge record is usable only when it won its exam, matches the
    current feature design, and carries complete weights."""
    required = ("coef", "intercept", "mean", "std",
                "rho_holdout", "rho_baseline", "feature_fingerprint")
    return (isinstance(d, dict)
            and d.get("promoted") is True
            and d.get("feature_fingerprint") == _feature_fingerprint()
            and all(k in d for k in required))


def _weight_from(d: dict) -> float:
    rho_j, rho_b = d.get("rho_holdout"), d.get("rho_baseline")
    if rho_j is None or rho_b is None:
        return 0.0
    adv = rho_j - rho_b
    return float(min(_BLEND_CAP, max(_BLEND_FLOOR,
                                     _BLEND_FLOOR + _BLEND_ADV_GAIN * adv)))


def _load_shipped() -> "dict | None":
    """The shipped master judge — part of the algorithm, no opt-in needed.
    A stale fingerprint (feature design changed since shipping) is treated
    as absent, loudly."""
    try:
        if os.environ.get("FIRSTCUT_MASTER_JUDGE_OFF", "").strip():
            return None
        if not _SHIPPED_PATH.exists():
            return None
        d = json.loads(_SHIPPED_PATH.read_text(encoding="utf-8"))
        if not _valid_judge_dict(d):
            print("[master_judge] shipped master judge is invalid/stale "
                  "for the current feature design — ignoring")
            return None
        d["_source"] = "shipped"
        return d
    except Exception as err:
        print(f"[master_judge] could not read shipped master judge ({err})")
        return None


def active() -> tuple:
    """(judge_dict, blend_weight) for THIS grading run, or (None, 0.0).

    Precedence, per the two-phase contract:
      1. kill switch FIRSTCUT_MASTER_JUDGE_OFF=1 → nothing
      2. a local cache challenger won its exam AND the user opted in
         (FIRSTCUT_MASTER_JUDGE=1) → the local judge — it is newer than
         whatever was shipped
      3. the shipped master judge → always on, no flags, no ratings
      4. nothing
    """
    if os.environ.get("FIRSTCUT_MASTER_JUDGE_OFF", "").strip():
        return None, 0.0
    cache = load()
    if cache and os.environ.get("FIRSTCUT_MASTER_JUDGE", "").strip() == "1":
        return cache, _weight_from(cache)
    shipped = _load_shipped()
    if shipped:
        return shipped, _weight_from(shipped)
    return None, 0.0


def promote_to_shipped(cache_path: "Path | None" = None,
                       shipped_path: "Path | None" = None) -> dict:
    """Bake a PROMOTED local challenger into the shipped defaults.

    This is the deliberate, operator-run step that turns a won exam into a
    master-algo release: the weights move from one machine's cache into
    data/ (shipped with the app), so every install grades with them and no
    user ever rates anything. REFUSES to ship an unpromoted record — a
    challenger that lost its exam can never become the master.
    """
    src = cache_path or _WEIGHTS_PATH
    dst = shipped_path or _SHIPPED_PATH
    try:
        if not src.exists():
            return {"shipped": False,
                    "reason": f"no cache record at {src} — run "
                              f"scripts/master_backtest.py --fit first"}
        d = json.loads(src.read_text(encoding="utf-8"))
    except Exception as err:
        return {"shipped": False, "reason": f"could not read cache record ({err})"}
    if not _valid_judge_dict(d):
        return {"shipped": False,
                "reason": "the cache record is unpromoted, stale, or "
                          "incomplete — a challenger that lost its exam "
                          "can never become the master"}
    out = dict(d)
    out.pop("_source", None)
    out["shipped_from_cache_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_weights(out, dst)
    return {"shipped": True, "path": str(dst),
            "rho_holdout": d.get("rho_holdout"),
            "rho_baseline": d.get("rho_baseline"),
            "n": d.get("n")}


# ── Continuous loop: auto-refit after enough NEW ratings ─────────────────────
# Mirrors background_dpo_trainer's "fire once a batch has accumulated" pattern.
# A refit is always SAFE (worst case the challenger loses and nothing changes
# at grade time), so the only cost worth gating is the ~10-20 s of CPU.

_AUTOFIT_DELTA = max(5, int(os.environ.get("FIRSTCUT_MASTER_AUTOFIT_EVERY", "25") or 25))
_autofit_lock = threading.Lock()
_autofit_thread = None


def ratings_count() -> int:
    try:
        import ratings_store as rs
        return len(rs.load())
    except Exception:
        return 0


def maybe_autofit(n_now: "int | None" = None) -> dict:
    """Launch a background champion/challenger refit once >= _AUTOFIT_DELTA
    new ratings have banked since the last fit. Called from the star-rating
    endpoint; NEVER blocks the caller — the fit runs on a daemon thread and
    only ever rewrites cache/master_judge.json (a lost challenge is a record,
    not a regression). Returns {"triggered": bool, "reason": str}."""
    global _autofit_thread
    if os.environ.get("FIRSTCUT_MASTER_JUDGE", "").strip() != "1":
        # 2026-08-30: ratings are placeholder data — background refits only
        # make sense once the judge is explicitly opted in. Pure measurement
        # (scripts/master_backtest.py) stays available regardless.
        return {"triggered": False,
                "reason": "MasterJudge not opted in "
                          "(set FIRSTCUT_MASTER_JUDGE=1)"}
    with _autofit_lock:
        if _autofit_thread is not None and _autofit_thread.is_alive():
            return {"triggered": False, "reason": "fit already running"}
        n_now = ratings_count() if n_now is None else int(n_now)
        n_then = 0
        try:
            if _WEIGHTS_PATH.exists():
                n_then = int(json.loads(
                    _WEIGHTS_PATH.read_text(encoding="utf-8")).get("n") or 0)
        except Exception:
            n_then = 0
        delta = n_now - n_then
        if delta < _AUTOFIT_DELTA:
            return {"triggered": False,
                    "reason": f"{delta} new ratings since last fit "
                              f"(threshold {_AUTOFIT_DELTA})"}

        def _run():
            try:
                stats = fit()
                print(f"[master_judge] auto-refit: promoted="
                      f"{stats.get('promoted')} rho={stats.get('rho_holdout')} "
                      f"vs baseline {stats.get('rho_baseline')} (n={stats.get('n')})")
            except Exception as err:
                print(f"[master_judge] auto-refit failed: {err}")

        _autofit_thread = threading.Thread(target=_run, daemon=True,
                                           name="master-judge-autofit")
        _autofit_thread.start()
        return {"triggered": True, "reason": f"{delta} new ratings since last fit"}
