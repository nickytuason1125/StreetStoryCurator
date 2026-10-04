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

# 2026-10 signals the incumbent judge never saw (subject_sharpness / pans /
# living subject). The pipeline writes _pan, _smeared_subject and _bg_streak
# only when they apply, so ABSENT means "no" (0.0). The measured ones must be
# present or the row is skipped — a missing measurement is not a zero.
EXTRA = ["_subject_sharp", "_subject_streak", "_bg_streak", "_living",
         "_pan", "_smeared_subject"]
_ABSENT_IS_ZERO = {"_pan", "_smeared_subject", "_bg_streak"}

# A fit uses only the design columns present in at least this share of its
# rows, and records that list. AADB/Exemplar come from optional models that
# are absent on most installs; requiring them silently dropped EVERY row
# (2026-10-03: 0 of 157 freshly graded rated photos were usable).
_MIN_FEATURE_COVERAGE = 0.90

# The stacked design matrix is [machine score, *FEATURES, *arch weights]: the
# incumbent's own vote is feature 0, so the ridge learns RESIDUAL corrections
# on top of it — e.g. Human/Culture measures ρ = −0.15 against this
# photographer's stars while the hand formula weights it 0.30–0.35 in several
# archetypes. Without the score term the judge only sees the aspects, which
# measured WEAKER than the incumbent (0.78 vs 0.87) and correctly failed
# promotion.
DESIGN = (["(machine score)"] + FEATURES
          + [f"arch:{a}" for a in ARCHES] + EXTRA)

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
    return np.array([_bd_value(bd, name) for name in DESIGN[1:]], dtype=np.float64)


def _bd_value(bd: dict, name: str) -> float:
    """Look up one named feature/arch value from a breakdown dict by NAME
    (not by position in the global FEATURES/DESIGN order) — NaN if missing
    or non-numeric, the same convention feature_vector() uses. Lets
    predict_many() score using a judge's own stored `features` list instead
    of always assuming the current live design."""
    if name.startswith("arch:"):
        v = (bd.get("_arch_w") or {}).get(name[len("arch:"):])
    else:
        v = bd.get(name)
        if v is None and name in _ABSENT_IS_ZERO:
            v = 0.0
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v) if isinstance(v, (int, float)) else float("nan")


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
        if not bd:
            # Features snapshotted onto the rating itself — survives a cleared
            # catalog/LanceDB and a moved file (ratings_store.attach_features),
            # with the score from the SAME grade (not the rating-time snapshot,
            # which may come from an older grader).
            bd = rs.get_features(path) or {}
            if bd and row is None:
                score = rs.get_current_score(path)
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

def _feature_fingerprint(design: "list | None" = None) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(design or DESIGN).encode()).hexdigest()[:16]


def _shoot_day(path: str) -> str:
    """Grouping key for honest held-out evaluation: parent folder + capture
    date (file mtime, which a camera card and copy2 both preserve). Falls back
    to the folder alone when the file is not reachable."""
    import accuracy as acc
    key = acc.shoot_of(path)
    try:
        key += "|" + time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))
    except OSError:
        pass
    return key


def _group_folds(groups: list, k: int, seed: int) -> list:
    """Fold index per row; every group lives in exactly one fold. Largest
    groups placed first into the currently smallest fold (balanced), ties in
    a seeded order so the split is reproducible."""
    counts: dict = {}
    for g in groups:
        counts[g] = counts.get(g, 0) + 1
    names = sorted(counts)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    names.sort(key=lambda g: -counts[g])
    sizes = [0] * k
    fold_of = {}
    for g in names:
        f = int(np.argmin(sizes))
        fold_of[g] = f
        sizes[f] += counts[g]
    return [fold_of[g] for g in groups]


def _shoot_split(rows: list, frac: float, seed: int):
    """Train/holdout indices with every shoot (parent folder) entirely on one
    side, so burst frames of the same moment never sit on both sides."""
    import accuracy as acc
    shoot = [acc.shoot_of(r["path"]) for r in rows]
    names = sorted(set(shoot))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    target = max(1, int(round(len(rows) * frac)))
    ho_s, count = set(), 0
    for nm in names:
        if count >= target or len(ho_s) >= len(names) - 1:
            break
        ho_s.add(nm)
        count += shoot.count(nm)
    ho = np.array([i for i, sh in enumerate(shoot) if sh in ho_s], dtype=np.intp)
    tr = np.array([i for i, sh in enumerate(shoot) if sh not in ho_s], dtype=np.intp)
    return tr, ho


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
    rows = [r for r in rows if r.get("breakdown")]
    # This fit's design: only columns present in most rows (optional models
    # such as AADB/Exemplar are absent on most installs). Recorded in the
    # output so predict_many() scores with exactly these names.
    names = []
    for name in DESIGN[1:]:
        cov = np.mean([np.isfinite(_bd_value(r["breakdown"], name)) for r in rows]) if rows else 0.0
        if cov >= _MIN_FEATURE_COVERAGE:
            names.append(name)
    rows = [r for r in rows
            if np.isfinite([_bd_value(r["breakdown"], nm) for nm in names]).all()]
    n = len(rows)
    if n < _MIN_PROMOTE_N:
        return {"n": n, "promoted": False,
                "reason": f"only {n} rows with a complete aspect breakdown; "
                          f"need >= {_MIN_PROMOTE_N}"}

    X_asp = np.array([[_bd_value(r["breakdown"], nm) for nm in names] for r in rows],
                     dtype=np.float64)
    y = np.array([r["stars"] for r in rows], dtype=np.float64)
    machine = np.array([r["score"] for r in rows], dtype=np.float64)
    # Stacked design: the incumbent's own score is feature 0 (see DESIGN).
    X = np.column_stack([machine, X_asp])
    design = ["(machine score)"] + names

    # Out-of-fold exam over SHOOT-DAYS (folder + capture date): every photo is
    # scored by a judge that never saw its shoot-day, so burst frames of one
    # moment can't be "recognised". A single whole-folder holdout was unfair
    # on real data — one LX3 folder held 514 of 826 ratings, the holdout took
    # 81% of rows and the judge trained on 156 (2026-10-03).
    import accuracy as acc
    from rating_calibration import thresholds_from_records
    groups = [_shoot_day(r["path"]) for r in rows]
    folds = _group_folds(groups, k=_CV_FOLDS, seed=seed)
    holdout_kind = f"{_CV_FOLDS}-fold by shoot-day ({len(set(groups))} groups)"
    if len({f for f in folds}) < 2:
        holdout_kind = "photos (single shoot-day)"
        folds = list(np.arange(n) % _CV_FOLDS)
    pred_oof = np.full(n, np.nan)
    # The fair baseline: the incumbent RECALIBRATED the same way (ridge on the
    # machine score alone, blended identically). Without it, blending in ANY
    # judge shifts scores across the Strong/Mid/Weak lines and can "win" on
    # recalibration alone — a pure-noise judge was promoted that way.
    cal_oof = np.full(n, np.nan)
    true_c, base_c, judge_raw, fold_thr = [0] * n, [0] * n, np.full(n, np.nan), {}
    for f in sorted(set(folds)):
        te = np.array([i for i in range(n) if folds[i] == f], dtype=np.intp)
        tr = np.array([i for i in range(n) if folds[i] != f], dtype=np.intp)
        if len(tr) < 10 or len(te) == 0:
            continue
        lam_f, best_f = lambdas[0], -2.0
        for lam in lambdas:                     # λ by CV inside the training folds only
            r = _cv_rho(X[tr], y[tr], lam, _CV_FOLDS, seed)
            if not np.isnan(r) and r > best_f:
                lam_f, best_f = lam, r
        pred_oof[te] = _ridge_predict(_ridge_fit(X[tr], y[tr] / 5.0, lam_f), X[te])
        cal_oof[te] = _ridge_predict(_ridge_fit(X[tr][:, :1], y[tr] / 5.0, lam_f), X[te][:, :1])
        st, mt, _ = thresholds_from_records(
            [{"path": rows[i]["path"], "stars": rows[i]["stars"],
              "score": rows[i]["score"], "rated_at": None} for i in tr])
        fold_thr[f] = (st, mt)
    ok = np.isfinite(pred_oof)
    if ok.sum() < _MIN_HOLDOUT_N:
        return {"n": n, "promoted": False,
                "reason": f"only {int(ok.sum())} photos could be scored out-of-fold; "
                          f"need >= {_MIN_HOLDOUT_N}"}
    rho_judge = spearman(pred_oof[ok], y[ok] / 5.0)
    rho_base = spearman(machine[ok], y[ok])
    w_blend = (_weight_from({"rho_holdout": rho_judge, "rho_baseline": rho_base})
               if not (np.isnan(rho_judge) or np.isnan(rho_base)) else 0.0)
    blended = (1.0 - w_blend) * machine + w_blend * np.clip(np.nan_to_num(pred_oof), 0.0, 1.0)
    calibrated = (1.0 - w_blend) * machine + w_blend * np.clip(np.nan_to_num(cal_oof), 0.0, 1.0)
    idx = np.flatnonzero(ok)
    raw_c = [0] * n
    for i in idx:
        st, mt = fold_thr[folds[i]]
        true_c[i] = acc.stars_to_class(int(y[i]))
        raw_c[i] = acc.score_to_class(float(machine[i]), st, mt)
        base_c[i] = acc.score_to_class(float(calibrated[i]), st, mt)
        judge_raw[i] = acc.score_to_class(float(blended[i]), st, mt)
    true_c = [true_c[i] for i in idx]
    raw_c = [raw_c[i] for i in idx]
    base_c = [base_c[i] for i in idx]
    judge_c = [int(judge_raw[i]) for i in idx]
    g_ok = [groups[i] for i in idx]
    # Resampling 1-7 groups gives a degenerate interval that LOOKS certain
    # (one group -> lo == hi). Below 8 shoot-days, resample photos instead.
    if len(set(g_ok)) < 8:
        g_ok = [str(i) for i in idx]
    boot = acc.paired_bootstrap(true_c, base_c, judge_c, g_ok, n=2000, seed=seed)
    # The ranking grading actually USES (the blend) must not get worse.
    rho_blend = spearman(blended[ok], y[ok])
    agree_raw = acc.agreement(raw_c, true_c)

    promoted = bool(not np.isnan(rho_blend) and not np.isnan(rho_base)
                    and rho_blend >= rho_base and boot["lo"] > 0)
    kind = "full"
    agree_b_rep, agree_j_rep, ci = acc.agreement(base_c, true_c), acc.agreement(judge_c, true_c), boot
    rho_j_rep = rho_judge
    boot_cal = None
    if not promoted:
        # Fallback candidate: recalibrate the incumbent's OWN score toward the
        # photographer's stars (2026-10-03: their Mid frames were scored into
        # Weak). Evaluated at the weight it would actually get at grade time —
        # equal ρ (a monotone recalibration) earns the blend floor — and
        # against the RAW incumbent, which is what grading uses today.
        w_cal = _weight_from({"rho_holdout": rho_base, "rho_baseline": rho_base})
        cal_blend = (1.0 - w_cal) * machine + w_cal * np.clip(np.nan_to_num(cal_oof), 0.0, 1.0)
        cal_c = [acc.score_to_class(float(cal_blend[i]), *fold_thr[folds[i]]) for i in idx]
        boot_cal = acc.paired_bootstrap(true_c, raw_c, cal_c, g_ok, n=2000, seed=seed)
        if boot_cal["lo"] > 0 and not np.isnan(rho_base):
            promoted, kind = True, "calibration"
            design, X = ["(machine score)"], X[:, :1]
            agree_b_rep, agree_j_rep, ci = agree_raw, acc.agreement(cal_c, true_c), boot_cal
            rho_j_rep = rho_base           # monotone recalibration: ranking unchanged

    # Deployed judge: refit on ALL rows (λ by CV on all of them).
    best_lam, best_cv = lambdas[0], -2.0
    cv_by_lam = {}
    for lam in lambdas:
        r = _cv_rho(X, y, lam, _CV_FOLDS, seed)
        cv_by_lam[lam] = None if np.isnan(r) else round(r, 4)
        if not np.isnan(r) and r > best_cv:
            best_lam, best_cv = lam, r
    model = _ridge_fit(X, y / 5.0, best_lam)
    strong_t, mid_t, _ = thresholds_from_records(
        [{"path": r["path"], "stars": r["stars"], "score": r["score"], "rated_at": None}
         for r in rows])
    ho = idx
    tr = idx

    out = {
        "n": n,
        "n_holdout": int(len(ho)),
        "n_train": int(n),
        "rho_blend": None if np.isnan(rho_blend) else round(rho_blend, 4),
        # The weight grading will ACTUALLY apply (_weight_from on this record).
        "blend_weight": round(float(_weight_from({"rho_holdout": rho_j_rep,
                                                   "rho_baseline": rho_base})), 4)
                        if not (np.isnan(rho_j_rep) or np.isnan(rho_base)) else 0.0,
        "lambda": best_lam,
        "cv_by_lambda": cv_by_lam,
        "rho_holdout": None if np.isnan(rho_j_rep) else round(rho_j_rep, 4),
        "rho_baseline": None if np.isnan(rho_base) else round(rho_base, 4),
        "promoted": promoted,
        "kind": kind,                                 # "full" judge or "calibration" only
        "agree_raw": round(agree_raw, 4),             # incumbent as-is
        "agree_recal": round(acc.agreement(base_c, true_c), 4),  # incumbent recalibrated
        "agree_full_judge": round(acc.agreement(judge_c, true_c), 4),
        "agree_base": round(agree_b_rep, 4),          # what the winner was compared to
        "agree_judge": round(agree_j_rep, 4),         # what grading gets if promoted
        "agree_ci": [round(ci["lo"], 4), round(ci["hi"], 4)],
        "calibration_ci": None if boot_cal is None else [round(boot_cal["lo"], 4), round(boot_cal["hi"], 4)],
        "n_groups": len(set(groups)),
        "holdout_kind": holdout_kind,
        "thresholds": [round(float(strong_t), 4), round(float(mid_t), 4)],
        "features": design,
        "coef": [round(float(c), 6) for c in model[0]],
        "intercept": round(float(model[1]), 6),
        "mean": [round(float(m), 6) for m in model[2]],
        "std": [round(float(s), 6) for s in model[3]],
        "feature_fingerprint": _feature_fingerprint(design),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not promoted:
        out["reason"] = (
            f"challenger lost on 3-class agreement over {len(ho)} held-out photos: "
            f"full judge {out['agree_full_judge']} vs recalibrated incumbent "
            f"{out['agree_recal']} (gain CI {out['agree_ci']}); recalibration alone "
            f"vs incumbent as-is CI {out['calibration_ci']}")

    # ρ trend: keep the last 20 challenge results so improvement over time is
    # a curve, not a single number. A lost challenge still records itself —
    # the history IS the continuous-improvement log.
    history = []
    prev = None
    wp = weights_path or _WEIGHTS_PATH
    try:
        if wp.exists():
            prev = json.loads(wp.read_text(encoding="utf-8"))
            history = list(prev.get("history") or [])[-19:]
    except Exception:
        history, prev = [], None
    history.append({"generated": out["generated"], "n": n,
                    "rho_holdout": out["rho_holdout"],
                    "rho_baseline": out["rho_baseline"],
                    "agree_base": out["agree_base"],
                    "agree_judge": out["agree_judge"],
                    "promoted": promoted})
    out["history"] = history

    # A LOST challenge must never displace a promoted judge: keep the winner's
    # weights and only log this attempt. (Overwriting it used to drop grading
    # back to the shipped judge after any unlucky background refit.)
    if not promoted and isinstance(prev, dict) and prev.get("promoted") is True             and _applicable_judge_dict(prev):
        kept = dict(prev)
        kept["history"] = history
        kept["last_challenge"] = {k: out.get(k) for k in (
            "generated", "n", "kind", "agree_raw", "agree_recal", "agree_full_judge",
            "agree_ci", "calibration_ci", "reason")}
        _write_weights(kept, wp)
        return out
    _write_weights(out, wp)
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
        # Each fit records its OWN design (only the columns its data had);
        # predict_many() scores by those names, so the check is that the
        # record is internally consistent, not that it matches today's DESIGN.
        if (not _applicable_judge_dict(d)
                or d.get("feature_fingerprint") != _feature_fingerprint(d.get("features"))):
            print("[master_judge] saved weights are incomplete or their feature "
                  "list does not match their fingerprint — treating as absent")
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
    judge_features = w["features"]  # the judge's OWN stored list, not DESIGN
    cols = []
    for name in judge_features:
        if name == "(machine score)":
            cols.append(base)
        else:
            cols.append(np.array([_bd_value(bd or {}, name) for bd in breakdowns],
                                 dtype=np.float64))
    X = np.column_stack(cols)
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
            and d.get("feature_fingerprint") == _feature_fingerprint(d.get("features"))
            and all(k in d for k in required))


def _applicable_judge_dict(d: dict) -> bool:
    """A judge record is safe to APPLY at grade time (not necessarily
    freshly-fit) when it won its exam and carries complete, internally
    consistent weights. Unlike _valid_judge_dict, this does NOT require the
    feature design to match the CURRENT live FEATURES: predict_many() scores
    using the judge's own stored `features` list, so a judge trained on a
    strict subset of today's features keeps producing its original, correct
    prediction — today's newly-added features it never saw are simply not
    part of its input, not scored as zero or dropped as incomplete."""
    required = ("coef", "intercept", "mean", "std",
                "rho_holdout", "rho_baseline", "features")
    return (isinstance(d, dict)
            and d.get("promoted") is True
            and all(k in d for k in required)
            and isinstance(d.get("features"), list)
            and len(d["features"]) == len(d.get("coef") or []))


def _weight_from(d: dict) -> float:
    rho_j, rho_b = d.get("rho_holdout"), d.get("rho_baseline")
    if rho_j is None or rho_b is None:
        return 0.0
    adv = rho_j - rho_b
    return float(min(_BLEND_CAP, max(_BLEND_FLOOR,
                                     _BLEND_FLOOR + _BLEND_ADV_GAIN * adv)))


def _load_shipped() -> "dict | None":
    """The shipped master judge — part of the algorithm, no opt-in needed.
    A stale fingerprint (feature design changed since shipping) is acceptable —
    predict_many() scores using the judge's own stored feature list, so older
    feature designs are safe by design. Only rejected when genuinely malformed
    (not promoted, missing required weight keys, or coef/features length
    mismatch)."""
    try:
        if os.environ.get("FIRSTCUT_MASTER_JUDGE_OFF", "").strip():
            return None
        if not _SHIPPED_PATH.exists():
            return None
        d = json.loads(_SHIPPED_PATH.read_text(encoding="utf-8"))
        if not _applicable_judge_dict(d):
            print("[master_judge] shipped master judge is malformed/"
                  "incomplete/unpromoted — ignoring")
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
    # 2026-10-03 (user: "The star ratings are legit trained"): a local judge
    # that PROVED a 3-class agreement gain on held-out shoots (bootstrap CI
    # lower bound > 0) grades by default. Older records without that proof
    # still need the explicit opt-in flag.
    proven = bool(cache and cache.get("agree_ci") and float(cache["agree_ci"][0]) > 0)
    if cache and (proven or os.environ.get("FIRSTCUT_MASTER_JUDGE", "").strip() == "1"):
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
    # No opt-in gate since 2026-10-03: ratings are real ground truth (user),
    # and a refit is always safe — a lost challenge is a record, it never
    # changes grading (active() only uses a PROVEN winner).
    if os.environ.get("FIRSTCUT_MASTER_JUDGE_OFF", "").strip():
        return {"triggered": False, "reason": "MasterJudge disabled (FIRSTCUT_MASTER_JUDGE_OFF)"}
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
