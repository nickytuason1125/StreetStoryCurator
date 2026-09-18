"""Pairwise preference model + holdout gate (2026-09-15, Stages 2 & 3).

Stage 2 — pairwise culling. Star ratings contain more than absolute levels:
"this 4★ beats that 2★" is a pairwise preference that survives even when the
absolute scale drifts between sessions. This module learns from those pairs:

    features per photo = [machine_score,
                          keeper_affinity,     # cosine to k nearest 4–5★
                          reject_affinity]     # cosine to k nearest 1–2★

and fits a RankNet-style pairwise logistic model (pure numpy: loss =
softplus(-(x_w · w − x_l · w)) + L2) so that photos the machine scores alike
but that LOOK like your keepers (or your rejects) separate correctly.

Stage 3 — the holdout gate. The retired PersonalHead failed because it was
deployed unconditionally and blended itself into every score. This model never
blends: it is deployed ONLY IF, on a holdout set of the newest ratings it was
not trained on, it orders the photographer's pairwise preferences better than
the machine score alone does, by a real margin. The gate decision and its
metrics are persisted (cache/preference_model.json) so every subsequent run
uses the same verdict until a retrain changes it.

PRODUCT RULE (2026-09-14, unchanged): verdicts come from the rating-anchored
thresholds alone. The preference model only (a) attaches `pref_score` for
within-band ordering and (b) flags `pref_cull` — "the model places this among
your rejects even though the ruler says Mid/Strong" — which is reported, never
applied. Verdict lines are the ratings' job; this module may only speak up.

Embeddings are SigLIP-2 vectors; anchors come from LanceDB via
lance_store.query_embeddings_by_paths. Embedding vectors from a DIFFERENT
encoder tier/space are rejected (vector-space tagging) — a cosine across
spaces is meaningless.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "cache" / "preference_model.json"

FEATURE_NAMES = ["score", "keeper_aff", "reject_aff"]
MIN_GAP = 2              # stars between winner and loser to count as a pair
K_AFFINITY = 8           # neighbours per anchor set for affinity features
MIN_HOLDOUT_PAIRS = 15   # fewer than this and the gate cannot judge
MIN_MARGIN = 0.02        # model must beat the baseline by this much
L2 = 1e-3
ITERS = 400
LR = 0.5


# ── Pair construction ────────────────────────────────────────────────────────

def build_pairs(stars, min_gap: int = MIN_GAP) -> list:
    """All (winner_idx, loser_idx) pairs whose star gap is at least `min_gap`.
    A 2★ gap is a clear preference; 1★ differences are noise on a 5-point
    scale and would swamp the signal with jitter pairs."""
    stars = np.asarray(stars)
    pairs = []
    n = len(stars)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if stars[i] - stars[j] >= min_gap:
                pairs.append((i, j))
    return pairs


def pairwise_accuracy(scores, pairs) -> float:
    """Fraction of pairs the score array orders correctly (ties = 0.5)."""
    if not pairs:
        return 0.5
    s = np.asarray(scores, dtype=np.float64)
    w = s[[p[0] for p in pairs]]
    l = s[[p[1] for p in pairs]]
    return float(np.mean((w > l).astype(float) + 0.5 * (w == l)))


# ── Affinity features ────────────────────────────────────────────────────────

def _norm_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def affinities(emb, keeper_bank, reject_bank, k: int = K_AFFINITY) -> tuple:
    """(keeper_aff, reject_aff): mean cosine to the k nearest members of each
    anchor bank. Banks may be empty → 0.0 for that side (the feature is then
    uninformative but stays finite for the model)."""
    v = np.asarray(emb, dtype=np.float64).ravel()
    nv = np.linalg.norm(v) or 1.0
    v = v / nv
    out = []
    for bank in (keeper_bank, reject_bank):
        if bank is None or len(bank) == 0:
            out.append(0.0)
            continue
        b = _norm_rows(np.asarray(bank, dtype=np.float64))
        sims = b @ v
        kk = min(k, len(sims))
        out.append(float(np.mean(np.sort(sims)[-kk:])))
    return out[0], out[1]


def build_anchor_banks(records, embs_by_path: dict) -> tuple:
    """Split rated photos with known embeddings into (keeper_bank, reject_bank,
    keeper_paths, reject_paths). Keepers: 4–5★; rejects: 1–2★. 3★ anchors are
    deliberately excluded — they are the ambiguous middle the model is supposed
    to help resolve, not a reference."""
    kp, rp, kb, rb = [], [], [], []
    for r in records:
        e = embs_by_path.get(r["path"])
        if e is None:
            continue
        if int(r["stars"]) >= 4:
            kp.append(r["path"]); kb.append(np.asarray(e, dtype=np.float64))
        elif int(r["stars"]) <= 2:
            rp.append(r["path"]); rb.append(np.asarray(e, dtype=np.float64))
    kb = np.stack(kb) if kb else np.zeros((0, 1))
    rb = np.stack(rb) if rb else np.zeros((0, 1))
    return kb, rb, kp, rp


# ── Model fit ────────────────────────────────────────────────────────────────

def fit_ranknet(X, pairs, l2: float = L2, iters: int = ITERS,
                lr: float = LR) -> np.ndarray:
    """Full-batch gradient descent on the pairwise logistic loss.
    X: (n, d) features; pairs: [(winner, loser)]. Returns weight vector (d,).
    The score feature is the machine's own — the model can only learn a
    correction to it, which is exactly the shape of the problem."""
    X = np.asarray(X, dtype=np.float64)
    w = np.zeros(X.shape[1], dtype=np.float64)
    w[0] = 1.0                        # start at "trust the machine score"
    if not pairs:
        return w
    wi = np.array([p[0] for p in pairs])
    li = np.array([p[1] for p in pairs])
    for it in range(iters):
        s = X @ w
        d = s[wi] - s[li]
        # dL/dΔ = σ(-Δ); Δ = x_w·w − x_l·w → grad wrt w
        g = 1.0 / (1.0 + np.exp(np.clip(d, -30, 30)))
        gw = g[:, None] * (X[li] - X[wi])
        grad = gw.mean(axis=0) + 2.0 * l2 * w
        step = lr * (1.0 - it / (iters * 1.5))     # mild decay
        w -= step * grad
    return w


def pref_scores(X, w) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64)


# ── Holdout split ────────────────────────────────────────────────────────────

def time_split_holdout(records, frac: float = 0.25, seed: int = 17) -> tuple:
    """Indices (train, holdout). Newest `frac` of ratings by rated_at are held
    out — the honest simulation of "rate future photos and see if the model
    predicts me". Legacy entries without timestamps fall back to a seeded
    random split so they still gate, just less honestly."""
    n = len(records)
    n_hold = max(1, int(round(n * frac))) if n >= 8 else 0
    if n_hold == 0:
        return list(range(n)), []
    idx = list(range(n))
    if all(isinstance(r.get("rated_at"), (int, float)) and r["rated_at"] > 0
           for r in records):
        idx.sort(key=lambda i: float(records[i]["rated_at"]))
    else:
        import random
        random.Random(seed).shuffle(idx)
    return idx[:-n_hold], idx[-n_hold:]


# ── Train + gate ─────────────────────────────────────────────────────────────

def train_and_gate(records, feats_by_path: dict, encoder_tag: str = "",
                   persist: bool = True) -> dict:
    """Train the pairwise model and decide deployment. `feats_by_path` maps
    path → (score, keeper_aff, reject_aff). Returns the gate dict (also
    persisted to cache/preference_model.json when `persist`). An inactive gate
    is a first-class outcome with its reason recorded — never an exception."""
    recs = [r for r in records
            if r["path"] in feats_by_path
            and isinstance(r.get("score"), (int, float)) and r["score"] > 0]
    info = {"trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "encoder_tag": encoder_tag, "active": False}

    if len(recs) < 20:
        info["reason"] = f"only {len(recs)} rated photos with features — need ≥ 20"
        if persist:
            _persist(info)
        return info

    stars = [int(r["stars"]) for r in recs]
    all_pairs = build_pairs(stars)
    info["n_pairs"] = len(all_pairs)
    if len(all_pairs) < 20:
        info["reason"] = f"only {len(all_pairs)} preference pairs (≥2★ gap) — need ≥ 20"
        if persist:
            _persist(info)
        return info

    tr, ho = time_split_holdout(recs)
    tr_set, ho_set = set(tr), set(ho)
    tr_pairs = [p for p in all_pairs if p[0] in tr_set and p[1] in tr_set]
    ho_pairs = [p for p in all_pairs if p[0] in ho_set and p[1] in ho_set]
    info["n_train_pairs"] = len(tr_pairs)
    info["n_holdout_pairs"] = len(ho_pairs)

    if len(ho_pairs) < MIN_HOLDOUT_PAIRS:
        info["reason"] = (f"holdout has {len(ho_pairs)} pairs — "
                          f"need ≥ {MIN_HOLDOUT_PAIRS} to judge fairly")
        if persist:
            _persist(info)
        return info

    X = np.array([feats_by_path[r["path"]] for r in recs], dtype=np.float64)
    score_col = X[:, 0]

    baseline = pairwise_accuracy(score_col, ho_pairs)
    # Pairs index the FULL record list; X[tr] compacts rows to the train
    # subset — remap pair indices or they point past the end of X[tr]
    # (real data hit this: not every rated photo has anchor vectors).
    tr_sorted = sorted(tr)
    _row_of = {orig: new for new, orig in enumerate(tr_sorted)}
    tr_pairs_c = [(_row_of[a], _row_of[b]) for a, b in tr_pairs]
    w = fit_ranknet(X[tr_sorted], tr_pairs_c)
    model = pairwise_accuracy(pref_scores(X, w), ho_pairs)
    info["baseline_holdout_acc"] = round(baseline, 4)
    info["model_holdout_acc"] = round(model, 4)
    info["margin"] = round(model - baseline, 4)
    info["weights"] = [round(float(x), 5) for x in w]
    info["feature_names"] = list(FEATURE_NAMES)

    if not np.all(np.isfinite(w)) or model < baseline + MIN_MARGIN:
        info["reason"] = (f"model holdout acc {model:.3f} does not beat the "
                          f"machine score's {baseline:.3f} by ≥ {MIN_MARGIN} "
                          f"— the ruler alone is already as good; staying "
                          f"thresholds-only")
        if persist:
            _persist(info)
        return info

    info["active"] = True
    if persist:
        _persist(info)
    return info


def load_gate() -> dict | None:
    """The persisted gate decision, or None. A gate trained on a DIFFERENT
    encoder's vector space is returned with active=False — affinities across
    spaces are meaningless, so a tier switch must force a retrain."""
    try:
        if not MODEL_PATH.exists():
            return None
        d = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if d.get("active"):
            try:
                from lance_store import current_encoder_tag
                tag = current_encoder_tag()
                if d.get("encoder_tag") and tag and d["encoder_tag"] != tag:
                    return {**d, "active": False,
                            "reason": "encoder changed since training — retrain needed"}
            except ImportError:
                pass    # vector-space check unavailable; trust the stored gate
        return d
    except Exception as exc:
        print(f"[pref] gate load failed ({exc})")
        return None


def _persist(info: dict) -> None:
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(info, indent=1), encoding="utf-8")
        import os
        os.replace(tmp, MODEL_PATH)
    except Exception as exc:
        print(f"[pref] gate persist failed ({exc})")


# ── Taste summary (read-only, for the vision critique) ──────────────────────

def taste_summary(records=None) -> dict:
    """What the photographer's keepers and rejects actually differ on. For
    every numeric breakdown factor (Technical, Composition, Lighting,
    Narrative, …) it averages your 4–5★ photos vs your 1–2★ photos and ranks
    by the GAP — so the vision critique can say "you keep frames stronger in
    Narrative" instead of generic praise. Read-only; {} when there is nothing
    to summarize."""
    try:
        import lance_store as _ls
        import ratings_store as _rs
        recs = records if records is not None else _rs.load_records()
        stars_by_path = {r["path"]: int(r["stars"]) for r in recs}
        if not stars_by_path:
            return {}
        k_sum: dict[str, float] = {}
        k_n: dict[str, int] = {}
        r_sum: dict[str, float] = {}
        r_n: dict[str, int] = {}
        n_k = n_r = 0
        for row in _ls.query_light_all():
            p = row.get("path")
            st = stars_by_path.get(p)
            if st is None:
                continue
            bd = row.get("breakdown") or {}
            if isinstance(bd, str):
                try:
                    bd = json.loads(bd)
                except Exception:
                    bd = {}
            if not isinstance(bd, dict):
                continue
            aspects = {k: float(v) for k, v in bd.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)
                       and not k.startswith("_")}   # skip _nima/_grader flags
            if not aspects:
                continue
            if st >= 4:
                n_k += 1
                for k, v in aspects.items():
                    k_sum[k] = k_sum.get(k, 0.0) + v
                    k_n[k] = k_n.get(k, 0) + 1
            elif st <= 2:
                n_r += 1
                for k, v in aspects.items():
                    r_sum[k] = r_sum.get(k, 0.0) + v
                    r_n[k] = r_n.get(k, 0) + 1
        if n_k == 0 or n_r == 0:
            return {}
        gaps = {}
        for k in set(k_sum) | set(r_sum):
            km = k_sum.get(k, 0.0) / k_n[k] if k_n.get(k) else 0.0
            rm = r_sum.get(k, 0.0) / r_n[k] if r_n.get(k) else 0.0
            gaps[k] = round(km - rm, 4)   # + = keepers score higher
        keeper_leans = sorted(
            [(k, g) for k, g in gaps.items() if g > 0.0],
            key=lambda kv: -kv[1])[:3]
        reject_leans = sorted(
            [(k, g) for k, g in gaps.items() if g < 0.0],
            key=lambda kv: kv[1])[:3]
        return {
            "keeper_leans": keeper_leans,
            "reject_leans": reject_leans,
            "n_keepers": n_k,
            "n_rejects": n_r,
        }
    except Exception as exc:
        print(f"[pref] taste_summary failed ({exc})")
        return {}



# ── Gallery orchestration (the one call the pipeline makes) ─────────────────

def apply_to_gallery(gallery, records, embs_by_path: dict,
                     encoder_tag: str = "", train: bool = True,
                     weak_label: str = "Weak") -> dict:
    """One entry point for a grade run. Builds affinity features, trains +
    gates the pairwise model (Stage 3), and — ONLY if the gate passes —
    attaches `pref_score` to every gallery photo with an embedding and sets
    `pref_cull` on photos the model places among your rejects while the ruler
    still calls them Mid/Strong. Verdicts are never modified here.

    embs_by_path: path → SigLIP-2 vector, for rated anchors AND gallery photos.
    Returns an info dict for logging / the response payload."""
    info: dict = {"stage": "preference", "trained": bool(train), "active": False}
    try:
        kb, rb, kp, rp = build_anchor_banks(records, embs_by_path)
        if len(kb) == 0 or len(rb) == 0:
            info["reason"] = (f"anchor banks empty (keepers {len(kb)} / "
                              f"rejects {len(rb)}) — nothing to learn from")
            return info

        # Training features: a rated photo must not measure similarity to
        # itself, or its own embedding dominates its k-NN affinity.
        kp_arr, rp_arr = np.array(kp, dtype=object), np.array(rp, dtype=object)
        kset, rset = set(kp), set(rp)
        feats: dict = {}
        for r in records:
            p = r["path"]
            e = embs_by_path.get(p)
            if e is None or not isinstance(r.get("score"), (int, float)):
                continue
            kxb = kb[kp_arr != p] if p in kset else kb
            rxb = rb[rp_arr != p] if p in rset else rb
            ka, ra = affinities(e, kxb, rxb)
            feats[p] = (float(r["score"]), ka, ra)

        gate = (train_and_gate(records, feats, encoder_tag)
                if train else (load_gate() or {}))
        info["gate"] = {k: v for k, v in gate.items() if k != "weights"}
        info["active"] = bool(gate.get("active"))
        info["reason"] = gate.get("reason", "")
        if not info["active"]:
            return info

        w = np.asarray(gate["weights"], dtype=np.float64)
        kset, rset = set(kp), set(rp)
        for g in gallery:
            e = embs_by_path.get(g["path"])
            if e is None:
                continue
            ka, ra = affinities(e, kb, rb)
            g["pref_score"] = round(float(np.dot(w, [float(g.get("score") or 0.0),
                                                     ka, ra])), 4)

        # Cull suggestion line: below the 80th percentile of YOUR rated
        # rejects' pref scores (i.e. more reject-like than 4 of your 5
        # typical rejects) while the verdict is not already Weak.
        rej_ps = []
        for p in rset:
            e = embs_by_path.get(p)
            if e is None:
                continue
            ka, ra = affinities(e, kb, rb)
            rej_ps.append(float(np.dot(w, [feats[p][0] if p in feats else 0.0,
                                           ka, ra])))
        if len(rej_ps) >= 10:
            thr = float(np.percentile(rej_ps, 80))
            n_flag = 0
            for g in gallery:
                if "pref_score" not in g or weak_label in str(g.get("grade", "")):
                    continue
                if g["pref_score"] <= thr:
                    g["pref_cull"] = True
                    n_flag += 1
            info["cull_threshold"] = round(thr, 4)
            info["n_cull_suggested"] = n_flag
        return info
    except Exception as exc:
        import traceback
        info["reason"] = f"preference stage failed: {exc}"
        print(f"[pref] {info['reason']}")
        traceback.print_exc()
        return info
