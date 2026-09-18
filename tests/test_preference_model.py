import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import preference_model as pm


# ── Pair construction ────────────────────────────────────────────────────────

def test_build_pairs_gap_two():
    # stars: 5,4,3,2,1 → pairs only where the gap is ≥ 2
    pairs = pm.build_pairs([5, 4, 3, 2, 1])
    assert (0, 2) in pairs and (0, 3) in pairs and (0, 4) in pairs
    assert (1, 3) in pairs and (1, 4) in pairs and (2, 4) in pairs
    assert (0, 1) not in pairs      # 1★ gap = noise, excluded
    assert (3, 4) not in pairs
    assert pm.build_pairs([4, 4, 4]) == []


def test_pairwise_accuracy_ties_half_credit():
    assert pm.pairwise_accuracy([3.0, 1.0], [(0, 1)]) == 1.0
    assert pm.pairwise_accuracy([1.0, 3.0], [(0, 1)]) == 0.0
    assert pm.pairwise_accuracy([2.0, 2.0], [(0, 1)]) == 0.5
    assert pm.pairwise_accuracy([1.0], []) == 0.5


# ── Affinity features ────────────────────────────────────────────────────────

def test_affinities_nearest_neighbour_mean():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 8))
    keeper_bank = np.vstack([base + 0.01 * rng.normal(size=(1, 8)),
                             base + 5.0])          # far outlier not in top-k
    reject_bank = -base + 0.01 * rng.normal(size=(1, 8))
    ka, ra = pm.affinities(base.ravel(), keeper_bank, reject_bank, k=1)
    assert ka > 0.99 and ra < 0.0
    ka0, ra0 = pm.affinities(base.ravel(), np.zeros((0, 1)), np.zeros((0, 1)))
    assert ka0 == 0.0 and ra0 == 0.0


def test_build_anchor_banks_excludes_3star():
    embs = {"a": np.ones(4), "b": np.ones(4) * 2, "c": np.ones(4) * 3}
    recs = [{"path": "a", "stars": 5, "score": 0.7},
            {"path": "b", "stars": 3, "score": 0.5},
            {"path": "c", "stars": 1, "score": 0.3}]
    kb, rb, kp, rp = pm.build_anchor_banks(recs, embs)
    assert kp == ["a"] and rp == ["c"] and len(kb) == 1 and len(rb) == 1


# ── RankNet fit ──────────────────────────────────────────────────────────────

def test_fit_ranknet_learns_the_informative_feature():
    rng = np.random.default_rng(1)
    n = 120
    # feature 1 (embedding affinity) carries the true ordering; feature 0
    # (machine score) is pure noise — the model must discover that.
    X = rng.normal(size=(n, 3))
    X[:, 1] = rng.normal(size=n)
    true = X[:, 1]
    stars = np.digitize(true, [-1, 0, 1]) + 1      # 1..4 star-ish
    pairs = pm.build_pairs(list(stars))
    w = pm.fit_ranknet(X, pairs)
    acc = pm.pairwise_accuracy(pm.pref_scores(X, w), pairs)
    assert acc > 0.9
    assert abs(w[1]) > abs(w[0])       # noise feature suppressed


# ── Stage 3: the holdout gate ────────────────────────────────────────────────

def _synthetic_world(signal_in_affinity: bool, n: int = 160):
    """Rated-photo records + features. When signal_in_affinity, the embedding
    affinity carries ordering info the machine score lacks — the gate should
    deploy the model. Otherwise the machine score is already perfect and the
    gate must refuse (the honest 'thresholds-only' outcome)."""
    rng = np.random.default_rng(2)
    recs, feats = [], {}
    for i in range(n):
        s = float(rng.uniform(0.05, 0.95))
        aff = rng.normal()
        truth = aff if signal_in_affinity else s
        st = 5 if truth > 0.6 else 4 if truth > 0.4 else 2 if truth > 0.2 else 1
        recs.append({"path": f"P{i}", "stars": st, "score": s,
                     "rated_at": 1_700_000_000 + i * 3600})
        feats[f"P{i}"] = (s, aff, -aff)
    return recs, feats


def test_gate_deploys_when_affinity_beats_score(tmp_path):
    recs, feats = _synthetic_world(signal_in_affinity=True)
    pm.MODEL_PATH = tmp_path / "gate.json"
    # encoder_tag "" (bypasses load_gate's vector-space check — in production
    # the pipeline always passes the real encoder tag)
    info = pm.train_and_gate(recs, feats, encoder_tag="", persist=True)
    assert info["active"] is True
    assert info["margin"] >= pm.MIN_MARGIN
    assert (tmp_path / "gate.json").exists()
    assert pm.load_gate()["active"] is True


def test_gate_refuses_when_score_already_optimal(tmp_path):
    recs, feats = _synthetic_world(signal_in_affinity=False)
    pm.MODEL_PATH = tmp_path / "gate.json"
    info = pm.train_and_gate(recs, feats, encoder_tag="", persist=True)
    assert info["active"] is False
    assert "does not beat" in info["reason"]


def test_gate_refuses_on_thin_holdout(tmp_path):
    recs, feats = _synthetic_world(signal_in_affinity=True, n=24)
    pm.MODEL_PATH = tmp_path / "gate.json"
    info = pm.train_and_gate(recs, feats, encoder_tag="t", persist=True)
    assert info["active"] is False
    assert "holdout" in info["reason"] or "pairs" in info["reason"]


def test_gate_survives_partial_features(tmp_path):
    """Real data (2026-09-15): only some rated photos have anchor vectors —
    pair indices must be remapped onto the compacted train matrix."""
    recs, feats = _synthetic_world(signal_in_affinity=True)
    # drop features for every 3rd record — features are sparse in practice
    sparse = {p: f for i, (p, f) in enumerate(feats.items()) if i % 3}
    pm.MODEL_PATH = tmp_path / "gate.json"
    info = pm.train_and_gate(recs, sparse, encoder_tag="", persist=False)
    # must not raise; either deploys or refuses with an honest reason
    assert isinstance(info.get("active"), bool)
    if info["active"]:
        assert info["margin"] >= pm.MIN_MARGIN


def test_time_split_prefers_newest():
    # epoch timestamps (never 0 — a 0/negative rated_at is treated as junk
    # and the split honestly falls back to random)
    recs = [{"path": str(i), "stars": 3, "score": 0.5,
             "rated_at": 1_700_000_000 + i} for i in range(40)]
    tr, ho = pm.time_split_holdout(recs, frac=0.25)
    assert min(ho) > max(tr)      # holdout is the newest quarter
    assert not set(tr) & set(ho) and len(tr) + len(ho) == 40


# ── Gallery orchestration ────────────────────────────────────────────────────

def test_apply_to_gallery_never_touches_verdicts(tmp_path):
    pm.MODEL_PATH = tmp_path / "gate.json"
    recs, feats = _synthetic_world(signal_in_affinity=True)
    embs = {r["path"]: np.random.default_rng(5).normal(size=(16,)) for r in recs}
    gallery = [{"path": "G0", "grade": "Strong ✅", "score": 0.8}]
    pm.apply_to_gallery(gallery, recs, embs, encoder_tag="t")
    assert gallery[0]["grade"] == "Strong ✅"      # product rule intact


def test_apply_to_gallery_attaches_only_when_active(tmp_path, monkeypatch):
    pm.MODEL_PATH = tmp_path / "gate.json"
    # Force-inactive gate: nothing may be attached to the gallery
    monkeypatch.setattr(pm, "train_and_gate",
                        lambda *a, **k: {"active": False, "reason": "x"})
    recs, feats = _synthetic_world(signal_in_affinity=True)
    embs = {r["path"]: np.random.default_rng(6).normal(size=(16,)) for r in recs}
    gallery = [{"path": f"G{i}", "grade": "Mid ⚠️", "score": 0.5}
               for i in range(4)]
    info = pm.apply_to_gallery(gallery, recs, embs, encoder_tag="t")
    assert info["active"] is False
    assert all("pref_score" not in g and "pref_cull" not in g for g in gallery)


def test_apply_to_gallery_active_path_scores_gallery(tmp_path, monkeypatch):
    pm.MODEL_PATH = tmp_path / "gate.json"
    recs, feats = _synthetic_world(signal_in_affinity=True)
    # Force the gate active with known weights so the scoring path runs
    monkeypatch.setattr(pm, "train_and_gate",
                        lambda *a, **k: {"active": True, "weights": [0.5, 2.0, -2.0]})
    rng = np.random.default_rng(7)
    embs = {r["path"]: rng.normal(size=(16,)) for r in recs}
    embs.update({f"G{i}": rng.normal(size=(16,)) for i in range(6)})
    gallery = [{"path": f"G{i}", "grade": "Strong ✅", "score": 0.7}
               for i in range(6)]
    info = pm.apply_to_gallery(gallery, recs, embs, encoder_tag="t")
    assert info["active"] is True
    assert all("pref_score" in g for g in gallery)

