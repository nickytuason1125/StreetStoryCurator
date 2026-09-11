"""
MasterJudge tests — the learned head may only ever help, never silently hurt.

The champion/challenger contract: weights promote ONLY when their held-out
Spearman against the photographer's stars beats the incumbent score's on the
same photos; unpromoted weights must be invisible at grade time; and the
human-anchored ruler must refuse to write a scale when the star data cannot
support one.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_master_judge.py -v
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import master_judge as mj  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _bd(tech: float, comp: float, light: float, narr: float, hc: float,
        arch: "dict | None" = None) -> dict:
    """A complete grade-time breakdown — every aspect mj.FEATURES currently
    lists, plus the archetype weights the hand formula uses (there is no
    'Aesthetic' key at grade time).

    The five named parameters are the original aspects and their meaning is
    fixed; any name later appended to mj.FEATURES (e.g. "AADB", "Exemplar")
    is filled in with a neutral default here so this fixture never goes
    stale again as FEATURES grows."""
    named = {"Technical": tech, "Composition": comp, "Lighting": light,
             "Narrative": narr, "Human/Culture": hc}
    bd = {name: named.get(name, 0.5) for name in mj.FEATURES}
    bd["_arch_w"] = arch or {"geo": 0.2, "night": 0.2, "layer": 0.2,
                              "messy": 0.2, "maxdoc": 0.2}
    return bd


def _synthetic_rows(n: int = 320, seed: int = 7) -> list:
    """Master baseline where the aspects genuinely carry the signal and the
    incumbent score only weakly does — the situation the judge exists for."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # stars driven by two aspects + noise; other aspects are noise
        latent = rng.normal(0, 1)
        stars = int(np.clip(round(3.0 + 1.1 * latent + rng.normal(0, 0.4)), 1, 5))
        tech = 0.4 + 0.10 * latent + rng.normal(0, 0.04)
        comp = 0.4 + 0.10 * latent + rng.normal(0, 0.04)
        light = 0.45 + rng.normal(0, 0.08)
        narr = 0.50 + rng.normal(0, 0.08)
        hc = 0.45 + rng.normal(0, 0.08)
        # archetype weights: mostly flat, mild latent signal in geo/night
        arch = {"geo": 0.2 + 0.02 * latent, "night": 0.2 + 0.01 * latent,
                "layer": 0.2, "messy": 0.2, "maxdoc": 0.2}
        # incumbent score: weakly correlated with truth (rho ~ 0.3)
        score = float(np.clip(0.5 + 0.08 * latent + rng.normal(0, 0.15), 0.05, 0.95))
        rows.append({"path": f"P{i}.jpg", "stars": stars, "source": "tpe_master",
                     "score": score, "grade": "", "from_snapshot": False,
                     "breakdown": _bd(tech, comp, light, narr, hc, arch)})
    return rows


# ── 1. the correlation itself ─────────────────────────────────────────────────

def test_spearman_matches_scipy_on_random_data():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    ours = mj.spearman(a, b)
    ref = float(spearmanr(a, b).statistic)
    assert abs(ours - ref) < 1e-10


def test_spearman_is_nan_without_signal():
    assert math.isnan(mj.spearman([0.5] * 10, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]))
    assert math.isnan(mj.spearman([1, 2, 3], [1, 2, 3, 4]))


# ── 2. features ───────────────────────────────────────────────────────────────

def test_feature_vector_preserves_canonical_order_and_flags_missing():
    v = mj.feature_vector(_bd(0.1, 0.2, 0.3, 0.4, 0.5))
    # only the first 5 (the original, explicitly-passed aspects) are pinned —
    # any later FEATURES growth (e.g. "AADB", "Exemplar") fills in beyond
    # that with _bd's neutral default, which this test doesn't pin a value
    # for.
    assert list(v[:5]) == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert np.isfinite(v).all()            # archetype weights came along
    assert len(v) == len(mj.DESIGN) - 1    # minus the machine-score column
    partial = mj.feature_vector({"Technical": 0.1})
    assert partial[0] == 0.1
    assert np.isnan(partial[1:]).all()     # everything missing is NaN, not 0


# ── 3. the stratified holdout ────────────────────────────────────────────────

def test_stratified_split_represents_every_bucket():
    y = np.array([1] * 10 + [2] * 40 + [3] * 60 + [4] * 30 + [5] * 20)
    tr, ho = mj._stratified_split(y, 0.25, seed=11)
    assert len(tr) + len(ho) == len(y)
    assert not set(tr) & set(ho)
    for val in (1, 2, 3, 4, 5):
        assert (y[ho] == val).sum() >= 1, f"star {val} missing from holdout"
    # a lone member of a bucket must stay in train
    y2 = np.array([5] * 30 + [1])
    tr2, ho2 = mj._stratified_split(y2, 0.25, seed=5)
    assert y2[tr2].tolist().count(1) == 1
    assert (y2[ho2] == 1).sum() == 0

# ── 4. champion/challenger fit ────────────────────────────────────────────────

def test_fit_promotes_when_judge_beats_baseline(tmp_path):
    rows = _synthetic_rows()
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "w.json")
    assert out["n"] == 320
    assert out["promoted"] is True
    assert out["rho_holdout"] > out["rho_baseline"]
    # weights on disk are loadable and carry the fingerprint
    saved = json.loads((tmp_path / "w.json").read_text(encoding="utf-8"))
    assert saved["promoted"] is True
    assert saved["feature_fingerprint"] == mj._feature_fingerprint()


def test_fit_refuses_to_promote_a_loser(tmp_path):
    rows = _synthetic_rows()
    # corrupt EVERY feature (aspects AND archetype weights) so the judge
    # learns pure noise while the incumbent keeps its weak-but-real signal —
    # the challenger must lose.
    rng = np.random.default_rng(99)
    for r in rows:
        for k in mj.FEATURES:
            r["breakdown"][k] = float(rng.normal(0.5, 0.25))
        r["breakdown"]["_arch_w"] = {a: float(rng.uniform(0.05, 0.45))
                                     for a in mj.ARCHES}
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "w.json")
    assert out["promoted"] is False
    assert "challenger lost" in out["reason"]
    # and an unpromoted file must be invisible at grade time
    assert mj.load(tmp_path / "w.json") is None
    assert mj.blend_weight(tmp_path / "w.json") == 0.0


def test_fit_refuses_tiny_baselines(tmp_path):
    out = mj.fit_from_rows(_synthetic_rows(n=30),
                           weights_path=tmp_path / "w.json")
    assert out["promoted"] is False
    assert ">= 60" in out["reason"]


def test_predict_many_scores_complete_rows_and_skips_incomplete(tmp_path):
    rows = _synthetic_rows()
    wp = tmp_path / "w.json"
    mj.fit_from_rows(rows, weights_path=wp)
    preds = mj.predict_many([r["breakdown"] for r in rows[:50]],
                            [r["score"] for r in rows[:50]], weights_path=wp)
    assert np.isfinite(preds).all()
    assert ((preds >= 0.0) & (preds <= 1.0)).all()
    mixed = mj.predict_many([rows[0]["breakdown"], {}, None],
                            [rows[0]["score"], 0.5, 0.5], weights_path=wp)
    assert np.isfinite(mixed[0])
    assert np.isnan(mixed[1]) and np.isnan(mixed[2])
    # a stacked judge CANNOT vote without the incumbent's base score
    no_base = mj.predict_many([rows[0]["breakdown"]], None, weights_path=wp)
    assert np.isnan(no_base[0])
    # monotone in the aspects that drive the synthetic truth
    strong = mj.predict_many([_bd(0.9, 0.9, 0.5, 0.5, 0.5)], [0.6], weights_path=wp)[0]
    weak = mj.predict_many([_bd(0.1, 0.1, 0.5, 0.5, 0.5)], [0.6], weights_path=wp)[0]
    assert strong > weak


def test_predict_many_rejects_misaligned_base_scores(tmp_path):
    rows = _synthetic_rows()
    wp = tmp_path / "w.json"
    mj.fit_from_rows(rows, weights_path=wp)
    with pytest.raises(ValueError, match="align"):
        mj.predict_many([rows[0]["breakdown"]], [0.5, 0.6], weights_path=wp)


def test_no_weights_means_no_vote(tmp_path):
    assert mj.load(tmp_path / "absent.json") is None
    assert mj.blend_weight(tmp_path / "absent.json") == 0.0
    assert np.isnan(mj.predict_many([_bd(1, 1, 1, 1, 1)], [0.5],
                                    weights_path=tmp_path / "absent.json")[0])


def test_stale_fingerprint_is_treated_as_absent(tmp_path):
    wp = tmp_path / "w.json"
    mj.fit_from_rows(_synthetic_rows(), weights_path=wp)
    d = json.loads(wp.read_text(encoding="utf-8"))
    d["feature_fingerprint"] = "deadbeefdeadbeef"
    wp.write_text(json.dumps(d), encoding="utf-8")
    assert mj.load(wp) is None

# ── 5. the human-anchored ruler ───────────────────────────────────────────────

def test_human_anchors_happy_path():
    rng = np.random.default_rng(1)
    disc = {1: list(rng.normal(-0.06, 0.008, 60)),
            2: list(rng.normal(-0.04, 0.008, 60)),
            3: list(rng.normal(-0.01, 0.008, 60)),
            4: list(rng.normal(0.02, 0.008, 60)),
            5: list(rng.normal(0.04, 0.008, 60))}
    lo, hi, info = mj.human_anchor_lo_hi(disc)
    assert lo < hi
    # anchored ends hug the human groups, not the corpus extremes
    assert lo < np.median(disc[3]) < hi
    assert info["n_high"] == 120 and info["n_low"] == 120


def test_human_anchors_refuse_small_groups():
    with pytest.raises(ValueError, match="too small"):
        mj.human_anchor_lo_hi({1: [-0.1] * 3, 2: [-0.1] * 3,
                               4: [0.1] * 3, 5: [0.1] * 3, 3: [0.0] * 10})


def test_human_anchors_refuse_non_monotone_stars():
    """If the mids don't land between the hits and the rejects, the
    discriminant is not measuring what the stars measure — no scale."""
    rng = np.random.default_rng(2)
    disc = {1: list(rng.normal(-0.06, 0.008, 40)),
            2: list(rng.normal(-0.04, 0.008, 40)),
            3: list(rng.normal(0.05, 0.008, 40)),   # mids ABOVE the highs
            4: list(rng.normal(0.02, 0.008, 40)),
            5: list(rng.normal(0.04, 0.008, 40))}
    with pytest.raises(ValueError, match="monotone"):
        mj.human_anchor_lo_hi(disc)


# ── 6. the continuous loop ────────────────────────────────────────────────────

def test_fit_records_history_across_refits(tmp_path):
    wp = tmp_path / "w.json"
    mj.fit_from_rows(_synthetic_rows(), weights_path=wp)
    first = json.loads(wp.read_text(encoding="utf-8"))
    assert len(first["history"]) == 1
    mj.fit_from_rows(_synthetic_rows(seed=8), weights_path=wp)
    second = json.loads(wp.read_text(encoding="utf-8"))
    assert len(second["history"]) == 2
    entry = second["history"][-1]
    assert {"generated", "n", "rho_holdout", "rho_baseline",
            "promoted"} <= set(entry)


def test_maybe_autofit_gates_on_new_rating_delta(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE", "1")
    wp = tmp_path / "w.json"
    wp.write_text(json.dumps({"n": 100, "promoted": False}), encoding="utf-8")
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", wp)
    called = {}
    monkeypatch.setattr(mj, "fit", lambda: called.setdefault("ran", True))
    # delta 10 < threshold 25 → no fit
    below = mj.maybe_autofit(n_now=110)
    assert below["triggered"] is False
    # delta 30 >= threshold → daemon fit fires
    above = mj.maybe_autofit(n_now=130)
    assert above["triggered"] is True
    t = mj._autofit_thread
    if t is not None:
        t.join(timeout=10)
    assert called.get("ran") is True


# ── 7. the two-phase master-algo contract ─────────────────────────────────────

def test_promotion_refuses_a_loser(tmp_path):
    """A challenger that lost its exam can never become the master."""
    rows = _synthetic_rows()
    rng = np.random.default_rng(99)
    for r in rows:
        for k in mj.FEATURES:
            r["breakdown"][k] = float(rng.normal(0.5, 0.25))
        r["breakdown"]["_arch_w"] = {a: float(rng.uniform(0.05, 0.45))
                                     for a in mj.ARCHES}
    wp = tmp_path / "w.json"
    mj.fit_from_rows(rows, weights_path=wp)          # loses → promoted False
    res = mj.promote_to_shipped(cache_path=wp, shipped_path=tmp_path / "s.json")
    assert res["shipped"] is False
    assert not (tmp_path / "s.json").exists()


def test_won_exam_ships_and_grades_without_opt_in(tmp_path, monkeypatch):
    """The full two-phase journey: a local challenger wins its exam, the
    operator bakes it, and from then on EVERY install grades with it —
    no flags, no ratings."""
    wp = tmp_path / "w.json"
    out = mj.fit_from_rows(_synthetic_rows(), weights_path=wp)
    assert out["promoted"] is True

    shipped = tmp_path / "master_judge_defaults.json"
    res = mj.promote_to_shipped(cache_path=wp, shipped_path=shipped)
    assert res["shipped"] is True
    assert res["rho_holdout"] > res["rho_baseline"]

    # fresh install: no cache record, no opt-in flags — shipped still grades
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(mj, "_SHIPPED_PATH", shipped)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE_OFF", raising=False)
    judge, w = mj.active()
    assert judge is not None
    assert judge["_source"] == "shipped"
    assert w > 0.0
    preds = mj.predict_many([_bd(0.9, 0.9, 0.5, 0.5, 0.5)], [0.6], weights=judge)
    assert np.isfinite(preds[0])


def test_cache_opt_in_beats_shipped_fresher_locally(tmp_path, monkeypatch):
    """A locally-promoted challenger (newer training data) outranks the
    shipped master — but only when the user explicitly opted in."""
    shipped = tmp_path / "master_judge_defaults.json"
    cache = tmp_path / "w.json"
    mj.fit_from_rows(_synthetic_rows(seed=11), weights_path=cache)   # promoted
    mj.promote_to_shipped(cache_path=cache, shipped_path=shipped)
    mj.fit_from_rows(_synthetic_rows(seed=12), weights_path=cache)   # newer local
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", cache)
    monkeypatch.setattr(mj, "_SHIPPED_PATH", shipped)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE_OFF", raising=False)

    # without opt-in → the SHIPPED master grades (local ratings never do)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    judge, _ = mj.active()
    assert judge["_source"] == "shipped"

    # with opt-in → the locally-promoted challenger takes over
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE", "1")
    judge, _ = mj.active()
    assert judge["_source"] == "cache"


def test_stale_shipped_judge_is_ignored(tmp_path, monkeypatch):
    wp = tmp_path / "w.json"
    mj.fit_from_rows(_synthetic_rows(), weights_path=wp)
    shipped = tmp_path / "master_judge_defaults.json"
    mj.promote_to_shipped(cache_path=wp, shipped_path=shipped)
    d = json.loads(shipped.read_text(encoding="utf-8"))
    d["feature_fingerprint"] = "stalestalestale000"
    shipped.write_text(json.dumps(d), encoding="utf-8")
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(mj, "_SHIPPED_PATH", shipped)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE_OFF", raising=False)
    judge, w = mj.active()
    assert judge is None and w == 0.0


def test_kill_switch_beats_everything(tmp_path, monkeypatch):
    shipped = tmp_path / "master_judge_defaults.json"
    cache = tmp_path / "w.json"
    mj.fit_from_rows(_synthetic_rows(), weights_path=cache)
    mj.promote_to_shipped(cache_path=cache, shipped_path=shipped)
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", cache)
    monkeypatch.setattr(mj, "_SHIPPED_PATH", shipped)
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE", "1")
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE_OFF", "1")
    judge, w = mj.active()
    assert judge is None and w == 0.0


def test_maybe_autofit_requires_opt_in(monkeypatch):
    """Ratings are placeholder data: no background refit unless the judge is
    explicitly enabled with FIRSTCUT_MASTER_JUDGE=1."""
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    out = mj.maybe_autofit(n_now=9999)
    assert out["triggered"] is False
    assert "opted in" in out["reason"]


def test_maybe_autofit_respects_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE", "1")
    wp = tmp_path / "w.json"
    wp.write_text(json.dumps({"n": 100, "promoted": False}), encoding="utf-8")
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", wp)
    called = {}
    monkeypatch.setattr(mj, "fit", lambda: called.setdefault("ran", True))
    out = mj.maybe_autofit(n_now=130)   # delta 30 >= 25 → fires
    assert out["triggered"] is True
    t = mj._autofit_thread
    if t is not None:
        t.join(timeout=10)
    assert called.get("ran") is True


def test_human_anchors_refuse_degenerate_span():
    disc = {1: [0.01] * 20, 2: [0.01] * 20, 3: [0.01] * 20,
            4: [0.01] * 20, 5: [0.01] * 20}
    with pytest.raises(ValueError, match="degenerate"):
        mj.human_anchor_lo_hi(disc)
