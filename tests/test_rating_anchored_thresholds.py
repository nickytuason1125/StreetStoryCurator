import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rating_calibration import thresholds_from_ratings


def _rated(n_lo, n_hi, lo_score=0.35, hi_score=0.65):
    """stars_by_path + aligned paths/scores: n_lo photos rated 1–2★ scoring
    just under lo_score, n_hi photos rated 4–5★ scoring just over hi_score.
    Jittered so no photo sits exactly on a cut-point (a tie would be bucketed
    Mid by design, which is correct — the ruler measures distributions, not
    ties)."""
    stars, paths, scores = {}, [], []
    for i in range(n_lo):
        p = f"C:/x/lo{i}.jpg"; stars[p] = 1 + (i % 2); paths.append(p); scores.append(round(lo_score - i * 0.004, 4))
    for i in range(n_hi):
        p = f"C:/x/hi{i}.jpg"; stars[p] = 4 + (i % 2); paths.append(p); scores.append(round(hi_score + i * 0.004, 4))
    return stars, paths, scores


def test_keepers_land_strong_and_rejects_land_weak():
    stars, paths, scores = _rated(12, 12)
    strong, mid, info = thresholds_from_ratings(stars, paths, scores)
    # ≈90% of keepers must clear strong; ≈90% of rejects must sit below mid
    keeper_scores = [s for p, s in zip(paths, scores) if stars[p] >= 4]
    reject_scores = [s for p, s in zip(paths, scores) if stars[p] <= 2]
    assert sum(s >= strong for s in keeper_scores) >= 10
    assert sum(s < mid for s in reject_scores) >= 10
    assert info.get("reason") is None if "reason" in info else True


def test_untouched_3star_mids_stay_mid():
    stars, paths, scores = _rated(12, 12)
    paths += [f"C:/x/mid{i}.jpg" for i in range(5)]
    scores += [0.50] * 5
    stars.update({f"C:/x/mid{i}.jpg": 3 for i in range(5)})
    strong, mid, _ = thresholds_from_ratings(stars, paths, scores)
    assert mid <= 0.50 <= strong  # mids keep their band


def test_too_few_ratings_keeps_defaults():
    stars, paths, scores = _rated(3, 4)
    strong, mid, info = thresholds_from_ratings(stars, paths, scores)
    assert (strong, mid) == (0.60, 0.41)
    assert "too small" in info["reason"]


def test_non_monotone_stars_refuse_to_calibrate():
    # the photographer's rejects score HIGHER than their keepers — the stars
    # don't order the machine's scores, so no ruler may be manufactured
    stars, paths, scores = _rated(12, 12, lo_score=0.70, hi_score=0.30)
    strong, mid, info = thresholds_from_ratings(stars, paths, scores)
    assert (strong, mid) == (0.60, 0.41)
    assert "not order" in info["reason"]


def test_pathological_batch_cannot_slide_the_ruler():
    # every keeper scores 1.0 and every reject 0.0: the anchors want 1.0/0.0,
    # but the shift caps keep the thresholds near the shipped values
    stars, paths, scores = _rated(12, 12, lo_score=0.05, hi_score=0.95)
    strong, mid, _ = thresholds_from_ratings(stars, paths, scores)
    assert 0.60 - 0.12 <= strong <= 0.60 + 0.12
    assert 0.41 - 0.12 <= mid <= 0.41 + 0.12
    assert mid < strong


def test_no_ratings_at_all_keeps_defaults():
    strong, mid, info = thresholds_from_ratings({}, ["C:/x/a.jpg"], [0.5])
    assert (strong, mid) == (0.60, 0.41)


# ── Global record-based fit (Stage 1 upgrade) ────────────────────────────────

import time as _time

from rating_calibration import thresholds_from_records


def _records(n_lo, n_hi, lo_score=0.35, hi_score=0.65, prefix="LX3", age_days=None):
    recs = []
    for i in range(n_lo):
        r = {"path": f"E:/{prefix}/lo{i}.jpg", "stars": 1 + (i % 2), "score": round(lo_score - i * 0.004, 4)}
        recs.append(r)
    for i in range(n_hi):
        r = {"path": f"E:/{prefix}/hi{i}.jpg", "stars": 4 + (i % 2), "score": round(hi_score + i * 0.004, 4)}
        recs.append(r)
    if age_days is not None:
        for r in recs:
            r["rated_at"] = _time.time() - age_days * 86400
    return recs


def test_records_cross_folder_snapshots_calibrate():
    # 278 LX3-style rejects + 53 keepers from OTHER folders: a folder with zero
    # rated photos of its own still gets a ruler fitted from the global table
    recs = _records(60, 20)
    strong, mid, info = thresholds_from_records(recs)
    assert 0.55 <= strong <= 0.72
    assert 0.29 <= mid <= 0.53
    assert mid < strong
    assert "reason" not in info


def test_records_shrinkage_pulls_thin_data_toward_defaults():
    # far fewer anchors than the old hard minimum: shrunk toward 0.60/0.41
    recs = _records(10, 10)
    strong, mid, info = thresholds_from_records(recs)
    # fitted edges would be ~0.65/0.35; with n=10 vs k=10 the result sits
    # halfway to the defaults
    assert abs(strong - 0.625) < 0.03
    assert abs(mid - 0.38) < 0.03


def test_records_recency_weighting_reduces_stale_votes():
    # a year-old batch pulling the ruler up vs the same batch fresh
    fresh = _records(30, 30, lo_score=0.35, hi_score=0.65, age_days=0)
    stale = _records(30, 30, lo_score=0.35, hi_score=0.65, age_days=3 * 365)
    strong_fresh, _, _ = thresholds_from_records(fresh)
    strong_stale, _, _ = thresholds_from_records(stale)
    # stale keepers' pull toward their edge is weaker → line sits closer to 0.60
    assert strong_stale < strong_fresh
    assert strong_fresh > 0.60  # fresh data still moves it up


def test_records_live_score_overrides_snapshot():
    recs = _records(60, 20, lo_score=0.35, hi_score=0.65)
    # the current run re-scored every keeper at 0.30 (a harsher scorer): the
    # fit must follow the LIVE scores, not the stale snapshots
    for i, r in enumerate(recs):
        if r["stars"] >= 4:
            r["score"] = round(0.30 - 0.001 * (i % 5), 4)
    strong, mid, info = thresholds_from_records(recs)
    assert info["keeper_edge"] < 0.35  # live scores used
    assert strong < 0.60               # ruler moved down with the live data


def test_records_rejects_thin_falls_back_to_defaults():
    recs = _records(2, 30)
    strong, mid, info = thresholds_from_records(recs)
    assert (strong, mid) == (0.60, 0.41)
    assert "too small" in info["reason"]

