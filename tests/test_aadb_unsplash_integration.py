# tests/test_aadb_unsplash_integration.py
"""Confirms the full chain — grading step -> breakdown key ->
master_judge.feature_vector -> ridge design matrix — behaves correctly both
with and without the new model artifacts present. This is the state of a
fresh clone (no models/aadb_head.npz, no models/exemplar_bank.npz) and must
grade exactly as it did before this plan, per the Global Constraints."""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_fresh_clone_state_degrades_to_prior_behavior(monkeypatch):
    import grade_pipeline_v2 as gp
    import master_judge as mj

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [
        {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5, "Narrative": 0.5, "Human/Culture": 0.5},
        {"Technical": 0.6, "Composition": 0.6, "Lighting": 0.6, "Narrative": 0.6, "Human/Culture": 0.6},
    ]

    monkeypatch.setattr("aadb_scorer.score", lambda e: None)      # fresh-clone state
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: None)  # fresh-clone state

    gp._apply_aadb_step(embs, [0, 1], breakdowns)
    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert "Exemplar" not in breakdowns[0]

    vec = mj.feature_vector(breakdowns[0])
    assert np.isnan(vec[mj.FEATURES.index("AADB")])
    assert np.isnan(vec[mj.FEATURES.index("Exemplar")])
    # the five original features are untouched
    for name in ["Technical", "Composition", "Lighting", "Narrative", "Human/Culture"]:
        assert not np.isnan(vec[mj.FEATURES.index(name)])


def test_both_models_present_populate_both_features(monkeypatch):
    import grade_pipeline_v2 as gp
    import master_judge as mj

    embs = np.array([[1.0, 0.0]], dtype=np.float32)
    breakdowns = [{"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
                   "Narrative": 0.5, "Human/Culture": 0.5}]

    monkeypatch.setattr("aadb_scorer.score", lambda e: np.array([0.77], dtype=np.float32))
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: np.array([0.33], dtype=np.float32))

    gp._apply_aadb_step(embs, [0], breakdowns)
    gp._apply_exemplar_step(embs, [0], breakdowns)

    vec = mj.feature_vector(breakdowns[0])
    assert vec[mj.FEATURES.index("AADB")] == 0.77
    assert vec[mj.FEATURES.index("Exemplar")] == 0.33


def test_aadb_and_exemplar_are_excluded_from_fusion_roles():
    """Regression guard: 'AADB' and 'Exemplar' must not match a live
    score-fusion role keyword (niche_registry._ROLE_KEYWORDS) by substring
    accident. Neither feature is a per-aspect axis — they're whole-photo
    MasterJudge inputs — so a future rename that happened to contain, say,
    'human' or 'moment' as a substring would silently route it into a live
    fusion slot instead of leaving it out ('')."""
    import niche_registry as nr

    assert nr.aspect_role("AADB") == ""
    assert nr.aspect_role("Exemplar") == ""
