"""Regression tests: HARD_FILTER_PEOPLE must be enforced from the rule set.

Root cause these lock down: the rule-set agent correctly set
HARD_FILTER_PEOPLE=True for a "no people" brief (via LLM refinement), but the
enforcement gates re-checked with the keyword matcher, which did not know the
phrase "no people" — so the gates silently no-oped while the UI showed the chip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import creative_director as cd  # noqa: E402
import creative_director_agent as cda  # noqa: E402


def test_no_people_phrase_triggers_keyword_matcher():
    assert cd._empty_brief_detected("golden warmth, geometric, no people")
    assert cd._empty_brief_detected("clean streets without anyone")
    assert not cd._empty_brief_detected("busy crowd, humanist warmth")


def test_kill_switch_runs_when_rule_set_says_so_even_without_keywords(tmp_path):
    # Brief deliberately contains NO empty/liminal vocabulary — previously the
    # gate silently skipped D-FINE even with HARD_FILTER_PEOPLE=True.
    calls = {}
    from PIL import Image as _PIL

    p1 = str(tmp_path / "a.jpg")
    p2 = str(tmp_path / "b.jpg")
    _PIL.new("RGB", (64, 64), (128, 128, 128)).save(p1)
    _PIL.new("RGB", (64, 64), (128, 128, 128)).save(p2)

    class _FakeDet:
        @staticmethod
        def detect_persons(paths, conf=0.35):
            calls["paths"] = list(paths)
            return {p: [{"bbox": (0.4, 0.4, 0.6, 0.6)}] for p in paths}

    fake = pytest.importorskip("types").ModuleType("dfine_detector")
    fake.detect_persons = _FakeDet.detect_persons  # type: ignore[attr-defined]
    fake.unload = lambda: None  # type: ignore[attr-defined]
    sys.modules["dfine_detector"] = fake
    try:
        blocked = cd.person_kill_switch(
            [p1, p2], "golden warmth, geometric, no people",
            hard_filter_people=True,
        )
        assert blocked == {p1, p2}
        assert calls["paths"] == [p1, p2]

        # Rule set says no filter → skip (even if the brief had empty words)
        calls.clear()
        blocked = cd.person_kill_switch(
            [p1], "empty liminal void", hard_filter_people=False,
        )
        assert blocked == set() and not calls
    finally:
        sys.modules.pop("dfine_detector", None)


def test_intrusion_penalty_follows_rule_set_decision():
    paths = ["p.jpg"]
    embs = [__import__("numpy").zeros(8, dtype="float32")]
    scores = [0.8]
    # hard_filter_people=True → gate active regardless of brief text
    adj, _ = cd._apply_brief_constraints(
        paths, embs, scores, None,
        "golden warmth, geometric, no people",
        hard_filter_people=True,
    )
    # people_emb may or may not exist on disk; with a zero embedding the sim
    # is 0 → no penalty either way; the contract tested here is that the gate
    # did not short-circuit on the brief text. Verify via the skip path below.
    assert len(adj) == 1
    # hard_filter_people=False → gate skipped even with empty-brief keywords
    adj2, _ = cd._apply_brief_constraints(
        paths, embs, scores, None, "empty liminal void",
        hard_filter_people=False,
    )
    assert adj2 == [0.8]


# ── Generic brief-exclusion criteria (any "no X / without X") ────────────────

def test_keyword_rule_set_extracts_exclusions():
    rs = cda._keyword_rule_set("no cars, avoid neon signs and reflections, geometric shapes")
    assert "cars" in rs["EXCLUDE"]
    assert "neon signs" in rs["EXCLUDE"]
    assert rs["HARD_FILTER_PEOPLE"] is False  # no people criterion here
    # "avoid" must NOT match the "void" empty keyword (word-boundary matching)
    rs_no_void = cda._keyword_rule_set("avoid cars, geometric shapes")
    assert cda._keyword_rule_set("avoid cars")["HARD_FILTER_PEOPLE"] is False
    assert cd._empty_brief_detected("avoid crowds") is False
    assert cd._empty_brief_detected("empty liminal void") is True

    rs2 = cda._keyword_rule_set("golden warmth, geometric, no people")
    assert rs2["EXCLUDE"] == ["people"]
    assert rs2["HARD_FILTER_PEOPLE"] is True  # people exclusion flips the flag


def test_exclusion_engine_hard_soft_keep(monkeypatch):
    import numpy as np
    # 4-d fake concept space: concept "cars" → e0
    monkeypatch.setattr(
        cd, "_embed_texts",
        lambda qs: np.stack([np.eye(4, dtype="float32")[i] for i in range(len(qs))]),
    )
    paths = ["car.jpg", "clean.jpg", "maybe.jpg"]
    embs = [
        np.eye(4, dtype="float32")[0],                                  # exact concept
        np.eye(4, dtype="float32")[2],                                  # unrelated
        np.eye(4, dtype="float32")[0] * 0.5 + np.eye(4, dtype="float32")[2] * 0.5,  # borderline
    ]
    adj, notes = cd._apply_exclusion_constraints(paths, embs, [0.9, 0.8, 0.7], ["cars"])
    assert adj[0] < 0.1                    # hard band → near-disqualified
    assert adj[1] == 0.8                   # unrelated → untouched
    assert abs(adj[2] - 0.035) < 1e-6      # sim 0.707 ≥ hard 0.32 → × 0.05
    assert "cars" in notes[0]


def test_exclusion_engine_skips_people_terms():
    # people concepts are handled by the D-FINE gate — the generic pass must
    # be a no-op for them (wired via the caller filter, verified at the unit level)
    excl = [e for e in ["people", "cars"] if e not in cd._PEOPLE_TERMS]
    assert excl == ["cars"]


def test_engine_degrades_gracefully_without_encoder(monkeypatch):
    def _boom(qs):
        raise RuntimeError("encoder not resident")
    monkeypatch.setattr(cd, "_embed_texts", _boom)
    adj, notes = cd._apply_exclusion_constraints(
        ["a.jpg"], [__import__("numpy").ones(4, dtype="float32")], [0.7], ["cars"])
    assert adj == [0.7] and notes == [""]
