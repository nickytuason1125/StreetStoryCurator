"""blend_anchor: shared brief/peg/book blender (2026-09-22).

The user's brief and reference peg previously influenced ONLY the Creative
Director selection — grading read specvlm_pipeline._CD_BRIEF, which no code
path ever set, and the sequence vibe_prompt endpoint was an admitted no-op.
These tests pin the shared blender and the wiring that makes the session
context reach every pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def _clean_session():
    import blend_anchor
    blend_anchor.clear_session_context()
    yield
    blend_anchor.clear_session_context()


def _unit(seed):
    v = np.random.default_rng(seed).normal(size=1536).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ── Session context ──────────────────────────────────────────────────────────

def test_session_context_roundtrip():
    from blend_anchor import set_session_context, get_session_context
    set_session_context("orange and reds", "59fb8077-aa75")
    ctx = get_session_context()
    assert ctx["brief"] == "orange and reds"
    assert ctx["peg_image_hash"] == "59fb8077-aa75"


def test_clear_session_context_empties_everything():
    from blend_anchor import (set_session_context, clear_session_context,
                              get_session_context)
    set_session_context("rainy night", "abc")
    clear_session_context()
    assert get_session_context() == {"brief": "", "peg_image_hash": ""}


# ── Peg stem matching (canonical impl shared with creative_director) ─────────


def test_peg_stem_match_boundaries():
    """Canonical semantics (identical to the original creative_director impl):
    match on full stem, or on the stem's LAST underscore-segment starting
    with the hash with a non-digit remainder."""
    from blend_anchor import peg_stem_match
    assert peg_stem_match("TPE26-1", "TPE26-1")
    assert peg_stem_match("x_TPE26-1", "TPE26-1")
    assert peg_stem_match("carousel_01_TPE26-1", "TPE26-1")
    assert not peg_stem_match("TPE26-105", "TPE26-1")     # substring trap
    assert not peg_stem_match("", "TPE26-1")
    # Empty-hash quirk inherited from the original implementation: the matcher
    # alone returns True for any stem, but every caller (resolve_peg_embedding,
    # creative_director's Step 3) guards an empty hash with an early return —
    # pinned here so the guard's importance is explicit.
    assert peg_stem_match("TPE26-1", "")


def test_creative_director_alias_matches_canonical():
    from blend_anchor import peg_stem_match
    from creative_director import _peg_stem_match
    for stem, h in [("TPE26-1", "TPE26-1"), ("TPE26-105", "TPE26-1"),
                    ("x_TPE26-1_y", "TPE26-1")]:
        assert _peg_stem_match(stem, h) == peg_stem_match(stem, h)


# ── blend_query_vector ───────────────────────────────────────────────────────

def test_no_peg_no_rag_returns_base_unchanged():
    """No session context → the vector is byte-identical (the old behaviour)."""
    from blend_anchor import blend_query_vector
    base = _unit(0)
    q, diag = blend_query_vector(base, "", use_rag=False)
    assert q is not None and np.allclose(q, base, atol=1e-6)
    assert diag["peg"] is False and diag["rag"] == 0
    assert diag["sources"] == ["base"]


def test_peg_blends_at_30_percent():
    from blend_anchor import blend_query_vector, PEG_WEIGHT
    base, peg = _unit(0), _unit(1)
    q, diag = blend_query_vector(base, "any brief", peg_vec=peg, use_rag=False)
    expected = (1 - PEG_WEIGHT) * base + PEG_WEIGHT * peg
    expected /= np.linalg.norm(expected)
    assert np.allclose(q, expected, atol=1e-5)
    assert diag["peg"] is True


def test_books_blend_at_30_percent_with_supplied_phrases(monkeypatch):
    """The SigLIP text tower is NOT loaded in unit tests — the phrase-vector
    dependency is stubbed so the blend maths itself is what's under test."""
    import blend_anchor as b
    from blend_anchor import blend_query_vector, RAG_WEIGHT
    base = _unit(2)
    phrase = _unit(3)[None, :]                          # (1, 1536)
    monkeypatch.setattr(b, "rag_phrase_vectors", lambda ph: phrase)
    q, diag = blend_query_vector(base, "moody rain", use_rag=True,
                                 phrases=["a rainy street"])
    expected = RAG_WEIGHT * phrase[0] + (1 - RAG_WEIGHT) * base
    expected /= np.linalg.norm(expected)
    assert np.allclose(q, expected, atol=1e-5)
    assert diag["rag"] == 1


def test_result_is_unit_normalised():
    from blend_anchor import blend_query_vector
    q, _ = blend_query_vector(_unit(4), "x", peg_vec=_unit(5), use_rag=False)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5


def test_no_base_and_no_encoder_returns_none_with_diag():
    """Degradation must be LOUD in diag, never a crash."""
    from blend_anchor import blend_query_vector
    q, diag = blend_query_vector(None, "some brief", use_rag=False)
    assert q is None
    assert "brief" not in diag["sources"]


def test_peg_dimension_mismatch_is_ignored():
    from blend_anchor import blend_query_vector
    base = _unit(6)
    q, diag = blend_query_vector(base, "", peg_vec=_unit(7)[:100], use_rag=False)
    assert diag["peg"] is False
    assert np.allclose(q, base, atol=1e-6)


# ── Diagnostics + context line ───────────────────────────────────────────────

def test_blend_diag_line_names_every_source():
    from blend_anchor import blend_query_vector, blend_diag_line
    _, diag = blend_query_vector(_unit(8), "brief words", peg_vec=_unit(9),
                                 use_rag=False)
    line = blend_diag_line(diag)
    assert "[blend]" in line and "peg=yes" in line and "base" in line


def test_context_line_empty_without_session():
    from blend_anchor import context_line
    assert context_line() == ""


def test_context_line_names_brief_and_peg():
    from blend_anchor import set_session_context, context_line
    set_session_context("orange and reds", "abc123")
    line = context_line()
    assert "orange and reds" in line and "reference photo" in line


# ── Router wiring (set_cd_brief now actually called) ─────────────────────────

def test_creative_router_sets_specvlm_brief():
    """The silent no-op this module exists to fix: specvlm_pipeline._CD_BRIEF
    must be set by the creative-direction router before grading runs.
    (Read as source — importing routers.creative pulls the whole server.)"""
    router_py = (Path(__file__).resolve().parent.parent /
                 "routers" / "creative.py").read_text(encoding="utf-8")
    assert "set_cd_brief(" in router_py, \
        "router must propagate the brief to grading"
    assert "set_session_context(" in router_py, \
        "router must set the blend session context"
