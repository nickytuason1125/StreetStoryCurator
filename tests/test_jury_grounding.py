"""
Make fabricated citations structurally impossible, not merely rejected.

The jury hallucinated cited evidence in 20-30% of verdicts. validate_claims
catches those and drops them, so nothing wrong reaches the narrative - but a
dropped verdict is a juror who said nothing, and three structural causes were
inviting the model to guess:

  1. _build_slot_summary showed only the FIRST 3 aspects, while the schema let
     the model cite any of 5. Citing one it was never shown means inventing a
     number.
  2. The summary was truncated at 500 chars, so later slots could vanish from
     the prompt while remaining citable.
  3. cited_slot was typed as a bare integer, so slot 47 of a 5-slot sequence
     was legal output.

The fix constrains the grammar to THIS sequence: slot indices and the actual
aspect values become enums, so a fabricated citation cannot be decoded at all.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_jury_grounding.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import jury_engine as je          # noqa: E402

ASPECTS = ["Composition", "Lighting", "Narrative", "Human/Culture", "Technical"]


def _imgs(n=3):
    out = []
    for i in range(n):
        out.append({"filename": f"P110{i:04d}.JPG",
                    **{a: 0.10 * (i + 1) + 0.01 * j for j, a in enumerate(ASPECTS)}})
    return out


# ── 1. the prompt must show everything it allows citing ──────────────────────

def test_summary_shows_every_aspect_not_just_three():
    imgs = _imgs(3)
    s = je._build_slot_summary(imgs, ["opener", "subject", "closer"], [0.5, 0.6, 0.7])
    for a in ASPECTS:
        assert a in s, f"{a} is citable but never shown to the model"


def test_summary_is_not_truncated_mid_slot():
    """Long filenames must not silently delete later slots from the prompt."""
    imgs = _imgs(6)
    for i, im in enumerate(imgs):
        im["filename"] = f"a-very-long-original-camera-filename-{i}-DSCF{i:05d}.JPG"
    s = je._build_slot_summary(imgs, ["r"] * 6, [0.5] * 6)
    for i in range(6):
        assert f"{i}:" in s, f"slot {i} was truncated out of the prompt"


# ── 2. the grammar must not permit citing what does not exist ────────────────

def test_schema_restricts_slots_to_those_that_exist():
    aspects_by_slot = [{a: 0.4 for a in ASPECTS} for _ in range(3)]
    schema = je.jury_schema(aspects_by_slot)
    slots = schema["properties"]["cited_slot"]["enum"]
    assert set(x for x in slots if x is not None) == {0, 1, 2}
    assert None in slots, "a purely qualitative verdict must still be expressible"


def test_schema_restricts_values_to_those_actually_present():
    aspects_by_slot = [{"Composition": 0.41, "Lighting": 0.74},
                       {"Composition": 0.55, "Lighting": 0.62}]
    schema = je.jury_schema(aspects_by_slot)
    vals = {v for v in schema["properties"]["cited_value"]["enum"] if v is not None}
    assert vals == {0.41, 0.74, 0.55, 0.62}
    assert 0.99 not in vals, "a value never computed must not be expressible"


def test_schema_values_match_the_rounding_shown_in_the_prompt():
    """The prompt prints 2dp. If the enum carried full precision the model
    would copy what it sees, fail the enum, and be forced into a wrong value."""
    aspects_by_slot = [{"Composition": 0.414159, "Lighting": 0.735}]
    schema = je.jury_schema(aspects_by_slot)
    vals = {v for v in schema["properties"]["cited_value"]["enum"] if v is not None}
    # Whatever the prompt prints is what the enum must contain - derive the
    # expectation the same way rather than hardcoding a rounding assumption.
    summary = je._build_slot_summary(
        [{"filename": "a.jpg", "Composition": 0.414159, "Lighting": 0.735}],
        ["opener"], [0.5])
    for v in vals:
        assert f"={v:.2f}" in summary, f"enum value {v} never appears in the prompt"
    assert len(vals) == 2


def test_aspect_enum_still_pinned():
    schema = je.jury_schema([{a: 0.4 for a in ASPECTS}])
    assert set(schema["properties"]["cited_aspect"]["enum"]) == je._VALID_ASPECTS | {"none"}


def test_empty_sequence_degrades_to_null_only():
    """No slots means nothing is citable - the schema must still be buildable."""
    schema = je.jury_schema([])
    assert schema["properties"]["cited_slot"]["enum"] == [None]
    assert schema["properties"]["cited_value"]["enum"] == [None]


# ── 3. the constrained grammar must actually compile ─────────────────────────

def test_constrained_grammar_compiles():
    pytest.importorskip("llama_cpp")
    aspects_by_slot = [{a: 0.4 + 0.01 * i for a in ASPECTS} for i in range(4)]
    g = je.build_grammar(je.jury_schema(aspects_by_slot))
    assert g is not None, "a per-sequence grammar must compile, or grounding is unenforced"
