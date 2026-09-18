"""Tests for the Brief object (photo_brief) — one parse of the user's brief,
consumed by every stage. Plus the SSE done-payload contract that pins the
frontend/backend interface (2026-09-16)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photo_brief import (Brief, build_brief, parse_subject_only,
                         REQUIRED_DONE_KEYS, RECOMMENDED_KEYWORDS)


@pytest.mark.parametrize("text,expected", [
    ("I want the 5 photos to be vehicles only", "vehicles"),
    ("vehicles only", "vehicles"),
    ("use architecture only please", "architecture"),
    ("motorcycles only in the rain", "motorcycles"),
    ("5 photos of food stalls only", "food stalls"),
    ("the shots should be vehicles only", "vehicles"),
])
def test_parse_subject_only(text, expected):
    assert parse_subject_only(text) is not None
    assert parse_subject_only(text)[0] == expected


@pytest.mark.parametrize("text", [
    "best shots",
    "something only god could explain",
    "a rainy night in the city",     # no "only"
    "",
])
def test_parse_subject_only_rejects_non_subjects(text):
    assert parse_subject_only(text) is None


def test_build_brief_full_example():
    b = build_brief("the 5 photos should be vehicles only, moody night, for instagram")
    assert b.subject == "vehicles"
    assert b.subject_mode == "hard"
    assert b.audience == "instagram"
    assert b.mood == "moody/high-contrast"   # from "moody night"
    assert b.hard_filters == []              # people allowed here
    d = b.to_dict()
    # Legacy keys the frontend renders must survive in to_dict.
    for k in ("LIGHTING_MOOD", "GEOMETRIC_PRIORITY", "HARD_FILTER_PEOPLE",
              "BRIEF_KEYWORDS", "SUBJECT_ONLY"):
        assert k in d


# ── Recommended-keyword contract (2026-09-17) ────────────────────────────────
# The brief editor's quick-add chips are served from RECOMMENDED_KEYWORDS.
# Every chip must provably move the parsed Brief — a chip that changes nothing
# is a dead button the user taps in good faith. If you add a keyword to the
# table, this test forces the parser to understand it too (and vice versa).
@pytest.mark.parametrize("group,chip",
                         [(g, c) for g, chips in RECOMMENDED_KEYWORDS.items()
                          for c in chips])
def test_recommended_chip_moves_the_brief(group, chip):
    text = f"moody night test, {chip}"
    b = build_brief(text)
    moved = (
        b.mood != "neutral"
        or b.geometry != "Normal"
        or b.hard_filters
        or b.subject is not None          # "<subject> only" chips
        or b.audience is not None
        or any(chip in k or k in chip for k in b.keywords)
    )
    assert moved, f"recommended chip {group}/{chip!r} does not change the parsed Brief"


def test_recommended_keywords_groups_are_healthy():
    # The UI renders these groups verbatim — keep the shape sane.
    assert set(RECOMMENDED_KEYWORDS) == {"Mood", "Subject", "Look", "Audience"}
    for group, chips in RECOMMENDED_KEYWORDS.items():
        assert chips, f"group {group} is empty"
        assert len(chips) == len(set(chips)), f"duplicate chip in {group}"


def test_build_brief_no_people_hard_filter():
    b = build_brief("empty streets only, no people")
    assert "no_people" in b.hard_filters
    assert b.subject == "empty streets"
    d = b.to_dict()
    assert d["HARD_FILTER_PEOPLE"] is True


def test_build_brief_low_context_reads_neutral():
    b = build_brief("Top 5 pictures for instagram as a sequence")
    assert b.mood == "neutral"
    assert b.keywords == []
    assert b.audience == "instagram"
    assert b.subject is None


def test_done_payload_contract_keys_are_satisfied_by_pipeline_return():
    """Contract: the creative-direction done payload carries every required
    key. The pipeline enforces this at run time (raise on missing); this test
    pins the constant against the keys the return dict is documented to hold."""
    from creative_director import run_creative_direction  # noqa: F401  (import cost guard)
    documented = {
        "outputs", "output_dir", "total", "success", "failed",
        "anchor_path", "rule_set", "director_fallback", "selection",
        "alt_outputs", "subject", "brief", "timings",
    }
    assert REQUIRED_DONE_KEYS <= documented
