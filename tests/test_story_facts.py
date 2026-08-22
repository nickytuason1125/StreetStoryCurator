r"""
Measured facts about a candidate photo, for Story and Competition mode.

Why this module exists
----------------------
The Art Director received {"id", "score", "style", "profile"} per photo, and
`profile` was ALWAYS "" because nothing in the repo ever wrote semantic_profile.
It was choosing a photo story from a spreadsheet with one adjective per row.

Why framing comes from OPTICS, not from the embedding
-----------------------------------------------------
The first design derived shot type from SigLIP zero-shot probes. It was
validated against 488 real Strong photos and REJECTED: the "close" class
returned a waterfall and then a wide-angle sun-flare landscape, and "portrait"
could not win a single frame out of 488 (every margin negative). Better prompts,
explicit scale wording and shared negatives did not fix it. CLIP-family models
are strong on CONTENT and weak on FRAMING, and this is that weakness.

Focal length is not an inference. 24mm IS wide. The camera wrote it down.

SigLIP keeps the jobs it is good at -- brief match, mood, content -- which is
where the existing text rerank already uses it successfully.

Run:  venv\Scripts\python.exe -m pytest tests/test_story_facts.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import story_facts as sf  # noqa: E402


# ── framing from focal length ────────────────────────────────────────────────

def test_wide_lens_is_wide():
    assert sf.framing_from_focal(24) == "wide"
    assert sf.framing_from_focal(16) == "wide"


def test_normal_lens_is_medium():
    assert sf.framing_from_focal(50) == "medium"


def test_long_lens_is_close():
    assert sf.framing_from_focal(85) == "close"
    assert sf.framing_from_focal(200) == "close"


def test_boundaries_are_explicit():
    """A boundary that moves silently changes every story. Pin them."""
    assert sf.framing_from_focal(sf.WIDE_MAX_MM) == "wide"
    assert sf.framing_from_focal(sf.WIDE_MAX_MM + 1) == "medium"
    assert sf.framing_from_focal(sf.MEDIUM_MAX_MM) == "medium"
    assert sf.framing_from_focal(sf.MEDIUM_MAX_MM + 1) == "close"


def test_unknown_focal_is_none_not_a_guess():
    """RAW files and scans often carry no focal length. Guessing 'medium'
    would put an unknown frame into a narrative slot on invented evidence."""
    assert sf.framing_from_focal(None) is None
    assert sf.framing_from_focal(0) is None
    assert sf.framing_from_focal("garbage") is None


def test_focal_accepts_the_string_exif_reader_returns():
    """exif_reader formats focal as '24mm', not as a number."""
    assert sf.framing_from_focal("24mm") == "wide"
    assert sf.framing_from_focal("85 mm") == "close"


# ── sessions from capture time ───────────────────────────────────────────────

def test_sessions_split_on_a_real_gap():
    """A shoot is what turns a pile of files into something with a beginning."""
    hour = 3600.0
    ts = [1000.0, 1000.0 + 60, 1000.0 + 120,      # morning
          1000.0 + 6 * hour, 1000.0 + 6 * hour + 90]   # evening
    assert sf.sessions_from_timestamps(ts) == [0, 0, 0, 1, 1]


def test_sessions_tolerate_missing_timestamps():
    assert sf.sessions_from_timestamps([0.0, 0.0, 0.0]) == [0, 0, 0]
    assert sf.sessions_from_timestamps([]) == []


def test_sessions_do_not_assume_sorted_input():
    hour = 3600.0
    ts = [1000.0 + 6 * hour, 1000.0, 1000.0 + 60]
    out = sf.sessions_from_timestamps(ts)
    assert out[1] == out[2], "the two morning frames belong together"
    assert out[0] != out[1], "the evening frame is its own session"


# ── the aggregate ────────────────────────────────────────────────────────────

def test_facts_carry_what_grading_already_computed():
    """personal_score is a model of the user's taste, computed per photo and
    currently discarded by the director. Same for exif_ts."""
    rows = [{"path": "/a.jpg", "score": 0.71, "personal_score": 0.63,
             "exif_ts": 1000.0, "focal_35mm": "24mm"}]
    facts = sf.facts_for_pool(rows)
    assert len(facts) == 1
    f = facts[0]
    assert f.framing == "wide"
    assert f.personal_score == 0.63
    assert f.score == 0.71
    assert f.session == 0


def test_missing_everything_is_survivable_and_explained():
    """A photo with no EXIF must not crash the run, and must say why it is
    unplaceable rather than silently defaulting into a slot."""
    facts = sf.facts_for_pool([{"path": "/b.jpg"}])
    f = facts[0]
    assert f.framing is None
    assert f.reason, "an absent fact needs a stated reason"
