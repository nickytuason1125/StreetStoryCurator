r"""
Tonal consistency: does this frame belong to this set, by look and not by subject?

Why this exists
---------------
A real Story run scored 0.891 cohesion and still mixed warm night colour,
black-and-white, and a bright daytime frame in one six-image sequence. Every
frame genuinely matched the brief -- "quiet streets, people at a distance" --
because SigLIP embeddings encode SUBJECT AND SCENE. They do not encode tonality,
colour, or time of day, which is most of what makes photographs read as one body
of work.

So cosine cohesion was measuring the wrong kind of similarity, exactly as the
design doc warned it might: "looks similar" is not "belongs together".

Stats come from cache/thumbs -- 7,791 existing 4 KB webp thumbnails, named
{stem}_{md5(path)[:10]}.webp by app.get_or_create_thumb. Reading those is
effectively free; decoding full frames would not be.

Run:  venv\Scripts\python.exe -m pytest tests/test_tonal_stats.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tonal_stats as ts  # noqa: E402


def _write(tmp_path, name, rgb, size=(24, 24)):
    from PIL import Image
    img = Image.new("RGB", size, rgb)
    p = tmp_path / name
    img.save(str(p), "WEBP")
    return p


def test_monochrome_reads_as_zero_chroma(tmp_path):
    grey = _write(tmp_path, "g.webp", (128, 128, 128))
    luma, chroma = ts.stats_from_thumb(grey)
    assert chroma < 0.02, f"grey should have no colour, got {chroma:.3f}"
    assert 0.4 < luma < 0.6


def test_saturated_colour_reads_as_high_chroma(tmp_path):
    orange = _write(tmp_path, "o.webp", (230, 120, 20))
    _, chroma = ts.stats_from_thumb(orange)
    assert chroma > 0.3, f"a strong colour should register, got {chroma:.3f}"


def test_dark_and_bright_separate_on_luma(tmp_path):
    dark = _write(tmp_path, "d.webp", (18, 18, 20))
    bright = _write(tmp_path, "b.webp", (225, 225, 220))
    assert ts.stats_from_thumb(dark)[0] < 0.15
    assert ts.stats_from_thumb(bright)[0] > 0.8


def test_unreadable_thumb_is_none_not_a_guess(tmp_path):
    assert ts.stats_from_thumb(tmp_path / "missing.webp") is None
    bad = tmp_path / "bad.webp"
    bad.write_bytes(b"not an image")
    assert ts.stats_from_thumb(bad) is None


def test_thumb_path_matches_the_apps_naming():
    """app.get_or_create_thumb: {stem with spaces as underscores}_{md5[:10]}.webp"""
    import hashlib
    p = r"C:\photos\JPN 2024\4STAR (7 of 392).jpg"
    want = "4STAR_(7_of_392)_" + hashlib.md5(p.encode()).hexdigest()[:10] + ".webp"
    assert ts.thumb_path_for(p).name == want


# ── the fit measure the selector uses ────────────────────────────────────────

def test_a_frame_matching_the_set_scores_higher_than_one_that_clashes():
    warm_night = np.array([[0.20, 0.45], [0.22, 0.42], [0.18, 0.48]])  # luma, chroma
    similar = np.array([0.21, 0.44])
    mono_daylight = np.array([0.75, 0.01])
    assert ts.tonal_fit(similar, warm_night) > ts.tonal_fit(mono_daylight, warm_night)


def test_fit_is_neutral_when_the_frame_is_unmeasured():
    """A photo with no thumbnail must not be pushed out of a set for having no
    thumbnail. Unknown is not the same as clashing."""
    chosen = np.array([[0.20, 0.45]])
    assert ts.tonal_fit(None, chosen) == 0.5
    assert ts.tonal_fit(np.array([np.nan, np.nan]), chosen) == 0.5


def test_fit_is_neutral_for_an_empty_set():
    assert ts.tonal_fit(np.array([0.2, 0.4]), np.zeros((0, 2))) == 0.5
