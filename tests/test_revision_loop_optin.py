r"""
The contact-sheet critique must not run by default.

Measured through the real /api/creative-direction/stream endpoint on a folder of
graded photos: the request did not finish in TEN MINUTES. The server log shows
why -- the revision loop renders the chosen set as one contact sheet and pushes
it through a vision model on CPU:

    encoding image slice ...          170,524 ms
    decoding image batch 1/3 ...        5,300 ms
    decoding image batch 2/3 ...       20,985 ms
    decoding image batch 3/3 ...        6,110 ms

~200 seconds per iteration, up to _MAX_ITERS iterations, for a feature that
suggests at most one slot swap. Everything else in a Story run is now measured
in single-digit seconds, so this alone decides whether the product is usable.

It is not deleted -- it works, and on a machine with a GPU it is cheap. It is
opt-in, the same shape as deep_grade: off by default, available to anyone
willing to wait.

Run:  venv\Scripts\python.exe -m pytest tests/test_revision_loop_optin.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director as cd  # noqa: E402
import run_profile  # noqa: E402


def test_setting_is_declared():
    assert "FRAMEGRADE_STORY_REVISION" in run_profile.SETTINGS


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("FRAMEGRADE_STORY_REVISION", raising=False)
    assert cd._revision_enabled() is False


def test_opt_in_turns_it_on(monkeypatch):
    monkeypatch.setenv("FRAMEGRADE_STORY_REVISION", "1")
    assert cd._revision_enabled() is True


def test_probe_never_raises(monkeypatch):
    """A settings failure must not be what stops a Story run."""
    def boom(_):
        raise RuntimeError("no run_profile")
    monkeypatch.setattr(run_profile, "setting", boom)
    assert cd._revision_enabled() is False
