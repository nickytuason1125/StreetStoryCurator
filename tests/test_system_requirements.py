"""
The machine has to be told it is under spec BEFORE the cull, not during it.

The floors are real and were paid for: the encoder refuses under ~1.5 GB free,
the app gates grading at 1.8 GB, and 5 GB is what "comfortable" actually means
(the SigLIP encode subprocess needs ~2 GB during model load, plus the grade
worker's ~1 GB baseline). None of that was documented anywhere a user could
read it, and nothing checked it at launch — so an under-spec machine started
fine, opened a folder, and died partway through a run that had already cost
real time.

These lock the shape of the check: it classifies honestly, it never blocks
launch, and the numbers stay tied to the ones the app enforces elsewhere.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_system_requirements.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import system_check as sc  # noqa: E402


# ── Classification ───────────────────────────────────────────────────────────

def test_a_comfortable_machine_is_ok():
    r = sc.assess(total_gb=16.0, free_gb=8.0, disk_free_gb=60.0)
    assert r.level == "ok"
    assert r.blocking is False


def test_enough_to_run_but_tight_is_flagged_not_failed():
    r = sc.assess(total_gb=8.0, free_gb=2.5, disk_free_gb=60.0)
    assert r.level == "tight"
    assert r.blocking is False
    assert "close" in r.message.lower()


def test_under_the_grade_floor_is_called_out():
    r = sc.assess(total_gb=8.0, free_gb=1.2, disk_free_gb=60.0)
    assert r.level == "insufficient"
    assert "1.8" in r.message or "1.5" in r.message


def test_low_disk_is_reported_even_when_memory_is_fine():
    r = sc.assess(total_gb=32.0, free_gb=20.0, disk_free_gb=3.0)
    assert r.level != "ok"
    assert "disk" in r.message.lower()


# ── The rule that matters most ───────────────────────────────────────────────

def test_the_check_never_blocks_launch():
    """A warning is the product. Refusing to start would be worse than the bug.

    The photographer may want to browse an existing library, adjust ratings or
    export a sequence - none of which loads an encoder. Gating the whole app on
    a grading floor would take those away to prevent a failure that has its own
    error message at the point it actually matters.
    """
    for free in (0.1, 0.9, 1.4, 1.9, 4.0, 32.0):
        assert sc.assess(total_gb=8.0, free_gb=free, disk_free_gb=1.0).blocking is False


def test_message_is_always_actionable_and_never_empty():
    for args in [(16.0, 8.0, 60.0), (8.0, 2.5, 60.0), (8.0, 1.2, 60.0), (32.0, 20.0, 2.0)]:
        r = sc.assess(*args)
        assert r.message and r.message.strip()
        assert not r.message.endswith((" ", "\n"))


def test_floors_match_what_the_app_actually_enforces():
    """If someone moves a floor, this fails rather than the docs going stale."""
    assert sc.ENCODER_FLOOR_GB == 1.5
    assert sc.GRADE_FLOOR_GB == 1.8
    assert sc.COMFORTABLE_FREE_GB == 5.0


def test_unknown_readings_do_not_crash_or_alarm():
    r = sc.assess(total_gb=None, free_gb=None, disk_free_gb=None)
    assert r.level == "unknown"
    assert r.blocking is False
