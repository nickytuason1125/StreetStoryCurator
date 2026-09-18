"""Regression tests: the RAM watchdog's model-unload wiring (2026-09-14).

These pin the idle-tracking API that the watchdog depends on — if the touch
points drift out of the modules, the watchdog would either never unload
(RAM creep returns) or unload constantly (features feel broken).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_critique_engine_idle_api():
    import time
    import critique_engine as ce
    assert ce.is_loaded() is False          # fresh import: nothing resident
    assert ce.idle_seconds() > 1e11         # never-used reads as "forever idle"
    ce._touch()
    assert ce.idle_seconds() < 5            # a touch resets the clock
    time.sleep(0.05)
    assert 0 <= ce.idle_seconds() < 5


def test_niche_detector_idle_api():
    import time
    import fast_niche_detector as fnd
    assert fnd.is_ready() is False
    assert fnd.idle_seconds() > 1e11
    fnd._touch()
    assert fnd.idle_seconds() < 5
    time.sleep(0.05)
    assert 0 <= fnd.idle_seconds() < 5


def test_unload_is_safe_when_nothing_is_loaded():
    """unload()/release() on a cold process must be no-ops, not errors —
    the watchdog calls them unconditionally under pressure."""
    import critique_engine as ce
    import fast_niche_detector as fnd
    ce.unload()      # must not raise
    fnd.release()    # must not raise
    assert ce.is_loaded() is False
    assert fnd.is_ready() is False
