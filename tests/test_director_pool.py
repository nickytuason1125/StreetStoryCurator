r"""
How many candidates the Art Director sees, and why it is not 25.

Measured on the target laptop, same model, same prompt, only the manifest
size changing:

    25 candidates   36.1 s
    12 candidates    4.7 s
     8 candidates    4.3 s

and the opener was correct at all three sizes. Attention is quadratic in
sequence length, so prefill cost is superlinear in candidate count -- halving
the manifest cut the time by 7.7x, not 2x.

The old rule was `min(max(n_target * 4, 25), len(pool))`, which gave the
director 28 candidates for a 7-image story and put a single call over the
budget for the entire run.

This is a real trade, not a free win: fewer candidates is less to choose from.
It is a declared setting so it can be raised deliberately, rather than a
constant nobody knows is there.

Run:  venv\Scripts\python.exe -m pytest tests/test_director_pool.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director as cd  # noqa: E402
import run_profile  # noqa: E402


def test_default_pool_is_the_fast_size():
    assert cd._director_pool_size(n_target=7, pool_len=500) == 12


def test_never_smaller_than_the_story_needs():
    """A 10-image story cannot be chosen from 12 without the director having
    almost no choice at all."""
    assert cd._director_pool_size(n_target=10, pool_len=500) >= 13


def test_never_larger_than_the_pool():
    assert cd._director_pool_size(n_target=7, pool_len=5) == 5


def test_setting_is_declared_so_it_can_be_raised():
    assert "FRAMEGRADE_DIRECTOR_POOL" in run_profile.SETTINGS


def test_setting_overrides_the_default(monkeypatch):
    monkeypatch.setenv("FRAMEGRADE_DIRECTOR_POOL", "25")
    assert cd._director_pool_size(n_target=7, pool_len=500) == 25
