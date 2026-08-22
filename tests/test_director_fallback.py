r"""
A degraded Story run must SAY it degraded.

Two failure modes, both measured on the target hardware rather than imagined:

  1. Silent fallback. When the text model cannot load, ask_local_art_director
     returns top-N by raw score and prints one line into a subprocess log. The
     user sees a curated-looking sequence with no indication that nothing
     curated it. That is the whole of "the picks are wrong, the order is
     arbitrary, it can't explain itself".

  2. A RAM floor that is a constant instead of a measurement. The floor was a
     flat 6.0 GB -- right for the 5.73 GB checkpoint it was written against, and
     wrong for every other model. A 0.73 GB model needs nowhere near 6 GB free,
     but the constant refuses to load it, so shrinking the model would not have
     helped by itself. Same class of bug as commit 3f42c7f, where the torch
     floor sat BELOW the crash it existed to prevent.

Run:  venv\Scripts\python.exe -m pytest tests/test_director_fallback.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import local_llm  # noqa: E402
import creative_director as cd  # noqa: E402


class _FakeWeights:
    """Stand-in for the GGUF path. Only its SIZE is ever read.

    The first version of this helper called f.truncate() to make a file of the
    right size. NTFS allocated it for real: two tests wrote 6.5 GB between them
    and the run died with "No space left on device" on a drive that had 7 GB
    free. Never materialise a model-sized file in a test -- describe it.
    """

    def __init__(self, gb, name="fake.gguf"):
        self._size = int(gb * 2 ** 30)
        self.name = name

    def exists(self):
        return True

    def stat(self):
        import types
        return types.SimpleNamespace(st_size=self._size)


# ── the RAM floor must follow the weights ────────────────────────────────────

def test_ram_floor_scales_with_weight_size(monkeypatch):
    """A small model must not inherit a big model's floor."""
    small = _FakeWeights(0.73)                     # LFM2.5-VL-1.6B
    monkeypatch.setattr(local_llm, "model_path", lambda: small)
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 0.0)

    need = local_llm.required_ram_gb()
    assert need < 2.0, f"a 0.73 GB model should not need {need:.1f} GB free"
    assert need > 0.73, "must still leave headroom above the weights themselves"


def test_ram_floor_is_higher_for_a_bigger_model(monkeypatch):
    big = _FakeWeights(5.34)                       # DeepSeek-R1-8B Q5, in GiB
    monkeypatch.setattr(local_llm, "model_path", lambda: big)
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 0.0)

    need = local_llm.required_ram_gb()
    assert need > 6.0, "the 5.73 GB checkpoint genuinely does need ~6.6 GB"


def test_explicit_override_still_wins(monkeypatch):
    small = _FakeWeights(0.73)
    monkeypatch.setattr(local_llm, "model_path", lambda: small)
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 9.0)
    assert local_llm.required_ram_gb() == 9.0


# ── the refusal must be legible ──────────────────────────────────────────────

def test_skip_reason_names_the_numbers(monkeypatch):
    """'It didn't run' is not actionable. 'Needed 6.6, had 1.4' is."""
    big = _FakeWeights(5.34)
    monkeypatch.setattr(local_llm, "model_path", lambda: big)
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 0.0)
    monkeypatch.setattr(local_llm, "_free_ram_gb", lambda: 1.4)
    monkeypatch.setattr(local_llm, "_llm", None)
    monkeypatch.setattr(local_llm, "_load_attempted", False)

    need = local_llm.required_ram_gb()
    assert local_llm._load() is None
    reason = local_llm.last_skip_reason()
    assert reason, "a refusal with no reason is the bug this test exists for"
    assert "1.4" in reason, reason
    assert f"{need:.1f}" in reason, f"must name what it needed: {reason!r}"


# ── the caller must be able to report it ─────────────────────────────────────

class _NoModel:
    """Stands in for local_llm when the weights cannot load."""
    @staticmethod
    def generate(*a, **k):
        return None

    @staticmethod
    def last_skip_reason():
        return "only 1.4 GB free, needs ~6.6 GB"


class _Chooser:
    """Stands in for local_llm when the model answers."""
    @staticmethod
    def generate(*a, **k):
        return "[2,0]"

    @staticmethod
    def last_skip_reason():
        return None


def _pool():
    return [{"id": i, "path": f"/img/{i}.jpg", "score": s, "breakdown": {}}
            for i, s in enumerate([0.30, 0.90, 0.60])]


def test_fallback_returns_a_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "local_llm", _NoModel)
    paths, reason = cd.ask_local_art_director("sys", _pool(), "Story", limit=2)

    assert paths == ["/img/1.jpg", "/img/2.jpg"], "still falls back to score order"
    assert reason, "the fallback must be reportable, not just printed"
    assert "1.4" in reason, reason


def test_no_reason_when_the_model_actually_chose(monkeypatch):
    monkeypatch.setitem(sys.modules, "local_llm", _Chooser)
    paths, reason = cd.ask_local_art_director("sys", _pool(), "Story", limit=2)

    assert paths == ["/img/2.jpg", "/img/0.jpg"], "model's order, not score order"
    assert reason is None, f"a real selection must not claim a fallback: {reason!r}"
