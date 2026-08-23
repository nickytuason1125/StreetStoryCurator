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


def test_floor_matches_what_loading_actually_costs(monkeypatch):
    r"""MEASURED, not chosen. Peak RSS while loading at n_ctx=4096, on a machine
    with headroom:

        LFM2.5-VL-1.6B   file 0.68 GiB -> peak 1.25 GiB   (1.84x)
        Qwen3-4B         file 2.33 GiB -> peak 4.56 GiB   (1.96x)

    The first version of this floor was file x 1.15 + 0.5, which I invented and
    shipped as though measured -- the exact mistake commit 3f42c7f documents.
    It gave 3.17 GB for Qwen3-4B, so at 3.58 GB free the gate PASSED and the
    load then drove the machine to 0.00 GB available.

    A floor that admits the failure it exists to prevent is not a floor.
    """
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 0.0)

    monkeypatch.setattr(local_llm, "model_path", lambda: _FakeWeights(0.68))
    assert local_llm.required_ram_gb() >= 1.25, "must cover the measured LFM peak"

    monkeypatch.setattr(local_llm, "model_path", lambda: _FakeWeights(2.33))
    need = local_llm.required_ram_gb()
    assert need >= 4.56, f"must cover the measured Qwen peak, got {need:.2f}"
    assert need < 6.0, "but not so conservative that nothing ever loads"


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


# ── a transient refusal must not latch for the process lifetime ──────────────

def test_ram_refusal_is_retried_when_memory_frees_up(monkeypatch):
    r"""Observed live: a Story run reported "only 2.0 GB RAM free" while 4.04 GB
    was actually free, and returned in 9.9s without loading anything.

    _load() sets _load_attempted before the RAM check and then returns None on
    every later call, so ONE refusal while Chrome was open disabled the text
    model for the life of the server. Closing Chrome does not help; only
    restarting the app does. That is indistinguishable, to a user, from the
    feature being broken.

    A missing file or an unloadable build is permanent and should latch.
    Free memory is not.
    """
    weights = _FakeWeights(0.73)
    monkeypatch.setattr(local_llm, "model_path", lambda: weights)
    monkeypatch.setattr(local_llm, "_setting", lambda n, d: 0.0)
    monkeypatch.setattr(local_llm, "_llm", None)
    monkeypatch.setattr(local_llm, "_load_attempted", False)

    monkeypatch.setattr(local_llm, "_free_ram_gb", lambda: 0.2)
    assert local_llm._load() is None, "should refuse when memory is short"

    # memory frees up; the next call must try again rather than stay latched
    loaded = {}

    class _FakeLlama:
        def __init__(self, **kw):
            loaded["yes"] = True

    import types
    fake = types.ModuleType("llama_cpp")
    fake.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setattr(local_llm, "_free_ram_gb", lambda: 8.0)

    assert local_llm._load() is not None, "a RAM refusal must not latch"
    assert loaded.get("yes"), "it never even tried to construct the model"
    assert local_llm.last_skip_reason() is None


def test_missing_weights_still_latch(monkeypatch):
    """A file that is not there will not appear between calls; retrying that
    on every generate would spam the log for no reason."""
    class _Absent:
        name = "gone.gguf"
        def exists(self):
            return False
    monkeypatch.setattr(local_llm, "model_path", lambda: _Absent())
    monkeypatch.setattr(local_llm, "_llm", None)
    monkeypatch.setattr(local_llm, "_load_attempted", False)
    assert local_llm._load() is None
    assert local_llm._load_attempted is True


# ── the model's pick list is not trusted on faith ────────────────────────────

class _ShortList:
    """The model returns fewer ids than asked for. Observed live: three
    IDENTICAL Story requests returned 6, then 4, then 1 photograph."""
    @staticmethod
    def generate(*a, **k):
        return "[1]"

    @staticmethod
    def last_skip_reason():
        return None


class _DirtyList:
    """Duplicates and out-of-range ids in one answer."""
    @staticmethod
    def generate(*a, **k):
        return "[2, 2, 99, -4, 0]"

    @staticmethod
    def last_skip_reason():
        return None


def _pool5():
    return [{"id": i, "path": f"/img/{i}.jpg", "score": s, "breakdown": {}}
            for i, s in enumerate([0.10, 0.90, 0.80, 0.70, 0.60])]


def test_a_short_pick_list_is_topped_up(monkeypatch):
    """Asking for 4 and getting 1 is not a curatorial judgement, it is a small
    model losing count. The user sees a sequence, not a parse error, so nothing
    tells them the difference."""
    monkeypatch.setitem(sys.modules, "local_llm", _ShortList)
    paths, reason = cd.ask_local_art_director("sys", _pool5(), "Story", limit=4)

    assert len(paths) == 4, f"asked for 4, got {len(paths)}: {paths}"
    assert paths[0] == "/img/1.jpg", "the model's own pick must stay first"
    assert len(set(paths)) == 4, "topping up must not repeat a photo"
    assert reason and "1" in reason, f"the top-up must be reported: {reason!r}"


def test_duplicates_and_out_of_range_are_dropped_then_topped_up(monkeypatch):
    monkeypatch.setitem(sys.modules, "local_llm", _DirtyList)
    paths, reason = cd.ask_local_art_director("sys", _pool5(), "Story", limit=4)

    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert "/img/2.jpg" in paths and "/img/0.jpg" in paths
    assert reason, "a repaired answer is not a clean one; say so"


def test_a_complete_answer_is_left_alone(monkeypatch):
    """No top-up, no note, when the model did its job."""
    class _Good:
        @staticmethod
        def generate(*a, **k):
            return "[3,1,4,2]"

        @staticmethod
        def last_skip_reason():
            return None

    monkeypatch.setitem(sys.modules, "local_llm", _Good)
    paths, reason = cd.ask_local_art_director("sys", _pool5(), "Story", limit=4)
    assert paths == ["/img/3.jpg", "/img/1.jpg", "/img/4.jpg", "/img/2.jpg"]
    assert reason is None


def test_top_up_cannot_exceed_the_pool(monkeypatch):
    monkeypatch.setitem(sys.modules, "local_llm", _ShortList)
    paths, reason = cd.ask_local_art_director("sys", _pool5(), "Story", limit=9)
    assert len(paths) == 5, "only five candidates exist"
    assert reason
