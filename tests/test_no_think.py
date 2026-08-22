r"""
Hybrid reasoning models must not think on a selection task.

Measured on Qwen3-4B over the same 6 trials, ground truth known:

    thinking ON   6/6 correct   31.9 s/answer
    thinking OFF  6/6 correct    1.7 s/answer

Identical accuracy, 19x the time. The think block is pure tax here: the task
is "return one id", and the model reaches the same id either way. Shipping
with it on would have put a single Art Director call over the entire budget
for a Story run.

The failure this guards is silent: nothing errors when a thinking model thinks,
it is just slow, and slow reads as "the machine is busy".

Run:  venv\Scripts\python.exe -m pytest tests/test_no_think.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import local_llm  # noqa: E402


class _Named:
    def __init__(self, name):
        self.name = name


def test_thinking_suppressed_for_qwen3(monkeypatch):
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("Qwen_Qwen3-4B-Q4_K_M.gguf"))
    out = local_llm._suppress_thinking("You are an editor.")
    assert "/no_think" in out, out
    assert out.startswith("You are an editor."), "must not drop the caller's prompt"


def test_thinking_switch_not_added_for_other_models(monkeypatch):
    """A model that does not understand /no_think would just see stray text."""
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("LFM2.5-VL-1.6B-Q4_K_M.gguf"))
    assert local_llm._suppress_thinking("You are an editor.") == "You are an editor."


def test_no_double_append(monkeypatch):
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("Qwen_Qwen3-4B-Q4_K_M.gguf"))
    once = local_llm._suppress_thinking("Pick one. /no_think")
    assert once.count("/no_think") == 1, once


def test_handles_absent_system_prompt(monkeypatch):
    """Callers may pass system=None; the switch still has to land somewhere."""
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("Qwen_Qwen3-4B-Q4_K_M.gguf"))
    out = local_llm._suppress_thinking(None)
    assert out and "/no_think" in out


def test_unreadable_model_path_is_not_fatal(monkeypatch):
    """Probing the name must never be the thing that breaks generation."""
    def boom():
        raise OSError("no weights")
    monkeypatch.setattr(local_llm, "model_path", boom)
    assert local_llm._suppress_thinking("Pick one.") == "Pick one."


# ── the think block must not reach callers ───────────────────────────────────

def test_empty_think_block_is_stripped():
    """With /no_think, Qwen3 still emits an EMPTY <think></think> wrapper.

    Observed live: generate() returned '<think>\n\n</think>\n\nREADY'.
    creative_director strips it (a regex written for DeepSeek-R1), but
    local_llm is what suppresses thinking, so it owes callers clean text --
    otherwise every future caller has to know this quirk.
    """
    assert local_llm._strip_thinking("<think>\n\n</think>\n\nREADY") == "READY"


def test_populated_think_block_is_stripped():
    assert local_llm._strip_thinking(
        "<think>weighing options</think>\n[1,2,3]") == "[1,2,3]"


def test_text_without_a_think_block_is_untouched():
    assert local_llm._strip_thinking("[1,2,3]") == "[1,2,3]"


def test_unclosed_think_tag_does_not_eat_the_answer():
    """A truncated generation can leave <think> open. Dropping everything after
    it would discard the only content we have."""
    out = local_llm._strip_thinking("<think>ran out of tokens")
    assert "ran out of tokens" in out


def test_none_survives():
    assert local_llm._strip_thinking(None) is None
