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
