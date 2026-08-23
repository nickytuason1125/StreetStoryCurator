r"""
The Judge's Verdict must not pay the thinking tax either.

Measured through the real endpoint: a Story run took 169.8s, of which the
Judge's Verdict was 104s -- for max_tokens=200. That is ~2 tokens/second, the
signature of a reasoning model thinking before it answers.

jury_engine builds its OWN Llama via _load_llm() instead of going through
local_llm, so local_llm._suppress_thinking never reached it. Same class as the
four modules that hardcoded a weight path and the two that held separate
instances of one model: a second path around the shared one.

Measured elsewhere on the same model, same task: thinking ON 31.9s/answer,
thinking OFF 1.7s, at identical accuracy 6/6.

Run:  venv\Scripts\python.exe -m pytest tests/test_jury_no_think.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import jury_engine  # noqa: E402
import local_llm  # noqa: E402


class _Named:
    def __init__(self, name):
        self.name = name


def test_prompts_carry_the_switch_for_a_thinking_model(monkeypatch):
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("Qwen_Qwen3-4B-Q4_K_M.gguf"))
    out = jury_engine._maybe_no_think("You are a juror. Answer in JSON.")
    assert "/no_think" in out


def test_other_models_are_left_alone(monkeypatch):
    monkeypatch.setattr(local_llm, "model_path",
                        lambda: _Named("LFM2.5-VL-1.6B-Q4_K_M.gguf"))
    p = "You are a juror."
    assert jury_engine._maybe_no_think(p) == p


def test_never_raises(monkeypatch):
    def boom():
        raise OSError("no weights")
    monkeypatch.setattr(local_llm, "model_path", boom)
    assert jury_engine._maybe_no_think("x") == "x"


def test_think_block_is_stripped_from_the_verdict():
    """Even suppressed, Qwen3 emits an empty <think></think> wrapper, and the
    verdict is shown to the user verbatim."""
    assert jury_engine._clean("<think>\n\n</think>\n\nA quiet sequence.") \
        == "A quiet sequence."
