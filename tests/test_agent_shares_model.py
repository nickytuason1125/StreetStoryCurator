r"""
One text model per process, loaded once.

creative_director_agent built its OWN Llama from a hardcoded path
(models/deepseek-r1-8b-q5.gguf) while local_llm held another instance of the
same file. A single Story run therefore did:

    load agent GGUF -> rule set -> director brief
      -> unload_agent_model()
      -> load local_llm GGUF -> art director selection

Two multi-GB instances of one file, and a load/unload/reload cycle inside one
run, on a laptop where the model barely fits at all.

The hardcoded path is also why the DeepSeek weights could not simply be deleted
after the model swap: the registry pointed at Qwen3-4B while this module still
reached for DeepSeek by name.

Run:  venv\Scripts\python.exe -m pytest tests/test_agent_shares_model.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import creative_director_agent as cda  # noqa: E402
import local_llm  # noqa: E402


def test_no_hardcoded_weight_path():
    """A second place naming a model file is a second place to forget."""
    assert not hasattr(cda, "_GGUF"), (
        "the agent must resolve its model through model_registry, not by name")


def test_no_second_model_instance():
    assert not hasattr(cda, "_llm_singleton"), (
        "two instances of one multi-GB file is the bug this removes")


def test_unload_does_not_evict_the_shared_model():
    """unload_agent_model() ran mid-pipeline, immediately before the art
    director needed the very same weights. Evicting a SHARED model there would
    force a reload seconds later -- strictly worse than doing nothing."""
    sentinel = object()
    local_llm._llm = sentinel
    try:
        cda.unload_agent_model()
        assert local_llm._llm is sentinel, "the shared model must survive"
    finally:
        local_llm._llm = None


def test_rule_set_still_works_without_a_model(monkeypatch):
    """No weights installed is a supported state, not an error: the rule set
    falls back to keyword parsing."""
    monkeypatch.setattr(local_llm, "generate", lambda *a, **k: None)
    rs = cda.generate_rule_set("rain, quiet streets, no people")
    assert set(rs) >= {"HARD_FILTER_PEOPLE", "GEOMETRIC_PRIORITY", "LIGHTING_MOOD"}
