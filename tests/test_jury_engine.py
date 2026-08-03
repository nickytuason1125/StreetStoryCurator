"""
Jury reliability.

The jury path had no tests at all, which is how a grammar that crashed
llama.cpp outright went unnoticed: _run_persona caught the crash, retried
unconstrained, and the module kept reporting success while the "evidence-first,
schema guaranteed" property in its own docstring was not in force.

These tests use a FAKE llm. Loading a real GGUF would make them slow enough to
skip, and every property here is about the module's own logic - parsing,
validation, degradation - not about any model's output.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_jury_engine.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import jury_engine as je                                    # noqa: E402
from signal_validator import Claim, validate_claims          # noqa: E402


class FakeLLM:
    """Returns canned completions; records whether a grammar was passed."""

    def __init__(self, replies, fail_on_grammar=False):
        self.replies = list(replies)
        self.fail_on_grammar = fail_on_grammar
        self.grammar_calls = 0
        self.plain_calls = 0

    def __call__(self, prompt, grammar=None, **kw):
        if grammar is not None:
            self.grammar_calls += 1
            if self.fail_on_grammar:
                raise OSError("exception: access violation reading 0x0")
        else:
            self.plain_calls += 1
        text = self.replies.pop(0) if self.replies else "{}"
        return {"choices": [{"text": text}]}


def _verdict_json(score=0.7, aspect="Composition", slot=0, value=0.50):
    return json.dumps({"verdict": "Slot 0 carries the sequence.", "score": score,
                       "cited_aspect": aspect, "cited_slot": slot,
                       "cited_value": value})


# ── parsing ───────────────────────────────────────────────────────────────────

def test_parses_a_well_formed_verdict():
    v = je._parse_verdict(_verdict_json(), "Purist")
    assert v["score"] == 0.7
    assert v["cited_aspect"] == "Composition"
    assert v["cited_slot"] == 0
    assert v["cited_value"] == 0.50


def test_prose_instead_of_json_is_rejected_not_guessed():
    """Unconstrained small models return prose. That must yield None, not a
    fabricated verdict - a made-up score would enter the narrative silently."""
    assert je._parse_verdict("The sequence flows nicely from opener to close.",
                             "Purist") is None


def test_think_preamble_is_stripped():
    raw = "<think>let me consider the slots</think>" + _verdict_json()
    v = je._parse_verdict(raw, "Purist")
    assert v is not None and v["score"] == 0.7


def test_invented_aspect_name_is_dropped():
    """The model may only cite aspects that exist. An unknown name becomes
    None rather than being passed to the validator as if it were real."""
    v = je._parse_verdict(_verdict_json(aspect="Vibes"), "Purist")
    assert v is not None
    assert v["cited_aspect"] is None


def test_out_of_range_score_is_clamped():
    assert je._parse_verdict(_verdict_json(score=9.5), "P")["score"] == 1.0
    assert je._parse_verdict(_verdict_json(score=-3.0), "P")["score"] == 0.0


def test_truncated_json_yields_none():
    assert je._parse_verdict('{"verdict":"good","score":0.7,"cited_asp',
                             "Purist") is None


# ── grammar ───────────────────────────────────────────────────────────────────

def test_grammar_builds():
    """The hand-written GBNF crashed llama.cpp's sampler. The schema-derived
    grammar must actually build, or the schema guarantee is fiction again."""
    pytest.importorskip("llama_cpp")
    je._grammar = None
    g = je._load_grammar()
    assert g is not None, "grammar failed to build - jury runs unconstrained"


def test_schema_pins_the_aspect_enum():
    """A model must not be able to invent an aspect at the grammar level."""
    enum = je._JURY_SCHEMA["properties"]["cited_aspect"]["enum"]
    assert set(enum) == je._VALID_ASPECTS | {"none"}


def test_grammar_failure_does_not_latch_forever(monkeypatch):
    """A transient grammar failure must not disable grammar for the whole
    process. This latched via a module global, so one blip in a long-running
    server permanently downgraded every later verdict."""
    monkeypatch.setattr(je, "_grammar_fails", 0)
    monkeypatch.setattr(je, "_grammar", object())    # non-None, so it is used

    bad = FakeLLM([_verdict_json()], fail_on_grammar=True)
    je._run_persona(bad, je._PERSONAS[0], "0:opener:a.jpg:score=0.5:Composition=0.50",
                    "brief", "street", "color")
    assert bad.plain_calls == 1, "must fall back to unconstrained on failure"

    good = FakeLLM([_verdict_json()])
    je._run_persona(good, je._PERSONAS[0], "0:opener:a.jpg:score=0.5:Composition=0.50",
                    "brief", "street", "color")
    assert good.grammar_calls == 1, (
        "grammar must be retried on a later call, not disabled for the process")


def test_synthesis_uses_the_grammar(monkeypatch):
    """The synthesis round feeds the final narrative, so it needs the same
    schema guarantee as the personas. It was calling the model unconstrained."""
    pytest.importorskip("llama_cpp")
    monkeypatch.setattr(je, "_grammar_fails", 0)
    monkeypatch.setattr(je, "_grammar", object())
    fake = FakeLLM([_verdict_json(score=0.6)])
    je._run_synthesis(fake, [{"persona": "P", "score": 0.4, "verdict": "x"},
                             {"persona": "S", "score": 0.9, "verdict": "y"}], "brief")
    assert fake.grammar_calls == 1, "synthesis must be grammar-constrained too"


# ── validation wiring ─────────────────────────────────────────────────────────

def test_validator_rejects_a_fabricated_value():
    aspects = [{"Composition": 0.41, "Lighting": 0.74}]
    ok = validate_claims([Claim("t", "Composition", 0.45, 0)], aspects)
    bad = validate_claims([Claim("t", "Composition", 0.95, 0)], aspects)
    assert ok.passed
    assert not bad.passed


def test_validator_rejects_an_out_of_range_slot():
    aspects = [{"Composition": 0.41}]
    assert not validate_claims([Claim("t", "Composition", 0.41, 7)], aspects).passed


def test_panel_degrades_to_empty_when_the_model_is_absent(monkeypatch):
    """No GGUF must mean 'no jury', never a crash in the creative-direction run."""
    monkeypatch.setattr(je, "_load_llm", lambda: None)
    verdicts, rejudged = je.run_jury_panel([], "brief", [], None, [])
    assert verdicts == [] and rejudged is False


def test_panel_excludes_verdicts_that_fail_validation(monkeypatch):
    """A hallucinated citation must not reach the narrative."""
    imgs = [{"filename": "a.jpg", "Composition": 0.41, "Lighting": 0.74}]
    monkeypatch.setattr(je, "_load_llm",
                        lambda: FakeLLM([_verdict_json(value=0.99)] * 3))
    monkeypatch.setattr(je, "unload", lambda: None)
    verdicts, _ = je.run_jury_panel(imgs, "brief", ["opener"], None, [0.5])
    assert verdicts == [], "a verdict citing 0.99 against a real 0.41 must be dropped"
