r"""
Reliability tests for the structured-judge machinery (2026 protocol).

Covers the failure modes that must degrade GRACEFULLY in production:
  - Ollama unreachable -> structured returns None (no exception escapes)
  - transient Ollama failures -> bounded retry with backoff recovers
  - malformed / hostile schema responses -> filtered, never crash the run
  - prose fallback engages when structured mode fails
  - a UTF-8 BOM in config.json no longer silently disables config loading

No real model is needed: urllib is mocked at the boundary.

Run:  venv\Scripts\python.exe -m pytest tests/test_story_judge_reliability.py -v
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vision_story_mode as vsm


class _FakeResp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


def _ollama_reply(response_text: str) -> _FakeResp:
    return _FakeResp(json.dumps({"response": response_text}).encode())


def _packet():
    return [
        vsm.SlotResult(
            "Subject/Interaction", "a.jpg", 0.9, True, "Subject/Interaction", "ok",
            subject_bbox=[80, 600, 700, 980],
            anchor_bbox=[0, 0, 1000, 200], luminance=0.55),
    ]


# ------------------------------------------------- Ollama unreachable

def test_structured_verdict_returns_none_when_ollama_is_down(monkeypatch):
    def boom(req, timeout):
        raise OSError("connection refused")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    # the structured path retries 3x with backoff; shrink the sleeps for tests
    monkeypatch.setattr(vsm.time, "sleep", lambda s: None)
    assert vsm._structured_verdict(_packet(), "brief") is None


def test_structured_verdict_rejects_unparseable_ollama_reply(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _ollama_reply("not json at all"))
    monkeypatch.setattr(vsm.time, "sleep", lambda s: None)
    assert vsm._structured_verdict(_packet(), "brief") is None


# ------------------------------------------------- retry recovery

def test_structured_verdict_survives_transient_failures(monkeypatch):
    calls = {"n": 0}
    good = json.dumps({
        "summary": "Tense street frame.",
        "claims": [{"fact_ref": "S1_area", "text": "The argument dominates the frame",
                    "value": 23.6}],
    })
    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("transient")
        return _ollama_reply(good)
    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(vsm.time, "sleep", lambda s: None)
    verdict = vsm._structured_verdict(_packet(), "brief")
    assert verdict is not None and "23.6" in verdict
    assert calls["n"] == 3


# ------------------------------------------------- prose fallback

def test_generate_verdict_falls_back_to_prose_when_structured_fails(monkeypatch):
    monkeypatch.setattr(vsm, "_structured_verdict", lambda *a, **k: None)
    prose = "Grounded prose verdict citing the spatial facts of the packet."
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout: _ollama_reply(prose))
    old = vsm._STR_CFG.get("JUDGE_MODE")
    try:
        vsm._STR_CFG["JUDGE_MODE"] = "structured"
        verdict = vsm.generate_judges_verdict(_packet(), "brief")
    finally:
        if old is None:
            vsm._STR_CFG.pop("JUDGE_MODE", None)
        else:
            vsm._STR_CFG["JUDGE_MODE"] = old
    assert verdict == prose


def test_generate_verdict_uses_structured_result_when_available(monkeypatch):
    structured = "Structured verdict with fact S1_area = 23.6 (fact S1_area = 23.6)."
    monkeypatch.setattr(vsm, "_structured_verdict", lambda *a, **k: structured)
    def must_not_be_called(req, timeout):
        raise AssertionError("prose path must not run when structured succeeds")
    monkeypatch.setattr(urllib.request, "urlopen", must_not_be_called)
    old = vsm._STR_CFG.get("JUDGE_MODE")
    try:
        vsm._STR_CFG["JUDGE_MODE"] = "structured"
        verdict = vsm.generate_judges_verdict(_packet(), "brief")
    finally:
        if old is None:
            vsm._STR_CFG.pop("JUDGE_MODE", None)
        else:
            vsm._STR_CFG["JUDGE_MODE"] = old
    assert verdict == structured


# ------------------------------------------------- config resilience

def test_bom_in_config_json_no_longer_disables_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_bytes(b"\xef\xbb\xbf" + json.dumps({
        "DOMINANT_AREA_PCT": 20.0, "JUDGE_MODEL": "test-model", 
        "JUDGE_MODE": "structured",
    }).encode("utf-8"))
    monkeypatch.setattr(vsm, "_CONFIG_PATH", cfg)
    strings = vsm._load_str_config()
    assert strings["JUDGE_MODEL"] == "test-model"
    assert strings["JUDGE_MODE"] == "structured"
    thresholds = vsm._load_thresholds()
    assert thresholds["DOMINANT_AREA_PCT"] == 20.0


def test_missing_config_json_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(vsm, "_CONFIG_PATH", tmp_path / "nope.json")
    assert vsm._load_str_config() == {}
    assert vsm._load_thresholds()["DOMINANT_AREA_PCT"] == 20.0


# ------------------------------------------------- hostile schema responses

def test_build_structured_verdict_never_crashes_on_hostile_payloads():
    reg = vsm._fact_registry(_packet())
    hostiles = [
        None, "string", 42, [], {},
        {"summary": None, "claims": None},
        {"summary": "s", "claims": [None, 1, "x", {}]},
        {"summary": "s", "claims": [{"fact_ref": None, "text": None, "value": None}]},
        {"summary": "s", "claims": [{"fact_ref": "S1_area", "text": "t", "value": "NaN"}]},
        {"summary": "s", "claims": [{"fact_ref": "S1_area", "text": "t", "value": float("inf")}]},
    ]
    for payload in hostiles:
        out = vsm._build_structured_verdict(payload, reg)  # must not raise
        assert out is None or isinstance(out, str)

# ------------------------------------------------- circuit breaker (F1+F6)

def _reset_breaker():
    vsm._JUDGE_BREAKER["fail"] = 0


def test_circuit_breaker_opens_after_configured_failures(monkeypatch):
    _reset_breaker()
    monkeypatch.setitem(vsm._CFG, "JUDGE_BREAKER_TRIPS", 2)
    calls = {"n": 0}
    def down(req, timeout):
        calls["n"] += 1
        raise OSError("down")
    monkeypatch.setattr(urllib.request, "urlopen", down)
    monkeypatch.setattr(vsm.time, "sleep", lambda s: None)
    assert vsm._structured_verdict(_packet(), "brief") is None
    assert vsm._structured_verdict(_packet(), "brief") is None
    n_before = calls["n"]
    assert vsm._structured_verdict(_packet(), "brief") is None   # breaker open
    assert calls["n"] == n_before                                # Ollama untouched
    _reset_breaker()


def test_circuit_breaker_resets_on_success(monkeypatch):
    _reset_breaker()
    monkeypatch.setitem(vsm._CFG, "JUDGE_BREAKER_TRIPS", 3)
    vsm._judge_breaker_record(False)
    vsm._judge_breaker_record(False)
    assert not vsm._judge_breaker_open()
    vsm._judge_breaker_record(True)
    assert vsm._JUDGE_BREAKER["fail"] == 0
    _reset_breaker()


def test_over_slow_calls_trip_the_breaker(monkeypatch):
    _reset_breaker()
    monkeypatch.setitem(vsm._CFG, "JUDGE_BREAKER_TRIPS", 2)
    monkeypatch.setitem(vsm._CFG, "JUDGE_SLOW_S", 5.0)
    vsm._judge_breaker_record(True, elapsed=1.0)    # fine, resets
    assert vsm._JUDGE_BREAKER["fail"] == 0
    vsm._judge_breaker_record(True, elapsed=99.0)   # slow -> trips
    vsm._judge_breaker_record(True, elapsed=99.0)   # slow again -> open
    assert vsm._judge_breaker_open()
    _reset_breaker()


def test_best_of_n_election_prefers_more_grounded_sample(monkeypatch):
    _reset_breaker()
    reg = vsm._fact_registry(_packet())
    good = vsm._build_structured_verdict({
        "summary": "Good.",
        "claims": [
            {"fact_ref": "S1_area", "text": "a", "value": 23.6},
            {"fact_ref": "S1_hgap", "text": "b", "value": 400},
            {"fact_ref": "S1_lum", "text": "c", "value": 0.55},
        ]}, reg)
    weak = vsm._build_structured_verdict({
        "summary": "Weak.",
        "claims": [{"fact_ref": "S1_area", "text": "a", "value": 23.6}]}, reg)
    assert good is not None and weak is not None
    queue = [weak, good]                     # first sample weak, second good
    monkeypatch.setattr(vsm, "_structured_call", lambda *a, **k: queue.pop(0))
    monkeypatch.setitem(vsm._CFG, "JUDGE_SAMPLES", 2.0)
    old_mode = vsm._STR_CFG.get("JUDGE_MODE")
    try:
        vsm._STR_CFG["JUDGE_MODE"] = "structured"
        verdict = vsm.generate_judges_verdict(_packet(), "brief")
    finally:
        if old_mode is None:
            vsm._STR_CFG.pop("JUDGE_MODE", None)
        else:
            vsm._STR_CFG["JUDGE_MODE"] = old_mode
        _reset_breaker()
    assert verdict == good                   # the election chose the better one


def test_gatekeeper_check_flags_missing_model_files(monkeypatch, tmp_path):
    monkeypatch.setattr(vsm, "_QWEN_GGUF", tmp_path / "absent.gguf")
    monkeypatch.setattr(vsm, "_QWEN_MMPROJ", tmp_path / "absent.mmproj")
    missing = vsm._check_gatekeeper()
    assert missing is not None and len(missing) == 2
    (tmp_path / "ok.gguf").write_bytes(b"x")
    (tmp_path / "ok.mmproj").write_bytes(b"x")
    monkeypatch.setattr(vsm, "_QWEN_GGUF", tmp_path / "ok.gguf")
    monkeypatch.setattr(vsm, "_QWEN_MMPROJ", tmp_path / "ok.mmproj")
    assert vsm._check_gatekeeper() is None
