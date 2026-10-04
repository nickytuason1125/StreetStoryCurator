"""Phase A (detect_worker during the encode) may only ever be reused on an
EXACT slice match — the guarantee that it never changes a grade."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import grade_pipeline_v2 as gp  # noqa: E402


def test_planned_slices_match_iqa_resumable(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_IQA_SLICE", "3")
    assert gp._planned_iqa_slices(list("abcdefg")) == [list("abc"), list("def"), list("g")]
    monkeypatch.setenv("FIRSTCUT_IQA_SLICE", "1000")
    assert gp._planned_iqa_slices(list("abc")) == [list("abc")]


def test_lookup_only_on_exact_slice(monkeypatch):
    class _Done:
        def poll(self): return 0
    gp._PHASE_A.update({"proc": _Done(), "results": {
        ("a", "b"): {"person_detected": {"a": True, "b": False},
                     "subject_bboxes": {"a": [[0, 0, 1, 1]]}, "subject_sharpness": {}}}})
    try:
        assert gp._phase_a_lookup(["a", "b"])["person_detected"]["a"] is True
        assert gp._phase_a_lookup(["b", "a"]) is None          # order differs
        assert gp._phase_a_lookup(["a"]) is None               # subset (e.g. resume)
    finally:
        gp._PHASE_A.update({"proc": None, "results": None})


def test_no_phase_a_means_no_precomputed():
    gp._phase_a_reset()
    assert gp._phase_a_lookup(["a"]) is None
