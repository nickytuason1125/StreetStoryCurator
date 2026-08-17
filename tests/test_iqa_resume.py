"""
IQA must be resumable WITHOUT changing a single score.

A killed cull used to throw away the whole quality-scoring pass. It now runs in
slices with a checkpoint, so a re-run continues where it stopped. The safety
argument is that IQA's only cross-image step (_batch_normalize) is a FIXED
affine, not batch statistics — so a photo's score cannot depend on which slice
it lands in. These tests hold that argument to account.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_iqa_resume.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import grade_pipeline_v2 as g  # noqa: E402


# ── the property that makes slicing legitimate ───────────────────────────────

def test_batch_normalize_is_slice_independent():
    """If this ever becomes batch-relative, slicing silently changes scores."""
    from vision_grading_heads import _batch_normalize
    rng = np.random.default_rng(0)
    whole = rng.uniform(0.1, 0.9, 200).astype(np.float32)
    full = _batch_normalize(whole)
    sliced = np.concatenate([_batch_normalize(whole[i:i + 40]) for i in range(0, 200, 40)])
    assert np.allclose(full, sliced, atol=1e-7), (
        "IQA normalisation became batch-dependent — slicing/resume is no longer "
        "score-preserving and _iqa_resumable must be revisited"
    )


# ── the resume machinery, with the GPU subprocess faked out ──────────────────

def _fake_iqa(monkeypatch, calls: list, fail_after: int = 10**9):
    """Deterministic stand-in: score = 0.5 + (len(name) % 10)/100, per path."""
    def fake(image_paths, image_embeddings, prompt_embedding, clip_scores,
             genre_ref_embs, lum_stats, comp_eligible_paths, vlm_breakdowns):
        calls.append(list(image_paths))
        if len(calls) > fail_after:
            raise RuntimeError("simulated IQA subprocess crash")
        q = np.array([0.5 + (len(Path(p).name) % 10) / 100 for p in image_paths],
                     dtype=np.float32)
        return {"quality": q, "tech": q, "aesthetic": q,
                "breakdowns": [{"Technical": round(float(x), 3)} for x in q],
                "composition_overrides": {image_paths[0]: 0.85} if image_paths else {},
                "chiaroscuro_flags": {p: False for p in image_paths},
                "person_detected": {p: True for p in image_paths},
                "subject_bboxes": {}}
    monkeypatch.setattr(g, "_iqa_via_subprocess", fake)


def _args(n: int):
    paths = [f"C:/shoot/img_{i:04d}.jpg" for i in range(n)]
    return dict(image_paths=paths,
                image_embeddings=np.zeros((n, 8), dtype=np.float32),
                prompt_embedding=None,
                clip_scores=np.full(n, 0.5, dtype=np.float32),
                genre_ref_embs=None,
                lum_stats=[(128.0, 60.0)] * n,
                comp_eligible_paths=set(paths),
                vlm_breakdowns=[{} for _ in range(n)])


@pytest.fixture(autouse=True)
def _isolate_ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_iqa_ckpt_path",
                        lambda key: tmp_path / f"{key or 'default'}.json")


def test_sliced_result_matches_unsliced(monkeypatch):
    """The whole point: resume must not move a score."""
    calls_a, calls_b = [], []
    a = _args(50)

    _fake_iqa(monkeypatch, calls_a)
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "0")          # no slicing
    whole = g._iqa_resumable(ckpt_key="k_whole", **a)

    _fake_iqa(monkeypatch, calls_b)
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "7")          # many slices
    sliced = g._iqa_resumable(ckpt_key="k_sliced", **a)

    assert len(calls_a) == 1 and len(calls_b) > 5, "slicing did not engage"
    assert np.allclose(whole["quality"], sliced["quality"], atol=1e-7)
    assert whole["breakdowns"] == sliced["breakdowns"]
    assert whole["person_detected"] == sliced["person_detected"]


def test_resume_skips_completed_work(monkeypatch):
    """A crash mid-pass must not redo what already succeeded."""
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "10")
    a = _args(50)

    calls1 = []
    _fake_iqa(monkeypatch, calls1, fail_after=3)             # dies on the 4th slice
    with pytest.raises(RuntimeError):
        g._iqa_resumable(ckpt_key="k_resume", **a)
    done_first = sum(len(c) for c in calls1[:3])
    assert done_first == 30

    calls2 = []
    _fake_iqa(monkeypatch, calls2)                           # re-run
    out = g._iqa_resumable(ckpt_key="k_resume", **a)
    rescored = sum(len(c) for c in calls2)
    assert rescored == 20, f"redid work: {rescored} photos rescored, expected 20"
    assert len(out["quality"]) == 50
    assert all(q > 0.0 for q in out["quality"]), "resumed entries lost their scores"


def test_resumed_scores_equal_a_clean_run(monkeypatch):
    """Resuming must give the same answer as never having crashed."""
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "10")
    a = _args(40)

    _fake_iqa(monkeypatch, [])
    clean = g._iqa_resumable(ckpt_key="k_clean", **a)

    _fake_iqa(monkeypatch, [], fail_after=2)
    with pytest.raises(RuntimeError):
        g._iqa_resumable(ckpt_key="k_crash", **a)
    _fake_iqa(monkeypatch, [])
    resumed = g._iqa_resumable(ckpt_key="k_crash", **a)

    assert np.allclose(clean["quality"], resumed["quality"], atol=1e-7)
    assert clean["breakdowns"] == resumed["breakdowns"]


def test_order_is_preserved(monkeypatch):
    """Results must line up with the caller's path order, not slice order."""
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "6")
    a = _args(30)
    _fake_iqa(monkeypatch, [])
    out = g._iqa_resumable(ckpt_key="k_order", **a)
    expect = [0.5 + (len(Path(p).name) % 10) / 100 for p in a["image_paths"]]
    assert np.allclose(out["quality"], expect, atol=1e-7)


def test_corrupt_checkpoint_restarts_cleanly(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "10")
    (tmp_path / "k_bad.json").write_text("{{{ not json", encoding="utf-8")
    calls = []
    _fake_iqa(monkeypatch, calls)
    out = g._iqa_resumable(ckpt_key="k_bad", **_args(20))
    assert len(out["quality"]) == 20
    assert sum(len(c) for c in calls) == 20, "corrupt checkpoint was trusted"


def test_clear_removes_the_checkpoint(monkeypatch):
    monkeypatch.setenv("FRAMEGRADE_IQA_SLICE", "10")
    _fake_iqa(monkeypatch, [])
    g._iqa_resumable(ckpt_key="k_clr", **_args(20))
    assert g._iqa_ckpt_path("k_clr").exists()
    g._iqa_ckpt_clear("k_clr")
    assert not g._iqa_ckpt_path("k_clr").exists()
