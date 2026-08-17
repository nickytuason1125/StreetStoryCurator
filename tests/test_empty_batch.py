"""Locks for the empty-batch crash class.

A 785-folder cull died three times on the same shape of bug, each time from a
different line:

  1. ``vlm_scores_rated.min()`` on a zero-length array
  2. ``embs[np.array(to_rate_indices)]`` where the empty list became float64
  3. ``scores_arr.min()`` after every remaining row was dropped as unreadable

All three had one trigger: a folder where nothing survives to be scored. That is
a NORMAL outcome, not an error — a directory holding only AppleDouble ``._*``
sidecars produces it, and so does a folder whose images are already cached. The
first two failures were inside diagnostic ``print`` statements, so a logging line
was killing a multi-hour job.

These tests pin the trigger, the NumPy behaviour the fix depends on, and the
presence of the bail-out itself.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_empty_batch.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pipeline_stages as ps          # noqa: E402

_V2_SRC = (_ROOT / "src" / "grade_pipeline_v2.py").read_text(encoding="utf-8")


# ── 1. The trigger ───────────────────────────────────────────────────────────

def test_all_rows_unreadable_empties_the_batch():
    """The exact condition that killed the run: every file unreadable.

    E:\\1017\\airy held 18 new files, all AppleDouble sidecars, all of them
    zero-vector. The drop then legitimately returns n == 0 and everything
    downstream has to cope.
    """
    paths = [f"._DSC{i}.jpg" for i in range(18)]
    embs = np.zeros((18, 8), dtype=np.float32)      # all unreadable

    kept, kept_embs, n = ps.drop_unreadable_rows(paths, embs, np)

    assert n == 0
    assert kept == []
    assert kept_embs.shape[0] == 0


def test_partial_unreadable_keeps_the_good_rows():
    """The drop must not be all-or-nothing — only the zero vectors leave."""
    paths = ["good_a.arw", "._sidecar.jpg", "good_b.arw"]
    embs = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    kept, kept_embs, n = ps.drop_unreadable_rows(paths, embs, np)

    assert n == 2
    assert kept == ["good_a.arw", "good_b.arw"]
    assert kept_embs.shape == (2, 2)


# ── 2. Why the index fix works ───────────────────────────────────────────────

def test_empty_index_must_carry_an_integer_dtype():
    """``np.array([])`` is float64, and NumPy rejects float arrays as indices.

    This is the whole reason the second crash existed. If a future change
    reverts to the bare ``np.array(...)`` form, this test explains why the
    resulting error message mentions dtypes rather than emptiness.
    """
    embs = np.zeros((5, 4), dtype=np.float32)

    with pytest.raises(IndexError, match="integer"):
        _ = embs[np.array([])]

    assert embs[np.asarray([], dtype=np.intp)].shape == (0, 4)
    assert embs[[]].shape == (0, 4)          # a plain empty list is fine


def test_vectorised_scoring_is_shape_safe_at_zero_rows():
    """Axis reductions survive an empty batch; bare reductions do not.

    The scoring block runs matrix maths over M rated images. At M == 0 that
    maths is fine — it is only the identity-less reductions in the summary
    prints that blow up, which is why guarding the prints was sufficient.
    """
    empty = np.zeros((0, 5), dtype=np.float32)
    assert empty.max(axis=1, keepdims=True).shape == (0, 1)
    assert np.linalg.norm(np.zeros((0, 16), dtype=np.float32), axis=1).shape == (0,)

    with pytest.raises(ValueError, match="zero-size array"):
        _ = np.zeros(0, dtype=np.float32).min()


# ── 3. The fixes are still in the source ─────────────────────────────────────

def test_run_v2_bails_out_when_the_batch_empties():
    """A zero-length batch must return, not walk the rest of the pipeline.

    Guarding each print in turn only moves the crash to the next reduction;
    the degenerate walk is the actual defect.
    """
    assert "if n == 0:" in _V2_SRC, (
        "The n == 0 bail-out after drop_unreadable_rows is gone. Without it a "
        "folder of unreadable files walks the whole scoring pipeline with empty "
        "arrays and dies in a diagnostic print."
    )
    bail = _V2_SRC.split("drop_unreadable_rows(paths, embs, np)", 1)[1][:1500]
    assert "if n == 0:" in bail, "The bail-out is no longer directly after the drop."
    assert "pipeline" in bail, "The bail-out must still return a well-formed result dict."


def test_no_unguarded_integer_index_on_the_rated_subset():
    """``np.array(to_rate_indices)`` must not come back in executable code."""
    code = re.sub(r"#.*", "", _V2_SRC)          # strip comments; they cite the old form
    assert "np.array(to_rate_indices)" not in code, (
        "Found a bare np.array(to_rate_indices). Empty -> float64 -> IndexError. "
        "Use the pre-built _rate_idx (np.intp) instead."
    )


@pytest.mark.parametrize("var", ["vlm_scores_rated", "scores_arr"])
def test_summary_reductions_are_size_guarded(var: str):
    """No diagnostic print may call .min() on a possibly-empty array.

    Both of these took down a multi-hour cull from inside a logging statement.
    """
    for m in re.finditer(rf"{re.escape(var)}\.min\(\)", _V2_SRC):
        window = _V2_SRC[max(0, m.start() - 600):m.start()]
        assert f"{var}.size" in window or "if n == 0:" in window, (
            f"{var}.min() at offset {m.start()} is not protected by a .size check. "
            f"An empty batch makes it raise 'zero-size array to reduction operation'."
        )
