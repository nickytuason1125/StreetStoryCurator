"""
Grades must be reproducible: the same photos must score the same every run.

This is a regression guard for a bug I introduced and then had to hunt down.
Chasing memory safety, four batch sizes were made to derive from *currently free
RAM*. Free RAM differs between runs, so the batch COMPOSITION differed, so GPU
kernels selected different algorithms, so embeddings and quality scores shifted
in their last bits — and borderline photos flipped grade. Two identical culls of
the same 514 photos disagreed on up to 76 of them.

Sizing that depends on free RAM is therefore FORBIDDEN for anything that feeds a
model batch. The memory argument that motivated it is also gone: draft decoding
cut per-image cost ~20x and the ONNX encoder dropped 2.70 GB -> 1.20 GB.

Each test below asserts a knob returns the SAME value regardless of how much RAM
psutil claims is free. Env overrides are still honoured — a user opting into a
smaller batch accepts the reproducibility cost knowingly.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_determinism.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# (module, function, call-arg) for every sizing knob that feeds a model batch.
_KNOBS = [
    ("dfine_detector",       "_chunk_size",       514),   # person-detection batch
    ("vision_grading_heads", "_iqa_chunk_size",   514),   # IQA decode window
    ("grade_pipeline_v2",    "_dedup_chunk_size", 514),   # dedup row block
]

_FREE_RAM_VALUES = [0.3, 1.0, 2.5, 6.0, 32.0]   # GB


def _value_under_fake_ram(module: str, func: str, arg: int, free_gb: float) -> str:
    """Call the knob in a clean process with psutil reporting `free_gb`."""
    code = f"""
import sys, types
sys.path.insert(0, r"{_ROOT / 'src'}")
import psutil
psutil.virtual_memory = lambda: types.SimpleNamespace(available=int({free_gb} * 1e9))
import {module} as m
print(m.{func}({arg}))
"""
    env = dict(os.environ)
    for k in ("FRAMEGRADE_DFINE_CHUNK", "FRAMEGRADE_IQA_CHUNK",
              "FRAMEGRADE_DEDUP_CHUNK", "SIGLIP_ENC_BATCH"):
        env.pop(k, None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300, env=env)
    assert out.returncode == 0, f"{module}.{func} failed: {out.stderr[-800:]}"
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("module,func,arg", _KNOBS)
def test_batch_size_does_not_depend_on_free_ram(module, func, arg):
    """A knob that changes with free RAM makes grades unreproducible."""
    seen = {free: _value_under_fake_ram(module, func, arg, free)
            for free in _FREE_RAM_VALUES}
    distinct = set(seen.values())
    assert len(distinct) == 1, (
        f"{module}.{func} returns different values depending on free RAM: {seen}\n"
        f"That changes batch composition between runs, which changes GPU kernel "
        f"selection, which flips borderline grades. Use a fixed size."
    )


def test_encode_batch_does_not_depend_on_free_ram():
    """Same rule for the SigLIP encode batch."""
    seen = {free: _value_under_fake_ram("siglip2_encoder", "_auto_enc_batch_probe", 0, free)
            for free in _FREE_RAM_VALUES} if False else None
    # _auto_enc_batch takes no argument; call it directly.
    vals = {}
    for free in _FREE_RAM_VALUES:
        code = f"""
import sys, types
sys.path.insert(0, r"{_ROOT / 'src'}")
import psutil
psutil.virtual_memory = lambda: types.SimpleNamespace(available=int({free} * 1e9))
import siglip2_encoder as s
print(s._auto_enc_batch())
"""
        env = dict(os.environ); env.pop("SIGLIP_ENC_BATCH", None)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=str(_ROOT), timeout=300, env=env)
        assert out.returncode == 0, out.stderr[-800:]
        vals[free] = out.stdout.strip().splitlines()[-1]
    assert len(set(vals.values())) == 1, (
        f"_auto_enc_batch varies with free RAM: {vals} — embeddings would differ "
        f"between runs")


@pytest.mark.parametrize("module,func,arg,env_var", [
    ("dfine_detector",       "_chunk_size",       514, "FRAMEGRADE_DFINE_CHUNK"),
    ("vision_grading_heads", "_iqa_chunk_size",   514, "FRAMEGRADE_IQA_CHUNK"),
    ("grade_pipeline_v2",    "_dedup_chunk_size", 514, "FRAMEGRADE_DEDUP_CHUNK"),
])
def test_env_override_still_works(module, func, arg, env_var):
    """Pinning must not remove the escape hatch for a memory-tight machine."""
    code = f"""
import sys; sys.path.insert(0, r"{_ROOT / 'src'}")
import {module} as m
print(m.{func}({arg}))
"""
    env = dict(os.environ); env[env_var] = "64"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300, env=env)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip().splitlines()[-1] == "64", (
        f"{env_var} was ignored: {out.stdout}")


def test_knobs_stay_within_a_sane_memory_budget():
    """Fixed does not mean unbounded — the sizes must still be affordable."""
    import dfine_detector, vision_grading_heads, grade_pipeline_v2
    assert dfine_detector._chunk_size(10_000) <= 32
    assert vision_grading_heads._iqa_chunk_size(10_000) <= 512
    assert grade_pipeline_v2._dedup_chunk_size(10_000) <= 1024


def test_small_inputs_are_not_padded_up():
    """A 5-photo folder must not request a 512-image block."""
    import dfine_detector, vision_grading_heads
    assert dfine_detector._chunk_size(5) <= 5
    assert vision_grading_heads._iqa_chunk_size(5) <= 16   # floor is 16
