"""
Running on hardware that is not this machine.

The provider chain (CUDA -> DirectML -> CoreML -> ROCm -> CPU) had never been
exercised off CUDA, and when it finally was, two things were wrong:

  * the giant ONNX graph on CPUExecutionProvider peaks at 6.5-7.6 GB and runs
    11-14 s/image — at the batch sizes in use it just fails with
    'bad allocation'. It passed the 1.5 GB ONNX RAM floor first, because that
    floor was measured on CUDA.
  * tier selection offered Pro on a machine with no GPU at all.

DirectML/CoreML/ROCm need hardware not available here, so these tests pin the
DECISIONS (which tier, which batch, which graph) rather than the arithmetic.
CPU-vs-CUDA numerical parity is covered separately by a real encode run.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_portability.py -v
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import tier_select  # noqa: E402


@pytest.fixture(autouse=True)
def _all_tiers_installed(monkeypatch):
    monkeypatch.setattr(tier_select, "available", lambda t: True)
    monkeypatch.setattr(tier_select, "_has_hf", lambda t: True)


# ── tier choice without a GPU ───────────────────────────────────────────────
@pytest.mark.parametrize("free_gb", [3.0, 5.0, 12.0, 64.0])
def test_no_gpu_never_selects_above_fast(free_gb):
    """Even with 64 GB free, Pro on a CPU is ~2 hours for a 514-photo cull."""
    tier, lbl, reason = tier_select.select(free_gb=free_gb, gpu=False)
    assert tier == "low" and lbl == "Fast"
    assert "no GPU" in reason


def test_gpu_ladder_is_unchanged():
    assert tier_select.select(free_gb=8.0, gpu=True)[0] == "high"
    assert tier_select.select(free_gb=2.5, gpu=True)[0] == "mid"
    assert tier_select.select(free_gb=1.5, gpu=True)[0] == "low"


def test_cpu_ram_requirement_is_higher_than_gpu():
    """CPU holds activations in system RAM; sizing it off the GPU number is
    what produced a bare 'bad allocation' instead of a usable message."""
    for tier in ("high", "mid", "low"):
        assert tier_select._CPU_RAM_NEED[tier] > tier_select.ram_need_gb(tier)


def test_no_gpu_and_too_little_ram_says_so_with_the_cpu_number():
    tier, lbl, reason = tier_select.select(free_gb=2.0, gpu=False)
    assert tier == "low"
    assert "3.0 GB" in reason and "CPU" in reason


def test_gpu_probe_never_initialises_cuda_in_this_process():
    """Regression: tier_select runs in the grade worker, the PARENT of the
    isolated encode subprocess. Calling torch.cuda.* here initialises CUDA in
    the parent, and the parent then faults 0xC0000005 when the encode child
    exits. Verified directly: the same cull exited 139 with a live
    torch.cuda.is_available() call and 0 with it bypassed. The probe must stay
    in a subprocess.
    """
    import subprocess
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "import tier_select as ts; ts.has_gpu();"
         "import torch; print('INIT', torch.cuda.is_initialized())"
         % str(_ROOT / "src")],
        capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("INIT")]
    assert line, f"probe process failed: {(r.stdout + r.stderr)[-300:]}"
    assert line[0] == "INIT False", (
        "has_gpu() initialised CUDA in-process — this reintroduces the "
        "0xC0000005 parent-fault crash")


def test_gpu_probe_result_is_cached_on_disk():
    """The subprocess probe costs seconds; hardware does not change per run."""
    import tier_select as ts
    ts.has_gpu()
    assert ts._GPU_PROBE_CACHE.exists()
    import json
    assert "gpu" in json.loads(ts._GPU_PROBE_CACHE.read_text(encoding="utf-8"))


def test_has_gpu_env_override():
    import os
    for val, expect in (("1", True), ("0", False), ("true", True), ("no", False)):
        os.environ["FRAMEGRADE_ASSUME_GPU"] = val
        try:
            assert tier_select.has_gpu() is expect
        finally:
            os.environ.pop("FRAMEGRADE_ASSUME_GPU", None)


def test_select_never_raises_on_any_combination():
    for gpu in (True, False):
        for free in (0.0, 0.5, 1.9, 3.0, 100.0):
            tier, lbl, reason = tier_select.select(free_gb=free, gpu=gpu)
            assert tier in tier_select._TIERS and lbl and reason


# ── encoder decisions, keyed on device ──────────────────────────────────────
def _worker(monkeypatch, *, gpu: bool, tier: str = "high", sel: str = ""):
    import os
    monkeypatch.setenv("SIGLIP_TIER", tier)
    monkeypatch.setenv("FRAMEGRADE_ENCODER", sel)
    monkeypatch.setenv("FRAMEGRADE_ASSUME_GPU", "1" if gpu else "0")
    sys.modules.pop("encode_worker", None)
    mod = importlib.import_module("encode_worker")
    mod._GPU_CACHE = None
    return mod


def test_batch_is_keyed_on_device_not_free_ram(monkeypatch):
    """Sizing the batch from free RAM made two identical culls disagree on
    47 of 514 photos. Keyed on device it is stable for a given machine."""
    assert _worker(monkeypatch, gpu=True)._default_batch() == 8
    assert _worker(monkeypatch, gpu=False)._default_batch() == 16


def test_giant_onnx_graph_is_not_used_without_a_gpu(monkeypatch, tmp_path):
    m = _worker(monkeypatch, gpu=False, tier="high")
    monkeypatch.setattr(m.os.path, "exists", lambda p: True)
    assert m._onnx_enabled() is False, (
        "the ONNX giant needs 6.5-7.6 GB and 11-14 s/img on CPU")


def test_onnx_graph_is_used_on_gpu(monkeypatch):
    m = _worker(monkeypatch, gpu=True, tier="high")
    monkeypatch.setattr(m.os.path, "exists", lambda p: True)
    assert m._onnx_enabled() is True


def test_onnx_text_gate_matches_the_vision_gate(monkeypatch):
    """Mismatched gates gave 1024-d images and 1536-d text — every probe
    dot-product a shape error, so the tier could not work at all."""
    for tier, expect in (("high", True), ("mid", False), ("low", False)):
        m = _worker(monkeypatch, gpu=True, tier=tier)
        monkeypatch.setattr(m.os.path, "exists", lambda p: True)
        assert m._onnx_text_enabled() is expect
        assert m._onnx_enabled() is expect, "text and vision must agree"
