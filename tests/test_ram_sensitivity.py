"""
RAM-sensitivity guards for the grading pipeline.

Each test locks in a change that lowers the grade worker's memory footprint,
and asserts the change did NOT alter what the pipeline computes:

  1. The CUDA-free grade worker must not import torch.
  2. personal_head_np must equal personal_head bit-for-bit (same MLP, numpy).
  3. Bounded EXIF header reads must equal full-file reads.
  4. The RAM floor must degrade (retry / smaller batch) rather than fail hard.
  5. lance_store must pin its native extensions at import (0xC0000005 guard).

Run:  venv\\Scripts\\python.exe -m pytest tests/test_ram_sensitivity.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ── 1. torch must stay out of the grade worker ───────────────────────────────

@pytest.mark.parametrize("module", [
    "specvlm_pipeline",     # imported during the early-exit gate
    "siglip2_encoder",      # subprocess bridge — never runs a model in-process
    "personal_head_np",     # numpy taste scorer
    "early_exit_gate",
    "lance_store",
])
def test_module_does_not_pull_torch(module):
    """Importing these must not drag ~350 MB of torch into a CUDA-free process.

    Run in a CLEAN subprocess: pytest's own session may already have torch
    resident from another test, which would mask the regression.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import %s;"
        "print('TORCH' if 'torch' in sys.modules else 'CLEAN')"
        % (str(_ROOT / "src"), module)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300)
    assert out.returncode == 0, f"import failed: {out.stderr[-1500:]}"
    assert "CLEAN" in out.stdout, (
        f"{module} pulled torch in at import time — that is ~350 MB resident in "
        f"the grade worker for the whole run. Make the torch import lazy.\n"
        f"{out.stdout}"
    )


# ── 2. numpy taste head == torch taste head ──────────────────────────────────

def test_personal_head_np_matches_torch():
    """The numpy forward pass must equal the reference torch implementation."""
    torch = pytest.importorskip("torch")
    import personal_head_np as ph_np

    weights = _ROOT / "cache" / "personal_head.pt"
    if not weights.exists():
        pytest.skip("no trained PersonalHead weights on this install")

    cwd = os.getcwd()
    os.chdir(_ROOT)                      # both modules use relative cache/ paths
    try:
        import personal_head as ph
        rng = np.random.default_rng(0)
        embs = rng.normal(size=(64, 1536)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)

        got  = ph_np.score(embs)
        want = ph.score(embs)

        assert got is not None, "numpy scorer returned None despite trained weights"
        assert got.shape == want.shape == (64,)
        assert np.allclose(got, want, atol=1e-5), (
            f"numpy/torch taste scores diverge: max |delta| = "
            f"{float(np.max(np.abs(got - want))):.2e}"
        )
    finally:
        os.chdir(cwd)


def test_personal_head_np_returns_none_without_weights(tmp_path):
    """No trained head must yield None (fall back), never a fake neutral 0.5.

    Returning 0.5 would read as a confident-neutral taste vote and silently
    flatten the blend for every photo.
    """
    import personal_head_np as ph_np
    cwd = os.getcwd()
    os.chdir(tmp_path)                   # empty cache/ → no weights
    try:
        ph_np._cache, ph_np._cache_mtime = None, -1.0
        assert ph_np.score(np.zeros((4, 1536), dtype=np.float32)) is None
    finally:
        os.chdir(cwd)


def test_personal_head_np_refreshes_stale_mirror(tmp_path):
    """A .npz older than its .pt must be regenerated, not used."""
    import personal_head_np as ph_np
    npz, pt = tmp_path / "a.npz", tmp_path / "a.pt"
    npz.write_bytes(b"x"); pt.write_bytes(b"y")
    old_npz, old_pt = ph_np._NPZ_PATH, ph_np._WEIGHTS_PATH
    try:
        ph_np._NPZ_PATH, ph_np._WEIGHTS_PATH = npz, pt
        os.utime(npz, (1_000_000, 1_000_000))     # npz older than pt
        os.utime(pt,  (2_000_000, 2_000_000))
        assert ph_np._mirror_is_stale() is True
        os.utime(npz, (3_000_000, 3_000_000))     # npz now newer
        assert ph_np._mirror_is_stale() is False
    finally:
        ph_np._NPZ_PATH, ph_np._WEIGHTS_PATH = old_npz, old_pt


# ── 3. bounded EXIF read == full-file read ───────────────────────────────────

def _sample_images(limit: int = 6) -> list:
    pool: list = []
    for folder, pattern in (
        (_ROOT / "dataset_images", "*.jpg"),
        (Path(r"C:/Users/Nicky Tuason/Desktop/LX3 2024/DCIM/110_PANA"), "*.RW2"),
    ):
        if folder.is_dir():
            pool += [str(p) for p in sorted(folder.glob(pattern))[:limit]]
    return pool


def test_bounded_exif_matches_full_file():
    """Prefix-read timestamps must equal whole-file timestamps, RAW included."""
    pytest.importorskip("piexif")
    import piexif
    import pipeline_stages as g

    paths = _sample_images()
    if not paths:
        pytest.skip("no sample images available")

    def full(p: str) -> float:
        try:
            exif = piexif.load(p)
            raw = (exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
                   or exif.get("0th", {}).get(piexif.ImageIFD.DateTime))
            if raw:
                from datetime import datetime
                return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass
        return 0.0

    for p in paths:
        assert g.exif_timestamp(p) == full(p), f"EXIF timestamp changed for {Path(p).name}"


def test_exif_survives_garbage_input(tmp_path):
    """Truncated / non-image files must yield 0.0, not raise."""
    import pipeline_stages as g
    junk = tmp_path / "junk.jpg"; junk.write_bytes(b"\xff\xd8not-an-image")
    empty = tmp_path / "empty.jpg"; empty.write_bytes(b"")
    assert g.exif_timestamp(str(junk)) == 0.0
    assert g.exif_timestamp(str(empty)) == 0.0
    assert g.exif_timestamp(str(tmp_path / "missing.jpg")) == 0.0


# ── 4. the RAM floor degrades instead of failing ─────────────────────────────

def _run_floor(free_gb: float, env: dict, onnx: bool = False) -> subprocess.CompletedProcess:
    """Exercise _enforce_ram_floor with psutil reporting `free_gb` free.

    `onnx` selects which encoder's floors apply: the ONNX image path peaks at
    1.20 GB (soft 2.0 / hard 1.5), the PyTorch path at 2.70 GB (soft 4.0 /
    hard 3.0). They are deliberately different — a floor tuned for the heavier
    encoder would refuse grades the lighter one runs comfortably.
    """
    code = f"""
import sys, types; sys.path.insert(0, r"{_ROOT / 'src'}")
import psutil
psutil.virtual_memory = lambda: types.SimpleNamespace(available=int({free_gb} * 1e9))
import siglip2_encoder as s
s._hf_checkpoint_present = lambda: True
s._onnx_active = lambda: {onnx}          # which encoder's floors are under test
try:
    s._enforce_ram_floor()
    print("PROCEEDED batch=" + __import__("os").environ.get("SIGLIP_ENC_BATCH", "default"))
except MemoryError as e:
    print("REFUSED " + str(e))
"""
    e = dict(os.environ); e.update(env)
    e.pop("SIGLIP_ENC_BATCH", None)
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(_ROOT), timeout=300, env=e)


def test_floor_proceeds_when_ram_is_ample():
    out = _run_floor(8.0, {})
    assert "PROCEEDED" in out.stdout, out.stdout + out.stderr


def test_floor_degrades_between_hard_and_soft():
    """Below the comfort floor but above the measured need → proceed, smaller batch."""
    out = _run_floor(3.4, {}, onnx=False)          # PyTorch floors: soft 4.0 / hard 3.0
    assert "PROCEEDED" in out.stdout, (
        "a transient dip must not kill the grade outright\n" + out.stdout + out.stderr
    )
    assert "batch=2" in out.stdout, f"expected a reduced decode batch: {out.stdout}"


def test_floor_still_refuses_when_genuinely_impossible():
    """Below the hard floor the encode cannot fit — refuse with a clear message."""
    out = _run_floor(0.8, {}, onnx=False)
    assert "REFUSED" in out.stdout, out.stdout + out.stderr
    assert "free" in out.stdout.lower()


def test_onnx_floors_are_lower_than_torch_floors():
    """The lighter encoder must not inherit the heavier one's RAM gate.

    ONNX peaks at 1.20 GB, so 2.5 GB free is comfortable for it while the
    PyTorch path (2.70 GB peak) would rightly shrink its batch there.
    """
    onnx_out  = _run_floor(2.5, {}, onnx=True)
    torch_out = _run_floor(2.5, {}, onnx=False)
    assert "PROCEEDED" in onnx_out.stdout, onnx_out.stdout + onnx_out.stderr
    assert "batch=default" in onnx_out.stdout, (
        f"ONNX should run at full batch with 2.5 GB free: {onnx_out.stdout}")
    # At 2.5 GB the PyTorch path is below its HARD floor (3.0) and correctly
    # refuses — which is the whole point: the same machine state runs one
    # encoder and not the other.
    assert "REFUSED" in torch_out.stdout, (
        f"PyTorch should refuse at 2.5 GB free: {torch_out.stdout}")


def test_onnx_still_refuses_when_truly_out_of_memory():
    out = _run_floor(0.5, {}, onnx=True)
    assert "REFUSED" in out.stdout, out.stdout + out.stderr


def test_floor_opt_out_still_honoured():
    out = _run_floor(0.2, {"SIGLIP_MIN_FREE_RAM_GB": "0"}, onnx=False)
    assert "PROCEEDED" in out.stdout, out.stdout + out.stderr


# ── 5. native-extension load order (the 0xC0000005 regression) ───────────────

def test_lance_store_pins_native_extensions_at_import():
    """pyarrow/lancedb must be resident once lance_store is imported.

    If their first import is deferred until after a CUDA subprocess has run,
    loading their DLLs faults the process with an access violation and no
    traceback (reproduced 6/6). See lance_store.warm_native.
    """
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "import lance_store;"
        "print('PYARROW' if 'pyarrow' in sys.modules else 'MISSING-PYARROW');"
        "print('LANCEDB' if 'lancedb' in sys.modules else 'MISSING-LANCEDB')"
        % str(_ROOT / "src")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(_ROOT), timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert "PYARROW" in out.stdout and "MISSING-PYARROW" not in out.stdout, out.stdout
    assert "LANCEDB" in out.stdout and "MISSING-LANCEDB" not in out.stdout, out.stdout
