"""
The process boundary, enforced rather than remembered.

The pipeline's stability rests on one rule: the grade worker is CUDA-FREE.
SigLIP, IQA and detection all run in isolated subprocesses, and if the PARENT
initialises a CUDA context, it faults with 0xC0000005 when a child exits — no
traceback, no clue.

That rule has been broken twice by code that looked harmless:
  * VRAMManager.purge_vram() calling ipc_collect()
  * tier_select.has_gpu() calling torch.cuda.is_available()

and once more subtly, by a guard written as
``is_available() and is_initialized()`` — which defeats itself, because
is_available() is the call that initialises, and it was evaluated first.

So the rule is checked two ways here: statically (no initialising call may
appear in a parent-side module) and dynamically (importing the parent-side
modules and exercising them must leave CUDA uninitialised).

Run:  venv\\Scripts\\python.exe -m pytest tests/test_process_boundary.py -v
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

# Modules that run INSIDE the grade worker (the parent). Anything here must
# never create a CUDA context.
PARENT_SIDE = [
    "run_profile.py", "tier_select.py", "lance_store.py", "catalog_store.py",
    "raw_support.py", "face_signals.py", "personal_head_np.py",
    "grade_pipeline_v2.py", "siglip2_encoder.py",
    # The jury/creative path runs INSIDE the server process, which also spawns
    # grade_runner.py as a CUDA subprocess:
    #   server.py:2070 -> creative_director.run_creative_direction
    #                  -> creative_director.py:1469 -> jury_engine
    # Both modules carried the full bug: a bare is_available() AND the
    # self-defeating `is_available() and is_initialized()` guard. They were
    # missed for so long only because this list is opt-in.
    "jury_engine.py", "creative_director_agent.py",
]

# Modules that ARE the isolated subprocess — CUDA is their job.
SUBPROCESS_SIDE = ["encode_worker.py", "iqa_worker.py"]

# Calls that create a CUDA context as a side effect of "just asking".
INITIALISING = re.compile(
    r"torch\.cuda\.(is_available|device_count|init|synchronize|current_device)\s*\(")


def _code_lines(path: Path):
    """Source lines with comments, docstrings and embedded subprocess source
    removed — those mention the forbidden calls precisely to explain them."""
    import io
    import tokenize
    out = {}
    try:
        with open(path, "rb") as fh:
            toks = list(tokenize.tokenize(fh.readline))
    except Exception:
        return out
    skip = set()
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            for ln in range(tok.start[0], tok.end[0] + 1):
                skip.add(ln)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if i not in skip:
            out[i] = line
    return out


@pytest.mark.parametrize("name", PARENT_SIDE)
def test_no_cuda_initialising_call_in_parent_side_module(name):
    p = _SRC / name
    if not p.exists():
        pytest.skip(f"{name} not present")
    bad = [(ln, s.strip()) for ln, s in _code_lines(p).items() if INITIALISING.search(s)]
    assert not bad, (
        f"{name} calls a CUDA-initialising function in the grade worker: {bad}. "
        f"Use torch.cuda.is_initialized(), or probe in a subprocess "
        f"(see tier_select._probe_gpu_subprocess).")


def test_guard_order_is_not_self_defeating():
    """`is_available() and is_initialized()` reads like a guard and is not one."""
    for name in PARENT_SIDE:
        p = _SRC / name
        if not p.exists():
            continue
        for ln, s in _code_lines(p).items():
            if "is_initialized" in s and "is_available" in s:
                assert s.index("is_initialized") < s.index("is_available"), (
                    f"{name}:{ln} evaluates is_available() first, which "
                    f"initialises CUDA before the guard can prevent it")


def test_importing_parent_modules_leaves_cuda_uninitialised():
    """The dynamic check: static analysis cannot see through a call chain."""
    code = (
        "import sys; sys.path.insert(0, r'%s');\n"
        "import run_profile, tier_select, lance_store, raw_support, face_signals\n"
        "p = run_profile.current()\n"
        "p.onnx_enabled(); p.encode_batch; p.ram_hard_gb; p.lance_table\n"
        "tier_select.select(free_gb=8.0)\n"
        "import torch\n"
        "print('INIT', torch.cuda.is_initialized())\n" % str(_SRC))
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=600)
    line = [l for l in r.stdout.splitlines() if l.startswith("INIT")]
    assert line, f"probe failed: {(r.stdout + r.stderr)[-500:]}"
    assert line[0] == "INIT False", (
        "building the profile and selecting a tier initialised CUDA in the "
        "parent — this is the 0xC0000005 crash class")


def test_subprocess_side_modules_are_allowed_to_use_cuda():
    """Guards the guard: if this ever fails, the rule was applied too broadly
    and the actual workers have been prevented from using the GPU."""
    found = False
    for name in SUBPROCESS_SIDE:
        p = _SRC / name
        if p.exists() and "torch.cuda" in p.read_text(encoding="utf-8"):
            found = True
    assert found, "no subprocess worker uses torch.cuda — is the GPU path gone?"
