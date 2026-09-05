"""Build the CPU-only engine flavor.

NVIDIA machines get the CUDA engine (16.6 GB, ~0.6 s/photo). Everyone else —
AMD/Intel GPUs, no GPU, small RAM — gets THIS: the same pipeline on CPU-only
torch. ~3-4 GB installed, ~11 s/photo. Same grading quality, same verdicts.

How it works:
  1. create venv-cpu/ next to venv/ (isolated: its torch must be the CPU wheel)
  2. install requirements.txt + torch/torchvision from the /whl/cpu index
  3. run the SAME FirstCut.spec with that interpreter (PyInstaller bundles
     whatever torch flavor the build venv holds — no spec changes needed)
  4. rename the exe to curator-api.exe and tag the dir .cpu-flavor
  5. zip into parts with the cpu flavor prefix (zip_engine.py --flavor cpu)

Usage:  venv\\Scripts\\python.exe scripts\\build_engine_cpu.py
~40 min end to end, most of it pip + PyInstaller. Safe to re-run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
CPU_PY = ROOT / "venv-cpu" / "Scripts" / "python.exe"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(list(map(str, cmd)), cwd=str(ROOT),
                       creationflags=NO_WINDOW, **kw)
    if r.returncode != 0:
        sys.exit(f"  ✗ step failed (exit {r.returncode}): {cmd[0]}")


def main() -> None:
    # 1. isolated build venv — the CUDA venv's torch must not be touched
    if not CPU_PY.exists():
        print("  creating venv-cpu …")
        run([sys.executable, "-m", "venv", str(ROOT / "venv-cpu")])

    # 2. requirements + CPU torch. Torch LAST with the /cpu index so pip's
    #    dependency resolution cannot "upgrade" it back to a CUDA wheel.
    print("  installing requirements (CPU venv) …")
    run([CPU_PY, "-m", "pip", "install", "--quiet",
         "-r", str(ROOT / "requirements.txt")])
    print("  installing CPU torch (≈2.5 GB download) …")
    run([CPU_PY, "-m", "pip", "install", "--quiet",
         "torch==2.5.1", "torchvision==0.20.1",
         "--index-url", TORCH_INDEX])

    # sanity: the CPU venv must NOT hold a CUDA build
    probe = subprocess.run(
        [CPU_PY, "-c", "import torch; print(torch.version.cuda)"],
        capture_output=True, text=True, creationflags=NO_WINDOW)
    if "None" not in probe.stdout:
        sys.exit(f"  ✗ venv-cpu has a CUDA torch ({probe.stdout.strip()}) — aborting")

    # 3. same spec, different interpreter
    print("  running PyInstaller (FirstCut.spec) with the CPU venv …")
    run([CPU_PY, "-m", "PyInstaller", "FirstCut.spec", "--noconfirm",
         "--distpath", "dist-cpu"])

    exe = ROOT / "dist-cpu" / "FirstCut" / "FirstCut.exe"
    want = exe.parent / "curator-api.exe"
    if exe.exists():
        exe.rename(want)
    if not want.exists():
        sys.exit("  ✗ engine exe missing after build")

    # 4. flavor tag — the shell's flavor probe and humans both look for this
    (want.parent / ".cpu-flavor").write_text(
        "CPU-only engine: same grading, ~11 s/photo. Built by build_engine_cpu.py.",
        encoding="utf-8")

    print("\n  ✓ CPU engine ready: dist-cpu/FirstCut/curator-api.exe")
    print("  next: python scripts/zip_engine.py --flavor cpu")


if __name__ == "__main__":
    main()