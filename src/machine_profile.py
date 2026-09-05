"""MachineProfile — one authoritative hardware/OS capability report.

Every part of FirstCut that needs to know "what kind of machine am I on"
(the setup wizard, the engine banner, the shell's engine-flavor choice)
asks this module instead of re-implementing detection. One process spawn
for the GPU query (WMI), cached to a marker file, so repeated calls cost
nothing.

The verdict drives real behavior:
  "full"         x64 + NVIDIA CUDA-class GPU  -> CUDA engine, ~0.6 s/photo
  "cpu-flavor"   x64, non-CUDA GPU or none   -> small CPU engine, ~11 s/photo
  "unsupported"  32-bit OS, pre-Win10, or ARM64-Windows-with-no-engine-yet
                 -> the shell refuses to download and says why

FIRSTCUT_FORCE_CPU=1 simulates a GPU-less machine (used by the verification
harness to prove the CPU flavor path on a CUDA box without uninstalling the
driver).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_NO_WIN = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@dataclass(frozen=True)
class MachineProfile:
    os_name: str          # "windows" | "macos" | "linux" | other
    os_version: str
    arch: str             # platform.machine(): AMD64 | ARM64 | x86 | arm64 ...
    gpu_vendor: str       # "nvidia" | "amd" | "intel" | "cpu" (none found)
    gpu_name: str         # raw adapter name, for the checklist display
    vram_gb: float        # best-effort (WMI AdapterRAM is unreliable >4 GB)
    ram_gb: float
    disk_free_gb: float
    cuda_capable: bool
    verdict: str          # "full" | "cpu-flavor" | "unsupported"
    reason: str           # human sentence for the checklist / refusal screen

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _gpu_via_wmi() -> tuple[str, float, str]:
    """(vendor, vram_gb, adapter_name) via a PowerShell one-shot.

    Returns ("cpu", 0.0, ...) when nothing NVIDIA/AMD/Intel is found or the
    query fails — WMI absent/broken must degrade to the CPU verdict, never
    crash the wizard or the shell.
    """
    try:
        ps = shutil.which("powershell") or "powershell"
        out = subprocess.run(
            [ps, "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM | ConvertTo-Json"],
            capture_output=True, text=True, timeout=20,
            creationflags=_NO_WIN,
        ).stdout
        data = json.loads(out or "{}")
        cards = data if isinstance(data, list) else [data]
    except Exception:
        return "cpu", 0.0, "GPU query failed"

    rank = {"nvidia": 3, "amd": 2, "intel": 1, "cpu": 0}
    best_vendor, best_vram, best_name = "cpu", 0.0, "Integrated / unknown adapter"
    for card in cards:
        if not isinstance(card, dict):
            continue
        name = str(card.get("Name", ""))
        try:
            vram = float(card.get("AdapterRAM") or 0) / 1e9
        except Exception:
            vram = 0.0
        low = name.lower()
        if "nvidia" in low or "geforce" in low or "quadro" in low or "rtx" in low:
            vendor = "nvidia"
        elif "amd" in low or "radeon" in low:
            vendor = "amd"
        elif "intel" in low or "arc" in low:
            vendor = "intel"
        else:
            vendor = "cpu"
        # Prefer discrete: NVIDIA always wins; otherwise the most VRAM.
        if rank[vendor] > rank[best_vendor] or (
                vendor == best_vendor and vram > best_vram):
            best_vendor, best_vram, best_name = vendor, vram, name or best_name
    return best_vendor, best_vram, best_name


def _cache_path() -> Path:
    base = os.environ.get("FIRSTCUT_DATA_DIR") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "FirstCut" / "machine.json"


def detect(force: bool = False) -> MachineProfile:
    """Build (or reuse the cached) capability profile.

    Cached because the GPU WMI query spawns a process; hardware does not
    change between the wizard's checklist and the shell's flavor decision
    minutes later. force=True re-runs the queries.
    """
    cache = _cache_path()
    if not force and cache.exists():
        try:
            return MachineProfile(**json.loads(cache.read_text(encoding="utf-8")))
        except Exception:
            pass  # stale/corrupt cache — rebuild below

    force_cpu = os.environ.get("FIRSTCUT_FORCE_CPU", "").strip() == "1"
    os_name = _os_name()
    arch = platform.machine().upper()

    ram_gb, disk_free_gb = 0.0, 0.0
    try:
        import psutil as _ps  # already a dependency of the app
        ram_gb = _ps.virtual_memory().total / 1e9
    except Exception:
        pass
    try:
        disk_free_gb = shutil.disk_usage(str(Path.home())).free / 1e9
    except Exception:
        pass

    if force_cpu:
        gpu_vendor = "cpu"
        vram = 0.0
        gpu_name = "simulated CPU-only (FIRSTCUT_FORCE_CPU=1)"
    elif os_name == "windows":
        gpu_vendor, vram, gpu_name = _gpu_via_wmi()
    else:
        gpu_vendor, vram, gpu_name = "cpu", 0.0, "no WMI (non-Windows)"

    cuda_capable = gpu_vendor == "nvidia" and vram >= 4 and not force_cpu

    # ── verdict ───────────────────────────────────────────────────────────
    if arch not in ("AMD64", "X64", "X86_64"):
        verdict = "unsupported"
        reason = (f"{arch} is not supported by this build yet — FirstCut ships "
                  "for 64-bit Intel/AMD Windows. An ARM build is planned.")
    elif os_name != "windows":
        verdict = "unsupported"
        reason = (f"{os_name.capitalize()} builds are planned; this installer "
                  "targets 64-bit Windows 10/11.")
    elif os_name == "windows" and sys.getwindowsversion().major < 10:
        verdict = "unsupported"
        reason = "Windows 10 or newer is required."
    elif cuda_capable:
        verdict = "full"
        reason = f"{gpu_name} — GPU acceleration active."
    else:
        verdict = "cpu-flavor"
        reason = ("No CUDA-capable NVIDIA GPU detected — the compact CPU engine "
                  "will be used (full quality, slower culling).")

    profile = MachineProfile(
        os_name=os_name, os_version=platform.version(), arch=arch,
        gpu_vendor=gpu_vendor, gpu_name=gpu_name, vram_gb=vram,
        ram_gb=ram_gb, disk_free_gb=disk_free_gb,
        cuda_capable=cuda_capable, verdict=verdict, reason=reason,
    )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(profile.as_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass  # read-only data dir — detection still worked
    return profile


def engine_flavor(profile: MachineProfile | None = None) -> str:
    """Which engine flavor the shell should download: "cuda" | "cpu" | "none"."""
    p = profile or detect()
    if p.verdict == "unsupported":
        return "none"
    return "cuda" if p.verdict == "full" else "cpu"


if __name__ == "__main__":
    print(json.dumps(detect(force=True).as_dict(), indent=2))