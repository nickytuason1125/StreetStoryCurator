"""Launch-time system check — tell the photographer BEFORE the cull, not during.

The floors here are not guesses. Each one is enforced somewhere in the app
already; the problem was that nothing said so until a run was underway:

  1.5 GB free   the SigLIP encoder refuses to load below this and the failure
                looks like a hang (see the run-framegrade skill's gotchas)
  1.8 GB free   the grade floor the RAM chip and the pre-grade modal gate on
  5.0 GB free   what "comfortable" means: ~2 GB for the encode subprocess
                during model load, plus the grade worker's ~1 GB baseline,
                plus room for the OS to not swap

This module holds no state, touches no GPU, and imports nothing heavy — it is
safe to call before any model import. Never call torch.cuda here (see the
0xC0000005 note in CLAUDE.md); GPU probing belongs in a subprocess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Kept as module constants, and asserted by tests, so moving a floor in the app
# without moving it here fails the suite instead of silently making the docs a
# lie.
ENCODER_FLOOR_GB = 1.5
GRADE_FLOOR_GB = 1.8
COMFORTABLE_FREE_GB = 5.0
MODELS_DISK_GB = 22.0
DISK_HEADROOM_GB = 5.0


@dataclass(frozen=True)
class Assessment:
    level: str          # "ok" | "tight" | "insufficient" | "unknown"
    message: str
    blocking: bool      # always False — see the note below


def assess(total_gb: Optional[float],
           free_gb: Optional[float],
           disk_free_gb: Optional[float]) -> Assessment:
    """Classify this machine. NEVER blocking.

    Browsing an existing library, adjusting ratings and exporting a sequence
    all work fine with no memory to spare, because none of them loads an
    encoder. Refusing to start would remove those to prevent a failure that
    already has a clear error message at the moment it actually matters.
    """
    if total_gb is None or free_gb is None:
        return Assessment("unknown",
                          "Could not read system memory; skipping the launch check.",
                          False)

    notes: list[str] = []
    level = "ok"

    if free_gb < GRADE_FLOOR_GB:
        level = "insufficient"
        notes.append(
            f"Only {free_gb:.1f} GB of memory is free. Grading needs "
            f"{GRADE_FLOOR_GB} GB and the vision model will not load under "
            f"{ENCODER_FLOOR_GB} GB. Browsing and rating still work; close a "
            f"few apps before starting a cull."
        )
    elif free_gb < COMFORTABLE_FREE_GB:
        level = "tight"
        notes.append(
            f"{free_gb:.1f} GB free. A cull will run, but it may drop to a "
            f"lighter encoder tier. Close a few heavy apps for the full "
            f"pipeline ({COMFORTABLE_FREE_GB:.0f} GB free is comfortable)."
        )

    if disk_free_gb is not None and disk_free_gb < DISK_HEADROOM_GB:
        level = "insufficient" if level == "ok" else level
        notes.append(
            f"Only {disk_free_gb:.1f} GB of disk is free. Thumbnails, the "
            f"vector store and the catalog all grow with the library."
        )

    if not notes:
        return Assessment("ok",
                          f"System check OK - {free_gb:.1f} GB of "
                          f"{total_gb:.1f} GB memory free.",
                          False)
    return Assessment(level, " ".join(notes), False)


def read_and_assess() -> Assessment:
    """Take a live reading and classify it. Degrades to 'unknown', never raises."""
    total = free = disk = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total = vm.total / (1024 ** 3)
        free = vm.available / (1024 ** 3)
    except Exception:
        pass
    try:
        import shutil
        from pathlib import Path
        disk = shutil.disk_usage(Path(__file__).resolve().parent.parent).free / (1024 ** 3)
    except Exception:
        pass
    return assess(total, free, disk)
