"""One-shot memory self-check: what a cull costs vs what this machine has.

Run:  venv/Scripts/python.exe scripts/memory_report.py

Support diagnostics for the "can't grade / it froze" family of reports: prints
the measured cost of every plan, the machine's current free RAM, the plan the
admission ladder would pick right now, and the worker memory ceiling.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import memory_plan as mp          # noqa: E402
import run_profile as rp                   # noqa: E402


def main() -> None:
    spec = rp.spec_for(os.environ.get("SIGLIP_TIER", "") or rp.TIERS[0])
    print(f"tier          : {spec.tier} ({spec.label})  — set SIGLIP_TIER to pin")
    try:
        from siglip2_encoder import _default_ram_floor_gb
        print(f"encoder floor : {_default_ram_floor_gb():.1f} GB (hard) — refuse below this")
    except Exception as exc:
        print(f"encoder floor : unknown ({exc})")
    _ceil_env = os.environ.get("FIRSTCUT_WORKER_RAM_CEILING_GB")
    _ceil = float(_ceil_env) if _ceil_env else 3.5
    print(f"worker ceiling: {_ceil:g} GB per worker process (kernel-enforced)")

    print()
    print("plan costs (free RAM needed):")
    print(f"  full quality,  500 photos : {rp.required_ram_gb(500, scan_mode=False):.1f} GB")
    print(f"  full quality,   50 photos : {rp.required_ram_gb(50,  scan_mode=False):.1f} GB")
    print(f"  scan (any size)           : {rp.required_ram_gb(0,   scan_mode=True):.1f} GB")

    free = mp.free_ram_gb()
    if free is None:
        print("\nfree RAM      : COULD NOT BE MEASURED — admission will fail open")
        return
    commit = mp.commit_headroom_gb()
    print(f"free RAM now  : {free:.1f} GB (admission gates on the scarcer of RAM / commit)")
    if commit is not None:
        print(f"commit headroom: {commit:.1f} GB (paging-file room)")
        if commit < free - 0.5:
            print("  ⚠ the PAGEFILE is the bottleneck, not RAM: set a fixed pagefile")
            print("    (initial 24576 MB / max 32768 MB) in System → Advanced →")
            print("    Virtual Memory, then reboot. 'The paging file is too small'")
            print("    crashes are commit exhaustion, not closed-apps problems.")

    for n in (50, 500):
        plan = mp.plan_for(n, requested_scan=False)
        if plan is None:
            print(f"a {n}-photo cull : REFUSED — even a Scan does not fit. Close apps and retry.")
        else:
            tag = plan["plan"] + ("  (auto-downgraded)" if plan["degraded"] else "")
            print(f"a {n}-photo cull : {tag}  ({plan['need_gb']:.1f} GB needed)")

    if free < 2.0:
        print("\nadvice: close memory-heavy apps (browsers, editors). The header RAM")
        print("chip stays red until a cull can fit; grading degrades before it refuses.")


if __name__ == "__main__":
    main()
