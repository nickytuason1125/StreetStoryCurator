#!/usr/bin/env python3
"""
calibrate_backend.py — Telemetry and calibration script.

Decouples pipeline thresholds from source code:
  1. Parse final_story_manifest.json for approved_by_vision=True entries.
  2. Compute per-slot mean/std for area_pct and h_gap.
  3. Derive calibrated thresholds from real-world data.
  4. Write config.json atomically — vision_story_mode.py reads this at runtime.
  5. Write calibration_telemetry.json for canvas_renderer.py.

Usage:
    python calibrate_backend.py
    python calibrate_backend.py --dry-run
    python calibrate_backend.py --manifest path/to/other.json
    python calibrate_backend.py --config-out path/to/config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT           = Path(__file__).parent
MANIFEST_PATH  = ROOT / "final_story_manifest.json"
CONFIG_PATH    = ROOT / "config.json"
TELEMETRY_PATH = ROOT / "calibration_telemetry.json"

SLOT_NAMES = ["Opener", "Subject/Interaction", "Detail/Accent", "Closer/Resolution"]

# Fallback values written to config.json when calibration data is insufficient.
# These match the defaults baked into vision_story_mode.py._THRESHOLD_DEFAULTS.
_FALLBACKS: dict[str, float] = {
    "DOMINANT_AREA_PCT": 20.0,
    "H_CONTRAST_THRESH": 161.0,
    "OPENER_MAX_AREA":   30.0,
}


# =============================================================================
# 1. Load manifest → approved entries
# =============================================================================

def load_approved(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        print(f"[calibrate] ERROR: manifest not found at {manifest_path}")
        sys.exit(1)

    with manifest_path.open(encoding="utf-8") as f:
        data = json.load(f)

    sequence = data.get("sequence", [])
    approved = [e for e in sequence if e.get("approved_by_vision")]

    print(f"[calibrate] Manifest : {manifest_path.name}")
    print(f"[calibrate] Total    : {len(sequence)} entries")
    print(f"[calibrate] Approved : {len(approved)} entries (approved_by_vision=true)")

    if not approved:
        print("[calibrate] WARNING: no approved entries — fallback thresholds will be written.")

    return approved


# =============================================================================
# 2. Per-slot statistics
# =============================================================================

def compute_slot_stats(entries: list[dict]) -> dict[str, dict]:
    """
    Returns { slot_name: { 'area_pct': {mean, std, n, values},
                           'h_gap':    {mean, std, n, values} } }
    """
    buckets: dict[str, dict[str, list]] = {}
    for e in entries:
        slot  = e.get("assigned_slot", "Unknown")
        facts = e.get("spatial_facts", {})
        if slot not in buckets:
            buckets[slot] = {"area_pct": [], "h_gap": []}
        a = facts.get("subject_area_pct")
        h = facts.get("h_gap")
        if a is not None:
            buckets[slot]["area_pct"].append(float(a))
        if h is not None:
            buckets[slot]["h_gap"].append(int(h))

    stats: dict[str, dict] = {}
    for slot in sorted(buckets.keys()):
        stats[slot] = {}
        for metric, nums in buckets[slot].items():
            n = len(nums)
            if n == 0:
                stats[slot][metric] = {"mean": None, "std": None, "n": 0, "values": []}
            elif n == 1:
                stats[slot][metric] = {
                    "mean": round(nums[0], 2), "std": 0.0, "n": 1, "values": nums,
                }
            else:
                arr  = np.array(nums, dtype=float)
                lo   = float(np.percentile(arr, 10))
                hi   = float(np.percentile(arr, 90))
                core = arr[(arr >= lo) & (arr <= hi)]
                if len(core) == 0:
                    core = arr  # nothing survived filter — use all
                m = float(np.median(core))
                s = float(np.std(core, ddof=1)) if len(core) > 1 else 0.0
                stats[slot][metric] = {
                    "mean":   round(m, 2),   # median of IQR-filtered core
                    "std":    round(s, 2),
                    "n":      int(len(core)),
                    "values": [round(v, 2) for v in core.tolist()],
                }
    return stats


def print_slot_stats(stats: dict[str, dict]) -> None:
    sep = "  " + "-" * 55
    print()
    print(sep)
    print("  Per-slot telemetry (approved_by_vision=True only)")
    print(sep)
    print(f"  {'Slot':<22}  {'area_pct':>13}  {'h_gap':>14}")
    print(sep)
    for slot in SLOT_NAMES:
        if slot not in stats:
            print(f"  {slot:<22}  {'(no data)':<13}  {'(no data)':<14}")
            continue
        a = stats[slot]["area_pct"]
        h = stats[slot]["h_gap"]
        a_str = f"{a['mean']:5.1f} +/- {a['std']:4.1f}" if a["n"] else "   n/a"
        h_str = f"{h['mean']:+7.0f} +/- {h['std']:4.0f}" if h["n"] else "     n/a"
        print(f"  {slot:<22}  {a_str:<13}  {h_str:<14}")
    print(sep)
    print()


# =============================================================================
# 3. Derive calibrated thresholds
# =============================================================================

def derive_thresholds(stats: dict[str, dict]) -> dict[str, float]:
    """
    Compute threshold values from real-world approved data.

    DOMINANT_AREA_PCT
        Subject/Interaction mean + 0.5*std.
        Places the gate just above a typical dominant frame so wide Openers
        don't trigger it.  Floor: 15.0.

    H_CONTRAST_THRESH
        mean(|h_gap|) / 1.5 across slots where |mean h_gap| > 80.
        Tightens sensitivity relative to the observed contrast magnitude.
        Floor: 80.0.

    OPENER_MAX_AREA
        Opener mean + 1.0*std.
        Generous cap that accommodates real-world wide establishing shots.
        Floor: 15.0.
    """
    thresholds: dict[str, float] = {}

    # DOMINANT_AREA_PCT
    si = stats.get("Subject/Interaction", {}).get("area_pct", {})
    if si.get("n", 0) >= 1 and si["mean"] is not None:
        thresholds["DOMINANT_AREA_PCT"] = max(
            15.0, round(si["mean"] + 0.5 * (si.get("std") or 0.0), 1)
        )
    else:
        thresholds["DOMINANT_AREA_PCT"] = _FALLBACKS["DOMINANT_AREA_PCT"]

    # H_CONTRAST_THRESH
    contrast_vals: list[float] = []
    for slot_stats in stats.values():
        h = slot_stats.get("h_gap", {})
        if h.get("n", 0) >= 1 and h["mean"] is not None and abs(h["mean"]) > 80:
            contrast_vals.append(abs(float(h["mean"])))
    if contrast_vals:
        arr = np.array(contrast_vals, dtype=float)
        lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
        core = arr[(arr >= lo) & (arr <= hi)]
        if len(core) == 0:
            core = arr
        thresholds["H_CONTRAST_THRESH"] = max(80.0, round(float(np.median(core)) / 1.5, 0))
    else:
        thresholds["H_CONTRAST_THRESH"] = _FALLBACKS["H_CONTRAST_THRESH"]

    # OPENER_MAX_AREA
    op = stats.get("Opener", {}).get("area_pct", {})
    if op.get("n", 0) >= 1 and op["mean"] is not None:
        thresholds["OPENER_MAX_AREA"] = max(
            15.0, round(op["mean"] + (op.get("std") or 0.0), 1)
        )
    else:
        thresholds["OPENER_MAX_AREA"] = _FALLBACKS["OPENER_MAX_AREA"]

    return thresholds


# =============================================================================
# 4. Diff vs current config.json
# =============================================================================

def load_current_config(config_path: Path) -> dict[str, float]:
    if not config_path.exists():
        return {}
    try:
        with config_path.open(encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()
                    if isinstance(v, (int, float))}
    except (json.JSONDecodeError, OSError):
        return {}


def print_thresholds(old: dict[str, float], new: dict[str, float]) -> None:
    keys = ["DOMINANT_AREA_PCT", "H_CONTRAST_THRESH", "OPENER_MAX_AREA"]
    labels = {
        "DOMINANT_AREA_PCT": "Subject dominant gate  (area_pct > X)",
        "H_CONTRAST_THRESH": "H-contrast gate        (|h_gap|  > X)",
        "OPENER_MAX_AREA":   "Opener area cap prompt (< X% of frame)",
    }
    print("  Threshold calibration:")
    for k in keys:
        o = old.get(k)
        n = new.get(k)
        o_str = f"{o:6.1f}" if o is not None else "  none"
        n_str = f"{n:6.1f}" if n is not None else "  none"
        tag   = "unchanged" if o == n else f"CHANGED {o_str} -> {n_str}"
        print(f"    {labels[k]:<42} {tag}")
    print()


# =============================================================================
# 5. Atomic config.json write
# =============================================================================

def save_config(
    thresholds: dict[str, float],
    config_path: Path,
    dry_run: bool = False,
) -> None:
    payload = json.dumps(thresholds, indent=2)
    if dry_run:
        print(f"[calibrate] DRY RUN — config.json would contain:\n{payload}\n")
        return
    tmp = config_path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(str(tmp), str(config_path))
    print(f"[calibrate] config.json written -> {config_path}")


# =============================================================================
# 6. Atomic calibration_telemetry.json write
# =============================================================================

def save_telemetry(
    stats: dict[str, dict],
    thresholds: dict[str, float],
    output: Path,
    dry_run: bool = False,
) -> None:
    record = {
        "slot_stats": {
            slot: {
                metric: {k: v for k, v in data.items() if k != "values"}
                for metric, data in metrics.items()
            }
            for slot, metrics in stats.items()
        },
        "calibrated_thresholds": thresholds,
    }
    if dry_run:
        return
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(output))
    print(f"[calibrate] Telemetry written  -> {output.name}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate vision pipeline thresholds from manifest telemetry"
    )
    parser.add_argument(
        "--manifest",      default=str(MANIFEST_PATH),
        help="Path to final_story_manifest.json",
    )
    parser.add_argument(
        "--config-out",    default=str(CONFIG_PATH),
        help="Where to write config.json  (default: project root)",
    )
    parser.add_argument(
        "--telemetry-out", default=str(TELEMETRY_PATH),
        help="Where to write calibration_telemetry.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print proposed values without writing any files",
    )
    args = parser.parse_args()

    manifest_path  = Path(args.manifest)
    config_path    = Path(args.config_out)
    telemetry_path = Path(args.telemetry_out)
    dry_run        = args.dry_run

    print()
    print("=" * 58)
    print("  calibrate_backend.py -- Vision Pipeline Calibration")
    print("=" * 58)

    # 1. Load
    approved = load_approved(manifest_path)

    # 2. Statistics
    stats = compute_slot_stats(approved)
    print_slot_stats(stats)

    # 3. Derive new thresholds
    new_thresholds = derive_thresholds(stats)

    # 4. Show diff vs current config
    old_thresholds = load_current_config(config_path)
    print_thresholds(old_thresholds, new_thresholds)

    # 5. Write config.json atomically
    save_config(new_thresholds, config_path, dry_run=dry_run)

    # 6. Write telemetry
    save_telemetry(stats, new_thresholds, telemetry_path, dry_run=dry_run)

    print()
    print("=" * 58)
    if dry_run:
        print("  DRY RUN complete -- no files written")
    else:
        changed = [k for k, v in new_thresholds.items() if old_thresholds.get(k) != v]
        if changed:
            print(f"  {len(changed)} threshold(s) updated in config.json")
        else:
            print("  No changes -- thresholds already match calibrated values")
    print("=" * 58)
    print()


if __name__ == "__main__":
    main()
