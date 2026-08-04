"""
What kind of photography is in every folder on this machine?

fast_niche_detector.detect() recommends a niche for a folder by CLIP-matching a
small adaptive sample. It shipped for the folder-select UI and was never used
anywhere else, so the grading path judged every folder by street-photography
criteria whatever it actually held.

This runs the detector across the whole machine. It is deliberately CHEAP: the
detector samples a handful of frames per folder rather than reading all of them,
so a 16,000-photo library audits in minutes instead of the hours a full grade
would take.

Usage:
    venv\\Scripts\\python.exe scripts/audit_photo_folders.py --min-photos 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".rw2", ".cr2", ".nef", ".arw",
             ".dng", ".heic", ".tif", ".tiff"}

# Folders that are not someone's photography. Auditing them wastes time and
# buries the real folders in the report.
SKIP_PARTS = {"node_modules", "venv", ".git", "__pycache__", "AppData",
              "cache", "previews", "thumbs", "_quarantine", "site-packages",
              "Movies", "$RECYCLE.BIN", "Windows", "Program Files"}


def find_folders(roots: list[Path], min_photos: int) -> dict[str, list[str]]:
    acc: dict[str, list[str]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            parts = set(Path(dirpath).parts)
            if parts & SKIP_PARTS:
                dirnames[:] = []          # do not descend
                continue
            hits = [os.path.join(dirpath, f) for f in filenames
                    if Path(f).suffix.lower() in PHOTO_EXT]
            if hits:
                acc[dirpath].extend(hits)
    return {k: v for k, v in acc.items() if len(v) >= min_photos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-photos", type=int, default=25)
    ap.add_argument("--sample", type=int, default=12,
                    help="frames per folder handed to the detector")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    home = Path.home()
    roots = [home / "Desktop", home / "Pictures", home / "Documents",
             home / "Downloads"]

    print("[audit] scanning…", flush=True)
    folders = find_folders(roots, args.min_photos)
    total = sum(len(v) for v in folders.values())
    print(f"[audit] {len(folders)} folders, {total} photos "
          f"(>= {args.min_photos} photos each)\n", flush=True)

    import fast_niche_detector as fnd
    fnd.warmup()

    rows = []
    for i, (folder, paths) in enumerate(
            sorted(folders.items(), key=lambda kv: -len(kv[1])), 1):
        t0 = time.monotonic()
        try:
            res = fnd.detect(sorted(paths), sample_limit=args.sample)
        except Exception as err:
            res = None
            print(f"  [{i}/{len(folders)}] {Path(folder).name}: detect failed ({err})")
        niche = (res or {}).get("preset") or (res or {}).get("niche") or "?"
        conf = float((res or {}).get("confidence") or 0.0)
        rows.append({"folder": folder, "n": len(paths),
                     "niche": niche, "confidence": conf})
        print(f"  [{i}/{len(folders)}] {len(paths):5d}  {niche:<18} "
              f"conf {conf:.2f}  {folder[-52:]}  ({time.monotonic()-t0:.1f}s)",
              flush=True)

    print(f"\n{'PHOTOS':>7}  {'NICHE':<18} {'CONF':>5}  FOLDER")
    for r in sorted(rows, key=lambda r: -r["n"]):
        print(f"{r['n']:7d}  {r['niche']:<18} {r['confidence']:5.2f}  "
              f"{r['folder'][-56:]}")

    by_niche: dict = defaultdict(int)
    for r in rows:
        by_niche[r["niche"]] += r["n"]
    print("\nphotos by detected niche:")
    for k, v in sorted(by_niche.items(), key=lambda kv: -kv[1]):
        print(f"  {v:7d}  {k}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
