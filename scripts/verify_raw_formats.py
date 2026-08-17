"""
Verify which RAW formats this install can ACTUALLY decode.

raw_support.RAW_EXTS declares 25 extensions. Declaring is not supporting: the
list is what gets routed to LibRaw, and LibRaw's coverage varies by format and
by camera body. Shipping "25 formats supported" when one has ever been run
against a real file is the kind of claim that fails in a customer's hands.

This walks real files and reports, per format:

  VERIFIED    decoded to a usable image, and by which route
              (preview = embedded JPEG, no demosaic — the cheap path;
               half/full = demosaic, slower and much more memory;
               preview-small = only a navigation thumbnail was available)
  FAILED      files present but none could be decoded
  NO SAMPLES  nothing on disk to test — status genuinely unknown

Run:
  venv\\Scripts\\python.exe scripts/verify_raw_formats.py
  venv\\Scripts\\python.exe scripts/verify_raw_formats.py D:\\shoots E:\\cards
  venv\\Scripts\\python.exe scripts/verify_raw_formats.py --per-format 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from raw_support import RAW_EXTS, load_rgb  # noqa: E402

# LibRaw does not decode RED's .r3d (a proprietary video container needing RED's
# own SDK). It is listed in RAW_EXTS so such files are at least recognised and
# reported rather than silently treated as stills; it is expected to FAIL here.
_KNOWN_UNSUPPORTED = {".r3d"}


def _catalog_folders() -> list:
    cat = _ROOT / "cache" / "catalog.json"
    if not cat.exists():
        return []
    try:
        photos = json.loads(cat.read_text(encoding="utf-8")).get("photos", [])
    except Exception:
        return []
    return sorted({str(Path(p["path"]).parent) for p in photos if p.get("path")})


def _find(roots: list, per_format: int) -> dict:
    found = defaultdict(list)
    for root in roots:
        rp = Path(root)
        if not rp.is_dir():
            continue
        for f in rp.rglob("*"):
            try:
                ext = f.suffix.lower()
                if (ext in RAW_EXTS and f.is_file()
                        and len(found[ext]) < per_format):
                    found[ext].append(f)
            except Exception:
                continue
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", help="folders to scan (default: catalog folders)")
    ap.add_argument("--per-format", type=int, default=3)
    args = ap.parse_args()

    roots = args.roots or _catalog_folders()
    if not roots:
        print("No folders to scan. Pass paths, or grade a folder first.")
        return 2

    print(f"Scanning {len(roots)} folder(s) for {len(RAW_EXTS)} RAW formats…\n")
    found = _find(roots, args.per_format)

    ok, failed, missing = [], [], []
    for ext in sorted(RAW_EXTS):
        files = found.get(ext, [])
        if not files:
            missing.append(ext)
            continue
        routes, sizes, errs = [], [], 0
        for f in files:
            t0 = time.monotonic()
            try:
                img, src = load_rgb(str(f))
            except Exception as e:
                img, src = None, f"raised {type(e).__name__}"
            dt = time.monotonic() - t0
            if img is None:
                errs += 1
                routes.append(src)
            else:
                routes.append(f"{src} {img.size[0]}x{img.size[1]} {dt:.2f}s")
                sizes.append(min(img.size))
        if errs == len(files):
            failed.append((ext, routes[0]))
            print(f"  FAILED      {ext}  ({len(files)} file(s))  -> {routes[0]}")
        else:
            ok.append(ext)
            print(f"  VERIFIED    {ext}  ({len(files)} file(s))  -> {routes[0]}")

    if missing:
        print("\n  NO SAMPLES  " + " ".join(missing))

    print("\n" + "=" * 66)
    print(f"  verified {len(ok)} / {len(RAW_EXTS)} formats"
          f"   failed {len(failed)}   untested {len(missing)}")
    unexpected = [e for e, _ in failed if e not in _KNOWN_UNSUPPORTED]
    if unexpected:
        print(f"  UNEXPECTED FAILURES: {' '.join(unexpected)}")
        print("  These are declared as supported but could not be decoded.")
    print("=" * 66)
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
