"""Package the PyInstaller engine into chunked zip resources.

Why: tauri-build's resources glob stack-overflows on the 8k-file engine
tree, AND NSIS (32-bit) cannot mmap files > 2 GB Ã¢â‚¬â€ so neither the loose
tree nor one big zip can ship. Chunked independent zips (~1.6 GB each)
avoid both limits; the shell extracts every part on first launch
(lib.rs start_sidecar) and NSIS recompresses for distribution.

STORED mode on purpose: the payload is mostly already-compressed DLLs,
so deflate would buy ~nothing and cost 20+ minutes.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--flavor", choices=("cuda", "cpu"), default="cuda",
                 help="cuda = E:/firstcut-engine staging; cpu = dist-cpu (build_engine_cpu.py output)")
_args = _ap.parse_args()

if _args.flavor == "cpu":
    SRC = Path(__file__).resolve().parent.parent / "dist-cpu" / "FirstCut"
else:
    SRC = Path(r"E:\\firstcut-engine")  # engine staging (C: is too small to hold the loose tree)
DST_DIR = Path(__file__).resolve().parent.parent / "frontend/src-tauri/resources" / _args.flavor
PART_LIMIT = 1_600_000_000  # stay well under NSIS's 2 GB mmap limit
TRIPLE = "x86_64-pc-windows-msvc"


def main() -> int:
    if not SRC.exists():
        print(f"engine dir missing: {SRC}")
        return 1
    for old in DST_DIR.glob("curator-engine*.zip"):
        old.unlink()
        print(f"removed stale {old.name}")

    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total/1e9:.2f} GB -> parts of <= {PART_LIMIT/1e9:.1f} GB")

    part_idx = 0
    written = 0
    z: zipfile.ZipFile | None = None

    def open_part() -> zipfile.ZipFile:
        nonlocal part_idx, written, z
        part_idx += 1
        written = 0
        name = f"curator-engine-{part_idx:02d}.zip"
        print(f"writing {name}")
        return zipfile.ZipFile(DST_DIR / name, "w", compression=zipfile.ZIP_STORED, allowZip64=True)

    z = open_part()
    prefix = "curator-api-" + TRIPLE + "/"
    for i, p in enumerate(files, 1):
        size = p.stat().st_size
        if written + size > PART_LIMIT and z is not None:
            z.close()
            z = open_part()
        rel = prefix + p.relative_to(SRC).as_posix()
        assert z is not None
        z.write(p, rel)
        written += size
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}")
    assert z is not None
    z.close()

    parts = sorted(DST_DIR.glob("curator-engine-*.zip"))
    for p in parts:
        print(f"  {p.name}: {p.stat().st_size/1e9:.2f} GB")
    print(f"done: {len(parts)} parts, {sum(p.stat().st_size for p in parts)/1e9:.2f} GB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
