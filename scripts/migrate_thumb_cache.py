"""One-off: shard the flat thumbnail cache into 2-hex-character subdirectories.

Before 2026-09-06 every thumbnail lived directly in cache/thumbs. A
380k-photo library turned that into ~400k entries in ONE NTFS directory and
every `exists()` stat became user-visibly slow (the "previews take forever"
report). The server now writes sharded names (`ab/abcdef1234_448.webp`) and
migrates flat files lazily on first miss; this script does the bulk move in
one pass so the flat directory empties immediately.

What it does to each file directly in cache/thumbs:
  - `<hash>_448.webp`     → move into `<first-2-hex>/`   (current grid thumbs)
  - `<hash>_L<px>.jpg`    → move into `<first-2-hex>/`   (loupe previews)
  - `<hash>_<other>.webp` → delete (old thumbnail sizes — dead since the
    size is part of the cache key; THUMB_PX is 448 now)
  - `<hash>.webp`         → delete (pre-size-key era names — unreachable)
  - `<name>_<hash>.webp`  → delete (oldest era, filename in the key —
    unreachable: the current key is the bare md5 of the source path)
  - `*.tmp.webp`          → delete (interrupted atomic writes)
  - anything else         → leave alone (only counted)

Idempotent: re-running is safe. A file whose sharded target already exists is
deleted, not duplicated. Run with the server STOPPED:

    venv\\Scripts\\python.exe scripts\\migrate_thumb_cache.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

THUMB_DIR = Path(__file__).resolve().parent.parent / "cache" / "thumbs"

_GRID = re.compile(r"^([0-9a-f]{10})_448\.webp$")
_LOUPE = re.compile(r"^([0-9a-f]{10})_L\d+\.jpg$")
_OLD_SIZE = re.compile(r"^[0-9a-f]{10}_\d+\.webp$")
_NO_SIZE = re.compile(r"^[0-9a-f]{10}\.webp$")
_NAME_ERA = re.compile(r"_[0-9a-f]{10}\.webp$")


def main() -> None:
    if not THUMB_DIR.is_dir():
        print(f"no thumb dir at {THUMB_DIR} — nothing to do")
        return
    moved = deleted = skipped = 0
    for entry in os.scandir(THUMB_DIR):
        if not entry.is_file(follow_symlinks=False):
            continue
        name = entry.name
        m = _GRID.match(name) or _LOUPE.match(name)
        if m:
            dest = THUMB_DIR / m.group(1)[:2] / name
            if dest.exists():
                os.remove(entry.path)      # shard already has it — flat copy is dead weight
                deleted += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.rename(entry.path, dest)
                moved += 1
            continue
        if name.endswith(".tmp.webp") or _OLD_SIZE.match(name) or _NO_SIZE.match(name) \
                or (name.endswith(".webp") and _NAME_ERA.search(name)):
            os.remove(entry.path)
            deleted += 1
            continue
        skipped += 1
    print(f"moved={moved} deleted={deleted} skipped={skipped}")


if __name__ == "__main__":
    main()
