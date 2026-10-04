"""Content identity for a photo file — survives copy, move and rename.

size + the first 64 KB. That window holds the RAW/JPEG header with the EXIF
capture time and camera serial, so two different shots never collide in
practice, while a byte-identical copy (card -> laptop) always matches. An
edited export is a different file and correctly gets a new identity.
"""
from __future__ import annotations

import hashlib
import os

_HEAD = 65536


def fingerprint(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(_HEAD)
    except OSError:
        return None
    h = hashlib.sha1(str(size).encode() + b"|" + head).hexdigest()[:20]
    return f"fp1:{h}"
