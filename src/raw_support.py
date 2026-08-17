"""
RAW camera format support via rawpy (LibRaw).

Provides drop-in replacements for PIL Image.open() and cv2.imread()
that transparently handle RAW files from Canon, Nikon, Sony, Fujifilm, etc.

Usage:
    from raw_support import open_image, imread_bgr, RAW_EXTS
    img = open_image(path)          # PIL Image, works for RAW and non-RAW
    bgr = imread_bgr(path)          # cv2-compatible BGR array
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

RAW_EXTS: frozenset[str] = frozenset({
    ".cr2", ".cr3",              # Canon
    ".nef", ".nrw",              # Nikon
    ".arw", ".srf", ".sr2",      # Sony
    ".dng",                      # Adobe DNG / Leica / Hasselblad / Apple ProRAW
    ".raf",                      # Fujifilm
    ".orf",                      # Olympus / OM System
    ".rw2",                      # Panasonic / Leica (compact)
    ".pef",                      # Pentax / Ricoh
    ".srw",                      # Samsung
    ".3fr",                      # Hasselblad (medium format)
    ".mef",                      # Mamiya
    ".rwl",                      # Leica (M-series)
    ".erf",                      # Epson
    ".kdc", ".dcs", ".dcr",      # Kodak
    ".x3f",                      # Sigma / Foveon
    ".iiq",                      # Phase One
    ".cap", ".liq",              # Phase One (older)
    ".r3d",                      # RED digital cinema
})


def _rawpy_decode(path: str, half_size: bool = False) -> np.ndarray:
    """
    Decode RAW file to HWC uint8 RGB via rawpy.
    half_size=True is ~4× faster and sufficient for thumbnails.
    Raises ImportError if rawpy is not installed.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            "rawpy is required for RAW file support. "
            "Install it with:  pip install rawpy"
        )
    with rawpy.imread(path) as raw:
        return raw.postprocess(
            use_camera_wb=True,
            half_size=half_size,
            output_bps=8,
            no_auto_bright=False,
        )


def open_image(path: str, mode: str = "RGB", half_size: bool = False):
    """
    Open any image as a PIL Image.
    RAW files are decoded via rawpy; all others use PIL.Image.open().
    """
    from PIL import Image
    if Path(path).suffix.lower() in RAW_EXTS:
        rgb = _rawpy_decode(path, half_size=half_size)
        img = Image.fromarray(rgb, "RGB")
        return img.convert(mode) if mode != "RGB" else img
    img = Image.open(path)
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img


def extract_embedded_preview(path: str, mode: str = "RGB"):
    """Return the camera's EMBEDDED preview (JPEG/bitmap) as a PIL Image WITHOUT
    demosaicing the sensor data — memory-safe for RAW. Returns None if the file
    has no embedded preview or cannot be read.

    Uses rawpy.extract_thumb(), which reads only the embedded thumbnail and never
    calls unpack()/postprocess() on the full Bayer array (the OOM-prone step), so
    it costs ~the embedded JPEG size in RAM, not the full uncompressed sensor."""
    try:
        import rawpy, io
        from PIL import Image
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            img = Image.open(io.BytesIO(thumb.data))
        else:  # ThumbFormat.BITMAP
            img = Image.fromarray(thumb.data)
        if mode and img.mode != mode:
            img = img.convert(mode)
        return img
    except Exception:
        return None


# An embedded preview smaller than this is a navigation thumbnail, not a usable
# image. Some bodies embed 160px JPEGs; running person detection on those finds
# nothing and reports "no people" with full confidence.
_MIN_PREVIEW_SIDE = 320


def load_rgb(path: str, mode: str = "RGB", min_side: int = _MIN_PREVIEW_SIDE):
    """Best-effort decode of ANY image to PIL. Returns (image | None, source).

    RAW_EXTS lists 25 extensions but LibRaw does not decode all of them equally
    — some bodies embed only a tiny thumbnail, some formats (notably .r3d, and
    .x3f/.liq in places) LibRaw handles partially or not at all. Every RAW
    consumer previously called extract_embedded_preview() alone, so any of those
    cases returned None and the caller carried on with nothing: the photo was
    scored as an empty scene rather than flagged as unreadable. That is the
    silent-wrong-answer shape this project keeps getting bitten by.

    The chain tries progressively more expensive options and reports which one
    worked, so "we could not read this" is always distinguishable from
    "we read it and there was nothing there":

        embedded preview (cheap, no demosaic)  -> "preview"
        half-size demosaic                     -> "half"
        full demosaic                          -> "full"
        PIL (DNG and some RAWs are TIFF-based) -> "pil"
        nothing worked                         -> (None, "unreadable")
    """
    from PIL import Image

    def _as(img, tag):
        """Enforce `mode` on EVERY route, so callers never have to check."""
        try:
            if img is not None and mode and img.mode != mode:
                img = img.convert(mode)
        except Exception:
            return None, "unreadable"
        return img, tag

    if Path(path).suffix.lower() not in RAW_EXTS:
        try:
            return _as(Image.open(path), "pil")
        except Exception:
            return None, "unreadable"

    # Every stage is individually guarded: this function's contract is that it
    # returns a verdict, never raises. A stage that throws is just a stage that
    # did not work.
    try:
        img = extract_embedded_preview(path, mode)
    except Exception:
        img = None
    if img is not None and min(img.size) >= min_side:
        return _as(img, "preview")

    for half, tag in ((True, "half"), (False, "full")):
        try:
            return _as(Image.fromarray(_rawpy_decode(path, half_size=half), "RGB"), tag)
        except Exception:
            continue
    if img is not None:
        return _as(img, "preview-small")   # tiny, but better than nothing
    try:                                   # DNG/TIFF-based RAWs PIL can sometimes open
        return _as(Image.open(path), "pil")
    except Exception:
        return None, "unreadable"


def imread_bgr(path: str, half_size: bool = False) -> Optional[np.ndarray]:
    """
    cv2.imread() replacement that handles RAW files.
    Returns HWC uint8 BGR array (same layout as cv2), or None on failure.
    """
    try:
        if Path(path).suffix.lower() in RAW_EXTS:
            rgb = _rawpy_decode(path, half_size=half_size)
            return rgb[:, :, ::-1].copy()
        import cv2
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_exif_timestamp(path: str) -> Optional[float]:
    """
    Extract EXIF DateTimeOriginal as a Unix timestamp.
    For RAW files, tries the embedded JPEG thumbnail (which carries full EXIF)
    before falling back to file mtime.
    """
    import os
    from datetime import datetime

    # Standard PIL path (works for JPEG, TIFF, DNG)
    try:
        from PIL import Image
        with Image.open(path) as img:
            exif_data = img.getexif()
            dt_str = exif_data.get(36867)
            if dt_str:
                return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass

    # RAW: extract embedded JPEG preview and read EXIF from it
    if Path(path).suffix.lower() in RAW_EXTS:
        try:
            import rawpy
            import io
            from PIL import Image
            with rawpy.imread(path) as raw:
                thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                buf = io.BytesIO(thumb.data)
                with Image.open(buf) as img:
                    exif_data = img.getexif()
                    dt_str = exif_data.get(36867)
                    if dt_str:
                        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass

    return os.path.getmtime(path)
