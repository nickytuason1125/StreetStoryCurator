"""EXIF reading, extracted from server.py so it can be tested without a server.

Three things the previous implementation got wrong, all locked by
tests/test_exif_reader.py:

1. Aperture was formatted with f"{fn:.1g}" — one significant digit. f/1.4 became
   "f/1", f/2.8 became "f/3", and f/11 became "f/1e+01". Aperture is one of the
   three numbers a photographer reads off a frame.
2. Every ExposureTime went through Fraction.limit_denominator, which is right
   below a second and nonsense above it: a 2.5s night exposure printed "5/2s".
3. RAW files produced nothing. PIL cannot open .ARW/.CR2/.NEF/.ORF/.RAF, and the
   failure was swallowed by a bare `except Exception: return {}` — so a parser
   failure and a camera that genuinely wrote no EXIF were indistinguishable.

Reader dispatch: PIL for the formats it understands, exifread for RAW (pure
Python, no binary, works offline), and rawpy/LibRaw as a last resort. Failures
are logged at debug rather than swallowed silently.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# LibRaw's set, matching src/raw_support.RAW_EXTS. Kept here too so the reader
# can decide which parser to use without importing the decode stack.
RAW_EXTS: frozenset[str] = frozenset({
    ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw", ".nef", ".nrw",
    ".orf", ".rw2", ".raf", ".dng", ".pef", ".srw", ".raw", ".rwl",
    ".iiq", ".3fr", ".fff", ".mef", ".mos", ".mrw", ".x3f", ".erf", ".kdc",
})

_PROGRAMS = {
    1: "Manual", 2: "Program", 3: "Aperture priority", 4: "Shutter priority",
    5: "Creative", 6: "Action", 7: "Portrait", 8: "Landscape",
}
_METERING = {
    1: "Average", 2: "Center-weighted", 3: "Spot", 4: "Multi-spot",
    5: "Multi-segment", 6: "Partial", 255: "Other",
}
_ORIENTATION = {
    1: "Horizontal", 2: "Mirrored", 3: "Rotated 180°", 4: "Mirrored vertical",
    5: "Mirrored + 90° CCW", 6: "Rotated 90° CW",
    7: "Mirrored + 90° CW", 8: "Rotated 90° CCW",
}
_COLOR_SPACE = {1: "sRGB", 2: "Adobe RGB", 65535: "Uncalibrated"}


# ── Formatters (pure — the regression surface, so they are unit-tested) ──────

def _num(v: Any) -> Optional[float]:
    """Coerce EXIF rationals, exifread Ratios, bytes and strings to a float."""
    if v is None:
        return None
    try:
        if isinstance(v, tuple) and len(v) == 2:      # piexif/PIL rational
            return float(v[0]) / float(v[1]) if v[1] else None
        if hasattr(v, "num") and hasattr(v, "den"):   # exifread Ratio
            return float(v.num) / float(v.den) if v.den else None
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _trim(x: float) -> str:
    """1.40 -> '1.4', 2.0 -> '2', 11.0 -> '11'."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def format_aperture(fnumber: Any) -> Optional[str]:
    """f/1.4, f/2.8, f/11 — never f/1, f/3 or f/1e+01."""
    v = _num(fnumber)
    if v is None or v <= 0:
        return None
    return f"f/{_trim(v)}"


def format_shutter(seconds: Any) -> Optional[str]:
    """1/250s below a second, 2.5s above it.

    A fraction is how a shutter is marked on a dial up to 1s, and how nobody
    writes a long exposure: 5/2s is not a thing anyone has ever said out loud.
    """
    t = _num(seconds)
    if t is None or t <= 0:
        return None
    if t >= 1:
        return f"{_trim(t)}s"
    return f"1/{round(1 / t)}s"


def format_focal(mm: Any) -> Optional[str]:
    """35mm, but 4.2mm stays 4.2mm — rounding a compact to 4mm loses real detail."""
    v = _num(mm)
    if v is None or v <= 0:
        return None
    return f"{_trim(v)}mm"


def _format_size(nbytes: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if nbytes >= div:
            return f"{nbytes / div:.1f} {unit}"
    return f"{nbytes} B"


def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    out = str(v).strip("\x00").strip()
    return out or None


def _camera(make: Optional[str], model: Optional[str]) -> Optional[str]:
    make, model = (make or "").strip(), (model or "").strip()
    if make and model and model.upper().startswith(make.split()[0].upper()):
        return model or None
    return f"{make} {model}".strip() or None


# ── PIL path (JPEG/PNG/TIFF/HEIF) ────────────────────────────────────────────

def _from_pil(p: Path) -> dict:
    from PIL import Image

    with Image.open(p) as img:
        raw = img.getexif()
        if not raw:
            return {}
        exif = raw.get_ifd(0x8769)
        gps = raw.get_ifd(0x8825)

        def g(tag, src=None):
            return (src if src is not None else raw).get(tag)

        dt = _s(g(0x9003, exif) or g(0x0132))
        date = time_s = None
        if dt:
            parts = dt.split(" ")
            date = parts[0].replace(":", "-")
            if len(parts) > 1:
                time_s = parts[1][:8]
                sub = _s(g(0x9291, exif))
                if sub:
                    time_s += f".{sub[:2]}"
                off = _s(g(0x9011, exif))
                if off:
                    time_s += f" {off}"

        return {
            "camera":          _camera(_s(g(0x010F)), _s(g(0x0110))),
            "lens":            _s(g(0xA434, exif)),
            "firmware":        _s(g(0x0131)),
            "body_serial":     _s(g(0xA431, exif)),
            "lens_serial":     _s(g(0xA435, exif)),
            "artist":          _s(g(0x013B)),
            "copyright":       _s(g(0x8298)),
            "aperture":        format_aperture(g(0x829D, exif)),
            "shutter":         format_shutter(g(0x829A, exif)),
            "iso":             _s(g(0x8827, exif)),
            "ev":              _ev(g(0x9204, exif)),
            "program":         _PROGRAMS.get(_int(g(0x8822, exif))),
            "metering":        _METERING.get(_int(g(0x9207, exif))),
            "white_balance":   _wb(g(0xA403, exif)),
            "flash":           _flash(g(0x9209, exif)),
            "focal":           format_focal(g(0x920A, exif)),
            "focal_35mm":      format_focal(g(0xA405, exif)),
            "subject_distance": _distance(g(0x9206, exif)),
            "orientation":     _ORIENTATION.get(_int(g(0x0112))),
            "color_space":     _COLOR_SPACE.get(_int(g(0xA001, exif))),
            "date":            date,
            "time":            time_s,
            "gps":             _gps(gps),
        }


def _int(v: Any) -> Optional[int]:
    n = _num(v)
    return int(n) if n is not None else None


def _ev(v: Any) -> Optional[str]:
    """0 EV is shown, not hidden — absence must mean 'not recorded'."""
    n = _num(v)
    return None if n is None else f"{n:+.1f} EV".replace("+0.0", "0.0")


def _wb(v: Any) -> Optional[str]:
    n = _int(v)
    return None if n is None else ("Auto" if n == 0 else "Manual")


def _flash(v: Any) -> Optional[str]:
    n = _int(v)
    return None if n is None else ("Fired" if n & 0x1 else "No flash")


def _distance(v: Any) -> Optional[str]:
    n = _num(v)
    if n is None or n <= 0:
        return None
    return "Infinity" if n > 6000 else f"{_trim(n)} m"


def _gps(ifd: Any) -> Optional[str]:
    if not ifd:
        return None
    try:
        def dms(x):
            return _num(x[0]) + _num(x[1]) / 60 + _num(x[2]) / 3600
        lat, lon = ifd.get(2), ifd.get(4)
        if not lat or not lon:
            return None
        la = dms(lat) * (-1 if _s(ifd.get(1)) == "S" else 1)
        lo = dms(lon) * (-1 if _s(ifd.get(3)) == "W" else 1)
        return f"{la:.5f}, {lo:.5f}"
    except Exception:
        log.debug("GPS block unreadable", exc_info=True)
        return None


# ── exifread path (RAW) ──────────────────────────────────────────────────────

def _from_exifread(p: Path) -> dict:
    import exifread

    with open(p, "rb") as fh:
        tags = exifread.process_file(fh, details=False)
    if not tags:
        return {}

    def t(name):
        v = tags.get(name)
        return v.values if v is not None else None

    def one(name):
        v = t(name)
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    dt = _s(t("EXIF DateTimeOriginal") or t("Image DateTime"))
    date = time_s = None
    if dt:
        parts = str(dt).split(" ")
        date = parts[0].replace(":", "-")
        if len(parts) > 1:
            time_s = parts[1][:8]
            off = _s(t("EXIF OffsetTimeOriginal"))
            if off:
                time_s += f" {off}"

    return {
        "camera":     _camera(_s(t("Image Make")), _s(t("Image Model"))),
        "lens":       _s(t("EXIF LensModel")),
        "firmware":   _s(t("Image Software")),
        "body_serial": _s(t("EXIF BodySerialNumber")),
        "artist":     _s(t("Image Artist")),
        "copyright":  _s(t("Image Copyright")),
        "aperture":   format_aperture(one("EXIF FNumber")),
        "shutter":    format_shutter(one("EXIF ExposureTime")),
        "iso":        _s(one("EXIF ISOSpeedRatings")),
        "ev":         _ev(one("EXIF ExposureBiasValue")),
        "program":    _PROGRAMS.get(_int(one("EXIF ExposureProgram"))),
        "metering":   _METERING.get(_int(one("EXIF MeteringMode"))),
        "white_balance": _wb(one("EXIF WhiteBalance")),
        # Decode the bitfield rather than str()-ing exifread's list wrapper,
        # which rendered as a literal "[16]" in the panel.
        "flash":      _flash(one("EXIF Flash")),
        "focal":      format_focal(one("EXIF FocalLength")),
        "focal_35mm": format_focal(one("EXIF FocalLengthIn35mmFilm")),
        "orientation": _ORIENTATION.get(_int(one("Image Orientation"))),
        "date":       date,
        "time":       time_s,
        # The camera's own count. LibRaw reports the visible sensor area, which
        # on a Sony ARW is ~16px wider than what the body recorded (5184 vs
        # 5168) because it includes a border the camera does not count.
        **_dims(_int(one("EXIF ExifImageWidth")), _int(one("EXIF ExifImageLength"))),
    }


def _dims(w: Optional[int], h: Optional[int]) -> dict:
    if not w or not h:
        return {}
    return {"dimensions": f"{w} x {h}", "megapixels": f"{(w * h) / 1e6:.1f} MP"}


def _from_rawpy(p: Path) -> dict:
    """Last resort. LibRaw exposes about seven fields and no lens model."""
    import rawpy

    with rawpy.imread(str(p)) as r:
        o = r.other
        return {
            "camera":   _camera(_s(getattr(o, "make", None)), _s(getattr(o, "model", None))),
            "aperture": format_aperture(getattr(o, "aperture", None)),
            "shutter":  format_shutter(getattr(o, "shutter", None)),
            "iso":      _s(int(getattr(o, "iso_speed", 0)) or None),
            "focal":    format_focal(getattr(o, "focal_len", None)),
        }


# ── Facts about the file itself, independent of any EXIF block ──────────────

def _file_facts(p: Path) -> dict:
    out: dict[str, Any] = {
        "file_size": _format_size(p.stat().st_size),
        "format": p.suffix.lstrip(".").upper() or None,
    }
    w = h = None
    if p.suffix.lower() in RAW_EXTS:
        try:
            import rawpy
            with rawpy.imread(str(p)) as r:
                h, w = r.sizes.height, r.sizes.width
        except Exception:
            log.debug("rawpy could not size %s", p.name, exc_info=True)
    else:
        from PIL import Image
        with Image.open(p) as img:
            w, h = img.size
    if w and h:
        out["dimensions"] = f"{w} x {h}"
        out["megapixels"] = f"{(w * h) / 1e6:.1f} MP"
    return out


# ── Public entry point ───────────────────────────────────────────────────────

def read_exif(path: str) -> dict:
    """Every field the file will give up, as a flat dict of display strings.

    Flat on purpose: /api/exif has other consumers (the filmstrip subtitle reads
    exif.camera / exif.aperture directly), so grouping is left to the UI.
    """
    p = Path(path)
    if not p.is_file():
        return {}

    data: dict[str, Any] = {}
    try:
        if p.suffix.lower() in RAW_EXTS:
            data = _from_exifread(p)
            if not data.get("camera"):
                data = {**_from_rawpy(p), **{k: v for k, v in data.items() if v}}
        else:
            data = _from_pil(p)
    except Exception:
        # Logged, not swallowed: "no EXIF" and "we could not parse it" are
        # different answers and the UI is entitled to tell them apart.
        log.debug("EXIF parse failed for %s", p.name, exc_info=True)
        data = {}

    try:
        # setdefault, not update: where the camera stated a value we keep it and
        # only fall back to what we can measure off the file ourselves.
        for k, v in _file_facts(p).items():
            data.setdefault(k, v)
    except Exception:
        log.debug("could not stat/size %s", p.name, exc_info=True)
        if not data:
            return {}

    return {k: v for k, v in data.items() if v is not None}
