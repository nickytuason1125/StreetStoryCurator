"""EXIF reading and formatting.

Written against three defects proved in the old server._read_exif:

  * aperture used f"{fn:.1g}" — ONE significant digit. f/1.4 printed as "f/1",
    f/2.8 as "f/3", and f/11 as "f/1e+01". Aperture is one of the three numbers
    a photographer actually reads.
  * shutter ran every ExposureTime through Fraction.limit_denominator, so a 2.5
    second night exposure printed as "5/2s".
  * RAW files returned {} — PIL cannot open .ARW/.CR2/.NEF/.ORF/.RAF, and the
    failure was swallowed by a bare `except Exception: return {}`, so "this
    camera wrote no EXIF" and "our parser cannot read this file" looked
    identical to the UI.

The formatters are pure so they can be tested without touching a file.
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from src.exif_reader import (
    format_aperture,
    format_focal,
    format_shutter,
    read_exif,
)


# ── Aperture ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("fnumber", "expected"), [
    (1.4,  "f/1.4"),   # was "f/1"
    (1.8,  "f/1.8"),   # was "f/2"
    (2.0,  "f/2"),
    (2.8,  "f/2.8"),   # was "f/3"
    (4.0,  "f/4"),
    (5.6,  "f/5.6"),   # was "f/6"
    (8.0,  "f/8"),
    (11.0, "f/11"),    # was "f/1e+01"
    (22.0, "f/22"),    # was "f/2e+01"
])
def test_aperture_keeps_the_stop(fnumber, expected):
    assert format_aperture(fnumber) == expected


def test_aperture_rejects_nonsense():
    assert format_aperture(None) is None
    assert format_aperture(0) is None
    assert format_aperture(-1) is None


# ── Shutter ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("seconds", "expected"), [
    (1 / 8000, "1/8000s"),
    (0.004,    "1/250s"),
    (0.008,    "1/125s"),
    (1 / 60,   "1/60s"),
    (0.5,      "1/2s"),
    (1.0,      "1s"),
    (1.3,      "1.3s"),    # was "13/10s"
    (2.5,      "2.5s"),    # was "5/2s"
    (30.0,     "30s"),
])
def test_shutter_reads_as_a_shutter_speed(seconds, expected):
    assert format_shutter(seconds) == expected


def test_shutter_rejects_nonsense():
    assert format_shutter(None) is None
    assert format_shutter(0) is None


# ── Focal length ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("mm", "expected"), [
    (35.0,  "35mm"),
    (50,    "50mm"),
    (4.2,   "4.2mm"),   # phone/compact — rounding to 4mm loses real information
    (10.5,  "10.5mm"),
    (200.0, "200mm"),
])
def test_focal_keeps_sub_millimetre_detail_only_when_present(mm, expected):
    assert format_focal(mm) == expected


# ── Extraction ──────────────────────────────────────────────────────────────

def _write_jpeg_with_exif(path: Path) -> None:
    """A real JPEG carrying known EXIF, built with piexif."""
    import piexif
    from PIL import Image

    Image.new("RGB", (64, 48), (90, 90, 90)).save(path, "JPEG")
    exif = {
        "0th": {
            piexif.ImageIFD.Make: b"SONY",
            piexif.ImageIFD.Model: b"ILCE-7M4",
            piexif.ImageIFD.Orientation: 1,
        },
        "Exif": {
            piexif.ExifIFD.FNumber: (18, 10),            # f/1.8
            piexif.ExifIFD.ExposureTime: (1, 250),       # 1/250s
            piexif.ExifIFD.ISOSpeedRatings: 3200,
            piexif.ExifIFD.FocalLength: (35, 1),         # 35mm
            piexif.ExifIFD.LensModel: b"FE 35mm F1.8",
            piexif.ExifIFD.DateTimeOriginal: b"2026:03:14 22:41:07",
            piexif.ExifIFD.OffsetTimeOriginal: b"+08:00",
            piexif.ExifIFD.SubSecTimeOriginal: b"62",
            piexif.ExifIFD.ExposureProgram: 3,           # aperture priority
            piexif.ExifIFD.MeteringMode: 5,
        },
        "GPS": {}, "1st": {}, "thumbnail": None,
    }
    piexif.insert(piexif.dump(exif), str(path))


def test_reads_a_jpeg_end_to_end(tmp_path):
    p = tmp_path / "frame.jpg"
    _write_jpeg_with_exif(p)
    got = read_exif(str(p))

    assert got["camera"] == "SONY ILCE-7M4"
    assert got["lens"] == "FE 35mm F1.8"
    assert got["aperture"] == "f/1.8"      # the headline regression
    assert got["shutter"] == "1/250s"
    assert got["iso"] == "3200"
    assert got["focal"] == "35mm"
    assert got["program"] == "Aperture priority"
    assert got["date"] == "2026-03-14"
    # Dimensions come from the image itself, not the EXIF block.
    assert got["dimensions"] == "64 x 48"
    assert "file_size" in got


def test_timezone_and_subsecond_reach_the_time(tmp_path):
    p = tmp_path / "tz.jpg"
    _write_jpeg_with_exif(p)
    got = read_exif(str(p))
    assert got["time"].startswith("22:41:07")
    assert "+08:00" in got["time"]


def test_a_file_with_no_exif_returns_empty_not_an_error(tmp_path):
    from PIL import Image
    p = tmp_path / "bare.jpg"
    Image.new("RGB", (8, 8)).save(p, "JPEG")
    got = read_exif(str(p))
    # No camera fields, but the reader still reports what it can see itself.
    assert "aperture" not in got
    assert got.get("dimensions") == "8 x 8"


def test_missing_file_degrades_quietly(tmp_path):
    assert read_exif(str(tmp_path / "nope.jpg")) == {}


def test_unreadable_bytes_do_not_raise(tmp_path):
    p = tmp_path / "junk.jpg"
    p.write_bytes(struct.pack("<8s", b"notajpeg"))
    assert read_exif(str(p)) == {}


# ── RAW ─────────────────────────────────────────────────────────────────────

def _find_a_raw() -> Path | None:
    """A real RAW off the user's corpus, if one is reachable.

    Card paths first, since that is where RAWs actually live on this machine —
    an empty bench directory is why this test silently skipped at first.
    """
    from src.exif_reader import RAW_EXTS
    for root in (Path(r"E:\GB"), Path(r"D:\framegrade_bench\raw"),
                 Path(r"D:\framegrade_bench\raw2")):
        if not root.exists():
            continue
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in RAW_EXTS and f.is_file():
                return f
    return None


def test_raw_files_produce_exif():
    """The whole point of adding exifread: PIL returns nothing for these.

    Verified against E:\\GB\\DSC07535.ARW — PIL raises UnidentifiedImageError,
    which the old bare `except` turned into "No EXIF data available", while this
    path recovers 20 fields including the lens and the timezone offset.
    """
    raw = _find_a_raw()
    if raw is None:
        pytest.skip("no RAW file reachable on this machine")
    got = read_exif(str(raw))
    assert got, f"RAW {raw.name} produced no EXIF at all"
    # A camera that wrote a RAW file always recorded at least its own name.
    assert "camera" in got, f"RAW {raw.name} produced {sorted(got)}"
    # And the fields PIL could never have reached.
    assert got.get("aperture", "").startswith("f/")
    assert got.get("shutter", "").endswith("s")


def test_raw_flash_is_decoded_not_dumped():
    """exifread hands back a list wrapper; str()-ing it printed a literal '[16]'."""
    raw = _find_a_raw()
    if raw is None:
        pytest.skip("no RAW file reachable on this machine")
    flash = read_exif(str(raw)).get("flash")
    if flash is not None:
        assert flash in ("Fired", "No flash"), f"raw flash bitfield leaked: {flash!r}"
