"""
raw_support.load_rgb is the single decode path every RAW consumer uses, and its
job is to make "could not read this" impossible to confuse with "read it fine,
nothing there". That distinction is the whole point: the previous code called
extract_embedded_preview() alone, so a body that embeds no preview returned None
and the photo was scored as an empty scene.

Only .rw2 has real sample files here, so the chain is exercised by controlling
what each stage returns rather than by shipping 25 test images.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_raw_loader.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import raw_support as R  # noqa: E402


def _img(w, h, colour=(120, 90, 60)):
    return Image.new("RGB", (w, h), colour)


def _raw(tmp_path, name="shot.rw2"):
    p = tmp_path / name
    p.write_bytes(b"\x49\x49\x2a\x00not-a-real-raw")
    return p


# ── the happy path: a usable embedded preview costs no demosaic ──────────────
def test_good_preview_is_used_and_no_demosaic_happens(monkeypatch, tmp_path):
    called = {"decode": 0}
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": _img(1920, 1280))
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: called.__setitem__("decode", 1))
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert src == "preview" and img.size == (1920, 1280)
    assert called["decode"] == 0, "must not demosaic when a good preview exists"


# ── the bug this function exists to fix ─────────────────────────────────────
def test_no_preview_falls_through_to_demosaic_instead_of_none(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": None)
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: np.zeros((800, 1200, 3), np.uint8))
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert img is not None, "a body with no embedded preview must still decode"
    assert src == "half" and img.size == (1200, 800)


def test_navigation_sized_preview_is_rejected_and_demosaiced(monkeypatch, tmp_path):
    """A 160px thumbnail finds no people and would report 'no people' confidently."""
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": _img(160, 120))
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: np.zeros((900, 1400, 3), np.uint8))
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert src == "half" and min(img.size) >= R._MIN_PREVIEW_SIDE


def test_half_size_failure_falls_back_to_full(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": None)

    def _decode(p, half_size=False):
        if half_size:
            raise MemoryError("half-size path unavailable")
        return np.zeros((600, 900, 3), np.uint8)

    monkeypatch.setattr(R, "_rawpy_decode", _decode)
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert src == "full" and img is not None


def test_tiny_preview_kept_when_demosaic_impossible(monkeypatch, tmp_path):
    """Better a small image than none — but the source says it is degraded."""
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": _img(160, 120))
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: (_ for _ in ()).throw(OSError("nope")))
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert img is not None and src == "preview-small"


def test_totally_undecodable_raw_reports_unreadable(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": None)
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: (_ for _ in ()).throw(OSError("nope")))
    img, src = R.load_rgb(str(_raw(tmp_path)))
    assert img is None and src == "unreadable", (
        "an unreadable RAW must be distinguishable from an empty scene")


def test_never_raises_on_any_garbage(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "extract_embedded_preview",
                        lambda p, m="RGB": (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: (_ for _ in ()).throw(RuntimeError("boom")))
    for name in ("a.rw2", "b.cr3", "c.x3f", "d.r3d"):
        try:
            img, src = R.load_rgb(str(_raw(tmp_path, name)))
        except Exception as e:                       # pragma: no cover
            pytest.fail(f"load_rgb raised for {name}: {e}")
        assert img is None and src == "unreadable"


# ── non-RAW files still work ────────────────────────────────────────────────
def test_ordinary_jpeg(tmp_path):
    p = tmp_path / "x.jpg"
    _img(900, 600).save(p)
    img, src = R.load_rgb(str(p))
    assert src == "pil" and img.size == (900, 600)


def test_corrupt_jpeg_is_unreadable_not_a_crash(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"definitely not a jpeg")
    img, src = R.load_rgb(str(p))
    assert img is None and src == "unreadable"


def test_mode_conversion_applies_on_every_route(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": _img(800, 600))
    img, _ = R.load_rgb(str(_raw(tmp_path)), mode="L")
    assert img.mode == "L"

    monkeypatch.setattr(R, "extract_embedded_preview", lambda p, m="RGB": None)
    monkeypatch.setattr(R, "_rawpy_decode",
                        lambda p, half_size=False: np.zeros((600, 800, 3), np.uint8))
    img, _ = R.load_rgb(str(_raw(tmp_path)), mode="L")
    assert img.mode == "L"


def test_every_declared_extension_is_routed_to_the_raw_chain(monkeypatch, tmp_path):
    """A format in RAW_EXTS must never fall through to PIL's still-image path."""
    seen = []
    monkeypatch.setattr(R, "extract_embedded_preview",
                        lambda p, m="RGB": seen.append(Path(p).suffix) or _img(800, 600))
    for ext in sorted(R.RAW_EXTS):
        R.load_rgb(str(_raw(tmp_path, f"f{ext}")))
    assert len(seen) == len(R.RAW_EXTS)
