"""raw_support.open_analysis — the one fast decode for every analysis step.

2026-10-04: Sony 61 MP JPEGs cost 100-270 ms per decode (even drafted) and the
cull decoded each one ~6 times. Their MPF preview (1616x1080, frame 1) is the
same in-camera render the paired ARW is graded from, and decodes in ~28 ms.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import raw_support as rs  # noqa: E402

RED, BLUE = (255, 0, 0), (0, 0, 255)


def _mpo(path: Path, main_size, preview_size):
    Image.new("RGB", main_size, RED).save(
        path, "MPO", save_all=True, append_images=[Image.new("RGB", preview_size, BLUE)])


def _close(a, b, tol=40):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_camera_jpeg_uses_mpf_preview_when_big_enough(tmp_path):
    p = tmp_path / "DSC1.JPG"
    _mpo(p, (4000, 2667), (1616, 1080))
    img = rs.open_analysis(str(p), 1600)
    assert img.mode == "RGB" and max(img.size) == 1616
    assert _close(img.getpixel((10, 10)), BLUE)


def test_small_preview_falls_back_to_the_main_image(tmp_path):
    p = tmp_path / "DSC2.JPG"
    _mpo(p, (4000, 2667), (640, 427))
    img = rs.open_analysis(str(p), 1600)
    assert _close(img.getpixel((10, 10)), RED)
    assert max(img.size) >= 1600          # drafted, never below the requested edge


def test_plain_jpeg_is_draft_decoded(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (4000, 3000), RED).save(p, "JPEG")
    img = rs.open_analysis(str(p), 512)
    assert 512 <= max(img.size) < 4000


def test_unreadable_file_returns_none(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    assert rs.open_analysis(str(p), 512) is None


def test_kill_switch(tmp_path, monkeypatch):
    p = tmp_path / "DSC3.JPG"
    _mpo(p, (4000, 2667), (1616, 1080))
    monkeypatch.setenv("FIRSTCUT_JPEG_PREVIEW", "0")
    assert rs.jpeg_preview(str(p), 512) is None
    assert _close(rs.open_analysis(str(p), 512).getpixel((10, 10)), RED)
