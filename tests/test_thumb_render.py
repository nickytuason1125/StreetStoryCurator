"""Contract tests for routers.library._render_thumb (2026-09-28 rewrite).

Locks the three rules that make card-dump thumbnails fast AND correct:
  1. a camera JPEG renders from its MPF preview, not its 160 px EXIF thumb
     and not a full decode — unless the preview is too small to be sharp
  2. one render serves a RAW+JPEG pair, but ONLY a pair the camera wrote
     together: a JPEG exported later next to its RAW is a different picture
  3. a bad file returns None — never raises into the queue worker
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402,F401  (mounts routers — must load before routers.library)
from routers import library as lib  # noqa: E402

RED, BLUE = (255, 0, 0), (0, 0, 255)


@pytest.fixture()
def thumb_dir(tmp_path, monkeypatch):
    d = tmp_path / "thumbs"
    d.mkdir()
    monkeypatch.setattr(lib, "THUMB_DIR", d)
    return d


def _mpo(path: Path, main_size, preview_size):
    """A camera-style JPEG: red main image + blue MPF preview as frame 1."""
    main = Image.new("RGB", main_size, RED)
    prev = Image.new("RGB", preview_size, BLUE)
    main.save(path, "MPO", save_all=True, append_images=[prev])


def _centre(p: Path):
    with Image.open(p) as im:
        im = im.convert("RGB")
        return im.size, im.getpixel((im.width // 2, im.height // 2))


def _near(a, b, tol=40):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_camera_jpeg_renders_from_mpf_preview(tmp_path, thumb_dir):
    src = tmp_path / "DSC0001.JPG"
    _mpo(src, (3000, 2000), (1616, 1080))
    out = lib._render_thumb(str(src))
    size, px = _centre(out)
    assert max(size) == lib.THUMB_PX
    assert _near(px, BLUE), "rendered from the main image, not the MPF preview"


def test_too_small_preview_falls_back_to_main_image(tmp_path, thumb_dir):
    src = tmp_path / "DSC0002.JPG"
    _mpo(src, (3000, 2000), (320, 240))
    size, px = _centre(lib._render_thumb(str(src)))
    assert max(size) == lib.THUMB_PX
    assert _near(px, RED)


def test_plain_jpeg_renders_sharp_size(tmp_path, thumb_dir):
    src = tmp_path / "plain.jpg"
    Image.new("RGB", (4000, 3000), RED).save(src, "JPEG")
    size, _ = _centre(lib._render_thumb(str(src)))
    assert size == (448, 336)


def test_second_render_is_a_cache_hit(tmp_path, thumb_dir):
    src = tmp_path / "plain.jpg"
    Image.new("RGB", (1000, 800), RED).save(src, "JPEG")
    first = lib._render_thumb(str(src))
    mtime = first.stat().st_mtime_ns
    assert lib._render_thumb(str(src)) == first
    assert first.stat().st_mtime_ns == mtime


def _cache_path(p: Path) -> Path:
    return lib.THUMB_DIR / lib._thumb_cache_name(p.resolve())


def test_camera_pair_shares_one_render(tmp_path, thumb_dir):
    jpg = tmp_path / "DSC0003.JPG"
    raw = tmp_path / "DSC0003.ARW"
    _mpo(jpg, (3000, 2000), (1616, 1080))
    raw.write_bytes(b"not decoded - the pair copy must not touch it")
    t = jpg.stat().st_mtime
    os.utime(raw, (t, t + 1.0))                  # written with the shutter press
    out = lib._render_thumb(str(jpg))
    assert _cache_path(raw).read_bytes() == out.read_bytes()


def test_later_export_is_not_treated_as_a_pair(tmp_path, thumb_dir):
    jpg = tmp_path / "DSC0004.jpg"
    raw = tmp_path / "DSC0004.ARW"
    _mpo(jpg, (3000, 2000), (1616, 1080))
    raw.write_bytes(b"x")
    t = jpg.stat().st_mtime
    os.utime(raw, (t, t - 3600))                 # RAW shot an hour before the export
    lib._render_thumb(str(jpg))
    assert not _cache_path(raw).exists()


def test_pair_requested_together_renders_once(tmp_path, thumb_dir, monkeypatch):
    """The browser asks for X.JPG and X.ARW at the same moment; before the pair
    key, both rendered (measured: only 36 of 64 pairs shared a render)."""
    import asyncio
    import time
    from thumb_queue import ThumbQueue

    jpg = tmp_path / "DSC0005.JPG"
    raw = tmp_path / "DSC0005.ARW"
    _mpo(jpg, (3000, 2000), (1616, 1080))
    raw.write_bytes(b"not decodable - must be served from the pair copy")
    t = jpg.stat().st_mtime
    os.utime(raw, (t, t))

    renders = []

    def counting_render(path):
        renders.append(Path(path).name)
        time.sleep(0.3)                       # both requests arrive mid-render
        return lib._render_thumb(path)

    monkeypatch.setattr(lib, "_THUMBS", ThumbQueue(counting_render, workers=4))

    async def both():
        return await asyncio.gather(lib.serve_thumb(path=str(jpg)),
                                    lib.serve_thumb(path=str(raw)))

    r_jpg, r_raw = asyncio.run(both())
    assert renders == ["DSC0005.JPG"]
    assert Path(r_jpg.path).read_bytes() == Path(r_raw.path).read_bytes()


def test_corrupt_file_returns_none(tmp_path, thumb_dir):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"\xff\xd8 definitely not a jpeg")
    assert lib._render_thumb(str(bad)) is None
    assert lib._render_thumb(str(tmp_path / "missing.jpg")) is None
    assert not list(thumb_dir.rglob("*.tmp.webp")), "temp file leaked"
