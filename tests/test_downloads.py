"""Downloads (2026-10-04: 'I can't download any picture'):
  - the app window had pywebview downloads OFF (ALLOW_DOWNLOADS False),
  - single downloads went through /api/photo, which serves a PREVIEW,
  - the zip was fetched through /api/photo, which refuses .zip (HTTP 400).
"""
import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
import server_impl  # noqa: E402,F401  (circular with routers — load first)
from routers import export, library  # noqa: E402


def _body(resp) -> bytes:
    return Path(resp.path).read_bytes()


def test_download_serves_the_original_file_as_an_attachment(tmp_path):
    from PIL import Image
    p = tmp_path / "DSC1.jpg"
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(p, quality=95)
    resp = asyncio.run(library.download_original(path=str(p)))
    assert _body(resp) == p.read_bytes()                  # original bytes, not a preview
    assert 'attachment; filename="DSC1.jpg"' in resp.headers["content-disposition"]


def test_download_refuses_non_images(tmp_path):
    f = tmp_path / "secrets.txt"
    f.write_text("x")
    with pytest.raises(HTTPException):
        asyncio.run(library.download_original(path=str(f)))


def test_zip_file_serves_only_from_the_export_folder(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "batch").mkdir(parents=True)
    z = out / "batch" / "batch_1.zip"
    z.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(server_impl, "_OUTPUT_DIR_ZIP", out)
    resp = asyncio.run(export.download_zip(name="batch_1.zip"))
    assert _body(resp) == z.read_bytes()
    for bad in ("../batch_1.zip", "..\\..\\x.zip", "batch_1.txt", "C:\\Windows\\win.ini"):
        with pytest.raises(HTTPException):
            asyncio.run(export.download_zip(name=bad))


def test_app_window_allows_downloads():
    src = (_ROOT / "src" / "local_launcher.py").read_text(encoding="utf-8")
    assert "webview.settings['ALLOW_DOWNLOADS'] = True" in src


def test_zip_file_serves_editorial_archives_inside_output(tmp_path, monkeypatch):
    """The story-carousel export writes output/editorial/<ts>/editorial_<ts>.zip
    and the app fetched it through /api/photo too (HTTP 400)."""
    out = tmp_path / "output"
    d = out / "editorial" / "20261004_010101"
    d.mkdir(parents=True)
    z = d / "editorial_20261004_010101.zip"
    z.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    monkeypatch.setattr(server_impl, "_OUTPUT_DIR_ZIP", out)
    resp = asyncio.run(export.download_zip(name=str(z)))
    assert _body(resp) == z.read_bytes()
    outside = tmp_path / "elsewhere.zip"
    outside.write_bytes(b"PK")
    with pytest.raises(HTTPException):
        asyncio.run(export.download_zip(name=str(outside)))
