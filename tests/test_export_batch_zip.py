"""Zip download: every batch-zip export died with NameError('_OUTPUT_DIR_ZIP')
because a module __getattr__ cannot resolve bare names inside the module."""
import asyncio
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
import server_impl  # noqa: E402,F401  (circular with routers — load first)
from routers import export  # noqa: E402


def test_batch_zip_writes_a_zip_of_the_photos(tmp_path, monkeypatch):
    from PIL import Image
    photos = []
    for i in range(2):
        p = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (32, 32), (i * 80, 10, 10)).save(p)
        photos.append(str(p))
    # Patch the SOURCE (server_impl). Patching export itself would create the
    # missing name and hide the bug (that is how this test first passed).
    monkeypatch.setattr(server_impl, "_OUTPUT_DIR_ZIP", tmp_path / "out")
    resp = asyncio.run(export.export_batch_zip({"paths": photos}))
    zips = list((tmp_path / "out" / "batch").glob("*.zip"))
    assert len(zips) == 1, resp
    assert sorted(zipfile.ZipFile(zips[0]).namelist()) == ["p0.jpg", "p1.jpg"]
