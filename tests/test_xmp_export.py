"""XMP sidecar export — the Lightroom/Capture One handoff contract.

These tests encode what "readable by Adobe" means, precisely because the
previous implementation was not: it emitted <xmp:Rating>-style elements with
undeclared namespace prefixes (fatal XML), wrote grade emoji into
photoshop:Label where Lightroom expects colour names, and derived ratings
from the machine score instead of the user's stars. Each test pins one of
those fixes.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from engine_utils import export_metadata          # noqa: E402

NS = {
    "x":         "adobe:ns:meta/",
    "rdf":       "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp":       "http://ns.adobe.com/xap/1.0/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "dc":        "http://purl.org/dc/elements/1.1/",
    "fc":        "https://firstcut.app/ns/1.0/",
}


def _write_sidecar(tmp_path: Path, meta: dict) -> ET.Element:
    img = tmp_path / "IMG_0001.jpg"
    img.write_bytes(b"\xff\xd8fake-jpeg-bytes")
    sidecar = export_metadata(str(img), meta)
    assert sidecar.endswith(".xmp"), f"expected .xmp sidecar, got {sidecar}"
    raw = Path(sidecar).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "Adobe readers expect a leading BOM"
    # ET.parse is the real gate: an undeclared namespace prefix is a fatal
    # ParseError here, which is exactly how Lightroom rejects such files.
    root = ET.fromstring(raw.decode("utf-8-sig"))
    desc = root.find(".//rdf:Description", NS)
    assert desc is not None
    return desc


def test_grade_maps_to_colour_label_and_stars_win(tmp_path: Path):
    d = _write_sidecar(tmp_path, {
        "grade": "Strong ✅", "score": 0.7123, "stars": 4,
        "critique": "Layered framing, window light",
        "breakdown": {"Lighting": 0.8, "Narrative": 0.65},
    })
    assert d.get(f"{{{NS['xmp']}}}Rating") == "4"
    assert d.get(f"{{{NS['photoshop']}}}Label") == "Green"
    assert d.get(f"{{{NS['dc']}}}description") == "Layered framing, window light"
    assert d.get(f"{{{NS['fc']}}}Grade") == "Strong"
    assert d.get(f"{{{NS['fc']}}}Score") == "0.7123"
    bd = json.loads(d.get(f"{{{NS['fc']}}}Breakdown"))
    assert bd == {"Lighting": 0.8, "Narrative": 0.65}


@pytest.mark.parametrize("grade,label", [
    ("Strong ✅", "Green"), ("Mid ⚠️", "Yellow"), ("Weak ❌", "Red"),
])
def test_all_three_buckets_map_to_lr_colours(tmp_path: Path, grade: str, label: str):
    d = _write_sidecar(tmp_path, {"grade": grade, "score": 0.5, "stars": 0})
    assert d.get(f"{{{NS['photoshop']}}}Label") == label


def test_zero_stars_omits_rating_never_fabricates_from_score(tmp_path: Path):
    """The old code wrote round(score*5) — a rating the user never gave."""
    d = _write_sidecar(tmp_path, {"grade": "Mid ⚠️", "score": 0.86, "stars": 0})
    assert d.get(f"{{{NS['xmp']}}}Rating") is None


def test_quotes_and_emoji_survive_escaping(tmp_path: Path):
    d = _write_sidecar(tmp_path, {
        "grade": "Strong ✅", "score": 0.9, "stars": 5,
        "critique": 'He said "wait" — then it clicked <fast>',
    })
    assert d.get(f"{{{NS['dc']}}}description") == 'He said "wait" — then it clicked <fast>'


def test_out_dir_writes_there_and_never_the_original(tmp_path: Path):
    img = tmp_path / "sub" / "IMG_0002.jpg"
    img.parent.mkdir()
    img.write_bytes(b"\xff\xd8fake")
    dest = tmp_path / "sidecars"
    sidecar = export_metadata(str(img), {"grade": "Weak ❌", "score": 0.2}, out_dir=str(dest))
    assert Path(sidecar) == dest / "IMG_0002.xmp"
    assert Path(sidecar).exists()
    assert img.read_bytes() == b"\xff\xd8fake"   # original untouched, byte-for-byte
