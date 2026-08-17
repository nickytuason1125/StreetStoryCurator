"""
XMP sidecars must be safe to run over a photographer's existing library.

The rules that matter, in order:
  1. originals are never touched
  2. an existing sidecar's other metadata (develop settings, crops, keywords)
     survives — we only update rating/label
  3. writes are atomic; a crash never leaves a half-written sidecar
  4. Lightroom's expected filename convention (photo.RW2 -> photo.xmp)

Run:  venv\\Scripts\\python.exe -m pytest tests/test_xmp_sidecar.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import xmp_sidecar as X  # noqa: E402


def test_filename_convention(tmp_path):
    assert X.sidecar_path(str(tmp_path / "P1100482.RW2")).name == "P1100482.xmp"
    assert X.sidecar_path(str(tmp_path / "shot.jpg")).name == "shot.xmp"


def test_original_file_is_never_touched(tmp_path):
    img = tmp_path / "a.RW2"
    img.write_bytes(b"RAWDATA-DO-NOT-TOUCH")
    before = img.read_bytes(), img.stat().st_mtime_ns
    X.write_sidecar(str(img), stars=5, grade="Strong ✅")
    assert img.read_bytes() == before[0], "the original image was modified"
    assert img.stat().st_mtime_ns == before[1], "the original's mtime changed"


def test_rating_and_label_written(tmp_path):
    img = tmp_path / "a.RW2"; img.write_bytes(b"x")
    sc = X.write_sidecar(str(img), stars=4, grade="Strong ✅")
    xml = sc.read_text(encoding="utf-8")
    assert "<xmp:Rating>4</xmp:Rating>" in xml
    assert "<xmp:Label>Green</xmp:Label>" in xml


@pytest.mark.parametrize("grade,label", [
    ("Strong ✅", "Green"), ("Mid ⚠️", "Yellow"), ("Weak ❌", "Red"),
])
def test_grade_maps_to_label(tmp_path, grade, label):
    img = tmp_path / "g.RW2"; img.write_bytes(b"x")
    sc = X.write_sidecar(str(img), stars=0, grade=grade)
    assert f"<xmp:Label>{label}</xmp:Label>" in sc.read_text(encoding="utf-8")


def test_existing_lightroom_metadata_survives(tmp_path):
    """The critical safety property: we must not destroy develop settings."""
    img = tmp_path / "b.RW2"; img.write_bytes(b"x")
    sc = tmp_path / "b.xmp"
    sc.write_text(
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
        '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
        '    crs:Exposure2012="+0.45"\n'
        '    crs:Temperature="5200"\n'
        '    xmp:Rating="1">\n'
        '   <dc:subject>keyword-one</dc:subject>\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n<?xpacket end="w"?>\n', encoding="utf-8")

    X.write_sidecar(str(img), stars=5, grade="Strong ✅")
    xml = sc.read_text(encoding="utf-8")

    assert 'crs:Exposure2012="+0.45"' in xml, "develop settings were destroyed"
    assert 'crs:Temperature="5200"' in xml, "white balance was destroyed"
    assert "keyword-one" in xml, "keywords were destroyed"
    assert 'xmp:Rating="5"' in xml or "<xmp:Rating>5</xmp:Rating>" in xml
    assert '"1"' not in xml.split("crs:")[0], "old rating left behind"


def test_attribute_form_rating_is_updated_not_duplicated(tmp_path):
    img = tmp_path / "c.RW2"; img.write_bytes(b"x")
    sc = tmp_path / "c.xmp"
    sc.write_text(
        '<rdf:RDF xmlns:rdf="r"><rdf:Description rdf:about=""'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="2">'
        '</rdf:Description></rdf:RDF>', encoding="utf-8")
    X.write_sidecar(str(img), stars=5)
    xml = sc.read_text(encoding="utf-8")
    assert xml.count("Rating") == 1, f"rating duplicated: {xml}"
    assert 'xmp:Rating="5"' in xml


def test_namespace_added_when_missing(tmp_path):
    img = tmp_path / "d.RW2"; img.write_bytes(b"x")
    sc = tmp_path / "d.xmp"
    sc.write_text('<rdf:RDF xmlns:rdf="r"><rdf:Description rdf:about="">'
                  '</rdf:Description></rdf:RDF>', encoding="utf-8")
    X.write_sidecar(str(img), stars=3)
    xml = sc.read_text(encoding="utf-8")
    assert "xmlns:xmp=" in xml and "<xmp:Rating>3</xmp:Rating>" in xml


def test_no_temp_file_left_behind(tmp_path):
    img = tmp_path / "e.RW2"; img.write_bytes(b"x")
    X.write_sidecar(str(img), stars=2, grade="Mid ⚠️")
    assert not list(tmp_path.glob("*.tmp")), "atomic write left a temp file"


def test_nothing_written_when_there_is_nothing_to_say(tmp_path):
    img = tmp_path / "f.RW2"; img.write_bytes(b"x")
    assert X.write_sidecar(str(img), stars=None, grade="") is None
    assert not (tmp_path / "f.xmp").exists()


def test_stars_are_clamped(tmp_path):
    img = tmp_path / "h.RW2"; img.write_bytes(b"x")
    sc = X.write_sidecar(str(img), stars=99)
    assert "<xmp:Rating>5</xmp:Rating>" in sc.read_text(encoding="utf-8")
    sc = X.write_sidecar(str(img), stars=-3)
    assert "<xmp:Rating>0</xmp:Rating>" in sc.read_text(encoding="utf-8")


def test_dry_run_writes_nothing(tmp_path):
    img = tmp_path / "i.RW2"; img.write_bytes(b"x")
    p = X.write_sidecar(str(img), stars=5, grade="Strong ✅", dry_run=True)
    assert p is not None and not p.exists()


def test_gallery_only_rated_filter(tmp_path):
    for n in ("r1", "r2", "u1"):
        (tmp_path / f"{n}.RW2").write_bytes(b"x")
    photos = [
        {"path": str(tmp_path / "r1.RW2"), "stars": 5, "grade": "Strong ✅"},
        {"path": str(tmp_path / "r2.RW2"), "stars": 3, "grade": "Mid ⚠️"},
        {"path": str(tmp_path / "u1.RW2"), "stars": 0, "grade": "Weak ❌"},
    ]
    assert X.write_for_gallery(photos, only_rated=True) == 2
    assert not (tmp_path / "u1.xmp").exists()
    assert X.write_for_gallery(photos, only_rated=False) == 3


def test_unwritable_location_does_not_raise(tmp_path):
    """A read-only card must degrade, not crash the cull."""
    assert X.write_sidecar(str(tmp_path / "nope" / "x.RW2"), stars=5) is None
