"""
The catalog must accumulate, not replace.

Grading folder B used to wipe folder A out of the gallery. These lock in the
behaviour a photo catalog is expected to have:

  1. Merging a new folder keeps every other folder.
  2. Re-grading a folder UPDATES its photos rather than duplicating them.
  3. A corrupt / truncated / missing catalog degrades to empty, never raises —
     and can still be written correctly afterwards.
  4. Writes are atomic (no .tmp left behind, no partial file).
  5. Embeddings never get written into the catalog.
  6. It can be rebuilt from LanceDB, which is the real source of truth.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_catalog_store.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import catalog_store as cs  # noqa: E402


def photo(path: str, score: float = 0.5, grade: str = "Mid ⚠️") -> dict:
    return {"id": path, "path": path, "filename": Path(path).name,
            "score": score, "grade": grade, "stars": 0}


def test_second_folder_does_not_erase_the_first(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/A/a1.jpg"), photo(r"C:/A/a2.jpg")], path=cat)
    cs.merge_write([photo(r"C:/B/b1.jpg")], path=cat)

    d = cs.load(cat)
    paths = {p["path"] for p in d["photos"]}
    assert paths == {r"C:/A/a1.jpg", r"C:/A/a2.jpg", r"C:/B/b1.jpg"}, \
        "grading folder B erased folder A — this is the bug being fixed"
    assert len(d["folders"]) == 2


def test_regrade_updates_in_place_without_duplicating(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/A/a1.jpg", 0.30, "Weak ❌")], path=cat)
    cs.merge_write([photo(r"C:/A/a1.jpg", 0.80, "Strong ✅")], path=cat)

    d = cs.load(cat)
    assert len(d["photos"]) == 1, "re-grading duplicated the photo"
    assert d["photos"][0]["score"] == 0.80
    assert "Strong" in d["photos"][0]["grade"]


def test_many_folders_accumulate(tmp_path):
    cat = tmp_path / "catalog.json"
    for i in range(5):
        cs.merge_write([photo(f"C:/F{i}/x.jpg"), photo(f"C:/F{i}/y.jpg")], path=cat)
    d = cs.load(cat)
    assert len(d["photos"]) == 10
    assert len(d["folders"]) == 5


@pytest.mark.parametrize("junk", ["", "{", "not json at all", "[]", '{"photos": "nope"}'])
def test_corrupt_catalog_degrades_to_empty_and_is_recoverable(tmp_path, junk):
    cat = tmp_path / "catalog.json"
    cat.write_text(junk, encoding="utf-8")
    assert cs.load(cat) == {"photos": [], "folders": []}
    # and a subsequent write must still produce a good catalog
    cs.merge_write([photo(r"C:/A/a1.jpg")], path=cat)
    assert len(cs.load(cat)["photos"]) == 1


def test_missing_catalog_is_fine(tmp_path):
    assert cs.load(tmp_path / "nope.json") == {"photos": [], "folders": []}


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/A/a1.jpg")], path=cat)
    assert cat.exists()
    assert not list(tmp_path.glob("*.tmp")), "temp file left behind"
    json.loads(cat.read_text(encoding="utf-8"))       # parses = not truncated


def test_embeddings_are_never_written(tmp_path):
    cat = tmp_path / "catalog.json"
    p = photo(r"C:/A/a1.jpg"); p["embedding"] = [0.1] * 1536
    cs.merge_write([p], path=cat)
    assert "embedding" not in cs.load(cat)["photos"][0]
    assert cat.stat().st_size < 4000, "embedding bloated the catalog"


def test_entries_without_a_path_are_skipped(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/A/a1.jpg"), {"filename": "orphan.jpg"}], path=cat)
    assert len(cs.load(cat)["photos"]) == 1


def test_rebuild_from_lance_restores_photos_and_stars(tmp_path, monkeypatch):
    """The catalog is a projection of LanceDB — it must be regenerable."""
    rows = [
        {"path": r"C:/A/a1.jpg", "score": 0.71, "grade": "Strong ✅",
         "personal_score": 0.6, "breakdown": '{"Technical": 0.7}', "exif_ts": 1.0},
        {"path": r"C:/B/b1.jpg", "score": 0.33, "grade": "Weak ❌",
         "personal_score": 0.4, "breakdown": {}, "exif_ts": 2.0},
    ]
    fake_ls = type("LS", (), {"query_all": staticmethod(lambda min_score=0.0: rows)})
    fake_rs = type("RS", (), {"load": staticmethod(lambda: {r"C:/A/a1.jpg": 5})})
    monkeypatch.setitem(sys.modules, "lance_store", fake_ls)
    monkeypatch.setitem(sys.modules, "ratings_store", fake_rs)

    cat = tmp_path / "catalog.json"
    n = cs.rebuild_from_lance(path=cat)
    assert n == 2
    d = cs.load(cat)
    by = {p["path"]: p for p in d["photos"]}
    assert by[r"C:/A/a1.jpg"]["score"] == 0.71
    assert by[r"C:/A/a1.jpg"]["breakdown"] == {"Technical": 0.7}, "breakdown JSON not parsed"
    assert by[r"C:/A/a1.jpg"]["stars"] == 5, "user's star rating not restored"
    assert len(d["folders"]) == 2


def test_rebuild_survives_an_unavailable_store(tmp_path, monkeypatch):
    broken = type("LS", (), {"query_all": staticmethod(
        lambda min_score=0.0: (_ for _ in ()).throw(RuntimeError("db down")))})
    monkeypatch.setitem(sys.modules, "lance_store", broken)
    assert cs.rebuild_from_lance(path=tmp_path / "c.json") == 0


# ── Purging a prefix ─────────────────────────────────────────────────────────
# The project's own smoke test grades five generated frames in a temp folder.
# It purged its LanceDB rows in a `finally:` block but never touched the
# catalog, so running the ship-readiness check permanently salted the user's
# library with synthetic noise — and once the temp dir was cleaned up, with
# entries pointing at files that no longer exist. Deleting a prefix is the
# catalog's own operation, so it lives here with the same atomic write.

def test_purge_prefix_removes_only_the_matching_rows(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/Photos/keep_1.jpg"),
                    photo(r"C:/Photos/keep_2.jpg"),
                    photo(r"C:/Temp/fg_selftest_abc/img/frame_0.jpg"),
                    photo(r"C:/Temp/fg_selftest_abc/img/flat.jpg")], path=cat)

    removed = cs.purge_prefix(r"C:/Temp/fg_selftest_abc", path=cat)

    assert removed == 2
    paths = {p["path"] for p in cs.load(cat)["photos"]}
    assert paths == {r"C:/Photos/keep_1.jpg", r"C:/Photos/keep_2.jpg"}


def test_purge_prefix_is_case_insensitive_and_slash_agnostic(tmp_path):
    """Windows hands the same folder back as C:\Temp and c:/temp."""
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:\Temp\fg_selftest_abc\img\frame_0.jpg"),
                    photo(r"C:/Photos/keep.jpg")], path=cat)

    removed = cs.purge_prefix("c:/temp/fg_selftest_abc", path=cat)

    assert removed == 1
    assert {p["path"] for p in cs.load(cat)["photos"]} == {r"C:/Photos/keep.jpg"}


def test_purge_prefix_no_match_leaves_the_file_untouched(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/Photos/keep.jpg")], path=cat)
    before = cat.read_text(encoding="utf-8")

    assert cs.purge_prefix(r"C:/Nowhere", path=cat) == 0
    assert cat.read_text(encoding="utf-8") == before


def test_purge_prefix_never_matches_a_partial_folder_name(tmp_path):
    """`C:/Temp/fg` must not eat `C:/Temp/fg_selftest_abc`'s neighbour."""
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/Temp/fg_real/keeper.jpg"),
                    photo(r"C:/Temp/fg/frame.jpg")], path=cat)

    removed = cs.purge_prefix(r"C:/Temp/fg", path=cat)

    assert removed == 1, "prefix matching must respect folder boundaries"
    assert {p["path"] for p in cs.load(cat)["photos"]} == {r"C:/Temp/fg_real/keeper.jpg"}


def test_purge_prefix_on_a_missing_catalog_is_a_no_op(tmp_path):
    assert cs.purge_prefix(r"C:/Anything", path=tmp_path / "nope.json") == 0


# ── Backing up before a destructive rebuild ──────────────────────────────────
# /api/regrade moved the catalog aside before clearing it; /api/scan did the
# same clearing with force_rescan=True and NO backup. A scan that then failed —
# a RAM refusal is the common case — destroyed the library with nothing to fall
# back to, which is consistent with a 21,416-photo catalog going to zero with no
# .pre-regrade.bak beside it. One helper now, so the two paths cannot drift.

def test_back_up_moves_the_catalog_aside(tmp_path):
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/Photos/a.jpg")], path=cat)

    assert cs.back_up("scan", path=cat) is True
    assert not cat.exists(), "the live catalog is moved, not copied"
    bak = cat.with_name("catalog.json.pre-regrade.bak")
    assert bak.exists()
    assert {p["path"] for p in json.loads(bak.read_text(encoding="utf-8"))["photos"]} \
        == {r"C:/Photos/a.jpg"}


def test_back_up_with_no_catalog_is_a_no_op(tmp_path):
    assert cs.back_up("scan", path=tmp_path / "nope.json") is False
    assert not (tmp_path / "catalog.json.pre-regrade.bak").exists()


def test_back_up_does_not_destroy_an_earlier_backup_when_there_is_nothing_to_save(tmp_path):
    """A second scan with no live catalog must not clobber the recovery copy."""
    cat = tmp_path / "catalog.json"
    cs.merge_write([photo(r"C:/Photos/precious.jpg")], path=cat)
    cs.back_up("regrade", path=cat)          # first failed run: catalog -> bak
    bak = cat.with_name("catalog.json.pre-regrade.bak")
    before = bak.read_text(encoding="utf-8")

    cs.back_up("scan", path=cat)             # second run, nothing live to move

    assert bak.read_text(encoding="utf-8") == before, \
        "the only copy of the user's grades was overwritten by an empty run"
