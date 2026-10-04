"""
Explicit per-folder file narrowing (2026-09-23) — the "check individual
photos" picker's server-side half. GradeRequest.explicit_paths maps a folder
to the specific files to grade in it instead of every image run_v2 finds
there; two units carry the actual behaviour and both are untrusted-input
boundaries, so both are tested directly rather than only through the full
streaming endpoint:

  * routers.grading._resolve_explicit_paths — validates the client's dict
    against the folders THIS request already resolved (never trust it blind).
  * grade_pipeline_v2._discover_images — Step 1's file list, re-validating
    again because this also runs from CLI/test entry points that skip the
    HTTP-layer check.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_explicit_paths.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (mounts routers — must load before routers.grading directly)
from routers.grading import _resolve_explicit_paths  # noqa: E402
from grade_pipeline_v2 import _discover_images  # noqa: E402


# ── routers.grading._resolve_explicit_paths ─────────────────────────────────
def test_keeps_paths_that_belong_to_a_resolved_folder(tmp_path):
    a = tmp_path / "a.jpg"; a.write_bytes(b"x")
    b = tmp_path / "b.jpg"; b.write_bytes(b"x")
    result = _resolve_explicit_paths(
        {str(tmp_path): [str(a), str(b)]}, [str(tmp_path)])
    assert result == {str(tmp_path.resolve()): [str(a.resolve()), str(b.resolve())]}


def test_drops_folder_key_not_in_all_folders(tmp_path):
    other = tmp_path / "other"; other.mkdir()
    f = other / "x.jpg"; f.write_bytes(b"x")
    # `other` was never validated as part of this request's folders.
    result = _resolve_explicit_paths({str(other): [str(f)]}, [str(tmp_path)])
    assert result == {}


def test_drops_path_that_belongs_to_a_different_folder(tmp_path):
    """A path whose real parent isn't the claimed folder is dropped, even
    though the folder key itself is valid — the traversal case."""
    real_folder = tmp_path / "real"; real_folder.mkdir()
    sneaky_folder = tmp_path / "sneaky"; sneaky_folder.mkdir()
    outside_file = sneaky_folder / "outside.jpg"; outside_file.write_bytes(b"x")
    result = _resolve_explicit_paths(
        {str(real_folder): [str(outside_file)]}, [str(real_folder)])
    assert result == {}


def test_drops_nonexistent_file(tmp_path):
    result = _resolve_explicit_paths(
        {str(tmp_path): [str(tmp_path / "ghost.jpg")]}, [str(tmp_path)])
    assert result == {}


def test_empty_and_none_input_are_both_a_no_op(tmp_path):
    assert _resolve_explicit_paths({}, [str(tmp_path)]) == {}
    assert _resolve_explicit_paths(None, [str(tmp_path)]) == {}


def test_a_folder_with_no_surviving_paths_is_absent_not_empty_list(tmp_path):
    """Downstream (_discover_images) treats an empty explicit_paths list the
    SAME as a non-empty one it can't glob from — must be a missing key, not
    a key mapped to []."""
    result = _resolve_explicit_paths(
        {str(tmp_path): [str(tmp_path / "ghost.jpg")]}, [str(tmp_path)])
    assert str(tmp_path.resolve()) not in result


def test_malformed_folder_key_does_not_raise():
    result = _resolve_explicit_paths({"\0bad\0path": ["x.jpg"]}, ["C:\\real"])
    assert result == {}


# ── grade_pipeline_v2._discover_images ───────────────────────────────────────
def test_no_explicit_paths_scans_the_whole_folder(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")   # non-image, must be excluded
    result = _discover_images(tmp_path, None)
    assert sorted(Path(p).name for p in result) == ["a.jpg", "b.png"]


def test_explicit_paths_narrows_to_exactly_those_files(tmp_path):
    a = tmp_path / "a.jpg"; a.write_bytes(b"x")
    b = tmp_path / "b.jpg"; b.write_bytes(b"x")
    (tmp_path / "c.jpg").write_bytes(b"x")   # exists but not requested
    result = _discover_images(tmp_path, [str(a), str(b)])
    assert sorted(Path(p).name for p in result) == ["a.jpg", "b.jpg"]


def test_explicit_paths_rejects_a_file_from_a_different_folder(tmp_path):
    """Defense in depth: even if something upstream failed to filter this
    out, _discover_images re-checks parent == folder itself."""
    other = tmp_path / "other"; other.mkdir()
    outside = other / "sneaky.jpg"; outside.write_bytes(b"x")
    inside = tmp_path / "real.jpg"; inside.write_bytes(b"x")
    result = _discover_images(tmp_path, [str(outside), str(inside)])
    assert result == [str(inside)]


def test_explicit_paths_rejects_a_nonexistent_file(tmp_path):
    real = tmp_path / "real.jpg"; real.write_bytes(b"x")
    ghost = tmp_path / "ghost.jpg"   # never created
    result = _discover_images(tmp_path, [str(real), str(ghost)])
    assert result == [str(real)]


def test_explicit_paths_rejects_non_image_extension(tmp_path):
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    doc = tmp_path / "a.txt"; doc.write_bytes(b"x")
    result = _discover_images(tmp_path, [str(img), str(doc)])
    assert result == [str(img)]


def test_empty_explicit_paths_falls_back_to_whole_folder_scan(tmp_path):
    """An empty list is falsy, matching _resolve_explicit_paths never handing
    back a folder with zero surviving paths — both sides agree on what
    'not narrowed' looks like."""
    (tmp_path / "a.jpg").write_bytes(b"x")
    result = _discover_images(tmp_path, [])
    assert [Path(p).name for p in result] == ["a.jpg"]
