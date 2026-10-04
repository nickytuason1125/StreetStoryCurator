"""Ratings follow the photo (content fingerprint), not its path."""
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import photo_identity as pid  # noqa: E402
import ratings_store as rs  # noqa: E402


def _photo(p: Path, seed: int) -> Path:
    p.write_bytes(bytes([seed]) * 200_000)
    return p


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")


def test_copy_has_same_fingerprint_edit_does_not(tmp_path):
    a = _photo(tmp_path / "DSC1.ARW", 7)
    b = tmp_path / "copy" / "renamed.ARW"
    b.parent.mkdir()
    shutil.copy2(a, b)
    c = _photo(tmp_path / "DSC2.ARW", 8)
    assert pid.fingerprint(str(a)) == pid.fingerprint(str(b))
    assert pid.fingerprint(str(a)) != pid.fingerprint(str(c))
    assert pid.fingerprint(str(tmp_path / "missing.ARW")) is None


def test_rating_follows_a_copied_file(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    a = _photo(tmp_path / "DSC1.ARW", 7)
    rs.set_rating(str(a), 1)
    moved = tmp_path / "laptop" / "DSC1.ARW"
    moved.parent.mkdir()
    shutil.move(a, moved)
    assert rs.stars_for_paths([str(moved)]) == {str(moved): 1}


def test_unrated_photos_are_absent(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    x = _photo(tmp_path / "X.ARW", 3)
    assert rs.stars_for_paths([str(x)]) == {}


def test_rerating_keeps_stored_features(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    a = _photo(tmp_path / "DSC1.ARW", 7)
    rs.set_rating(str(a), 2)
    rs.attach_features(str(a), {"Technical": 0.4})
    rs.set_rating(str(a), 5)
    assert rs.get_features(str(a)) == {"Technical": 0.4}


def test_unrated_photos_of_other_sizes_are_not_read(tmp_path, monkeypatch):
    """Review finding: every unrated photo in a cull was fingerprinted (64 KB
    read each) — ~320 MB of extra card reads on a 5,000-photo import."""
    _isolate(tmp_path, monkeypatch)
    rated = _photo(tmp_path / "R.ARW", 7)
    rs.set_rating(str(rated), 4)
    others = []
    for i in range(5):
        q = tmp_path / f"U{i}.ARW"
        q.write_bytes(bytes([9]) * (300_000 + i))       # sizes differ from the rated photo
        others.append(str(q))
    calls = []
    real = pid.fingerprint
    monkeypatch.setattr(pid, "fingerprint", lambda p: calls.append(p) or real(p))
    assert rs.stars_for_paths(others) == {}
    assert calls == []
