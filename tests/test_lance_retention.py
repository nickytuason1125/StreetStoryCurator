"""
LanceDB version retention.

photos.lance reached 859 MB holding ~10 MB of vectors — 409 versions left behind
because compact_files() merges fragments but never deletes old versions. These
tests pin the fix: history is reaped, current data is untouched, and a cleanup
failure can never fail a cull whose grades are already committed.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_lance_retention.py -v
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _make_table(tmp_path, n_versions: int = 5):
    """A real LanceDB table with several versions of history."""
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    db = lancedb.connect(str(tmp_path / "t.db"))
    schema = pa.schema([
        pa.field("path", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 4)),
    ])
    tbl = db.create_table("photos", schema=schema)
    for v in range(n_versions):
        tbl.merge_insert("path").when_matched_update_all().when_not_matched_insert_all().execute(
            pa.table({"path": [f"img{i}.jpg" for i in range(3)],
                      "embedding": [[float(v), 0.0, 0.0, 0.0] for _ in range(3)]})
        )
    return tbl


def test_cleanup_reaps_history_but_keeps_rows(tmp_path):
    tbl = _make_table(tmp_path, n_versions=5)
    before = len(tbl.list_versions())
    rows_before = tbl.count_rows()
    assert before > 1, "fixture must produce history to reap"

    tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=False)

    assert len(tbl.list_versions()) < before, "old versions must be reaped"
    assert tbl.count_rows() == rows_before, "cleanup must not lose rows"


def test_current_version_always_survives(tmp_path):
    """A zero-length retention window must not leave an unreadable table."""
    tbl = _make_table(tmp_path, n_versions=3)
    tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=False)

    assert len(tbl.list_versions()) >= 1
    assert tbl.count_rows() == 3
    assert tbl.to_arrow().num_rows == 3, "table must stay readable"


def test_retention_setting_is_declared():
    """Undeclared settings raise, so this also pins the spelling."""
    import run_profile
    days = run_profile.setting("FRAMEGRADE_LANCE_RETENTION_DAYS")
    assert isinstance(days, int)
    assert days == 7, "default retention window is 7 days"


def test_cleanup_failure_never_raises(monkeypatch):
    """Housekeeping runs AFTER grades are committed. It must degrade, not raise."""
    import lance_store

    class Boom:
        def optimize(self, **kw):
            raise RuntimeError("simulated lance failure")

        def compact_files(self):
            raise RuntimeError("simulated lance failure")

    monkeypatch.setattr(lance_store, "_open_table", lambda: Boom())
    lance_store.compact_after_write()      # must return normally


def test_no_second_lancedb_is_ever_created():
    """cache/lancedb_v2 was a parallel store with a conflicting schema.

    src/lance_migration.py defined it: table photos_v2, a `confidence` column
    the live schema lacks, and `breakdown` as a JSON string where the live store
    uses a struct. It created the directory at IMPORT time on a CWD-relative
    path. The module never ran only because line 1 was the literal text
    "but ar", so it raised SyntaxError on import — corruption that was committed,
    not a local accident.

    Deleted. This guards against it returning: a second store would silently
    split grades across two databases.
    """
    assert not (_ROOT / "src" / "lance_migration.py").exists(), \
        "lance_migration.py is dead code that creates a conflicting second store"
    assert not (_ROOT / "cache" / "lancedb_v2").exists(), \
        "a second LanceDB appeared - something is importing a migration module"
