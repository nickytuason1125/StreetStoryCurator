"""
The test suite must not be able to reach the photographer's stores.

A live library was found holding 203 synthetic rows: benchmark frames, smoke
test frames, and pytest fixtures from three separate runs. lance_store anchors
its path next to the source tree and a grade request's data_dir does not
redirect it, so every test that reached the store wrote into real data.

This asserts the isolation rather than trusting conftest.py to have worked.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

REAL_DB = _ROOT / "cache" / "lance.db"


def test_the_override_is_set_for_the_session():
    assert os.environ.get("FRAMEGRADE_LANCE_DIR"), \
        "conftest.py must set FRAMEGRADE_LANCE_DIR before any test imports lance_store"


def test_lance_store_points_somewhere_temporary():
    import lance_store
    used = Path(lance_store._DB_DIR).resolve()
    assert used != REAL_DB.resolve(), (
        f"lance_store is pointed at the REAL library ({used}). A test run will "
        f"write synthetic rows into the photographer's vector store."
    )


def test_the_real_store_is_not_under_the_temp_root():
    """Guards the guard: an override that happens to resolve back to the real
    path would pass the check above while changing nothing."""
    import lance_store
    used = Path(lance_store._DB_DIR).resolve()
    assert REAL_DB.resolve() not in used.parents and used != REAL_DB.resolve()
