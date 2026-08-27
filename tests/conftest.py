"""Session-wide isolation: tests must never touch the real stores.

`lance_store` anchors _DB_DIR to <unit>/cache/lance.db and `data_dir` in a grade
request does not redirect it, so anything that reached the store during a test
wrote into the photographer's live library. That is not theoretical — a working
library was found holding 203 synthetic rows, including fixtures from three
separate pytest runs (pytest-1, pytest-7, pytest-8) sitting alongside real
photographs.

The env var is set here, before any test imports lance_store, because that
module reads it once at import time. autouse fixtures run too late.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fg_test_lance_"))
os.environ["FRAMEGRADE_LANCE_DIR"] = str(_TMP / "lance.db")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Best-effort cleanup. A leftover temp dir is untidy; a polluted library
    is a bug, and that is the one this file exists to prevent."""
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
