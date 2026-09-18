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
os.environ["FIRSTCUT_LANCE_DIR"] = str(_TMP / "lance.db")
# The persisted LIBRARY TIER must not leak between tests either: it is state
# on disk (cache/library_tier.json), so point it at the scratch dir. Tests
# that exercise it set FIRSTCUT_LIB_TIER_FILE explicitly anyway (see
# test_library_tier.py); this default keeps every OTHER test hermetic.
os.environ["FIRSTCUT_LIB_TIER_FILE"] = str(_TMP / "library_tier.json")

# Scrub session-level SIGLIP/FIRSTCUT overrides (2026-09-13). The dev machine's
# terminal had SIGLIP_TIER=low set at session level; it leaked into the pytest
# process and its subprocesses, switching _PROFILE to the low tier so
# test_ram_sensitivity (which assumes the high-tier floors 4.0/3.0) and
# test_portability (batch 8) failed with low-tier numbers (batch 2, floor 1.2).
# Tests that WANT an override set it explicitly; a stray one from the shell
# session must never be able to fake a failure.
for _k in list(os.environ):
    if _k.startswith(("SIGLIP_", "FIRSTCUT_")) and _k not in ("FIRSTCUT_LANCE_DIR", "FIRSTCUT_LIB_TIER_FILE"):
        del os.environ[_k]

# Production pauses INDEFINITELY on OOM (the keep-trucking contract); in tests
# that is a hang — the suite runs while the dev machine itself is often below
# the hard floor, and any in-process _enforce_ram_floor would park forever.
# Bounded pause: tests assert readable outcomes, not infinite patience. Tests
# that need the pause semantics explicitly set their own budget
# (test_floor_pauses_then_resumes_when_memory_recovers).
os.environ["SIGLIP_PAUSE_MAX_WAIT_S"] = "0.05"


import pytest


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot and restore os.environ around EVERY test (2026-09-13).

    The grading pipeline legitimately mutates the process env at runtime:
    _enforce_ram_floor writes SIGLIP_ENC_BATCH=2 on a low-RAM machine, and
    tier_select.select/apply publish SIGLIP_TIER. On a memory-starved dev
    machine those fire during tests that import siglip2_encoder, and the
    values then leak into later, unrelated tests — test_portability's
    'assert 2 == 8' was literally the OOM-shrunk batch, and the floor tests
    saw the auto-selected low tier's 1.6/1.2 GB floors instead of high's
    4.0/3.0. Each test now starts from the scrubbed session state.
    """
    _snap = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(_snap)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Best-effort cleanup. A leftover temp dir is untidy; a polluted library
    is a bug, and that is the one this file exists to prevent."""
    import shutil
    shutil.rmtree(_TMP, ignore_errors=True)
