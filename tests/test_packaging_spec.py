"""
The PyInstaller spec has to keep up with the source tree.

The server split moved 3,800 lines out of server_impl.py into routers/, and
FrameGrade.spec was never updated. Nothing failed: tsc passed, the suite
passed, the dev server ran, 378 tests were green. Only the PACKAGED app was
broken, and nobody had built one since. server.py does `from server_impl
import app`, server_impl calls routers.mount_all(), and neither server_impl.py
nor routers/ was in datas or hiddenimports — so the shipped binary would have
died at startup with ModuleNotFoundError before painting a single pixel.

That is the worst shape a bug can take: invisible to every check that runs
often, fatal in the artifact users actually receive.

These tests read the spec as text and assert the modules the app cannot start
without are declared. They are deliberately dumb — no PyInstaller import, no
build — so they run in milliseconds on every suite and fail the moment someone
adds a router without telling the packager.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_packaging_spec.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = _ROOT / "FrameGrade.spec"


@pytest.fixture(scope="module")
def spec() -> str:
    if not _SPEC.exists():
        pytest.skip("FrameGrade.spec not present")
    return _SPEC.read_text(encoding="utf-8")


def _router_modules() -> list[str]:
    """Every routers/*.py the app actually ships, discovered, not hardcoded."""
    d = _ROOT / "routers"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.py") if p.stem != "__init__")


def test_the_app_entry_module_is_bundled(spec):
    """server.py is useless without it: `from server_impl import app`."""
    assert "server_impl" in spec, (
        "server_impl.py is neither bundled nor hidden-imported — the packaged "
        "app cannot start"
    )


def test_the_routers_package_is_bundled(spec):
    assert "routers" in spec, "routers/ is not in the spec at all"


@pytest.mark.parametrize("mod", _router_modules())
def test_each_router_is_declared(spec, mod):
    """mount_all() imports these inside a function, so static analysis misses
    them. Each has to be named explicitly or it will not be collected."""
    assert f"routers.{mod}" in spec, (
        f"routers/{mod}.py exists but 'routers.{mod}' is not in the spec's "
        f"hiddenimports — it will be missing from the packaged app"
    )


@pytest.mark.parametrize("mod", ["catalog_store", "system_check"])
def test_lazily_imported_modules_are_declared(spec, mod):
    """Imported inside functions (to keep startup light / avoid cycles), which
    is exactly the pattern PyInstaller's static scan cannot see."""
    assert mod in spec, f"{mod} is imported lazily and must be a hiddenimport"


def test_every_router_on_disk_is_accounted_for():
    """Guards the guard: if routers/ is empty the parametrised tests above all
    silently pass by collecting nothing."""
    mods = _router_modules()
    assert len(mods) >= 8, f"expected the split's routers, found {mods}"
