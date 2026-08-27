"""
Routers must not build filesystem paths from their own directory.

This code lived in server_impl.py at the repo root, where
`Path(__file__).parent` and `dirname(__file__)` meant the unit root. The
server split moved 3,800 lines into routers/ and every such expression came
along unchanged — silently resolving one level too deep. Four separate
failures came from it, and every one was invisible to the test suite because
the module still imports, the route still registers, and the wrong path is
only touched when a request arrives:

  routers/grading.py   grade_runner.py  -> routers/grade_runner.py
                       the Grade button could not start a grade at all
                       crash.log        -> routers/crash.log
  routers/extras.py    scripts/         -> routers/scripts/
                       "Download Missing Models" pointed at nothing
                       cwd=routers/     -> the critique child's
                       sys.path.insert(0,'src') became routers/src, so it
                       died with ModuleNotFoundError: critique_engine

The fix in both files is an explicit _UNIT_ROOT. This test keeps it that way.

Companion to test_router_globals_resolve.py, which covers the NAME half of
the same class.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_router_paths_anchored.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_ROUTERS = sorted((_ROOT / "routers").glob("*.py"))

# `Path(__file__).parent` / `dirname(__file__)` NOT immediately followed by a
# second .parent (or a second dirname), i.e. anything that stops at routers/.
_SELF_DIR = re.compile(
    r"Path\(__file__\)\.resolve\(\)\.parent(?!\.parent)"
    r"|Path\(__file__\)\.parent(?!\.parent)"
    r"|os\.path\.dirname\(os\.path\.abspath\(__file__\)\)"
)


def _strip_comments(src: str) -> str:
    """Comments explain this bug at length and quote the bad expression."""
    out = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("  #")[0])
    return "\n".join(out)


@pytest.mark.parametrize("router", _ROUTERS, ids=lambda p: p.stem)
def test_no_path_anchored_on_the_router_directory(router):
    body = _strip_comments(router.read_text(encoding="utf-8"))

    # An _UNIT_ROOT definition is the correct use and must be allowed through.
    body = re.sub(r"_UNIT_ROOT\s*=.*", "", body)

    hits = _SELF_DIR.findall(body)
    assert not hits, (
        f"routers/{router.name} builds a path from its OWN directory "
        f"({hits[0]!r}). Under routers/ that resolves one level too deep — "
        f"src/, scripts/, static/ and grade_runner.py all live at the unit "
        f"root. Use _UNIT_ROOT. The module will import and the route will "
        f"register either way; it only fails when a request arrives."
    )


def test_the_guard_sees_the_routers():
    assert len(_ROUTERS) >= 8, f"expected the split's routers, found {_ROUTERS}"
