"""
Every shared name a router uses must actually resolve.

The server split moved 3,800 lines out of server_impl.py into routers/. Names
that were module globals there became free variables here, and routers/*.py
resolves them through a PEP 562 module __getattr__ that forwards to
server_impl. When a name exists in NEITHER place, nothing complains at import
time — the module loads, the route registers, route-parity passes, and the
whole test suite is green. It raises NameError only when a request actually
reaches the line.

That is how `_CATALOG_PATH` shipped: defined solely in routers/misc.py while
routers/grading.py referenced it five times, so every grade started from the UI
died with

    NameError: name '_CATALOG_PATH' is not defined

...inside the SSE body, where it surfaced as a broken stream rather than a
traceback anyone would read.

This scans each router for SHOUTY_CASE names it uses but neither defines nor
imports, and asserts server_impl can supply them. Static and fast — no server,
no requests — so it runs on every suite.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_router_globals_resolve.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

_ROUTERS = sorted((_ROOT / "routers").glob("*.py"))


def _free_shared_names(path: Path) -> set[str]:
    """SHOUTY_CASE names this module reads but never binds itself."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                    bound.add(arg.arg)
        elif isinstance(node, ast.Global):
            bound.update(node.names)

    used = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        # The convention for shared state in this codebase: leading underscore,
        # upper case. Narrow on purpose — a broad scan drowns in locals.
        and n.id.startswith("_") and n.id.upper() == n.id and len(n.id) > 3
    }
    return used - bound


@pytest.mark.parametrize("router", _ROUTERS, ids=lambda p: p.stem)
def test_shared_names_resolve_in_server_impl(router):
    free = _free_shared_names(router)
    if not free:
        pytest.skip(f"{router.name} references no shared globals")

    import server_impl  # importing this mounts the routers, as the app does

    missing = sorted(n for n in free if not hasattr(server_impl, n))
    assert not missing, (
        f"routers/{router.name} uses {missing} which server_impl does not "
        f"define. The module imports fine and the route registers; it raises "
        f"NameError only when a request reaches that line."
    )
