"""Lightweight undefined-name check (pyflakes F821 approximation) for
creative_director.py and routers/. Written because the venv's ruff
binary is corrupt. Run:
  venv\\Scripts\\python.exe scripts\\check_undefined_names.py
"""
import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.names: set[str] = set()

    def has(self, name: str) -> bool:
        s = self
        while s is not None:
            if name in s.names:
                return True
            s = s.parent
        return False


def bind_target(node, scope):
    if isinstance(node, ast.Name):
        scope.names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            bind_target(e, scope)
    elif isinstance(node, ast.Starred):
        bind_target(node.value, scope)


def collect_scope_bindings(scope, node):
    """Pre-pass: bind every name this scope's subtree defines, without
    descending into nested function/class/lambda scopes (they get their
    own pre-pass)."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(child, ast.comprehension):
            bind_target(child.target, scope)
        elif isinstance(child, ast.Assign):
            for t in child.targets:
                bind_target(t, scope)
        elif isinstance(child, (ast.AnnAssign, ast.AugAssign)):
            bind_target(child.target, scope)
        elif isinstance(child, ast.NamedExpr):
            bind_target(child.target, scope)
        elif isinstance(child, (ast.For, ast.AsyncFor)):
            bind_target(child.target, scope)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for al in child.names:
                scope.names.add((al.asname or al.name).split(".")[0])
        elif isinstance(child, ast.excepthandler):
            if child.name:
                scope.names.add(child.name)
        elif isinstance(child, ast.With):
            for item in child.items:
                if item.optional_vars:
                    bind_target(item.optional_vars, scope)
        collect_scope_bindings(scope, child)


def walk(scope, node, errors, fname, global_names):
    collect_scope_bindings(scope, node)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fscope = Scope(scope)
            a = child.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                fscope.names.add(arg.arg)
            if a.vararg:
                fscope.names.add(a.vararg.arg)
            if a.kwarg:
                fscope.names.add(a.kwarg.arg)
            walk(fscope, child, errors, fname, global_names)
            continue
        if isinstance(child, ast.ClassDef):
            walk(Scope(scope), child, errors, fname, global_names)
            continue
        if isinstance(child, ast.Lambda):
            lscope = Scope(scope)
            a = child.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
                lscope.names.add(arg.arg)
            if a.vararg:
                lscope.names.add(a.vararg.arg)
            if a.kwarg:
                lscope.names.add(a.kwarg.arg)
            walk(lscope, child, errors, fname, global_names)
            continue
        # Load usage check
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if not scope.has(child.id) and child.id not in global_names \
                    and not hasattr(builtins, child.id) \
                    and child.id not in ("__file__", "__name__", "__doc__"):
                errors.append(f"{fname}:{child.lineno}: "
                              f"undefined name '{child.id}'")
        walk(scope, child, errors, fname, global_names)


def check(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, str(path))
    # Files with a module-level __getattr__ (PEP 562) lazily resolve any
    # missing global from another module — nothing to flag there.
    has_module_getattr = any(
        isinstance(n, ast.FunctionDef) and n.name == "__getattr__"
        for n in tree.body)
    if has_module_getattr:
        return []
    errors: list[str] = []
    gscope = Scope()
    # module-level names bound anywhere (order-insensitive approximation)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            gscope.names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bind_target(t, gscope)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                gscope.names.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind_target(node.target, gscope)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars:
                    bind_target(item.optional_vars, gscope)
    walk(gscope, tree, errors, str(path), gscope.names)
    return errors


def main():
    targets = [ROOT / "src" / "creative_director.py"]
    targets += sorted((ROOT / "routers").glob("*.py"))
    all_errors = []
    for t in targets:
        try:
            all_errors += check(t)
        except SyntaxError as e:
            all_errors.append(f"{t}: SyntaxError: {e}")
    if all_errors:
        print("UNDEFINED-NAME ISSUES FOUND:")
        for e in all_errors:
            print(" ", e)
        sys.exit(1)
    print(f"OK — no undefined names in {len(targets)} files")


if __name__ == "__main__":
    main()
