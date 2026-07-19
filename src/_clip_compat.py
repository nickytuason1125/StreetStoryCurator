"""
_clip_compat.py -- compatibility shim for the `openai-clip` package on modern setuptools.

The problem
-----------
openai-clip 1.0.1 begins clip/clip.py with:

    from pkg_resources import packaging

setuptools >= 66 removed that re-export, and Python 3.12 requires a setuptools at least
that new -- so there is NO setuptools version that both supports 3.12 AND still exposes
`pkg_resources.packaging`. Downgrading is not an option; the package cannot be "fixed"
by pinning versions.

The fix
-------
Re-attach the standalone `packaging` distribution (always installed -- it is a transitive
dependency of pip, torch, transformers, ...) onto `pkg_resources` under the exact name
clip expects, BEFORE `clip` is imported. clip only uses `packaging.version.parse`, so we
make sure the `.version` submodule is bound.

This lives in version control (not site-packages), so it survives `pip install
--force-reinstall clip` and full venv rebuilds. It is a no-op when pkg_resources already
provides `packaging` (older setuptools) or when clip is absent.

Usage
-----
Import this module immediately BEFORE `import clip`:

    import _clip_compat   # noqa: F401  (restores pkg_resources.packaging)
    import clip
"""
from __future__ import annotations


def _patch() -> bool:
    """Attach `packaging` onto pkg_resources if missing. Returns True if a patch was applied."""
    try:
        import pkg_resources
    except Exception:
        return False
    if getattr(pkg_resources, "packaging", None) is not None:
        return False  # older setuptools already provides it -- nothing to do
    try:
        import packaging
        import packaging.version  # bind the submodule clip dereferences (packaging.version.parse)
    except Exception:
        return False
    pkg_resources.packaging = packaging  # type: ignore[attr-defined]
    return True


PATCHED = _patch()
