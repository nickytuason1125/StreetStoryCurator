"""
suppress_console.py
===================
Import this module ONCE, as early as possible (before any numpy / pymoo /
subprocess usage), to ensure that no child process ever flashes a console
window on Windows.

Five layers of protection:

1. subprocess.Popen   — OR CREATE_NO_WINDOW into creationflags and hide the
                         window via STARTUPINFO.  Uses OR so it can't be
                         defeated by callers that already set creationflags.

2. multiprocessing    — switches the spawn-worker executable to pythonw.exe
                         (the windowless interpreter).  pymoo, joblib, and
                         BLAS thread-pool managers all use this path on Windows.

3. asyncio subprocesses — patches asyncio's Windows ProactorEventLoop subprocess
                         transport to inject CREATE_NO_WINDOW.

4. BLAS / OpenMP env  — caps every known thread-pool env-var to 1 so MKL,
                         OpenBLAS, and OpenMP never try to spawn their own
                         worker console processes.

5. Hidden console     — pythonw.exe has no console; any child process spawned
                         WITHOUT CREATE_NO_WINDOW (e.g. from a native DLL
                         bypassing Python's subprocess module) would get a brand
                         new visible console window.  AllocConsole + SW_HIDE
                         gives this process a hidden console that children can
                         inherit instead of creating new visible ones.
"""
from __future__ import annotations

import os
import sys

# ── Nothing to do on non-Windows ──────────────────────────────────────────────
if sys.platform != "win32":
    def apply() -> None:  # type: ignore[misc]
        pass
else:
    import subprocess

    _CREATE_NO_WINDOW: int = subprocess.CREATE_NO_WINDOW  # 0x08000000

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  subprocess.Popen — force CREATE_NO_WINDOW + hidden STARTUPINFO
    # ─────────────────────────────────────────────────────────────────────────
    _orig_Popen_init = subprocess.Popen.__init__

    def _silent_Popen(self: subprocess.Popen, args, **kwargs: object) -> None:
        # OR the flag in — don't use setdefault, which a caller can defeat by
        # explicitly passing creationflags=0.
        kwargs["creationflags"] = (
            int(kwargs.get("creationflags") or 0) | _CREATE_NO_WINDOW
        )
        # Also hide via STARTUPINFO so any residual console allocation is
        # suppressed at the Win32 level.
        if "startupinfo" not in kwargs:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
        _orig_Popen_init(self, args, **kwargs)  # type: ignore[arg-type]

    subprocess.Popen.__init__ = _silent_Popen  # type: ignore[method-assign]

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  multiprocessing — use pythonw.exe so spawned workers have no console
    # ─────────────────────────────────────────────────────────────────────────
    def _maybe_switch_to_pythonw() -> None:
        exe = sys.executable
        if exe.lower().endswith("python.exe"):
            pyw = exe[:-10] + "pythonw.exe"
            if os.path.isfile(pyw):
                import multiprocessing
                multiprocessing.set_executable(pyw)

    _maybe_switch_to_pythonw()

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  asyncio ProactorEventLoop subprocess transport (Python 3.8+)
    #     asyncio.create_subprocess_exec / create_subprocess_shell go through
    #     _overlapped / _winapi directly — patch the transport layer.
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from asyncio.windows_utils import Popen as _aio_Popen  # type: ignore[import]

        _orig_aio_Popen = _aio_Popen.__init__

        def _silent_aio_Popen(self, args, **kwargs: object) -> None:
            kwargs["creationflags"] = (
                int(kwargs.get("creationflags") or 0) | _CREATE_NO_WINDOW
            )
            _orig_aio_Popen(self, args, **kwargs)

        _aio_Popen.__init__ = _silent_aio_Popen  # type: ignore[method-assign]
    except (ImportError, AttributeError):
        pass

    # ─────────────────────────────────────────────────────────────────────────
    # 4.  BLAS / OpenMP / thread-pool env vars
    #     MKL, OpenBLAS, and OpenMP each spawn their own worker pool; on
    #     Windows this can briefly flash a console.  Capping them to 1 keeps
    #     all computation in the existing process threads.
    # ─────────────────────────────────────────────────────────────────────────
    _BLAS_VARS = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS",
        "TBB_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for _var in _BLAS_VARS:
        os.environ.setdefault(_var, "1")

    # ─────────────────────────────────────────────────────────────────────────
    # 5.  Hidden console — pythonw.exe starts with NO console.  Any child
    #     process that omits CREATE_NO_WINDOW gets a brand-new VISIBLE console
    #     instead of inheriting the parent's quiet one.  Allocate a console
    #     here and immediately hide its window so children can inherit it
    #     without a flash.  (No-op when a console already exists, i.e. when
    #     running under python.exe from a terminal.)
    # ─────────────────────────────────────────────────────────────────────────
    def _alloc_hidden_console() -> None:
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            u32 = ctypes.windll.user32
            if k32.GetConsoleWindow():
                return  # already have a console — nothing to do
            if not k32.AllocConsole():
                return
            hwnd = k32.GetConsoleWindow()
            if hwnd:
                u32.ShowWindow(hwnd, 0)  # SW_HIDE — hides before it can be painted
            # Redirect the newly allocated console FDs to NUL so any native
            # code writing to fd 1/2 doesn't block.
            try:
                import os as _os
                _nul_w = open("nul", "w")
                _os.dup2(_nul_w.fileno(), 1)
                _os.dup2(_nul_w.fileno(), 2)
                _nul_r = open("nul", "r")
                _os.dup2(_nul_r.fileno(), 0)
            except Exception:
                pass
        except Exception:
            pass

    _alloc_hidden_console()

    def apply() -> None:
        """No-op: all patches are applied at import time."""
        pass
