"""
FrameGrade — local desktop launcher.
Starts FastAPI, then opens the UI in a local pywebview window.
Errors are written to crash.log in the project root.
"""
import os, sys, time, threading, socket, traceback, urllib.request, subprocess
from pathlib import Path

def _resolve_root() -> Path:
    """Where the app's files actually are.

    From source that is the repo root, one level above src/. Inside a frozen
    PyInstaller bundle __file__ points into the archive, so the old expression
    resolved somewhere that contains none of the app's data — and the first
    casualty was frontend/dist/index.html. _build_frontend_if_needed() then
    concluded the frontend was missing and tried to run `npm run build` inside
    a windowed executable with no real stdio handles, which dies:

        OSError: [WinError 50] The request is not supported

    So the packaged app failed to start by trying to compile itself.
    sys._MEIPASS is where PyInstaller unpacks datas, which is exactly the
    directory the spec's paths are relative to.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


_ROOT = _resolve_root()
# Must be first — patches subprocess/multiprocessing/asyncio, allocates hidden
# console so any native child that omits CREATE_NO_WINDOW inherits it silently.
sys.path.insert(0, str(_ROOT))
import suppress_console
_LOG    = _ROOT / "crash.log"
_APP_ID = "StreetPhotography.FrameGrade.1"

# Register a unique App User Model ID before any window is created.
# Without this, Windows groups our window under "pythonw.exe" and uses
# Python's icon for the taskbar button and pinned shortcuts.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_ID)
    except Exception:
        pass


# Kill any stale port-8000 process BEFORE opening crash.log — ONLY in the
# detached SERVER process ("--server-only"). The DECOUPLED backend server now
# outlives the window (see main): the window process must NEVER kill a healthy
# running server, or closing the window would abort an in-flight grade. So the
# port-clear runs only when we are (re)starting the server itself, where an
# unresponsive stale holder genuinely needs clearing before we bind.
#
# CRITICAL GUARD — `__name__ == "__main__"`: on Windows, multiprocessing 'spawn'
# RE-IMPORTS this module inside every grade-worker child. Spawned children import
# this file as "__mp_main__", so this runs ONLY in a real launcher/server process.
if __name__ == "__main__" and "--server-only" in sys.argv and sys.platform == "win32":
    try:
        import subprocess as _sp_early
        _early_out = _sp_early.check_output(
            "netstat -ano | findstr :8000",
            shell=True, text=True, creationflags=0x08000000,
        )
        for _ln in _early_out.splitlines():
            _parts = _ln.split()
            if len(_parts) >= 5 and _parts[3] == "LISTENING":
                _sp_early.run(
                    ["taskkill", "/F", "/PID", _parts[4]],
                    creationflags=0x08000000, capture_output=True,
                )
        time.sleep(0.3)  # give OS time to release the file handle
    except Exception:
        pass

try:
    _log_fh = open(_LOG, "a", encoding="utf-8", buffering=1)
except PermissionError:
    # Fallback: write to a temp file so the app can still start and log.
    import tempfile as _tmp
    _log_fh = open(_tmp.mktemp(suffix=".log", prefix="ssc_"), "a", encoding="utf-8", buffering=1)
sys.stdout = _log_fh
sys.stderr = _log_fh


def _log(msg):
    try:
        _log_fh.write(msg + "\n")
        _log_fh.flush()
    except Exception:
        pass

# pythonw.exe has NULL C-level file descriptors (fd 1/2). C extensions that
# write to them directly bypass sys.stdout and crash the process. Redirect them
# to the log file so any low-level writes are safely captured instead.
if sys.platform == "win32":
    try:
        os.dup2(_log_fh.fileno(), 1)
        os.dup2(_log_fh.fileno(), 2)
    except Exception:
        pass

os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
_log(f"--- Local Launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
_log(f"Python: {sys.executable}")
_log(f"Python version: {sys.version}")
_log(f"Working directory: {os.getcwd()}")

# Route pywebview's internal logger (WebView2 init errors etc.) into crash.log
import logging as _logging
_wv_log_handler = _logging.FileHandler(str(_LOG), mode="a", encoding="utf-8")
_wv_log_handler.setFormatter(_logging.Formatter("[pywebview] %(levelname)s %(message)s"))
_logging.getLogger("pywebview").addHandler(_wv_log_handler)
_logging.getLogger("pywebview").setLevel(_logging.DEBUG)


def _kill_port(port):
    """Kill any process listening on the given port so we can bind to it."""
    try:
        import subprocess as _sp
        out = _sp.check_output(
            f"netstat -ano | findstr :{port}",
            shell=True, text=True,
            creationflags=0x08000000,
        )
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING":
                pid = parts[4]
                _sp.run(
                    ["taskkill", "/F", "/PID", pid],
                    creationflags=0x08000000,
                    capture_output=True,
                )
                _log(f"Killed stale process PID {pid} on port {port}")
    except Exception:
        pass


def _find_free_port(preferred=8000):
    """Always try to use the preferred port; kill anything blocking it first."""
    _kill_port(preferred)
    import time as _t
    _t.sleep(0.3)  # give the OS time to release the port
    return preferred


def _patch_webview_gpu():
    """Force WebView2 into SOFTWARE rendering (no GPU) — critical on a 6 GB card.

    Chromium (WebView2) holds a D3D/GPU context on the same GPU the ML models use.
    On a strict 6 GB VRAM budget that shared context intermittently faults the
    grade worker's CUDA operations (0xC0000005 — the "grade worker died" crash at
    Qwen load/inference). pywebview HARDCODES CoreWebView2 AdditionalBrowserArguments
    in its edgechromium platform module and ignores the WEBVIEW2_ADDITIONAL_BROWSER_
    ARGUMENTS env var, so we inject the disable-gpu switches into that source line
    BEFORE importing webview. Idempotent + re-applied every launch, so it self-heals
    after a pywebview upgrade/reinstall. find_spec locates the file WITHOUT importing
    webview, so the edit is picked up on the very next import in this same process."""
    try:
        import importlib.util
        _spec = importlib.util.find_spec("webview")
        if not _spec or not _spec.origin:
            return
        _ec = Path(_spec.origin).parent / "platforms" / "edgechromium.py"
        if not _ec.exists():
            return
        _src = _ec.read_text(encoding="utf-8")
        _marker = "'--disable-features=ElasticOverscroll'"
        if _marker in _src and "--disable-gpu" not in _src:
            _flags = (" --disable-gpu --disable-gpu-compositing "
                      "--disable-gpu-rasterization --disable-accelerated-2d-canvas "
                      "--disable-webgl")
            _src = _src.replace(
                _marker, "'--disable-features=ElasticOverscroll" + _flags + "'", 1)
            _ec.write_text(_src, encoding="utf-8")
            try:
                import py_compile
                py_compile.compile(str(_ec), doraise=False)
            except Exception:
                pass
            _log("WebView2 GPU acceleration DISABLED (software rendering) — 6GB-VRAM safety")
        else:
            _log("WebView2 GPU already software-rendering (patch present)")
    except Exception as exc:
        _log(f"_patch_webview_gpu skipped (non-fatal): {exc}")


def _clear_webview2_cache():
    """Remove stale WebView2 lock files from any pywebview temp dir under %TEMP%."""
    import tempfile, glob as _glob
    tmp = tempfile.gettempdir()
    try:
        for lockname in ("lockfile", "SingletonLock", "SingletonCookie"):
            for lock_path in _glob.glob(os.path.join(tmp, "tmp*", "**", lockname), recursive=True):
                try:
                    os.unlink(lock_path)
                    _log(f"Removed stale lock: {lock_path}")
                except Exception:
                    pass
    except Exception as exc:
        _log(f"_clear_webview2_cache: {exc}")


def _run_server(port):
    try:
        import uvicorn
        from server import app
        _log(f"uvicorn starting on port {port}")
        uvicorn.run(
            app, host="127.0.0.1", port=port,
            log_level="info",
            # Route uvicorn's own log records into crash.log
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {
                    "file": {
                        "class": "logging.FileHandler",
                        "filename": str(_LOG),
                        "mode": "a",
                        "encoding": "utf-8",
                        "formatter": "default",
                    }
                },
                "formatters": {
                    "default": {"format": "[uvicorn] %(levelname)s %(message)s"}
                },
                "loggers": {
                    "uvicorn":        {"handlers": ["file"], "level": "WARNING", "propagate": False},
                    "uvicorn.error":  {"handlers": ["file"], "level": "WARNING", "propagate": False},
                    "uvicorn.access": {"handlers": ["file"], "level": "INFO",    "propagate": False},
                },
            },
        )
    except Exception:
        _log("SERVER ERROR:\n" + traceback.format_exc())


def _build_frontend_if_needed():
    dist_index = _ROOT / "frontend" / "dist" / "index.html"
    if dist_index.exists():
        return

    _log("Building frontend because dist/index.html is missing...")
    import subprocess as sp
    build = sp.run(
        ["npm", "run", "build"],
        cwd=str(_ROOT / "frontend"),
        shell=False,
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        text=True,
    )
    if build.returncode != 0:
        _log("FRONTEND BUILD FAILED")
        _log(build.stdout or "")
        _log(build.stderr or "")
        raise RuntimeError("Frontend build failed; see crash.log for details.")
    _log("Frontend build succeeded")


def _start_frontend_watch():
    """
    Start 'npm run watch' (vite build --watch) as a background process.
    Vite watches frontend/src and rebuilds dist/ on every file save,
    so the backend always serves the latest bundle without a manual build step.

    DEVELOPMENT ONLY. A packaged app ships a built dist/ and carries no
    frontend/src to watch, no package.json and no node_modules — and the user
    may not have Node at all. Attempting it there produced, on every launch:

        Frontend watch failed to start: [WinError 50] The request is not supported

    Caught and logged, so the app still ran, but it is the first thing anyone
    reads in crash.log when something looks wrong and it sends them chasing a
    failure that has no consequences. The condition it needs is simply not
    present in a bundle, so do not ask.
    """
    if getattr(sys, "frozen", False):
        _log("Frontend watch skipped — packaged build serves a prebuilt dist/")
        return

    import subprocess as sp
    try:
        proc = sp.Popen(
            ["npm", "run", "watch"],
            cwd=str(_ROOT / "frontend"),
            stdout=_log_fh,
            stderr=_log_fh,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        _log(f"Frontend watch started (PID {proc.pid})")
    except Exception as exc:
        _log(f"Frontend watch failed to start: {exc}")


def _wait_for_server(url, retries=120, interval=0.25):
    for i in range(retries):
        try:
            urllib.request.urlopen(url, timeout=1)
            _log(f"Server ready after {round(i * interval, 1)}s")
            return True
        except Exception:
            time.sleep(interval)
    _log("ERROR: server startup timed out after {:.1f}s".format(retries * interval))
    return False


if sys.platform == "win32":
    import ctypes as _ct, ctypes.wintypes as _wt

    _u32             = _ct.windll.user32
    _k32             = _ct.windll.kernel32
    _WM_SETICON      = 0x0080
    _IMAGE_ICON      = 1
    _LR_LOADFROMFILE = 0x0010
    _GCL_HICON       = -14
    _GCL_HICONSM     = -34
    _WNDENUMPROC     = _ct.WINFUNCTYPE(_ct.c_bool, _wt.HWND, _wt.LPARAM)
    _SetCLP          = (_u32.SetClassLongPtrW
                        if _ct.sizeof(_ct.c_void_p) == 8
                        else _u32.SetClassLongW)

    def _our_hwnds() -> list:
        """Return every visible top-level HWND owned by this process."""
        our_pid = _k32.GetCurrentProcessId()
        found   = []

        def _cb(hwnd, _):
            pid = _wt.DWORD(0)
            _u32.GetWindowThreadProcessId(hwnd, _ct.byref(pid))
            if pid.value == our_pid and _u32.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True

        _u32.EnumWindows(_WNDENUMPROC(_cb), 0)
        return found

    def _stamp_icon(icon_path: str) -> None:
        """Apply our .ico to every visible window owned by this process."""
        try:
            hBig   = _u32.LoadImageW(None, icon_path, _IMAGE_ICON, 48, 48, _LR_LOADFROMFILE)
            hSmall = _u32.LoadImageW(None, icon_path, _IMAGE_ICON, 16, 16, _LR_LOADFROMFILE)
            for hwnd in _our_hwnds():
                _u32.SendMessageW(hwnd, _WM_SETICON, 1, hBig)    # ICON_BIG  → taskbar
                _u32.SendMessageW(hwnd, _WM_SETICON, 0, hSmall)  # ICON_SMALL → title bar
                if hBig:
                    _SetCLP(hwnd, _GCL_HICON,   hBig)    # class-level (survives redraws)
                if hSmall:
                    _SetCLP(hwnd, _GCL_HICONSM, hSmall)
        except Exception as exc:
            _log(f"_stamp_icon: {exc}")

else:
    def _our_hwnds() -> list: return []     # type: ignore[misc]
    def _stamp_icon(_: str)  -> None: pass  # type: ignore[misc]


def _icon_watcher(icon_path: str) -> None:
    """Daemon thread: wait for our window to appear, then stamp the icon every
    500 ms for 6 s — WebView2 can reset the class icon during its init phase."""
    if sys.platform != "win32":
        return
    # Phase 1: wait until at least one window exists (up to 10 s)
    for _ in range(100):
        if _our_hwnds():
            break
        time.sleep(0.1)
    # Phase 2: apply repeatedly while WebView2 finishes setting up
    for _ in range(12):
        _stamp_icon(icon_path)
        time.sleep(0.5)


def _server_healthy(url: str) -> bool:
    """True if a backend is already serving on `url` (decoupled reuse check)."""
    try:
        urllib.request.urlopen(url + "/api/config", timeout=2)
        return True
    except Exception:
        return False


def _spawn_detached_server(port: int) -> None:
    """Launch the backend (uvicorn + prestarted grade worker) as a DETACHED
    process that OUTLIVES this window process.

    DECOUPLING: previously uvicorn ran in a daemon thread of THIS launcher and the
    grade worker was its multiprocessing child, so when the WebView2 window died
    (webview.start returns → main exits) the server AND the in-flight grade died
    with it. Running the server as its own detached process means a window death —
    for ANY reason, including the OS killing WebView2 under memory pressure — never
    aborts a cull. DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP breaks the child off
    this process so it is not torn down when the window exits."""
    import subprocess as _sp
    _flags = 0
    if sys.platform == "win32":
        _flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    # DEVNULL for all stdio: the detached server redirects its own fd 1/2 to
    # crash.log at module load, so it must not inherit this window process's
    # handles (which vanish when the window exits).
    _sp.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--server-only"],
        cwd=str(_ROOT), creationflags=_flags, close_fds=True,
        stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
    )
    _log("Spawned DETACHED backend server (survives window death)")


def _run_server_only(port: int) -> None:
    """Entry for the detached backend process: build the frontend, then run
    uvicorn in the FOREGROUND of this (server) process forever."""
    _log(f"--- Backend server process (--server-only) on port {port} ---")
    _build_frontend_if_needed()
    _start_frontend_watch()
    _run_server(port)   # blocks in uvicorn.run for the life of the server


def _reexec_in_venv_if_needed() -> None:
    """Re-launch under the venv interpreter when started with the wrong Python.

    The system Python has fastapi and uvicorn but NO torch, so an app started
    with it comes all the way up — window opens, server binds, /api/config
    answers 200 — and then every grade dies on `import torch`. The failure is
    as far as possible from its cause.

    It propagates, too: the backend is spawned with sys.executable, so one
    wrong launch poisons both processes. A desktop shortcut pointing at
    `pythonw src\\local_launcher.py` is enough to cause it, and that is exactly
    how the running instance on this machine was started.

    Frozen builds bundle their own interpreter and are skipped.
    """
    if getattr(sys, "frozen", False):
        return
    try:
        import torch  # noqa: F401
        return                      # whatever we are on can grade; leave it
    except Exception:
        pass

    venv_py = _ROOT / "venv" / "Scripts" / ("pythonw.exe" if sys.platform == "win32" else "python")
    if not venv_py.exists():
        venv_py = _ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_py.exists() or Path(sys.executable).resolve() == venv_py.resolve():
        _log("WARNING: torch is unavailable and no venv interpreter was found. "
             "Grading will fail. Run scripts/setup_wizard.py or launch.bat.")
        return

    _log(f"Wrong interpreter ({sys.executable}) — no torch. Re-executing with {venv_py}")
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


def main():
    _reexec_in_venv_if_needed()
    port = 8000
    url = f"http://127.0.0.1:{port}"

    # DETACHED SERVER MODE — this process IS the backend (no window).
    if "--server-only" in sys.argv:
        try:
            _run_server_only(port)
        except Exception:
            _log("SERVER-ONLY FATAL:\n" + traceback.format_exc())
            raise
        return

    # WINDOW MODE.
    try:
        _patch_webview_gpu()        # force WebView2 software rendering (6GB-VRAM safety)
        _clear_webview2_cache()

        # Reuse a healthy already-running backend (it may still be finishing a cull
        # from a previous window that was closed); otherwise spawn a detached one.
        if _server_healthy(url):
            _log("Reusing running decoupled backend server")
        else:
            _spawn_detached_server(port)
            if not _wait_for_server(url):
                raise RuntimeError("Backend server did not become available in time.")

        import webview
        webview.settings['REMOTE_DEBUGGING_PORT'] = 9222

        class FolderApi:
            def pick_folder(self):
                try:
                    result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                    return result[0] if result else None
                except Exception as e:
                    _log(f"pick_folder error: {e}")
                return None

        _icon = str(_ROOT / "icon.ico")

        # Start the icon watcher before webview.start() so the icon is applied
        # the instant our window exists — and reapplied every 500 ms for 6 s
        # to survive WebView2's internal class-icon resets.
        threading.Thread(target=_icon_watcher, args=(_icon,), daemon=True).start()

        _log(f"Opening local pywebview window: {url}")
        win = webview.create_window(
            title="FrameGrade",
            url=url,
            width=1400,
            height=900,
            min_size=(960, 640),
            resizable=True,
            text_select=False,
            js_api=FolderApi(),
        )

        def _post_start():
            """Called by webview.start() in its own thread after the webview loop starts."""
            _log("[diag] _post_start called")
            loaded = win.events.loaded.wait(5)
            _log(f"[diag] loaded.wait returned {loaded}")
            if not loaded:
                _log("[diag] page never loaded!")
                return
            _stamp_icon(_icon)

            def _diag():
                time.sleep(0.1)
                try:
                    root_kids = win.evaluate_js("document.getElementById('root')?.children?.length ?? -1")
                    body_bg   = win.evaluate_js("getComputedStyle(document.body).backgroundColor")
                    onerrors  = win.evaluate_js("JSON.stringify(window.__errors||[])")
                    cerrors   = win.evaluate_js("JSON.stringify(window.__cerrors||[])")
                    body_text = win.evaluate_js("document.body.innerText?.slice(0,200)")
                    _log(f"[diag] bg={body_bg} root_children={root_kids}")
                    _log(f"[diag] onerrors={onerrors}")
                    _log(f"[diag] cerrors={cerrors}")
                    _log(f"[diag] body_text={body_text}")
                except Exception as exc:
                    _log(f"[diag] error: {exc}")
            threading.Thread(target=_diag, daemon=True).start()

        webview.start(icon=_icon, func=_post_start)

        # DECOUPLED: the backend is its own detached process, so the window can
        # exit immediately — uvicorn and any in-flight grade keep running in the
        # background. The user relaunches to reattach and hits "Resume" (or the
        # gallery is already populated if the cull finished). No keep-alive wait
        # and no server teardown here — that is the whole point of the decoupling.
        _log("Window closed — decoupled backend left running; relaunch to reattach")

    except Exception:
        _log("FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
