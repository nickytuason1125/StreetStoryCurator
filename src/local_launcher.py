"""
Street Story Curator — local desktop launcher.
Starts FastAPI, then opens the UI in a local pywebview window.
Errors are written to crash.log in the project root.
"""
import os, sys, time, threading, socket, traceback, urllib.request, subprocess
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent
_LOG    = _ROOT / "crash.log"
_APP_ID = "StreetPhotography.StreetStoryCurator.1"

# Register a unique App User Model ID before any window is created.
# Without this, Windows groups our window under "pythonw.exe" and uses
# Python's icon for the taskbar button and pinned shortcuts.
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_ID)
    except Exception:
        pass


def _log(msg):
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


_log_fh = open(_LOG, "a", encoding="utf-8", buffering=1)
sys.stdout = _log_fh
sys.stderr = _log_fh

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


def main():
    url = ""
    try:
        _clear_webview2_cache()
        _build_frontend_if_needed()

        port = _find_free_port()
        url = f"http://127.0.0.1:{port}"
        _log(f"Port: {port}")

        threading.Thread(target=_run_server, args=(port,), daemon=True).start()
        if not _wait_for_server(url):
            raise RuntimeError("Server did not become available in time.")

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
            title="Street Story Curator",
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
        _log("pywebview window closed — exiting")

    except Exception:
        _log("FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
