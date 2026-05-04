"""
Street Story Curator — native desktop launcher.
Starts FastAPI, then opens Microsoft Edge in --app mode (no address bar, no tabs).
Errors are written to crash.log in the project root.
"""
import os, sys, time, threading, socket, traceback, urllib.request, subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_LOG  = _ROOT / "crash.log"

def _log(msg):
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

_log_fh = open(_LOG, "a", encoding="utf-8", buffering=1)
sys.stdout = _log_fh
sys.stderr = _log_fh

os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))
_log(f"--- Launch {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
_log(f"Python: {sys.executable}")


def _find_free_port(preferred=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(port):
    try:
        import uvicorn
        from server import app
        _log(f"uvicorn starting on port {port}")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
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


def _find_browser():
    # Try Edge first, then Chrome, then Firefox
    candidates = [
        (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "edge"),
        (r"C:\Program Files\Microsoft\Edge\Application\msedge.exe", "edge"),
        (r"C:\Program Files\Google\Chrome\Application\chrome.exe", "chrome"),
        (r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", "chrome"),
        (r"C:\Program Files\Mozilla Firefox\firefox.exe", "firefox"),
    ]
    for p, browser_type in candidates:
        if Path(p).exists():
            return p, browser_type
    return None, None


def main():
    url = ""
    try:
        # Build frontend if missing
        _build_frontend_if_needed()

        port = _find_free_port()
        url  = f"http://127.0.0.1:{port}"
        _log(f"Port: {port}")

        # Start server
        threading.Thread(target=_run_server, args=(port,), daemon=False).start()
        if not _wait_for_server(url):
            raise RuntimeError("Server did not become available in time.")

        # Try pywebview first (no external browser dependency)
        try:
            _log("Trying pywebview...")
            import webview

            class FolderApi:
                def pick_folder(self):
                    try:
                        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                        return result[0] if result else None
                    except Exception as e:
                        _log(f"pick_folder error: {e}")
                    return None

            kwargs = dict(
                title="Street Story Curator",
                url=url,
                width=1400, height=900,
                min_size=(960, 640),
                resizable=True, text_select=False,
                js_api=FolderApi(),
            )
            icon_path = _ROOT / "icon.ico"
            if icon_path.exists():
                try:
                    webview.create_window(**kwargs, icon=str(icon_path))
                except TypeError:
                    webview.create_window(**kwargs)
            else:
                webview.create_window(**kwargs)
            webview.start()
            _log("pywebview window closed — exiting")
            return  # Success, exit

        except ImportError:
            _log("pywebview not available — falling back to browser")
        except Exception as e:
            _log(f"pywebview failed: {e} — falling back to browser")

        # Fallback to browser
        browser_path, browser_type = _find_browser()
        if browser_path:
            if browser_type == "edge":
                # Isolated Edge profile so the app window is always separate
                profile_dir = Path(os.environ.get("LOCALAPPDATA", _ROOT)) / "StreetStoryCurator" / "EdgeProfile"
                profile_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    browser_path,
                    f"--app={url}",
                    f"--user-data-dir={profile_dir}",
                    "--window-size=1400,900",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-extensions",
                ]
                _log(f"Opening Edge app mode: {url}")
            elif browser_type == "chrome":
                # Chrome app mode
                profile_dir = Path(os.environ.get("LOCALAPPDATA", _ROOT)) / "StreetStoryCurator" / "ChromeProfile"
                profile_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    browser_path,
                    f"--app={url}",
                    f"--user-data-dir={profile_dir}",
                    "--window-size=1400,900",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-extensions",
                ]
                _log(f"Opening Chrome app mode: {url}")
            elif browser_type == "firefox":
                # Firefox app mode
                profile_dir = Path(os.environ.get("LOCALAPPDATA", _ROOT)) / "StreetStoryCurator" / "FirefoxProfile"
                profile_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    browser_path,
                    f"--app={url}",
                    f"--profile={profile_dir}",
                    "--width=1400",
                    "--height=900",
                ]
                _log(f"Opening Firefox app mode: {url}")
            # Use DETACHED_PROCESS to ensure no console window appears
            DETACHED_PROCESS = 0x00000008 if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd,
                creationflags=DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
                close_fds=True
            )
            _log(f"{browser_type.capitalize()} window launched (PID: {proc.pid})")
            # Don't wait for browser - just log and exit
            _log("Browser launched — exiting launcher")
        else:
            # Fallback: pywebview
            _log("No browser found — falling back to pywebview")
            import webview

            class FolderApi:
                def pick_folder(self):
                    try:
                        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                        return result[0] if result else None
                    except Exception as e:
                        _log(f"pick_folder error: {e}")
                    return None

            webview.create_window(
                title="Street Story Curator",
                url=url,
                width=1400, height=900,
                min_size=(960, 640),
                resizable=True, text_select=False,
                js_api=FolderApi(),
            )
            webview.start()
            _log("pywebview window closed — exiting")

    except Exception:
        _log("FATAL:\n" + traceback.format_exc())
        try:
            import webbrowser
            if url:
                webbrowser.open(url)
            else:
                _log("No URL available; skipping webbrowser.open to avoid opening System32")
            while True:
                time.sleep(1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
