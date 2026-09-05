"""
platform_compat.py — the single place that knows which OS we're on.

The Windows port audit (2026-09) found every OS-specific call site. The
policy going forward: no module reaches for ctypes.windll, LOCALAPPDATA,
or CREATE_NO_WINDOW directly — they ask this module, which answers with
the native equivalent (or a documented fallback) for the running OS.

Supported: Windows 10+ (primary, fully verified), macOS 13+ Intel/ARM
(ported, CPU-tier verified — see PORTING.md for the real-hardware
verification checklist). Linux is best-effort via the POSIX paths.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_POSIX = os.name == "posix"

# subprocess flag that keeps a child from flashing a console window.
# Windows-only — passing it on POSIX raises ValueError, so callers must
# use this value, never the constant directly.
NO_WINDOW_FLAG: int = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0

_APP_DIR_NAME = "FirstCut"


def app_data_dir() -> Path:
    """Per-user writable dir for app state (browser profiles, caches).

    Windows: %LOCALAPPDATA%\\FirstCut   (matches all existing installs)
    macOS:   ~/Library/Application Support/FirstCut
    Linux:   ${XDG_DATA_HOME:-~/.local/share}/FirstCut
    """
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _APP_DIR_NAME
    if IS_MACOS:
        return Path.home() / "Library" / "Application Support" / _APP_DIR_NAME
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / _APP_DIR_NAME


def browser_app_mode_cmd(url: str) -> list[str] | None:
    """Command to open `url` as a chrome-style app window, per-OS.

    Returns the argv for subprocess.Popen, or None to let the caller fall
    back to whatever the OS default browser does with a plain URL.
    """
    if IS_MACOS:
        # /Applications paths; `open -na` also finds user-installed copies.
        for app in ("Google Chrome", "Microsoft Edge", "Chromium"):
            exe = Path("/Applications") / f"{app}.app" / "Contents" / "MacOS" / app
            if exe.exists():
                profile = app_data_dir() / f"{app.split()[0]}Profile"
                profile.mkdir(parents=True, exist_ok=True)
                return [
                    str(exe),
                    f"--app={url}",
                    f"--user-data-dir={profile}",
                    "--window-size=1400,900",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-extensions",
                ]
        return None   # Safari has no app mode; caller uses `open <url>`
    if IS_WINDOWS:
        # Historic behaviour lives in native_launcher._find_browser — keep it
        # there so the existing isolated-profile logic stays byte-identical.
        return None
    return None


def reveal_path(path: str | Path) -> None:
    """Best-effort: show `path` in the OS file manager. Never raises."""
    try:
        p = Path(path)
        if not p.exists():
            return
        if IS_WINDOWS:
            os.startfile(str(p if p.is_dir() else p.parent))  # noqa: S606
        elif IS_MACOS:
            subprocess.Popen(["open", str(p if p.is_dir() else p.parent)])
        else:
            subprocess.Popen(["xdg-open", str(p if p.is_dir() else p.parent)])
    except Exception:
        pass