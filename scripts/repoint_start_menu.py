"""Repoint the Start Menu FirstCut shortcut at the repo launcher (one-time)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lnk = Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\FirstCut.lnk"))
pyw = ROOT / "venv" / "Scripts" / "pythonw.exe"
script = ROOT / "src" / "local_launcher.py"
icon = ROOT / "icon.ico"

import win32com.client  # noqa: E402
ws = win32com.client.Dispatch("WScript.Shell")
l = ws.CreateShortcut(str(lnk))
l.TargetPath = str(pyw)
l.Arguments = f'"{script}"'
l.WorkingDirectory = str(ROOT)
l.IconLocation = str(icon if icon.exists() else pyw)
l.Description = "FirstCut — street photography culler (repo launcher)"
l.Save()
print(f"Start Menu shortcut repointed: {lnk} -> {pyw} {script}")
