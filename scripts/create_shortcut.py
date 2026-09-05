"""Create the FirstCut desktop shortcut (Windows)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pyw = ROOT / "venv" / "Scripts" / "pythonw.exe"
script = ROOT / "src" / "local_launcher.py"
icon = ROOT / "icon.ico"
desktop = Path.home() / "Desktop"
lnk_path = desktop / "FirstCut.lnk"

if not pyw.exists():
    print(f"venv pythonw not found at {pyw} — run the venv setup first")
    sys.exit(1)

shell = sys.modules.get("win32com")  # noqa: F841
try:
    import win32com.client  # type: ignore
except ImportError:
    # pywin32 is in requirements; fall back to PowerShell COM if missing
    import subprocess
    ps = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path}')
$s.TargetPath = '{pyw}'
$s.Arguments = '\"{script}\"'
$s.WorkingDirectory = '{ROOT}'
$s.IconLocation = '{icon if icon.exists() else pyw}'
$s.Description = 'FirstCut — street photography culler'
$s.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print(f"Shortcut created: {lnk_path} (via PowerShell)")
    sys.exit(0)

ws = win32com.client.Dispatch("WScript.Shell")
lnk = ws.CreateShortcut(str(lnk_path))
lnk.TargetPath = str(pyw)
lnk.Arguments = f'"{script}"'
lnk.WorkingDirectory = str(ROOT)
lnk.IconLocation = str(icon if icon.exists() else pyw)
lnk.Description = "FirstCut — street photography culler"
lnk.Save()
print(f"Shortcut created: {lnk_path}")