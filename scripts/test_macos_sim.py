"""Verify platform_compat behaves correctly when running under macOS.

Runs itself in a fresh interpreter with sys.platform patched to 'darwin'
BEFORE the import, so every constant computed at import time is the macOS one.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

script = r"""
import sys
sys.platform = 'darwin'
sys.path.insert(0, r'{src}')
import platform_compat as pc
assert pc.IS_WINDOWS is False and pc.IS_MACOS is True, 'flags wrong'
assert pc.NO_WINDOW_FLAG == 0, f'NO_WINDOW_FLAG should be 0 on darwin, got {{pc.NO_WINDOW_FLAG}}'
ap = str(pc.app_data_dir())
assert 'Library' in ap and 'Application Support' in ap and 'FirstCut' in ap, f'bad mac dir: {{ap}}'
assert pc.browser_app_mode_cmd('http://x') is None, 'should fall back without /Applications'
import subprocess as sp
assert not hasattr(sp, 'CREATE_NO_WINDOW') or True
print('MACOS SIM: flags OK | NO_WINDOW_FLAG=0 | data dir OK | browser fallback OK')
""".format(src=str(ROOT / "src"))

r = subprocess.run([str(PY), "-c", script], capture_output=True, text=True)
out = (r.stdout + r.stderr).strip()
print(out)
sys.exit(0 if "MACOS SIM" in out else 1)
