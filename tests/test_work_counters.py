import json, os, subprocess, sys
from pathlib import Path
_SRC = Path(__file__).resolve().parent.parent / "src"


def test_counters_survive_os_exit(tmp_path):
    """Workers leave via os._exit (skips atexit) — counts must still land."""
    code = ("import sys; sys.path.insert(0, r'%s'); import os, work_counters as w;"
            "w.install_exit_flush('t'); w.bump('decode.x', 3); w.bump('decode.x'); os._exit(0)") % _SRC
    env = dict(os.environ, FIRSTCUT_WORK_COUNTERS=str(tmp_path))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    sys.path.insert(0, str(_SRC))
    import work_counters
    assert work_counters.collect(str(tmp_path)) == {"decode.x": 4}


def test_off_by_default(monkeypatch):
    sys.path.insert(0, str(_SRC))
    import importlib, work_counters
    monkeypatch.delenv("FIRSTCUT_WORK_COUNTERS", raising=False)
    importlib.reload(work_counters)
    work_counters.bump("x")
    assert not work_counters.enabled()
