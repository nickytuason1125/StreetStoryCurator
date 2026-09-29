"""The detached backend must be able to spawn children.

local_launcher starts the server DETACHED with stdin=NUL; suppress_console
then allocates a hidden console, which replaces STD_INPUT_HANDLE with a
handle Windows refuses to duplicate. Every child spawned without an explicit
stdin (the jury critique's asyncio subprocess, subprocess.run(capture_output))
failed with WinError 50 / 6 until the check tested the real Win32 handle
(2026-09-28). This test reproduces that launch exactly.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows handle semantics")


def test_detached_server_can_spawn_children(tmp_path):
    out = tmp_path / "result.json"
    child = tmp_path / "child.py"
    child.write_text(textwrap.dedent(f"""
        import asyncio, json, subprocess, sys
        sys.path.insert(0, {str(_ROOT)!r})
        import suppress_console  # noqa: F401
        res = {{}}
        async def aio():
            p = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "print('ok')",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            return (await p.communicate())[0].decode().strip()
        for name, fn in (
            ("asyncio", lambda: asyncio.run(aio())),
            ("popen", lambda: subprocess.Popen([sys.executable, "-c", "print('ok')"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode().strip()),
            ("run", lambda: subprocess.run([sys.executable, "-c", "print('ok')"],
                capture_output=True, text=True).stdout.strip()),
        ):
            try:
                res[name] = fn()
            except Exception as e:
                res[name] = f"{{type(e).__name__}}: {{e}}"
        json.dump(res, open({str(out)!r}, "w"))
    """))
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = str(pyw if pyw.exists() else sys.executable)
    # Exactly how local_launcher._spawn_detached_server starts the backend.
    p = subprocess.Popen([exe, str(child)],
                         creationflags=0x00000008 | 0x00000200, close_fds=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    p.wait(90)
    assert json.loads(out.read_text()) == {"asyncio": "ok", "popen": "ok", "run": "ok"}
