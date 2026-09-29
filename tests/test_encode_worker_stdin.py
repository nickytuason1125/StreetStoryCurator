"""encode_worker.serve() must not deadlock its own imports.

serve() waits for jobs on a background thread. Reading the job pipe through
the process's standard input left a synchronous read pending on
STD_INPUT_HANDLE; scipy's OpenBLAS DLL start-up inspects that handle and
queued behind the read for ever — culls froze at 48% ("Preparing the style
reference…"). _stdin_lines() reads a private duplicate and points stdin at NUL.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"


def _child(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "child.py"
    p.write_text(textwrap.dedent(f"""
        import sys, threading, time, json
        sys.path.insert(0, {str(_SRC)!r})
        from encode_worker import _stdin_lines
    """) + textwrap.dedent(body))
    return p


@pytest.mark.skipif(sys.platform != "win32", reason="Windows synchronous-handle semantics")
def test_import_with_reader_waiting_does_not_deadlock(tmp_path):
    child = _child(tmp_path, """
        threading.Thread(target=lambda: [None for _ in _stdin_lines()], daemon=True).start()
        time.sleep(0.5)                 # the reader is now waiting on the pipe
        import sklearn.utils.validation # pulls scipy's OpenBLAS DLL
        print("imported", flush=True)
        import os; os._exit(0)
    """)
    p = subprocess.Popen([sys.executable, str(child)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(timeout=90)   # parent never writes: the real case
    except subprocess.TimeoutExpired:
        p.kill()
        pytest.fail("import deadlocked behind the stdin reader")
    assert b"imported" in out, err.decode(errors="replace")[-600:]


def test_lines_are_split_and_eof_ends_the_stream(tmp_path):
    out_file = tmp_path / "lines.json"
    child = _child(tmp_path, f"""
        got = list(_stdin_lines())
        json.dump(got, open({str(out_file)!r}, "w"))
    """)
    p = subprocess.Popen([sys.executable, str(child)], stdin=subprocess.PIPE)
    p.stdin.write(b'{"job": 1}\n{"jo')      # a line split across writes
    p.stdin.flush()
    p.stdin.write(b'b": 2}\n{"job": 3}')     # last line without a newline
    p.stdin.close()                          # EOF, as when the parent exits
    assert p.wait(60) == 0
    assert json.loads(out_file.read_text()) == ['{"job": 1}\n', '{"job": 2}\n', '{"job": 3}']
