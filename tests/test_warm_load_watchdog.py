"""A warm encoder that never finishes loading must not stall a cull.

2026-09-28: a warm worker deadlocked in its imports and the cull sat at 48%
("Preparing the style reference…"). The old guard guessed "still importing"
from RSS < 1 GB, but that worker still held a 1.2 GB ONNX session, so the
guard never fired and only the 30-minute no-resp budget remained. The worker
now touches <resp>.loaded once its model is ready; _warm_run gives up on any
attempt whose marker has not appeared within the load window.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import siglip2_encoder as se  # noqa: E402


def _fake_worker(tmp_path: Path, behaviour: str) -> Path:
    """A stand-in `encode_worker serve`: reads one job line, then behaves."""
    p = tmp_path / f"fake_{behaviour}.py"
    p.write_text(textwrap.dedent(f"""
        import json, sys, time
        import numpy as np
        job = json.loads(sys.stdin.readline())
        if {behaviour!r} == "wedge":
            time.sleep(3600)                      # never loads, never answers
        if {behaviour!r} == "slow":
            open(job["resp"] + ".loaded", "w").close()
            time.sleep(4)                         # loaded; encoding outlasts the window
        if {behaviour!r} == "ok":
            open(job["resp"] + ".loaded", "w").close()
        np.save(job["out"], np.zeros((1, 4), dtype=np.float32))
        json.dump({{"ok": True, "error": ""}}, open(job["resp"], "w"))
        time.sleep(3600)
    """))
    return p


@pytest.fixture()
def fake_warm(tmp_path, monkeypatch):
    procs = []

    def install(behaviour):
        script = _fake_worker(tmp_path, behaviour)

        def ensure(_worker_path):
            if se._WARM is None or se._WARM["proc"].poll() is not None:
                proc = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE)
                procs.append(proc)
                se._WARM = {"proc": proc, "stdin": proc.stdin, "spawn_ts": time.time()}
            return True

        def shutdown():
            if se._WARM is not None:
                se._WARM["proc"].kill()
                se._WARM["proc"].wait(10)
            se._WARM = None

        monkeypatch.setattr(se, "_warm_ensure", ensure)
        monkeypatch.setattr(se, "_warm_shutdown", shutdown)
        monkeypatch.setattr(se, "_WARM_LOAD_WINDOW_S", 2.0)
        monkeypatch.setattr(se, "_WARM", None)
        return procs

    yield install
    for p in procs:
        if p.poll() is None:
            p.kill()


def _job(tmp_path):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps(["x"]))
    return str(inp), str(tmp_path / "out.npy")


class _Enc:
    _WORKER = "unused"


def test_worker_that_never_loads_is_abandoned_fast(tmp_path, fake_warm):
    procs = fake_warm("wedge")
    inp, out = _job(tmp_path)
    t = time.time()
    err = se._warm_run(_Enc(), "text", inp, out, n_items=656)
    elapsed = time.time() - t
    assert isinstance(err, str) and err            # caller falls back to one-shot
    assert elapsed < 15, f"stalled {elapsed:.0f}s (old code: up to 30 min)"
    assert len(procs) == 2                          # BOTH attempts were guarded
    assert all(p.poll() is not None for p in procs)


def test_loaded_worker_is_never_killed_for_slow_encoding(tmp_path, fake_warm):
    fake_warm("slow")
    inp, out = _job(tmp_path)
    assert se._warm_run(_Enc(), "text", inp, out, n_items=10) is None


def test_healthy_worker_returns_normally(tmp_path, fake_warm):
    fake_warm("ok")
    inp, out = _job(tmp_path)
    assert se._warm_run(_Enc(), "text", inp, out, n_items=10) is None
