"""Deterministic tests for the warm-encoder watchdog (2026-09-16).

The 'respawn race' bug class is untestable with live model loads — it only
reproduces under memory pressure with real weight files. These tests fake the
worker (a never-ready proc, a pre-written resp, a wedged stdin) so every
watchdog behaviour is exercised in seconds, deterministically, on any machine.
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import siglip2_encoder as se


class _FakeProc:
    """Minimal subprocess stand-in: never exits unless told to."""

    def __init__(self, pid=999999, exit_after=None):
        self.pid = pid
        self._exit_after = exit_after
        self._t0 = time.time()
        self.killed = False

    def poll(self):
        if self._exit_after is not None and time.time() - self._t0 > self._exit_after:
            return 1
        return None

    def terminate(self):
        self.killed = True

    def kill(self):
        self.killed = True


class _FakeStdin:
    def __init__(self):
        self.written = []

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass


class _FakeEnc:
    """Just enough of SigLIP2Encoder for _warm_run/_warm_ensure."""

    def __init__(self):
        self._WORKER = "encode_worker.py"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the ready-marker at a throwaway dir."""
    monkeypatch.setenv("FIRSTCUT_DATA_DIR", str(tmp_path))
    return tmp_path


def _write_marker(data_dir, pid, ts=None):
    import json
    (data_dir / "encode_worker.ready.json").write_text(
        json.dumps({"pid": pid, "ts": ts if ts is not None else time.time()}),
        encoding="utf-8",
    )


def test_ready_wait_accepts_fresh_marker(data_dir):
    """A fresh marker carrying a live python pid (this test process) is
    accepted — the wrapper-pid mismatch can no longer reject a healthy
    worker."""
    _write_marker(data_dir, os.getpid())
    proc = _FakeProc(pid=123456)  # wrapper pid != marker pid, never exits
    t0 = time.time()
    assert se._warm_ready_wait(proc, timeout_s=5.0, spawn_ts=time.time() - 5) is True
    assert time.time() - t0 < 5.0


def test_ready_wait_rejects_stale_marker_within_bound(data_dir):
    """A stale marker is never accepted and the wait returns False within
    the hard timeout even though the worker never signals ready."""
    _write_marker(data_dir, os.getpid(), ts=0.0)  # ancient ts → stale
    proc = _FakeProc(pid=123456)
    t0 = time.time()
    assert se._warm_ready_wait(proc, timeout_s=2.0, spawn_ts=time.time()) is False
    assert time.time() - t0 < 2.0 + 3.0


def test_ready_wait_bounded_returns_false_when_no_marker(data_dir):
    """No marker at all + worker alive → bounded False (the daemon-thread
    wrapper guarantees the parent cannot hang past the timeout)."""
    proc = _FakeProc(pid=123456)
    t0 = time.time()
    assert se._warm_ready_wait_bounded(proc, spawn_ts=time.time(), timeout_s=2.0) is False
    assert time.time() - t0 < 2.0 + 3.0


def test_warm_run_kills_wedged_worker_on_no_resp(data_dir, monkeypatch):
    """A worker that takes the job but never resp's is killed by the
    wall-clock budget on each try, and _warm_run reports loss after two."""
    monkeypatch.setattr(se, "_NO_RESP_MIN_S", 1.0)
    monkeypatch.setattr(se, "_NO_RESP_SCALE", 1.0)
    monkeypatch.setattr(se, "_warm_no_cpu_s", lambda n: 1.0)

    kills = []

    def _fake_shutdown():
        kills.append(time.time())
        se._WARM = None

    monkeypatch.setattr(se, "_warm_shutdown", _fake_shutdown)
    monkeypatch.setattr(se, "_warm_ensure", lambda worker_path: True)

    se._WARM = {"proc": _FakeProc(pid=987654), "stdin": _FakeStdin(),
                "spawn_ts": time.time(), "real_pid": None}

    enc = _FakeEnc()
    t0 = time.time()
    err = se._warm_run(enc, "text", str(data_dir / "in.json"),
                       str(data_dir / "out.npy"), n_items=1)
    elapsed = time.time() - t0

    assert err is not None and "lost the job twice" in err
    # Two tries × ~2 s budget each — deterministically fast.
    assert elapsed < 15.0
    assert len(kills) == 2


def test_warm_run_succeeds_on_ok_resp(data_dir, monkeypatch):
    """A worker that resp's ok + writes the out file → immediate success.
    The worker is simulated by a thread that writes the resp after _warm_run
    has unlinked any stale one (mirroring the real worker's lifecycle)."""
    import json
    import threading as _threading
    import numpy as np
    monkeypatch.setattr(se, "_warm_ensure", lambda worker_path: True)
    monkeypatch.setattr(se, "_NO_RESP_MIN_S", 2.0)   # fail fast on regression
    monkeypatch.setattr(se, "_NO_RESP_SCALE", 1.0)
    monkeypatch.setattr(se, "_warm_no_cpu_s", lambda n: 1.0)
    se._WARM = {"proc": _FakeProc(pid=987654), "stdin": _FakeStdin(),
                "spawn_ts": time.time(), "real_pid": None}

    in_path = str(data_dir / "in.json")
    out_path = in_path + ".npy"
    np.save(out_path, np.zeros((1, 1536), dtype=np.float32))

    from pathlib import Path as _Path

    def _fake_worker():
        time.sleep(1.0)   # let _warm_run unlink any stale resp first
        (_Path(in_path + ".resp.json")).write_text(json.dumps({"ok": True}), encoding="utf-8")

    _threading.Thread(target=_fake_worker, daemon=True).start()

    enc = _FakeEnc()
    t0 = time.time()
    assert se._warm_run(enc, "text", in_path, out_path, n_items=1) is None
    assert time.time() - t0 < 20.0


def test_warm_run_reports_died_mid_job(data_dir, monkeypatch):
    """A worker that exits mid-job is detected by poll() and retried, then
    reported lost — no hang, no silent success."""
    monkeypatch.setattr(se, "_NO_RESP_MIN_S", 1.0)
    monkeypatch.setattr(se, "_NO_RESP_SCALE", 1.0)
    monkeypatch.setattr(se, "_warm_no_cpu_s", lambda n: 1.0)

    kills = []

    def _fake_shutdown():
        kills.append(time.time())
        se._WARM = None

    monkeypatch.setattr(se, "_warm_shutdown", _fake_shutdown)
    monkeypatch.setattr(se, "_warm_ensure", lambda worker_path: True)

    se._WARM = {"proc": _FakeProc(pid=987654, exit_after=0.5),
                "stdin": _FakeStdin(), "spawn_ts": time.time(), "real_pid": None}

    enc = _FakeEnc()
    t0 = time.time()
    err = se._warm_run(enc, "text", str(data_dir / "in2.json"),
                       str(data_dir / "out2.npy"), n_items=1)
    assert err is not None and "lost the job twice" in err
    assert time.time() - t0 < 15.0
    assert len(kills) >= 2
