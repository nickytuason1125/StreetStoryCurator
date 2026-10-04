"""
raw_decode_pool exists to parallelize RAW preview decode across processes
(not threads — LibRaw is documented elsewhere in this repo as not reliably
thread-safe; see early_exit_gate.py's _technical_inspect docstring). These
tests cover the two things that actually matter:

  1. The pool really is a separate-process round trip (proves pickling /
     spawn bootstrap works for the worker function, not just that the
     function is callable in-process).
  2. encode_worker's batch decode degrades correctly when the pool is
     unavailable or breaks mid-batch, instead of losing images silently.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_raw_decode_pool.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import raw_decode_pool as rdp  # noqa: E402
import encode_worker as ew     # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_pool():
    rdp.reset_pool()
    yield
    rdp.reset_pool()


# ── real subprocess round trip ───────────────────────────────────────────────
def test_decode_raw_thumb_runs_in_a_real_worker_process(tmp_path):
    """A nonexistent path can't decode, but the ERROR must come back through a
    real process boundary — proving the worker function is picklable and the
    spawn bootstrap actually works, not just that the function runs in-process."""
    pool = rdp.get_pool()
    fut = pool.submit(rdp._decode_raw_thumb, str(tmp_path / "does_not_exist.rw2"))
    kind, payload = fut.result(timeout=30)
    assert kind is None
    assert isinstance(payload, str) and payload


def test_pool_is_reused_across_calls():
    p1 = rdp.get_pool()
    p2 = rdp.get_pool()
    assert p1 is p2


def test_worker_count_respects_env_override(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_RAW_DECODE_WORKERS", "2")
    assert rdp.worker_count() == 2


# ── encode_worker._decode_raw_batch degrade paths ────────────────────────────
class _FakeFuture:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def result(self, timeout=None):
        if self._exc:
            raise self._exc
        return self._result


class _FakePool:
    """Stands in for a ProcessPoolExecutor without paying real spawn cost —
    encode_worker only ever calls .submit() on it."""
    def __init__(self, results):
        self._results = results   # path -> _FakeFuture

    def submit(self, fn, path):
        return self._results[path]


def test_decode_raw_batch_success_path(monkeypatch):
    jpeg_bytes = _tiny_jpeg_bytes()
    fake = _FakePool({"a.rw2": _FakeFuture(("jpeg", jpeg_bytes))})
    monkeypatch.setattr(rdp, "get_pool", lambda: fake)
    monkeypatch.setattr(rdp, "tie_workers_to_job", lambda: None)

    images, errors = ew._decode_raw_batch(["a.rw2"])
    assert errors == {}
    assert images[0] is not None and images[0].mode == "RGB"


def test_decode_raw_batch_per_file_timeout_is_isolated(monkeypatch):
    import concurrent.futures as _cf
    fake = _FakePool({
        "hang.rw2": _FakeFuture(exc=_cf.TimeoutError()),
        "ok.rw2":   _FakeFuture(("jpeg", _tiny_jpeg_bytes())),
    })
    monkeypatch.setattr(rdp, "get_pool", lambda: fake)
    monkeypatch.setattr(rdp, "tie_workers_to_job", lambda: None)

    images, errors = ew._decode_raw_batch(["hang.rw2", "ok.rw2"])
    assert images[0] is None and "timeout" in errors["hang.rw2"].lower()
    assert images[1] is not None, "one poison file must not fail its neighbours"


def test_decode_raw_batch_falls_back_to_serial_when_pool_is_broken(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    fake = _FakePool({
        "crash.rw2": _FakeFuture(exc=BrokenProcessPool("worker died")),
        "next.rw2":  _FakeFuture(("jpeg", _tiny_jpeg_bytes())),
    })
    monkeypatch.setattr(rdp, "get_pool", lambda: fake)
    monkeypatch.setattr(rdp, "tie_workers_to_job", lambda: None)
    reset_calls = {"n": 0}
    monkeypatch.setattr(rdp, "reset_pool", lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1))

    fallback_calls = []
    def _fake_serial_decode(path, timeout_s=30):
        fallback_calls.append(path)
        return None   # simulate: even the fallback can't read it
    monkeypatch.setattr(ew, "_decode_with_timeout", _fake_serial_decode)
    monkeypatch.setattr(ew, "_decode_last_err", lambda: "serial fallback failed")

    images, errors = ew._decode_raw_batch(["crash.rw2", "next.rw2"])
    assert reset_calls["n"] == 1, "a broken pool must be reset so the NEXT batch gets a fresh one"
    # Once broken, every remaining path in this batch — including the one
    # that already looked fine — must go through the serial fallback rather
    # than trusting a pool that just proved it can drop work.
    assert fallback_calls == ["crash.rw2", "next.rw2"]
    assert images == [None, None]


def test_decode_raw_batch_missing_pool_module_falls_back_entirely(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *a, **k):
        if name == "raw_decode_pool":
            raise ImportError("simulated: module unavailable")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setattr(ew, "_decode_with_timeout", lambda path, timeout_s=30: None)
    monkeypatch.setattr(ew, "_decode_last_err", lambda: "no pool available")

    images, errors = ew._decode_raw_batch(["a.rw2"])
    assert images == [None]
    assert errors["a.rw2"] == "no pool available"


def _tiny_jpeg_bytes() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


# ── encode_worker._decode_chunk routing ──────────────────────────────────────
def test_decode_chunk_routes_raw_and_non_raw_separately(monkeypatch, tmp_path):
    from PIL import Image
    jpg_path = tmp_path / "x.jpg"
    Image.new("RGB", (40, 30), (1, 2, 3)).save(jpg_path)
    raw_path = str(tmp_path / "y.rw2")

    def _fake_raw_batch(raw_paths):
        assert raw_paths == [raw_path]
        return [Image.new("RGB", (10, 10))], {}
    monkeypatch.setattr(ew, "_decode_raw_batch", _fake_raw_batch)

    pil, failed, first_err = ew._decode_chunk([str(jpg_path), raw_path])
    assert failed == [] and first_err is None
    assert pil[0].size == (40, 30)
    assert pil[1].size == (10, 10)
