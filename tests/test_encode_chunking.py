"""RAM-aware encode chunking + partial-run checkpoints (2026-09-11 mechanism).

A 2,754-photo encode used to run as ONE subprocess job while the machine's
free RAM bled to zero; the sustained-collapse watcher killed it and the UI
showed a silent grader. encode_images now chunks by memory_plan's RAM budget
and checkpoints partial embeddings, so a resume never re-encodes finished
chunks. These tests pin the logic with a FAKE encoder — no model, no GPU.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import siglip2_encoder as enc
import memory_plan


# ── plan_encode_chunks: RAM bands, override, unmeasurable default ────────────

def test_chunk_plan_bands(monkeypatch):
    cases = {7.0: 900, 6.0: 900, 5.0: 600, 4.5: 600, 3.5: 350, 3.0: 350,
             2.0: 200, 0.5: 200}
    for free, expected in cases.items():
        monkeypatch.setattr(memory_plan, "free_ram_gb", lambda f=free: f)
        assert memory_plan.plan_encode_chunks(5000) == expected, free


def test_chunk_plan_unmeasurable_falls_to_busy_default(monkeypatch):
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: None)
    assert memory_plan.plan_encode_chunks(5000) == 400


def test_chunk_plan_env_override_wins(monkeypatch):
    monkeypatch.setenv("SIGLIP_ENC_CHUNK", "37")
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: 8.0)
    assert memory_plan.plan_encode_chunks(5000) == 37


# ── bump_chunk_size: grow toward the ceiling only when proven comfortable ────

def test_bump_grows_when_comfortable(monkeypatch):
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    # free 5.0 >= floor 1.5 + 3.0 → one growth step, capped at 900
    assert memory_plan.bump_chunk_size(350, 5.0) == 490
    assert memory_plan.bump_chunk_size(700, 5.0) == 900   # capped


def test_bump_never_grows_when_tight(monkeypatch):
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    assert memory_plan.bump_chunk_size(350, 4.0) == 350   # 4.0 < 1.5 + 3.0
    assert memory_plan.bump_chunk_size(350, None) == 350  # unmeasurable → hold


def test_bump_never_shrinks(monkeypatch):
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    assert memory_plan.bump_chunk_size(600, 1.0) == 600


def test_bump_caps_at_900(monkeypatch):
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    assert memory_plan.bump_chunk_size(900, 8.0) == 900


# ── warm watchdog: no-CPU window scales with job size ────────────────────────

def test_warm_window_small_jobs_keep_90s():
    assert enc._warm_no_cpu_s(8) == 90.0
    assert enc._warm_no_cpu_s(64) == 90.0


def test_warm_window_scales_for_big_jobs():
    assert enc._warm_no_cpu_s(350) == 440.0   # 90 + 350, under the 600 cap
    assert enc._warm_no_cpu_s(900) == 600.0   # capped


def test_warm_window_env_override(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_WARM_NO_CPU_S", "45")
    assert enc._warm_no_cpu_s(350) == 45.0


# ── fake encoder: the chunk loop is pure logic, so test it without a model ──

class FakeEncoder:
    """Records _run calls; encodes deterministically so rows are checkable."""

    def __init__(self, fail_on_call=None):
        self.calls = []
        self._fail_on = fail_on_call
        self._run = self._fake_run

    def _fake_run(self, kind, chunk):
        assert kind == "images"
        if self._fail_on is not None and len(self.calls) >= self._fail_on:
            raise MemoryError("sustained OOM (simulated)")
        self.calls.append(list(chunk))
        return np.array([[float(int(hashlib.md5(p.encode()).hexdigest()[:8], 16)) % 1e6]
                         for p in chunk], dtype=np.float32)


@pytest.fixture()
def fake_ckpt(tmp_path, monkeypatch):
    """Redirect checkpoints into tmp and neutralise env-dependent guards."""
    monkeypatch.setattr(enc, "_encode_ckpt_path",
                        lambda paths: (tmp_path / "ck").mkdir(parents=True, exist_ok=True)
                        or tmp_path / "ck"
                        / (hashlib.md5("\n".join(sorted(paths)).encode()).hexdigest()[:16]
                           + ".npz"))
    monkeypatch.setattr(enc, "_stale_ckpt_sweep", lambda d: None)
    monkeypatch.setattr(enc, "_enforce_ram_floor", lambda: None)
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: 8.0)
    # The chunk loop calls retune_encode_batch() at boundaries; it writes
    # SIGLIP_ENC_BATCH to os.environ DIRECTLY (invisible to monkeypatch
    # teardown) — on this machine the ONNX graph exists, so a 4-chunk fake
    # run leaked batch=32 into later tests (test_batch_is_keyed_on_device
    # reads that env live). Stub it: chunk tests pin chunking, not retune.
    monkeypatch.setattr(memory_plan, "retune_encode_batch", lambda: 0)
    # Keep the env-override chunk size EXACT: bump_chunk_size would grow 40
    # to 56 (free RAM is mocked comfortable) and break the chunk-boundary math.
    monkeypatch.setattr(memory_plan, "bump_chunk_size", lambda planned, free: planned)
    return tmp_path


def _make(n, tmp):
    return [str(tmp / f"p{i:04d}.jpg") for i in range(n)]


def test_small_job_unchanged_single_call(fake_ckpt, tmp_path):
    paths = _make(50, tmp_path)
    e = FakeEncoder()
    out = enc.SigLIP2Encoder.encode_images(e, paths)   # unbound: self=e
    assert e.calls == [paths]
    assert out.shape == (50, enc.EMBED_DIM)
    assert out[7, 0] == e._fake_run("images", [paths[7]])[0, 0]


def test_large_job_chunked_and_in_order(fake_ckpt, tmp_path, monkeypatch):
    monkeypatch.setenv("SIGLIP_ENC_CHUNK", "40")
    paths = _make(130, tmp_path)
    e = FakeEncoder()
    out = enc.SigLIP2Encoder.encode_images(e, paths)
    assert [len(c) for c in e.calls] == [40, 40, 40, 10]
    for i, p in enumerate(paths):
        assert out[i, 0] == e._fake_run("images", [p])[0, 0], i
    assert not list((fake_ckpt / "ck").glob("*.npz"))  # cleaned on success


def test_crash_midjob_checkpoints_then_resume(fake_ckpt, tmp_path, monkeypatch):
    monkeypatch.setenv("SIGLIP_ENC_CHUNK", "40")
    paths = _make(130, tmp_path)
    # First attempt: dies on the third chunk (80 encoded, 50 lost).
    with pytest.raises(MemoryError):
        enc.SigLIP2Encoder.encode_images(FakeEncoder(fail_on_call=2), paths)
    ckpts = list((fake_ckpt / "ck").glob("*.npz"))
    assert len(ckpts) == 1
    with np.load(ckpts[0], allow_pickle=False) as z:
        assert len(z["paths"]) == 80

    # Second attempt resumes: only the 50 unfinished photos re-encode.
    e2 = FakeEncoder()
    out2 = enc.SigLIP2Encoder.encode_images(e2, paths)
    assert [len(c) for c in e2.calls] == [40, 10]
    assert not any(set(c) & set(paths[:80]) for c in e2.calls)
    for i, p in enumerate(paths):
        assert out2[i, 0] == e2._fake_run("images", [p])[0, 0], i
    assert not list((fake_ckpt / "ck").glob("*.npz"))


def test_corrupt_checkpoint_is_ignored(fake_ckpt, tmp_path, monkeypatch):
    monkeypatch.setenv("SIGLIP_ENC_CHUNK", "40")
    paths = _make(80, tmp_path)
    ck = enc._encode_ckpt_path(paths)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.write_bytes(b"not an npz file")
    e = FakeEncoder()
    out = enc.SigLIP2Encoder.encode_images(e, paths)
    assert [len(c) for c in e.calls] == [40, 40]      # full re-encode, no crash
    assert out.shape == (80, enc.EMBED_DIM)


def test_reordered_resume_still_aligns_rows(fake_ckpt, tmp_path, monkeypatch):
    """A resume presenting paths in a different order must not scramble
    embeddings — rows are keyed by path, never by position."""
    monkeypatch.setenv("SIGLIP_ENC_CHUNK", "40")
    paths = _make(80, tmp_path)
    with pytest.raises(MemoryError):
        enc.SigLIP2Encoder.encode_images(FakeEncoder(fail_on_call=1), paths)
    e2 = FakeEncoder()
    shuffled = paths[40:] + paths[:40]
    out = enc.SigLIP2Encoder.encode_images(e2, shuffled)
    for i, p in enumerate(shuffled):
        assert out[i, 0] == e2._fake_run("images", [p])[0, 0], i


def test_stale_sweep_removes_only_old_checkpoints(tmp_path):
    old = tmp_path / "old.npz"; old.write_bytes(b"x")
    new = tmp_path / "new.npz"; new.write_bytes(b"x")
    import os
    two_days_ago = __import__("time").time() - 49 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    enc._stale_ckpt_sweep(tmp_path)
    assert not old.exists() and new.exists()


# ── retune_encode_batch: adaptive batch with the determinism guard ───────────

class _FakeProfile:
    def __init__(self, onnx=True, batch=8):
        self._onnx, self.encode_batch = onnx, batch
    def onnx_enabled(self, **kw):
        return self._onnx


def test_retune_shrinks_in_tight_band(monkeypatch):
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: 1.0)
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    try:
        assert memory_plan.retune_encode_batch() == 2
        import os as _os
        assert _os.environ["SIGLIP_ENC_BATCH"] == "2"
    finally:
        # Direct pop, NOT monkeypatch.delenv: a trailing delenv registers an
        # undo that RESTORES the mid-test value at teardown, leaking it into
        # later tests (run_profile reads this env live — leaked "32" made
        # test_batch_is_keyed_on_device fail).
        __import__("os").environ.pop("SIGLIP_ENC_BATCH", None)


def test_retune_raises_only_on_onnx(monkeypatch):
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: 6.0)
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    fake = type(sys)("run_profile")
    fake.current = lambda: _FakeProfile(onnx=True, batch=8)
    monkeypatch.setitem(sys.modules, "run_profile", fake)
    try:
        got = memory_plan.retune_encode_batch()
        assert got == 16 and got > 8
        import os as _os
        assert _os.environ["SIGLIP_ENC_BATCH"] == "16"
    finally:
        __import__("os").environ.pop("SIGLIP_ENC_BATCH", None)


def test_retune_never_raises_on_torch(monkeypatch):
    """The determinism guard: memory-keyed TORCH batches once made two
    identical culls disagree on 47/514 photos — torch never gets raised."""
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: 8.0)
    monkeypatch.setattr(memory_plan, "_hard_floor_gb", lambda: 1.5)
    fake = type(sys)("run_profile")
    fake.current = lambda: _FakeProfile(onnx=False, batch=8)
    monkeypatch.setitem(sys.modules, "run_profile", fake)
    try:
        assert memory_plan.retune_encode_batch() == 0
        assert "SIGLIP_ENC_BATCH" not in __import__("os").environ
    finally:
        __import__("os").environ.pop("SIGLIP_ENC_BATCH", None)


def test_retune_unmeasurable_is_noop(monkeypatch):
    monkeypatch.delenv("SIGLIP_ENC_BATCH", raising=False)
    monkeypatch.setattr(memory_plan, "free_ram_gb", lambda: None)
    try:
        assert memory_plan.retune_encode_batch() == 0
        assert "SIGLIP_ENC_BATCH" not in __import__("os").environ
    finally:
        __import__("os").environ.pop("SIGLIP_ENC_BATCH", None)
