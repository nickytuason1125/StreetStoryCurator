"""
SigLIP-2 ViT-g/16 @384 Encoder (1536-d embeddings) — open_clip loader.

NOTE (2026-06-15): an HF-transformers loader was prototyped (loads the SAME
weights in ~3.8 GB vs ~9.5 GB, image-emb cosine 0.989 / text 0.9997). It worked
perfectly STANDALONE but destabilised the spawned multiprocessing grade-worker
(native crashes + server executor shutdown). Reverted to this stable open_clip
path. The HF checkpoint is kept at models/siglip2_hf_fp16 for a future retry once
the worker/multiprocessing interaction is understood.

VRAM Protocol: SigLIP2Encoder() loads → encode_images() → unload().
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional, List

# torch is imported LAZILY (inside _download_siglip2_if_needed, its only user).
# This class is a subprocess bridge — the model loads in encode_worker.py, never
# here — so the ~350 MB torch costs the CUDA-free grade worker was pure waste
# held for the whole run. See the same note in specvlm_pipeline.py.
import numpy as np

def _encode_ckpt_path(paths: List[str]) -> Path:
    """Checkpoint file for this encode job: one per tier + exact path set.

    Rows inside are keyed by path, so a resume restores them regardless of
    order. Lives in cache/encode_ckpts/ next to the catalog.
    """
    import hashlib as _hl
    _ckpt_dir = Path(__file__).resolve().parent.parent / "cache" / "encode_ckpts"
    try:
        _ckpt_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return _ckpt_dir / (
        f"enc_{os.environ.get('SIGLIP_TIER', 'x')}_"
        f"{_hl.md5('\\n'.join(sorted(paths)).encode()).hexdigest()[:16]}.npz")


def _stale_ckpt_sweep(ckpt_dir: Path, max_age_s: float = 48 * 3600) -> None:
    """Best-effort cleanup of abandoned encode checkpoints.

    A checkpoint only lives for the duration of one encode job (deleted on
    success), so anything older than ~2 days is debris from a crashed run
    whose job never came back. Old files are unlinked; failures never block
    the encode.
    """
    try:
        import time as _t
        now = _t.time()
        for f in ckpt_dir.glob("*.npz"):
            if now - f.stat().st_mtime > max_age_s:
                f.unlink()
    except Exception:
        pass


class _Tier:
    """Deprecated holder kept so old pickled/keyword references do not break."""
_PRETRAINED = "webli"

# ── Tiered model selection (Phase 1) ─────────────────────────────────────────
# SIGLIP_TIER picks the embedding model so the SAME codebase ships to capable
# and weak machines. All use the stable open_clip loader (no HF-worker risk).
#   high → ViT-g  (1536-d, ~7 GB RAM, ~4 GB VRAM)  — default, your machine
#   mid  → ViT-L  (1024-d, ~6 GB RAM, ~1.8 GB VRAM)
#   low  → ViT-B  ( 768-d, ~4 GB RAM, ~0.8 GB VRAM)  — laptops / weak GPU
# NOTE: tiers other than "high" need the Phase-2 dim-flexible LanceDB schema to
# be fully wired; the encoder itself is dim-agnostic and works today.
_TIERS = {
    "high": ("ViT-gopt-16-SigLIP2-384", 1536, "models/siglip2",   7.0),
    "mid":  ("ViT-L-16-SigLIP2-384",    1024, "models/siglip2_L", 6.0),
    "low":  ("ViT-B-16-SigLIP2-384",     768, "models/siglip2_B", 4.0),
}
_TIER = os.environ.get("SIGLIP_TIER", "high").strip().lower()
if _TIER not in _TIERS:
    _TIER = "high"
_MODEL_TAG, EMBED_DIM, _CACHE_DIR_STR, _DEFAULT_MIN_RAM = _TIERS[_TIER]

# run_profile owns every tier-derived value. The tables below are retained only
# as the open_clip fallback's own metadata; anything the rest of the system
# reads (checkpoint dir, RAM floors, embed dim, encoder source) comes from the
# profile, so there is one place to change when a tier is added.
import run_profile as _rp                                     # noqa: E402
_PROFILE = _rp.current()
EMBED_DIM = _PROFILE.embed_dim

MODEL_CACHE_DIR = Path(_CACHE_DIR_STR)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _hf_dir() -> str:
    return _PROFILE.hf_dir


def _active_loader() -> str:
    """Which loader encode_worker will actually use — must mirror its _load().

    The HF fp16 checkpoint and the open_clip fp32 checkpoint are the same
    weights in different formats, but they do NOT produce identical vectors
    (~0.989 image cosine). They are therefore DIFFERENT embedding spaces, and
    mixing them in one LanceDB table corrupts dedup, the archetype projection
    and the PersonalHead. Folding the loader into ENCODER_SOURCE makes
    grade_pipeline_v2's existing source-change guard fire on the switch, so it
    clears the probe/text caches and re-encodes everything into one space.
    """
    return "hf" if _PROFILE.has_lean_checkpoint else "openclip"


# Source tag includes the tier AND the loader so grade_pipeline's source-change
# guard re-encodes whenever the actual vector space changes.
ENCODER_SOURCE = _PROFILE.encoder_source


def _auto_enc_batch() -> int:
    """Encode batch — FIXED for reproducibility.

    This was picked from free RAM (8 / 4 / 2). That makes the batch COMPOSITION
    differ between runs, and although a ViT has no cross-image interaction
    mathematically, GPU kernels select different algorithms per batch shape, so
    embeddings shift in the last bits and borderline grades flip. Two identical
    culls disagreed on 47 of 514 photos with this and the dedup blocking still
    RAM-derived.

    A fixed 8 is affordable now: ONNX image encoding peaks at 1.20 GB total,
    against 2.70 GB before, so batch sizing is no longer the memory lever it
    was. SIGLIP_ENC_BATCH still overrides.
    """
    return _PROFILE.encode_batch


def _hf_checkpoint_present() -> bool:
    """True when the lean fp16 HF checkpoint exists (encode_worker prefers it)."""
    return _PROFILE.has_lean_checkpoint


def _onnx_active() -> bool:
    """True when image encoding will run through ONNX (see encode_worker)."""
    return _PROFILE.onnx_enabled()


# Floors live in run_profile.TierSpec — measured peaks, one table, one place.
def _hf_floors() -> tuple:
    return (_PROFILE.spec.ram_hard_gb, _PROFILE.spec.ram_soft_gb)


def _default_hard_floor_gb() -> float:
    """Absolute minimum: the encoder's MEASURED peak plus a small margin."""
    if _onnx_active():
        return 1.5                      # measured 1.20 GB
    return _hf_floors()[0] if _hf_checkpoint_present() else 1.5


def _default_ram_floor_gb() -> float:
    """Free-RAM floor required to load the encoder, matched to the loader in use.

    The lean HF fp16 checkpoint peaks near ~3.5 GB, so 4.0 is a true "this will
    fit" gate. The open_clip fallback was MEASURED at 10.3 GB on a 16 GB machine
    (it reads a 6.97 GB fp32 .bin into RAM before converting to fp16) — that is
    more than is ever free in practice, so a floor set to its real requirement
    would block every grade. Those runs do complete today by leaning on the
    pagefile, so the fallback floor stays deliberately permissive: it catches
    only a genuinely hopeless machine and lets everything else proceed (slowly)
    exactly as before. Populating the HF checkpoint via
    scripts/setup_siglip2_hf.py is what actually fixes the fallback's footprint.
    """
    # ONNX image encoding peaks at 1.20 GB (measured on a real 514-photo grade),
    # against 2.70 GB for the PyTorch path. Keeping the PyTorch-era floor would
    # refuse grades the machine can now comfortably run — the guard would have
    # become the binding constraint instead of the memory.
    if _onnx_active():
        return 2.0
    return _hf_floors()[1] if _hf_checkpoint_present() else 2.0


def _enforce_ram_floor() -> None:
    """Raise MemoryError if free RAM is below the floor. Called before EVERY
    encode (not just at construction) so a doomed model load fails cleanly
    instead of OOM-killing the grade worker — even when the encoder singleton is
    reused across runs and __init__ doesn't run again.

    SIGLIP_MIN_FREE_RAM_GB overrides; when unset the floor now DEFAULTS from the
    active loader (it used to be a no-op when unset, which is why an impossible
    load produced a 0xC0000005 crash instead of a readable error). Set it to 0
    to opt out entirely."""
    _floor_env = os.environ.get("SIGLIP_MIN_FREE_RAM_GB")
    try:
        _floor = float(_floor_env) if _floor_env else _default_ram_floor_gb()
    except (TypeError, ValueError):
        _floor = _default_ram_floor_gb()
    if _floor <= 0:
        return
    try:
        import psutil as _ps
    except Exception:
        return

    def _free() -> float:
        return _ps.virtual_memory().available / 1e9

    _avail = _free()
    if _avail >= _floor:
        return

    # ── Adaptive, not fatal ──────────────────────────────────────────────────
    # This used to raise immediately, so a transient dip (a browser tab, an
    # antivirus sweep, the previous stage's buffers not yet reclaimed) killed
    # the whole grade. Free RAM is volatile: collect our own garbage, wait
    # briefly, and re-check before giving up. Only if memory stays below the
    # HARD floor do we refuse — and the hard floor is the encoder's MEASURED
    # peak (2.71 GB for the HF fp16 loader) plus a margin, not the comfort
    # figure. Between hard and soft we proceed with a smaller decode batch and
    # say so, because a slow grade beats a refused one.
    import gc as _gc, time as _time
    _gc.collect()
    for _wait in (0.5, 1.5, 3.0):
        _avail = _free()
        if _avail >= _floor:
            print(f"[siglip2] RAM recovered to {_avail:.1f} GB — continuing")
            return
        _time.sleep(_wait)
    _avail = _free()
    if _avail >= _floor:
        return

    _hard_env = os.environ.get("SIGLIP_HARD_MIN_RAM_GB")
    try:
        _hard = float(_hard_env) if _hard_env else _default_hard_floor_gb()
    except (TypeError, ValueError):
        _hard = _default_hard_floor_gb()

    if _avail >= _hard:
        # Shrink the per-batch decode buffers; the model load itself is fixed.
        os.environ["SIGLIP_ENC_BATCH"] = "2"
        print(f"[siglip2] Low RAM ({_avail:.1f} GB < {_floor:.1f} GB soft floor) — "
              f"proceeding with a reduced encode batch instead of failing. "
              f"Expect a slower encode.")
        return

    raise MemoryError(
        f"Not enough free RAM for the vision model: only {_avail:.1f} GB free, "
        f"need at least ~{_hard:.1f} GB. Close a couple of apps and retry."
    )


def _siglip2_cache_exists() -> bool:
    if not MODEL_CACHE_DIR.exists():
        return False
    weight_exts = {".pt", ".bin", ".safetensors"}
    return any(
        f.suffix in weight_exts
        for f in MODEL_CACHE_DIR.rglob("*")
        if f.is_file() and not f.name.endswith(".incomplete")
    )


def _download_siglip2_if_needed() -> bool:
    if _siglip2_cache_exists():
        return True
    try:
        import torch          # setup-time only — see the module-header note
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            _MODEL_TAG, pretrained=_PRETRAINED, precision="fp16",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        del model
        gc.collect()
        # is_initialized(), never is_available(): this module runs in the grade
        # worker, the PARENT of the encode subprocess, and is_available() is the
        # call that creates a CUDA context there. If nothing initialised CUDA,
        # there is no cache to empty.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"⚠️  SigLIP-2 download failed: {e}")
        return False


# ── Persistent warm encoder (FIRSTCUT_WARM_ENCODER, default on) ─────────────
# One encode_worker in serve mode stays loaded across culls: imports + model
# load happen once (at boot freshness), and every later cull skips both. See
# encode_worker.serve for the file-based job protocol — responses go through a
# per-job .resp.json file so the parent never reads worker stdout (no pipe
# deadlock, stdout keeps flowing to crash.log for diagnostics).
_WORKER_PATH = Path(__file__).resolve().parent / "encode_worker.py"
_WARM: dict | None = None


def _warm_enabled() -> bool:
    return os.environ.get("FIRSTCUT_WARM_ENCODER", "1").strip() != "0"


def _warm_shutdown() -> None:
    global _WARM
    if _WARM is not None:
        try:
            _WARM["proc"].terminate()
        except Exception:
            pass
        _WARM = None


def _sweep_orphan_workers() -> int:
    """Kill ghost `encode_worker serve` processes. Returns how many were killed.

    A warm worker outlives its parent whenever the parent dies without
    running _warm_shutdown — a window closed mid-prewarm, a server
    hard-killed at boot (local_launcher taskkills the port holder), or a
    grade_runner terminated by a stream close. The orphan then sits on
    ~430 MB until its 600 s idle timeout, and if the machine is busy the
    idle exit can be preempted — so duplicates accumulate ("2.0 GB free
    with nothing open", 2026-09-10: two warm workers found alive).

    Matched by SCRIPT NAME in the cmdline (never by bare image name), the
    same rule local_launcher uses for backend remnants. An ORPHAN (parent
    no longer alive) is provably unreferenced and is killed. A worker whose
    parent is alive is left alone — we cannot know which owner is about to
    use it, and this sweep must never turn into a warm-vs-warm war.
    """
    import re as _re
    killed = 0
    try:
        import psutil as _psutil
    except Exception:
        return 0
    try:
        _marker = _re.compile(r"encode_worker\.py[\"']?\s+serve")
        for _p in _psutil.process_iter(["pid", "name", "cmdline", "ppid"]):
            try:
                _cmd = " ".join(_p.info.get("cmdline") or [])
                if not _marker.search(_cmd):
                    continue
                _ppid = _p.info.get("ppid")
                if _ppid_alive(_psutil, _p.info.get("ppid")):
                    continue   # parent alive — owned, leave it
                _p.kill()
                killed += 1
                print(f"[siglip2] swept orphan encode_worker pid={_p.info.get('pid')}", flush=True)
            except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                continue
            except Exception:
                pass
    except Exception:
        pass   # hygiene, not correctness — never block a spawn on the sweep
    return killed


def _ppid_alive(psutil_mod, ppid) -> bool:
    try:
        if ppid is None or ppid <= 0:
            return False
        psutil_mod.Process(int(ppid))
        return True
    except Exception:
        return False


def _warm_ensure(worker_path: Path) -> bool:
    """A live warm worker, spawning one if needed. False = could not spawn."""
    global _WARM
    import sys
    import win_job
    if _WARM is not None and _WARM["proc"].poll() is None:
        return True
    _warm_shutdown()
    _sweep_orphan_workers()   # ghost workers outlive dead parents — sweep on (re)spawn
    import subprocess as _sp
    env = dict(os.environ)
    env["SIGLIP_TIER"] = _TIER
    env.setdefault("PYTHONIOENCODING", "utf-8")
    _crash_log = Path(__file__).resolve().parent.parent / "crash.log"
    try:
        _lf = open(_crash_log, "a", encoding="utf-8", errors="replace")
        proc = win_job.popen(
            [sys.executable, str(worker_path), "serve"],
            stdin=_sp.PIPE, stdout=_lf, stderr=_lf,
            cwd=str(Path(__file__).resolve().parent.parent), env=env,
        )
    except Exception as exc:
        print(f"[siglip2] warm encoder spawn failed (non-fatal): {exc}", flush=True)
        return False
    _WARM = {"proc": proc, "stdin": proc.stdin}
    print(f"[siglip2] warm encoder spawned pid={proc.pid}", flush=True)
    return proc.poll() is None


def _warm_no_cpu_s(n_items: int) -> float:
    """How long a warm worker may show no CPU progress before it is declared
    lost (respawn race) and killed in favour of a one-shot spawn.

    90 s was calibrated for small jobs. The 2026-09-11 real run lost a
    350-photo chunk job to it: a big image encode on a tight machine waits
    on the GPU and a slow SD card, and can legitimately sit at near-zero
    CPU for minutes — the watchdog killed a HEALTHY worker and cost a full
    model reload. The window now scales with job size (90 s small jobs,
    up to ~7 min for 350+), and FIRSTCUT_WARM_NO_CPU_S overrides."""
    override = os.environ.get("FIRSTCUT_WARM_NO_CPU_S", "").strip()
    if override:
        try:
            return max(30.0, float(override))
        except ValueError:
            pass
    return 90.0 if n_items <= 64 else min(600.0, 90.0 + float(n_items))


def _warm_run(enc, mode: str, in_path: str, out_path: str,
              n_items: int = 0) -> "str | None":
    """One job through the warm worker (one respawn retry). None = produced;
    a str return is the worker's error text (so the caller can classify a
    deterministic per-file decode failure and fail fast instead of walking
    the whole one-shot ladder — see _run's unreadable-file breaker)."""
    global _WARM
    import json as _json
    import time as _time
    resp_path = in_path + ".resp.json"
    try:
        os.unlink(resp_path)
    except Exception:
        pass
    for _try in (1, 2):
        if not _warm_ensure(enc._WORKER):
            return "warm encoder could not be spawned"
        req = _json.dumps({"mode": mode, "in": in_path,
                           "out": out_path, "resp": resp_path})
        try:
            _WARM["stdin"].write((req + "\n").encode("utf-8"))
            _WARM["stdin"].flush()
        except Exception:
            print("[siglip2] warm encoder pipe dead — respawning", flush=True)
            _warm_shutdown()
            continue
        deadline = _time.time() + 3600
        # ── Liveness watchdog ─────────────────────────────────────────────────
        # A warm job that never starts (respawn race eats the request) leaves
        # the worker at serve-ready with the model loaded and the parent
        # polling for a resp file that will never be written — the eternal
        # "frozen at the same %" freeze, 3600 s per attempt. Distinguish a
        # HEALTHY long job (worker burns CPU) from a LOST one (worker at 0%
        # CPU): track the worker's total CPU time; if it hasn't moved for 90 s
        # while no resp has arrived, the job was lost — kill the worker and
        # fall through to the one-shot spawn ladder (retry ladder +
        # crash-loop breaker + checkpointed abort). A genuinely working CPU
        # encode never sits at 0 CPU for 90 s.
        _liveliness_floor = _time.time() + 60   # grace period: model warm-up
        # No-CPU window: scaled to the job (see _warm_no_cpu_s), and doubled
        # when the machine is memory-starved — a swapped-out worker is not
        # lost, it is SLOW, and killing it forces a one-shot respawn into the
        # same starvation (the memory watcher owns that regime, not this).
        _no_cpu_s = _warm_no_cpu_s(n_items)
        try:
            import memory_plan as _mp_warm
            _warm_free = _mp_warm.free_ram_gb()
            if _warm_free is not None and _warm_free < 2.0:
                _no_cpu_s = max(_no_cpu_s, 600.0)
        except Exception:
            pass
        # TWO-PHASE liveness (2026-09-11, refined after the real run):
        #   Phase 1 (fast): a RECEIVED job shows model-load/encode CPU within
        #     seconds of submission. Zero CPU for 45 s means the respawn race
        #     ate the request — kill fast; the one-shot spawn is already the
        #     faster path in that state.
        #   Phase 2 (patient): once the job is known-started (CPU > 1 s), use
        #     the scaled _no_cpu_s window — a working worker may legitimately
        #     stall on the GPU and a slow SD card.
        _received = False
        _fast_deadline = _time.time() + 45
        _last_cpu = 0.0
        _last_cpu_change = _time.time()
        try:
            import psutil as _psutil
            _warm_ps = _psutil.Process(_WARM["proc"].pid)
            _last_cpu = sum(_warm_ps.cpu_times())
        except Exception:
            _warm_ps = None
        while _time.time() < deadline:
            if os.path.exists(resp_path):
                try:
                    res = _json.load(open(resp_path, encoding="utf-8"))
                except Exception:
                    res = {}
                if res.get("ok") and os.path.exists(out_path):
                    return None
                print(f"[siglip2] warm encoder job failed: "
                      f"{str(res.get('error', ''))[-300:]}", flush=True)
                return str(res.get("error", "")) or "warm encoder job failed"
            if _WARM["proc"].poll() is not None:
                print("[siglip2] warm encoder died mid-job — respawning", flush=True)
                _warm_shutdown()
                break
            if _warm_ps is not None and _time.time() > _liveliness_floor:
                try:
                    cpu_now = sum(_warm_ps.cpu_times())
                    if cpu_now > 1.0:
                        _received = True
                    if cpu_now != _last_cpu:
                        _last_cpu = cpu_now
                        _last_cpu_change = _time.time()
                    if not _received and _time.time() > _fast_deadline:
                        print("[siglip2] warm worker showed no CPU 45s after "
                              "submission — job lost in respawn race; killing "
                              "worker, falling back to one-shot spawn", flush=True)
                        _warm_shutdown()
                        break
                    if (_received and cpu_now == _last_cpu
                            and _time.time() - _last_cpu_change > _no_cpu_s):
                        print(f"[siglip2] warm encoder idle {_no_cpu_s:.0f}s with no resp — "
                              "job lost in respawn race; killing worker, "
                              "falling back to one-shot spawn", flush=True)
                        _warm_shutdown()
                        break
                except Exception:
                    pass   # worker gone — poll() check handles it
            _time.sleep(0.25)
    return "warm encoder lost the job twice (died mid-job / idle no-resp)"


def warm_start() -> bool:
    """Spawn the warm worker and pre-load the model, without blocking.

    Called after the UI window opens (machine at its freshest) so the heavy
    imports + model load never land mid-session. Returns False when the warm
    path is disabled or the spawn failed (one-shot still works).
    """
    if not _warm_enabled():
        return False
    import threading as _threading

    def _prewarm():
        try:
            if not _warm_ensure(_WORKER_PATH):
                return
            import json as _json
            import tempfile as _tf
            fd, tin = _tf.mkstemp(suffix=".warm.json"); os.close(fd)
            tout = tin + ".npy"
            with open(tin, "w", encoding="utf-8") as f:
                _json.dump(["warmup"], f)
            resp = tin + ".resp.json"
            global _WARM
            req = _json.dumps({"mode": "text", "in": tin, "out": tout, "resp": resp})
            _WARM["stdin"].write((req + "\n").encode("utf-8"))
            _WARM["stdin"].flush()
            print("[siglip2] warm pre-load submitted", flush=True)
            # The tiny job's output is never read: its whole purpose is to
            # trigger _load() inside the worker. The idle timeout cleans up.
        except Exception as exc:
            print(f"[siglip2] warm pre-load failed (non-fatal): {exc}", flush=True)

    _threading.Thread(target=_prewarm, daemon=True, name="encoder-prewarm").start()
    return True


def _unreadable_user_msg(err: str, n: int) -> str:
    """User-facing message for a deterministic all-files-unreadable encode.
    Carries the failed file list extracted from the worker's error when the
    warm path supplied one."""
    import re as _re
    _m = _re.search(r"\[failed: (.+?)\]", err or "")
    _files = f" Affected file(s): {_m.group(1)}" if _m else ""
    return (f"{n} photo(s) could not be read — the file(s) are corrupt or "
            f"truncated, or the card/reader dropped out mid-copy. "
            f"Remove or re-copy them, then grade again.{_files}")


class SigLIP2Encoder:
    """SigLIP-2 image+text encoder — runs the model in an ISOLATED SUBPROCESS.

    The model never loads inside this (grade-worker) process. Each encode spawns
    src/encode_worker.py, which loads the model in a clean process (the efficient
    HF FP16 loader for the high tier — ~4 GB, vs ~9.5 GB in-process — that
    native-crashes inside the multiprocessing worker but is fine standalone),
    encodes, writes a .npy, and EXITS (freeing all RAM/VRAM). If the model OOMs,
    only the subprocess dies — this process reads the non-zero exit code and
    raises a clean error, so the worker never wedges.
    """

    _WORKER = Path(__file__).resolve().parent / "encode_worker.py"

    def __init__(self, device: str = "auto", quantize: bool = False, progress=None):
        _p = progress or (lambda f, d: None)
        # Device selection is handled entirely inside encode_worker.py.
        # NOTE: encode_worker.py uses os._exit(0) to bypass PyTorch's CUDA atexit,
        # which was crashing this (grade-worker) process via NVIDIA driver callbacks
        # when the subprocess exited. No CUDA init is needed here — the grade worker
        # defers all CUDA work to the encode subprocess (for SigLIP) and then to
        # Qwen/TOPIQ later. encode_worker.py handles device selection itself.
        self.device = device   # informational only; subprocess picks the real device
        _enforce_ram_floor()   # bail cleanly if RAM can't fit the model
        _p(0.07, "Image analysis ready…")

    # ── Subprocess bridge ────────────────────────────────────────────────────
    # Retries: crash.log shows encode_worker occasionally hard-crashing at model
    # load with a transient "CUDA error: out of memory" / "device(s) busy or
    # unavailable" (WDDM-level contention with another GPU subprocess), and the
    # very next attempt with identical inputs succeeding. One retry after a short
    # backoff turns that into a self-healing blip instead of a failed grade.
    #
    # 2026-09-07 escalation: on a commit-starved machine the worker now dies
    # during IMPORTS (sklearn/scipy MemoryError at 0.47 GB RSS, "The paging
    # file is too small") — a machine-state dip, not a code bug. Two attempts
    # 1.5 s apart cannot win that: the dip lasts seconds. The loop now runs
    # 4 attempts with escalating backoff (2s / 8s / 32s) and a gc.collect()
    # between them, because free RAM is volatile in exactly the window where
    # the retry lands. Attempt 1 failing twice in a row is the dip; attempt 4
    # lands on a rebalanced machine.
    #
    # But the retry loop is the SECOND line of defence. The first is the
    # persistent warm encoder (serve mode + _warm_* below): imports and the
    # model load happen ONCE at app-boot freshness, and every cull afterwards
    # reuses the loaded model — the import-time OOM class disappears and each
    # cull starts 30–60 s faster. The one-shot loop below remains the
    # fallback (warm worker dead AND respawn failed → exactly today's
    # behaviour, never worse).
    _MAX_ATTEMPTS = 4
    _RETRY_DELAY_S = 2.0

    def _run(self, mode: str, items: list) -> np.ndarray:
        import sys, json, tempfile, subprocess, time
        import win_job
        if not items:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        fd, in_path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        out_path = in_path + ".npy"
        # Write encode subprocess stdout/stderr directly to crash.log instead of
        # capturing them in a pipe. Critical: if Windows OOM-kills the grade worker
        # process, its pipe handles are closed → encode_worker gets SIGPIPE/broken
        # pipe and dies silently too (nothing visible). Writing to the log file means
        # the encode subprocess has its OWN handle to crash.log (inherited from
        # CreateProcess), so it keeps writing even after the grade worker is gone —
        # and next time we open crash.log we can read exactly what went wrong.
        _crash_log = Path(__file__).resolve().parent.parent / "crash.log"
        try:
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump(list(items), f)
            env = dict(os.environ)
            env["SIGLIP_TIER"] = _TIER
            env.setdefault("PYTHONIOENCODING", "utf-8")
            if mode == "images":
                env.setdefault("SIGLIP_ENC_BATCH", str(_auto_enc_batch()))
            # ── Preferred path: the persistent warm worker ──────────────────
            # No imports, no model load — the serve-mode worker is already
            # loaded. On any warm-path failure we fall through to the
            # one-shot loop below (today's behaviour, never worse).
            _warm_err = ""
            if _warm_enabled() and os.environ.get("FIRSTCUT_NO_WARM", "") != "1":
                try:
                    _warm_err = _warm_run(self, mode, in_path, out_path, n_items=len(items))
                    if _warm_err is None:
                        return np.load(out_path)
                    print(f"[siglip2] warm encoder unavailable ({str(_warm_err)[-160:]}) — "
                          f"falling back to a one-shot spawn", flush=True)
                except Exception as _warm_exc:
                    _warm_err = str(_warm_exc)
                    print(f"[siglip2] warm encoder error ({_warm_exc}) — falling back to a one-shot spawn", flush=True)
            _saw_unreadable = "files unreadable" in (_warm_err or "")
            _last_rc = None
            for _attempt in range(1, self._MAX_ATTEMPTS + 1):
                print(f"[siglip2] encode_worker start: mode={mode} n={len(items)} attempt={_attempt}", flush=True)
                import subprocess as _sp
                with open(_crash_log, "a", encoding="utf-8", errors="replace") as _lf:
                    r = win_job.run(
                        [sys.executable, str(self._WORKER), mode, in_path, out_path],
                        env=env, cwd=str(Path(__file__).resolve().parent.parent),
                        stdin=_sp.DEVNULL,   # explicit: a detached pythonw chain inherits a std handle that turns
                                             # non-duplicable in the grandchild (WinError 6/50 at pipe creation —
                                             # measured 2026-09-06). DEVNULL opens NUL fresh in THIS process.
                        stdout=_lf, stderr=_lf,
                        timeout=3600,
                    )
                print(f"[siglip2] encode_worker done: rc={r.returncode} npy={os.path.exists(out_path)}", flush=True)
                if r.returncode == 0 and os.path.exists(out_path):
                    return np.load(out_path)
                _last_rc = r.returncode
                # ── Deterministic unreadable-files breaker (2026-09-10) ──────
                # Exit 3 = every file failed to DECODE. Decoding is a pure
                # function of the file's bytes — a fresh process, a fresh
                # model load, and a CPU tier all read the same bytes, so the
                # ladder cannot help. Two consecutive deterministic failures
                # (the warm attempt + one one-shot confirmation) end the run
                # with a message that names the files, instead of the 8-model-
                # reload crawl that wedged a 2-photo grade for 12+ minutes.
                if r.returncode == 3:
                    if _saw_unreadable:
                        raise RuntimeError(_unreadable_user_msg(_warm_err, len(items)))
                    _saw_unreadable = True
                    print("[siglip2] encode_worker unreadable-files failure — "
                          "one confirmation retry, then fail fast", flush=True)
                    time.sleep(2.0)
                    continue
                if _attempt < self._MAX_ATTEMPTS:
                    # ── Crash-loop breaker ───────────────────────────────────
                    # The spawned worker already ran _enforce_ram_floor and
                    # died anyway — respawning into the same conditions just
                    # burns the 2s/8s/32s ladder and produces the classic
                    # "frozen at the same % forever" zombie. Check the floor
                    # HERE, in the parent, before deciding to respawn: below
                    # the hard floor the failure is a machine-state refusal,
                    # not a transient dip, so fail the attempt loop now and
                    # let the checkpointed-abort message reach the user.
                    try:
                        _enforce_ram_floor()
                    except MemoryError:
                        print(f"[siglip2] encode_worker attempt {_attempt} failed "
                              f"(rc={r.returncode}) and free RAM is below the hard "
                              f"floor — breaking the respawn loop (checkpoint saved, "
                              f"resume available)", flush=True)
                        raise
                    # The failure is usually a transient commit/RAM dip, not
                    # code (see the constants' comment). Collect our own
                    # garbage, then back off long enough for the machine to
                    # rebalance — standby lists refill and other processes
                    # free pages on a timescale of seconds, so a 1.5 s retry
                    # only ever won when the dip had already passed.
                    import gc as _gc
                    import memory_plan as _mp_run
                    _free = _mp_run.free_ram_gb()
                    _delay = self._RETRY_DELAY_S * (4 ** (_attempt - 1))   # 2s, 8s, 32s
                    _free_txt = "unmeasurable" if _free is None else f"{_free:.2f} GB free"
                    print(f"[siglip2] encode_worker attempt {_attempt} failed "
                          f"(rc={r.returncode}, {_free_txt}) — gc + retry in {_delay:.0f}s",
                          flush=True)
                    _gc.collect()
                    time.sleep(_delay)
                    # Wait-for-recovery: a fixed sleep loses to a dip that
                    # outlasts it. Hold the retry until the machine actually
                    # has room for a reduced-batch encode (floor + margin, up
                    # to 90 s per attempt) — riding out a transient VS Code/
                    # antivirus surge instead of burning all attempts while
                    # the machine sits at bottom.
                    _floor = _mp_run.min_admission_gb()
                    _recovery_deadline = time.time() + 90
                    while time.time() < _recovery_deadline:
                        _free = _mp_run.free_ram_gb()
                        if _free is None or _free >= _floor:
                            break
                        time.sleep(2)
            raise MemoryError(
                f"The vision encoder could not start after {self._MAX_ATTEMPTS} attempts "
                f"(last exit {_last_rc}) — the machine ran out of memory while loading it. "
                f"Everything graded so far is saved: close a few apps (or reboot if the "
                f"RAM chip has been red a while) and press Resume — the cull continues "
                f"from its checkpoint instead of starting over."
            )
        finally:
            for _f in (in_path, out_path):
                try: os.unlink(_f)
                except Exception: pass

    def encode_images(self, paths: List[str], batch_size: int = 0, progress=None) -> np.ndarray:
        """Return normalised (N, EMBED_DIM) float32 embeddings for image paths.

        SINGLE-SUBPROCESS encode (2026-07-02): one encode_worker.py process loads
        the SigLIP model ONCE and encodes ALL images, batching internally (only
        SIGLIP_ENC_BATCH images decoded in RAM at a time). This replaces the old
        per-150-chunk design, which respawned a fresh subprocess PER CHUNK and
        therefore RELOADED the whole ~3.5 GB model at every chunk boundary. On a
        16 GB machine each reload spike (landing on an already-tight system with
        the app + frontend running) drove free RAM to near-zero and killed the
        grade worker with a C-level 0xC0000005 access violation.

        RAM-AWARE CHUNKING (2026-09-11): the unlimited single run has its own
        failure — a 2,754-photo encode bled the MACHINE's free RAM from 2 GB
        to 0.6 GB over minutes until the sustained-collapse watcher killed the
        worker and the UI showed a silent grader. Large jobs are now split
        into chunks sized by memory_plan.plan_encode_chunks() (200–900 photos
        by free RAM). Each chunk keeps the full single-load protection ladder;
        between chunks the subprocess exits (freeing its RAM), the machine gets
        a recovery window, and PARTIAL EMBEDDINGS ARE CHECKPOINTED so a resume
        never re-encodes finished chunks. SIGLIP_ENC_CHUNK still overrides."""
        _enforce_ram_floor()   # per-encode guard (covers the reused-singleton path)
        n = len(paths)
        import memory_plan as _mp
        _CHUNK = _mp.plan_encode_chunks(n)
        if n > _CHUNK:
            print(f"[siglip2] encode chunking: {n} photos -> {_CHUNK}-photo chunks "
                  f"(free RAM {4 if _mp.free_ram_gb() is None else round(_mp.free_ram_gb(), 1)} GB)",
                  flush=True)

        # ── Partial-run checkpoint: a killed cull resumes at chunk granularity ──
        # Rows are keyed by path, so a resume fills them back regardless of order.
        _ckpt_path = _encode_ckpt_path(paths)

        _EMB_DIM = EMBED_DIM
        out    = np.zeros((n, _EMB_DIM), dtype=np.float32)
        filled = np.zeros(n, dtype=bool)
        if _ckpt_path.exists():
            try:
                # Context manager: the open handle would otherwise block the
                # atomic replace in _commit() for the rest of the job (Windows).
                with np.load(_ckpt_path, allow_pickle=False) as _z:
                    _saved = {str(p): i for i, p in enumerate(_z["paths"])}
                    _embs_ref = _z["embs"]
                    _hit = 0
                    for i, p in enumerate(paths):
                        j = _saved.get(p)
                        if j is not None and np.isfinite(_embs_ref[j]).all():
                            out[i], filled[i] = _embs_ref[j], True
                            _hit += 1
                if _hit:
                    print(f"[siglip2] encode checkpoint: resuming with {_hit}/{n} "
                          f"already encoded", flush=True)
            except Exception as _exc:
                print(f"[siglip2] encode checkpoint unreadable ({_exc}) — "
                      f"starting fresh", flush=True)
                out.fill(0); filled.fill(False)

        def _commit():
            """Persist every filled row so a kill never costs finished chunks."""
            try:
                rows = np.nonzero(filled)[0]
                if not len(rows):
                    return
                _tmp = _ckpt_path.with_suffix(".tmp.npz")
                np.savez(_tmp,
                         paths=np.array([paths[i] for i in rows], dtype=np.str_),
                         embs=out[rows])
                _tmp.replace(_ckpt_path)
            except Exception as _exc:
                print(f"[siglip2] chunk checkpoint skipped ({_exc})", flush=True)

        # Chunked jobs skip the warm worker entirely (FIRSTCUT_NO_WARM): the
        # respawn race ate warm jobs repeatedly in real runs (the worker sits
        # at serve-ready having never received the request), and each race
        # costs a 45-60 s fast-fail plus a fallback model reload anyway. With
        # chunks of only a few hundred photos, one straight one-shot load per
        # chunk is cheaper AND deterministic. Small jobs keep the warm path.
        _chunked = n > _CHUNK
        if _chunked:
            os.environ["FIRSTCUT_NO_WARM"] = "1"

        _stale_ckpt_sweep(_ckpt_path.parent)

        # Chunk size is RE-PLANNED at every boundary (smart: tracks the live
        # machine, not a snapshot from job start) and the batch is retuned
        # (fast: bigger when comfortable, 2 when tight). ETA rides in the
        # progress text once there is a measured rate.
        _todo = [i for i in range(n) if not filled[i]]
        _pos   = 0
        import time as _time
        _t0    = _time.time()
        _rate  = 0.0            # photos/sec, measured across finished chunks
        _chunk_size = max(1, _mp.plan_encode_chunks(n))   # grows via bump_chunk_size
        while _pos < len(_todo):
            _plan = _mp.plan_encode_chunks(n)
            try:
                _chunk_size = _mp.bump_chunk_size(max(_plan, _chunk_size), _mp.free_ram_gb())
            except Exception:
                _chunk_size = _plan
            _CHUNK = max(1, _chunk_size)
            _idx   = _todo[_pos:_pos + _CHUNK]
            _chunk = [paths[i] for i in _idx]
            if progress and n:
                _eta = (f" · ~{int(((len(_todo) - _pos) / _rate) // 60)} min left"
                        if _rate > 0 else "")
                progress(0.07 + 0.39 * (_pos) / max(len(_todo), 1),
                         f"Analyzing photos {_todo[_pos] + 1}–{_todo[-1] + 1} "
                         f"of {n}{_eta}")
            # Boundary guard: before the next model load, give the machine one
            # honest chance to recover if RAM sagged during the previous chunk.
            if _pos:
                try:
                    _mp.retune_encode_batch()
                    _free = _mp.free_ram_gb()
                    _floor = _mp.min_admission_gb()
                    _deadline = _time.time() + 180
                    while _free is not None and _free < _floor and _time.time() < _deadline:
                        _time.sleep(3)
                        _free = _mp.free_ram_gb()
                except Exception:
                    pass
            out[_idx] = np.asarray(self._run("images", _chunk), dtype=np.float32)
            filled[_idx] = True
            _commit()
            _pos += len(_idx)
            _elapsed = _time.time() - _t0
            if _elapsed > 5 and _pos:
                _rate = _pos / _elapsed

        if progress and n:
            progress(0.47, f"Analyzed {n}/{n} photos")
        try: _ckpt_path.unlink(missing_ok=True)
        except Exception: pass
        return out

    def encode_text(self, queries: List[str]) -> np.ndarray:
        """Return normalised (N, EMBED_DIM) float32 embeddings for text queries."""
        _enforce_ram_floor()
        return self._run("text", list(queries))

    def encode_text_groups(self, groups: "dict[str, list[str]]") -> "dict[str, np.ndarray]":
        """Encode multiple named prompt groups in ONE subprocess call (one model
        load) instead of one encode_text() call per group. Each encode_text()
        call spawns its own encode_worker.py subprocess and reloads the whole
        SigLIP-2 model from scratch — calling it N times in a row (e.g. the
        probe-cache-miss path's 9 prompt groups) means N full model reloads
        back to back, each an 8 GB (open_clip fallback) or 3.5 GB (HF loader)
        RAM spike. Flattening all groups into one list and splitting the
        result after a single _run() call collapses that to one reload."""
        _enforce_ram_floor()
        names: list = []
        flat:  list = []
        bounds: list = []
        for name, items in groups.items():
            start = len(flat)
            flat.extend(items)
            names.append(name)
            bounds.append((start, len(flat)))
        embs = self._run("text", flat)
        return {name: embs[s:e] for name, (s, e) in zip(names, bounds)}

    def unload(self) -> None:
        # No in-process model to free — the subprocess already exited.
        pass


def get_siglip2_encoder() -> SigLIP2Encoder:
    if not hasattr(get_siglip2_encoder, "_instance"):
        get_siglip2_encoder._instance = SigLIP2Encoder()
    return get_siglip2_encoder._instance
