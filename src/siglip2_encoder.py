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
    _MAX_ATTEMPTS = 2
    _RETRY_DELAY_S = 1.5

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
            _last_rc = None
            for _attempt in range(1, self._MAX_ATTEMPTS + 1):
                print(f"[siglip2] encode_worker start: mode={mode} n={len(items)} attempt={_attempt}", flush=True)
                with open(_crash_log, "a", encoding="utf-8", errors="replace") as _lf:
                    r = win_job.run(
                        [sys.executable, str(self._WORKER), mode, in_path, out_path],
                        env=env, cwd=str(Path(__file__).resolve().parent.parent),
                        stdout=_lf, stderr=_lf,
                        timeout=3600,
                    )
                print(f"[siglip2] encode_worker done: rc={r.returncode} npy={os.path.exists(out_path)}", flush=True)
                if r.returncode == 0 and os.path.exists(out_path):
                    return np.load(out_path)
                _last_rc = r.returncode
                if _attempt < self._MAX_ATTEMPTS:
                    print(f"[siglip2] encode_worker attempt {_attempt} failed (rc={r.returncode}) — retrying", flush=True)
                    time.sleep(self._RETRY_DELAY_S)
            raise RuntimeError(
                f"Vision encoder subprocess failed after {self._MAX_ATTEMPTS} attempts (exit {_last_rc}) — see crash.log"
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
        grade worker with a C-level 0xC0000005 access violation. One load = one
        transient, taken when RAM is freshest; sustained peak is just model +
        one batch. SIGLIP_ENC_CHUNK can still force chunking if ever needed."""
        _enforce_ram_floor()   # per-encode guard (covers the reused-singleton path)
        n = len(paths)
        # Default: no chunking (one subprocess, one model load). A very large
        # cap keeps the override available without splitting realistic folders.
        _CHUNK = int(os.environ.get("SIGLIP_ENC_CHUNK", "100000"))
        if n <= _CHUNK:
            embs = self._run("images", list(paths))
        else:
            parts = []
            for k in range(0, n, _CHUNK):
                chunk = paths[k:k + _CHUNK]
                if progress:
                    progress(
                        0.20 + 0.25 * k / n,
                        f"Analyzing photos {k + 1}–{min(k + _CHUNK, n)} of {n}…",
                    )
                parts.append(self._run("images", chunk))
            embs = np.concatenate(parts, axis=0)
        if progress and n:
            progress(0.47, f"Analyzed {n}/{n} photos")
        return embs

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
