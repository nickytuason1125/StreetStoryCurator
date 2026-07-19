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

import torch
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

MODEL_CACHE_DIR = Path(_CACHE_DIR_STR)
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Source tag includes the tier so grade_pipeline's source-change guard re-encodes
# only when the actual model changes.
ENCODER_SOURCE = f"openclip-{_TIER}-{_MODEL_TAG}"


def _auto_enc_batch() -> int:
    """Pick a SigLIP encode batch from free RAM for stability.

    Smaller batch on a memory-tight machine lowers both the GPU forward-pass VRAM
    and the transient decode RAM. CUDA is NOT queried here — the grade worker must
    never touch the CUDA driver before the encode subprocess starts. The subprocess
    (encode_worker.py) reads VRAM directly and can adjust if needed.
    Batch 8 is a safe ceiling for a 6 GB GPU; RAM gates further below."""
    try:
        import psutil as _ps
        avail = _ps.virtual_memory().available / 1e9
        return max(1, 8 if avail >= 5 else 4 if avail >= 3 else 2)
    except Exception:
        return 4  # safe default for 6 GB GPU


def _enforce_ram_floor() -> None:
    """Raise MemoryError if free RAM is below SIGLIP_MIN_FREE_RAM_GB. Called before
    EVERY encode (not just at construction) so a doomed model load fails cleanly
    instead of OOM-killing the grade worker — even when the encoder singleton is
    reused across runs and __init__ doesn't run again. No-op if the env var is
    unset or psutil is unavailable."""
    _floor = os.environ.get("SIGLIP_MIN_FREE_RAM_GB")
    if not _floor:
        return
    try:
        import psutil as _ps
        _avail = _ps.virtual_memory().available / 1e9
    except Exception:
        return
    if _avail < float(_floor):
        raise MemoryError(
            f"Not enough free RAM for the vision model: only {_avail:.1f} GB free, "
            f"need ~{float(_floor):.1f} GB. Close a couple of apps and retry."
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
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            _MODEL_TAG, pretrained=_PRETRAINED, precision="fp16",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
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
        _p(0.07, "SigLIP-2 ready (isolated encoder)…")

    # ── Subprocess bridge ────────────────────────────────────────────────────
    def _run(self, mode: str, items: list) -> np.ndarray:
        import sys, json, tempfile, subprocess
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
            print(f"[siglip2] encode_worker start: mode={mode} n={len(items)}", flush=True)
            with open(_crash_log, "a", encoding="utf-8", errors="replace") as _lf:
                r = subprocess.run(
                    [sys.executable, str(self._WORKER), mode, in_path, out_path],
                    env=env, cwd=str(Path(__file__).resolve().parent.parent),
                    stdout=_lf, stderr=_lf,
                    timeout=3600,
                )
            print(f"[siglip2] encode_worker done: rc={r.returncode} npy={os.path.exists(out_path)}", flush=True)
            if r.returncode != 0 or not os.path.exists(out_path):
                raise RuntimeError(
                    f"Vision encoder subprocess failed (exit {r.returncode}) — see crash.log"
                )
            return np.load(out_path)
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
                        f"Encoding images {k + 1}–{min(k + _CHUNK, n)} / {n}…",
                    )
                parts.append(self._run("images", chunk))
            embs = np.concatenate(parts, axis=0)
        if progress and n:
            progress(0.47, f"SigLIP-2: {n}/{n}")
        return embs

    def encode_text(self, queries: List[str]) -> np.ndarray:
        """Return normalised (N, EMBED_DIM) float32 embeddings for text queries."""
        return self._run("text", list(queries))

    def unload(self) -> None:
        # No in-process model to free — the subprocess already exited.
        pass


def get_siglip2_encoder() -> SigLIP2Encoder:
    if not hasattr(get_siglip2_encoder, "_instance"):
        get_siglip2_encoder._instance = SigLIP2Encoder()
    return get_siglip2_encoder._instance
