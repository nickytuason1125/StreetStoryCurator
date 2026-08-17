"""
One-time setup: builds the lean HF-format SigLIP-2 checkpoint that
src/encode_worker.py prefers when present.

Why this matters (MEASURED on a 16 GB machine, not estimated): loading the
open_clip checkpoint peaks at 10.3 GB — open_clip builds the giant model in
fp32 and reads a 6.97 GB fp32 .bin into RAM before casting to fp16. With ~5-7 GB
typically free that overflows into the pagefile, which is the stall/OOM during
grading. The HF path builds directly in fp16 with low_cpu_mem_usage and streams
weights from mmap-ed safetensors, peaking near ~3.5 GB.

Designed to survive interruption. Three idempotent stages, each safe to re-run:

  1 DOWNLOAD  snapshot_download into the shared HF cache, retried with backoff.
              Resumes from a partial cache, so a killed run loses nothing.
  2 CONVERT   load fp16 (low_cpu_mem_usage) and save_pretrained to a STAGING
              dir in 1 GB shards. Never touches the live path.
  3 VALIDATE  encode with the REAL runtime worker (src/encode_worker.py) four
              times as separate subprocesses — hf/oc x images/text. Each child
              exits before the next starts, so only ONE model is ever resident
              (an earlier version loaded both in-process and needed ~14 GB).
              Promote staging -> live only if cosine >= 0.98 both ways.

encode_worker.py decides purely on config.json existing, so a half-written
checkpoint at the live path would silently poison every future grade — hence
staging-then-promote, never write-in-place.

Run:  venv\\Scripts\\python.exe scripts/setup_siglip2_hf.py
Requires network (setup only — the runtime stays fully offline).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent.parent
_DEST   = _ROOT / "models" / "siglip2_hf_fp16"
_TMP    = _ROOT / "models" / "_siglip2_hf_fp16_staging"
_WORKER = _ROOT / "src" / "encode_worker.py"

_HF_REPO = "google/siglip2-giant-opt-patch16-384"
# Only what the runtime loader needs. Excludes flax/tf mirrors of the same
# weights, which would otherwise double or triple the bytes pulled.
_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "spiece*"]

_SAMPLE_PROMPTS = [
    "a street photograph", "a black and white portrait",
    "a busy city intersection", "a quiet rural landscape",
]
_MIN_COSINE = 0.98
_EXPECT_DIM = 1536
_MIN_FREE_DISK_GB = 20.0
_RETRIES = 5


def _free_disk_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:
        return 999.0


def _stage1_download() -> str:
    """Fetch into the shared HF cache. Resumable and retried; returns local dir."""
    from huggingface_hub import snapshot_download
    last = None
    for attempt in range(1, _RETRIES + 1):
        try:
            print(f"[1/3] download attempt {attempt}/{_RETRIES} — {_HF_REPO}", flush=True)
            return snapshot_download(
                _HF_REPO,
                allow_patterns=_ALLOW,
                max_workers=2,          # gentler on a laptop link; fewer partials
            )
        except Exception as e:
            last = e
            print(f"      failed: {type(e).__name__}: {e}", flush=True)
            if attempt < _RETRIES:
                _back = min(30, 3 * attempt)
                print(f"      retrying in {_back}s (partial progress is kept)…", flush=True)
                time.sleep(_back)
    raise RuntimeError(f"download failed after {_RETRIES} attempts: {last}")


def _stage2_convert(src_dir: str) -> None:
    """Load fp16 with low_cpu_mem_usage and write sharded safetensors to staging."""
    import torch
    from transformers import AutoModel, AutoProcessor

    if (_TMP / "config.json").exists():
        print("[2/3] staging already converted — skipping", flush=True)
        return
    if _TMP.exists():
        shutil.rmtree(_TMP)
    _TMP.mkdir(parents=True, exist_ok=True)

    print("[2/3] converting to fp16 (streamed, low_cpu_mem_usage)…", flush=True)
    model = AutoModel.from_pretrained(src_dir, dtype=torch.float16, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(src_dir, use_fast=True)
    model.save_pretrained(str(_TMP), max_shard_size="1GB", safe_serialization=True)
    processor.save_pretrained(str(_TMP))
    del model, processor
    print(f"      staged -> {_TMP}", flush=True)


def _encode(loader: str, mode: str, items: list, tag: str):
    """Run ONE encode_worker subprocess and return its embeddings.

    loader: "hf" | "oc". Each call is a fresh process that os._exit()s when done,
    so only one model is resident at a time. cwd MUST be the project root —
    encode_worker resolves the open_clip cache through a relative path.
    """
    import numpy as np
    fd, in_json = tempfile.mkstemp(suffix=".json"); os.close(fd)
    out_npy = in_json + ".npy"
    try:
        Path(in_json).write_text(json.dumps(list(items)), encoding="utf-8")
        env = dict(os.environ)
        env["SIGLIP_HF_DIR"]     = str(_TMP)
        env["SIGLIP_ENC_USE_OC"] = "1" if loader == "oc" else "0"
        env["SIGLIP_ENC_BATCH"]  = "2"
        env["PYTHONIOENCODING"]  = "utf-8"
        print(f"      [{tag}] {len(items)} {mode} via {loader} loader…", flush=True)
        r = subprocess.run([sys.executable, str(_WORKER), mode, in_json, out_npy],
                           cwd=str(_ROOT), env=env)
        if r.returncode != 0 or not os.path.exists(out_npy):
            raise RuntimeError(f"{loader}/{mode} encode failed (exit {r.returncode})")
        return np.load(out_npy)
    finally:
        for f in (in_json, out_npy):
            try: os.unlink(f)
            except Exception: pass


def _cosine(a, b) -> float:
    import numpy as np
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return float((a * b).sum(axis=-1).mean())


def _stage3_validate() -> bool:
    imgs = sorted((_ROOT / "dataset_images").glob("*.jpg"))[:4]
    if not imgs:
        print("      no sample images in dataset_images/ — cannot validate", file=sys.stderr)
        return False
    imgs = [str(p) for p in imgs]

    print("[3/3] validating vs the current open_clip output "
          "(one model resident at a time)…", flush=True)
    hf_img = _encode("hf", "images", imgs,            "1/4")
    oc_img = _encode("oc", "images", imgs,            "2/4")
    hf_txt = _encode("hf", "text",   _SAMPLE_PROMPTS, "3/4")
    oc_txt = _encode("oc", "text",   _SAMPLE_PROMPTS, "4/4")

    if hf_img.shape[1] != _EXPECT_DIM:
        print(f"      embedding dim {hf_img.shape[1]} != {_EXPECT_DIM} — wrong model",
              file=sys.stderr)
        return False

    img_cos = _cosine(hf_img, oc_img)
    txt_cos = _cosine(hf_txt, oc_txt)
    print(f"      image cosine vs open_clip: {img_cos:.4f}", flush=True)
    print(f"      text  cosine vs open_clip: {txt_cos:.4f}", flush=True)
    return img_cos >= _MIN_COSINE and txt_cos >= _MIN_COSINE


def main() -> None:
    if (_DEST / "config.json").exists():
        print(f"{_DEST} already populated — nothing to do.")
        return

    free = _free_disk_gb(_ROOT)
    if free < _MIN_FREE_DISK_GB:
        print(f"Only {free:.1f} GB free on disk; need ~{_MIN_FREE_DISK_GB:.0f} GB "
              f"for the download plus the converted copy.", file=sys.stderr)
        sys.exit(1)

    src = _stage1_download()
    _stage2_convert(src)

    if not _stage3_validate():
        print(f"\nValidation FAILED (need cosine >= {_MIN_COSINE}). Discarding "
              f"{_TMP}; the open_clip fallback stays in place and nothing changed.",
              file=sys.stderr)
        shutil.rmtree(_TMP, ignore_errors=True)
        sys.exit(1)

    if _DEST.exists():
        shutil.rmtree(_DEST)
    _TMP.rename(_DEST)
    print(f"\nDone — checkpoint promoted to {_DEST}")
    print("encode_worker.py will now use the lean fp16 loader (~3.5 GB) instead "
          "of the open_clip fallback (measured 10.3 GB peak).")


if __name__ == "__main__":
    main()
