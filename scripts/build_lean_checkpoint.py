"""
Build a lean fp16 checkpoint for ANY quality tier.

Generalises setup_siglip2_hf.py (which was hardcoded to the giant encoder). The
point is the same: open_clip loads a full fp32 checkpoint into RAM before
casting to fp16, so the "smaller" tiers were costing MORE memory than the giant
does on its lean path — Balanced at 4.0 GB vs Pro at 3.0 GB. A lean checkpoint
flips that, and tier_select then picks Balanced automatically on a busy machine.

Three idempotent stages, safe to re-run and safe to interrupt:

  1 DOWNLOAD  snapshot_download into the shared HF cache (resumable, retried).
  2 CONVERT   load fp16 with low_cpu_mem_usage, save sharded safetensors to a
              STAGING dir. Never touches the live path.
  3 VALIDATE  encode through the REAL runtime worker (src/encode_worker.py) in a
              separate process, then check the embeddings are sane:
                * correct width for the tier
                * finite and unit-norm
                * discriminative (distinct photos are not collapsed)
              Promote staging -> live only if all checks pass.

NOTE ON VALIDATION: for a tier that has never run here there is no reference
embedding set to compare against (unlike the giant, whose open_clip vectors were
already on disk). So this validates the checkpoint is *correct and usable*, not
that it matches another model — a different tier is a different model by design.

Run:  venv\\Scripts\\python.exe scripts/build_lean_checkpoint.py --tier mid
      venv\\Scripts\\python.exe scripts/build_lean_checkpoint.py --tier mid --promote
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

_ROOT   = Path(__file__).resolve().parent.parent
_WORKER = _ROOT / "src" / "encode_worker.py"

# tier -> (hf repo id, expected embed dim, staging dir, live dir)
_TIERS = {
    "mid":  ("google/siglip2-large-patch16-384", 1024,
             "models/_siglip2_L_hf_fp16_staging", "models/siglip2_L_hf_fp16"),
    "low":  ("google/siglip2-base-patch16-384",   768,
             "models/_siglip2_B_hf_fp16_staging", "models/siglip2_B_hf_fp16"),
    "high": ("google/siglip2-giant-opt-patch16-384", 1536,
             "models/_siglip2_hf_fp16_staging",   "models/siglip2_hf_fp16"),
}
_ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "spiece*"]
_RETRIES = 5
_MIN_FREE_DISK_GB = 12.0


def _free_disk_gb() -> float:
    try:
        return shutil.disk_usage(str(_ROOT)).free / 1e9
    except Exception:
        return 999.0


def _stage1_download(repo: str) -> str:
    from huggingface_hub import snapshot_download
    last = None
    for attempt in range(1, _RETRIES + 1):
        try:
            print(f"[1/3] download attempt {attempt}/{_RETRIES} — {repo}", flush=True)
            return snapshot_download(repo, allow_patterns=_ALLOW, max_workers=2)
        except Exception as e:
            last = e
            print(f"      failed: {type(e).__name__}: {e}", flush=True)
            if attempt < _RETRIES:
                back = min(30, 3 * attempt)
                print(f"      retrying in {back}s (partial progress is kept)…", flush=True)
                time.sleep(back)
    raise RuntimeError(f"download failed after {_RETRIES} attempts: {last}")


# fp16 bytes accumulated before a shard is flushed. Bounds peak RSS: only the
# tensors of the CURRENT shard are held.
_SHARD_BYTES = 700 * 1024 * 1024
# A single tensor larger than this is converted in row slices rather than whole.
# SigLIP-2's token embedding is 256000x1024 fp32 = 1.05 GB on its own, which is
# more than a loaded machine has free — every other tensor here is under 17 MB.
_ROW_CHUNK_BYTES = 64 * 1024 * 1024

_ST_DTYPE = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "BF16": "<u2",
             "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "|i1",
             "U8": "|u1", "BOOL": "|b1"}


def _read_header(path: Path) -> tuple:
    """(header dict, byte offset where the data blob starts). No mmap."""
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def _load_tensor_fp16(fh, base: int, meta: dict):
    """Read one tensor straight off disk and return it as fp16.

    Deliberately uses seek/read rather than safetensors' mmap: mapping the whole
    3.5 GB file is what fails on a loaded Windows box (OSError 1455 'paging file
    too small', or a hard segfault inside the native reader). Reading byte
    ranges needs no address-space reservation.
    """
    import numpy as np
    dt = _ST_DTYPE.get(meta["dtype"])
    if dt is None:
        raise ValueError(f"unsupported dtype {meta['dtype']}")
    start, end = meta["data_offsets"]
    shape = tuple(meta["shape"])
    nbytes = end - start

    # bf16 has no numpy dtype: widen to fp32 by placing the 16 bits in the high
    # half of the word, then narrow to fp16. Never hit for the F32 checkpoints
    # shipped today, but a silent misread here would be untraceable later.
    if meta["dtype"] == "BF16":
        fh.seek(base + start)
        raw = np.frombuffer(fh.read(nbytes), dtype="<u2").astype("<u4") << 16
        return raw.view("<f4").astype(np.float16).reshape(shape)

    itemsize = np.dtype(dt).itemsize
    if nbytes <= _ROW_CHUNK_BYTES or not shape or shape[0] < 2:
        fh.seek(base + start)
        arr = np.frombuffer(fh.read(nbytes), dtype=dt).reshape(shape)
        return arr.astype(np.float16) if arr.dtype != np.float16 else arr.copy()

    # Big tensor: fill a preallocated fp16 buffer a row-block at a time, so peak
    # is (fp16 result + one block) instead of (fp32 whole + fp16 whole).
    rows = shape[0]
    row_items = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    row_bytes = row_items * itemsize
    block = max(1, _ROW_CHUNK_BYTES // max(1, row_bytes))
    out = np.empty(shape, dtype=np.float16)
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block)
        fh.seek(base + start + r0 * row_bytes)
        chunk = np.frombuffer(fh.read((r1 - r0) * row_bytes), dtype=dt)
        out[r0:r1] = chunk.reshape((r1 - r0,) + shape[1:]).astype(np.float16)
        del chunk
    return out


def _stage2_convert_streaming(src_dir: str, staging: Path) -> None:
    """Convert fp32 -> fp16 without ever building the model.

    Produces a normal sharded HF checkpoint (shards + index.json + the config
    and tokenizer files copied verbatim), which stage 3 then validates through
    the real encode_worker before anything is promoted.
    """
    import numpy as np
    from safetensors.numpy import save_file

    src = Path(src_dir)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no .safetensors found in {src}")

    print(f"[2/3] converting to fp16 (low-RAM streaming, "
          f"{len(shards)} source file(s))…", flush=True)

    weight_map: dict = {}
    total_bytes = 0
    buf: dict = {}
    buf_bytes = 0
    out_idx = 0
    n_done = 0

    def flush():
        nonlocal buf, buf_bytes, out_idx
        if not buf:
            return
        out_idx += 1
        name = f"model-{out_idx:05d}.safetensors"
        save_file(buf, str(staging / name))
        for k in buf:
            weight_map[k] = name
        print(f"      shard {name}  ({buf_bytes/1e9:.2f} GB, {len(buf)} tensors)",
              flush=True)
        buf = {}
        buf_bytes = 0

    for sh in shards:
        hdr, base = _read_header(sh)
        with open(sh, "rb") as fh:
            for key in hdr:
                t = _load_tensor_fp16(fh, base, hdr[key])
                buf[key] = t
                buf_bytes += t.nbytes
                total_bytes += t.nbytes
                n_done += 1
                if buf_bytes >= _SHARD_BYTES:
                    flush()
    flush()

    # Sharded checkpoints are found through this index, not by globbing.
    (staging / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes},
                    "weight_map": weight_map}, indent=2), encoding="utf-8")

    # config / tokenizer / preprocessor come across untouched.
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, staging / f.name)

    # Tell loaders the weights are fp16 now; save_pretrained would have done
    # this. Left as float32 the model materialises in fp32 and the whole point
    # of the lean checkpoint is lost.
    cfg_path = staging / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["torch_dtype"] = "float16"
        cfg["dtype"] = "float16"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"      staged -> {staging}  ({total_bytes/1e9:.2f} GB fp16, "
          f"{n_done} tensors)", flush=True)


def _free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 999.0


def _should_stream(src_dir: str, free_gb: Optional[float] = None) -> tuple:
    """(stream?, free_gb, needed_gb) — kept pure so it is testable on its own.

    from_pretrained mmaps the whole fp32 checkpoint and then builds the model on
    top, so it needs roughly 1.4x the file size free. Below that it does not
    degrade — it dies, either with OSError 1455 ('paging file is too small') or
    a segfault inside the native safetensors reader. The choice is therefore
    made on measurement rather than by catching a failure: a segfault cannot be
    caught.
    """
    free = _free_ram_gb() if free_gb is None else free_gb
    src_gb = sum(f.stat().st_size for f in Path(src_dir).glob("*.safetensors")) / 1e9
    return free < src_gb * 1.4, free, src_gb * 1.4


def _stage2_convert(src_dir: str, staging: Path) -> None:
    if (staging / "config.json").exists():
        print("[2/3] staging already converted — skipping", flush=True)
        return
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    stream, free_gb, need_gb = _should_stream(src_dir)
    if stream:
        print(f"[2/3] {free_gb:.1f} GB free vs {need_gb:.1f} GB needed to load "
              f"the model — using the low-RAM streaming converter", flush=True)
        _stage2_convert_streaming(src_dir, staging)
        return

    import torch
    from transformers import AutoModel, AutoProcessor

    print("[2/3] converting to fp16 (streamed, low_cpu_mem_usage)…", flush=True)
    model = AutoModel.from_pretrained(src_dir, dtype=torch.float16, low_cpu_mem_usage=True)
    proc  = AutoProcessor.from_pretrained(src_dir, use_fast=True)
    model.save_pretrained(str(staging), max_shard_size="1GB", safe_serialization=True)
    proc.save_pretrained(str(staging))
    del model, proc
    size = sum(f.stat().st_size for f in staging.glob("*.safetensors")) / 1e9
    print(f"      staged -> {staging}  ({size:.2f} GB fp16)", flush=True)


def _encode(staging: Path, tier: str, mode: str, items: list):
    """Run ONE encode_worker subprocess against the STAGING checkpoint."""
    import numpy as np
    fd, in_json = tempfile.mkstemp(suffix=".json"); os.close(fd)
    out_npy = in_json + ".npy"
    try:
        Path(in_json).write_text(json.dumps(list(items)), encoding="utf-8")
        env = dict(os.environ)
        env["SIGLIP_HF_DIR"]     = str(staging)
        env["SIGLIP_TIER"]       = tier
        env["SIGLIP_ENC_USE_OC"] = "0"       # forbid any open_clip fallback
        env["SIGLIP_ENC_BATCH"]  = "2"
        env["PYTHONIOENCODING"]  = "utf-8"
        env["HF_HUB_OFFLINE"]    = "1"
        r = subprocess.run([sys.executable, str(_WORKER), mode, in_json, out_npy],
                           cwd=str(_ROOT), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        for line in (r.stdout or "").splitlines():
            if any(t in line for t in ("loader", "peak_wset", "FATAL", "Traceback")):
                print(f"        | {line}", flush=True)
        if r.returncode != 0 or not os.path.exists(out_npy):
            print((r.stderr or "")[-1500:], flush=True)
            raise RuntimeError(f"{mode} encode failed (exit {r.returncode})")
        return np.load(out_npy)
    finally:
        for f in (in_json, out_npy):
            try: os.unlink(f)
            except Exception: pass


def _stage3_validate(staging: Path, tier: str, dim: int) -> bool:
    import numpy as np
    imgs = [str(p) for p in sorted((_ROOT / "dataset_images").glob("*.jpg"))[:8]]
    if not imgs:
        print("      no sample images in dataset_images/ — cannot validate", file=sys.stderr)
        return False

    print("[3/3] validating the staged checkpoint through the runtime worker…", flush=True)
    img = _encode(staging, tier, "images", imgs)
    txt = _encode(staging, tier, "text",
                  ["a street photograph", "a quiet empty interior",
                   "a portrait in harsh sunlight"])

    ok = True
    def chk(name, cond, extra=""):
        nonlocal ok
        print(f"      {'ok  ' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        ok = ok and cond

    chk("embedding width", img.shape[1] == dim, f"{img.shape[1]} (expected {dim})")
    chk("text width matches", txt.shape[1] == img.shape[1], f"{txt.shape[1]}")
    chk("all finite", bool(np.all(np.isfinite(img)) and np.all(np.isfinite(txt))))
    norms = np.linalg.norm(img, axis=1)
    chk("unit-norm", bool(np.allclose(norms, 1.0, atol=1e-2)),
        f"min {norms.min():.4f} max {norms.max():.4f}")
    if len(img) >= 2:
        gram = img @ img.T
        off  = gram[~np.eye(len(gram), dtype=bool)]
        chk("discriminative", float(off.max()) < 0.999, f"max off-diag {off.max():.4f}")
    # A real encoder separates unrelated captions; a broken one collapses them.
    tg = txt @ txt.T
    toff = tg[~np.eye(len(tg), dtype=bool)]
    chk("text prompts separable", float(toff.max()) < 0.999, f"max {toff.max():.4f}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(_TIERS), required=True)
    ap.add_argument("--promote", action="store_true",
                    help="move staging -> live if validation passes")
    args = ap.parse_args()

    repo, dim, staging_rel, live_rel = _TIERS[args.tier]
    staging, live = _ROOT / staging_rel, _ROOT / live_rel

    print("=" * 68)
    print(f"Lean fp16 checkpoint — tier '{args.tier}' ({dim}-d)  {repo}")
    print("=" * 68)

    if (live / "config.json").exists():
        print(f"{live} already populated — nothing to do.")
        return 0
    free = _free_disk_gb()
    if free < _MIN_FREE_DISK_GB:
        print(f"Only {free:.1f} GB free on disk; need ~{_MIN_FREE_DISK_GB:.0f} GB.",
              file=sys.stderr)
        return 1

    src = _stage1_download(repo)
    _stage2_convert(src, staging)

    if not _stage3_validate(staging, args.tier, dim):
        print(f"\nValidation FAILED — leaving {staging} in place and NOT promoting. "
              f"Nothing about the running app changed.", file=sys.stderr)
        return 1

    print("\nVALIDATION PASSED.")
    if not args.promote:
        print(f"Dry run (no --promote): staged at {staging}")
        print(f"To promote:  venv\\Scripts\\python.exe scripts/build_lean_checkpoint.py "
              f"--tier {args.tier} --promote")
        return 0

    if live.exists():
        shutil.rmtree(live)
    staging.rename(live)
    print(f"PROMOTED -> {live}")
    print("tier_select will now offer this tier on its lean RAM requirement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
