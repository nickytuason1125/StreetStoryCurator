"""Isolated embedding subprocess.

Loads SigLIP in a CLEAN process (NOT the multiprocessing grade-worker, where the
efficient HF/accelerate loader native-crashes), encodes images/text, writes the
result to a .npy, and exits — freeing all model RAM. The grade-worker calls this
via subprocess.Popen and reads the output, so it never loads the model itself.

Usage:
    python encode_worker.py images <paths_json> <out_npy>
    python encode_worker.py text   <texts_json> <out_npy>

SIGLIP_TIER selects the model. For "high" it uses the efficient HF FP16 loader
(models/siglip2_hf_fp16, ~3.8 GB, 1536-d, ~0.99 cosine vs the open_clip ViT-g);
mid/low use the smaller open_clip ViT-L / ViT-B.
"""
import sys, os, json
import numpy as np
import torch

_TIER   = os.environ.get("SIGLIP_TIER", "high").strip().lower()
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HF_DIR = os.path.join(_ROOT, "models", "siglip2_hf_fp16")
_OC = {  # tier -> (open_clip tag, cache dir)
    "high": ("ViT-gopt-16-SigLIP2-384", "models/siglip2"),
    "mid":  ("ViT-L-16-SigLIP2-384",    "models/siglip2_L"),
    "low":  ("ViT-B-16-SigLIP2-384",    "models/siglip2_B"),
}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    """Return (kind, model, helper). kind 'hf' or 'oc'.

    DEFAULT = HF fp16 loader. CRITICAL for a 16 GB machine: the HF checkpoint is
    3.49 GB fp16 and loads directly to fp16 with low_cpu_mem_usage=True (~3.5 GB
    peak). The open_clip checkpoint is 6.97 GB fp32 — open_clip loads it fully
    into CPU RAM before converting to fp16 + moving to GPU, spiking to ~8 GB.
    That spike (not the loader's native stack) is what exhausts RAM and kills the
    grade worker with 0xC0000005 on the encode. So HF is the LEAN, safe path.
    Set SIGLIP_ENC_USE_OC=1 to force open_clip (only if the HF checkpoint is
    missing or for debugging)."""
    dev = _device()
    _force_oc = os.environ.get("SIGLIP_ENC_USE_OC", "0").strip() == "1"
    if (not _force_oc and _TIER == "high"
            and os.path.exists(os.path.join(_HF_DIR, "config.json"))):
        from transformers import AutoModel, AutoProcessor
        m = AutoModel.from_pretrained(
            _HF_DIR, dtype=torch.float16 if dev == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(dev).eval()
        proc = AutoProcessor.from_pretrained(_HF_DIR, use_fast=True)
        print(f"[encode_worker] HF loader ({_TIER}, {dev}) — lean 3.5 GB fp16", flush=True)
        return "hf", m, proc
    import open_clip
    tag, cd = _OC.get(_TIER, _OC["high"])
    m, _, prep = open_clip.create_model_and_transforms(
        tag, pretrained="webli", precision="fp16", cache_dir=cd)
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer(tag)
    print(f"[encode_worker] open_clip loader ({_TIER}, {dev}) — WARN 8 GB fp32 spike", flush=True)
    return "oc", m, (prep, tok)


def _norm(e):
    return e / (e.norm(dim=-1, keepdim=True) + 1e-9)


def encode_images(kind, m, helper, paths, batch=8):
    from PIL import Image
    from raw_support import RAW_EXTS, extract_embedded_preview
    dev = _device(); dt = next(m.parameters()).dtype
    out = []
    failed = []   # global indices whose pixels could not be read
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        pil = []
        for j, p in enumerate(chunk):
            img = None
            try:
                if os.path.splitext(p)[1].lower() in RAW_EXTS:
                    # RAW: embedded JPEG preview only — never demosaic (memory-safe).
                    img = extract_embedded_preview(p, "RGB")
                else:
                    im = Image.open(p)
                    try: im.draft("RGB", (512, 512))
                    except Exception: pass
                    img = im.convert("RGB")
            except Exception:
                img = None
            if img is None:
                # Unreadable / no embedded preview → mark for drop. A tiny black
                # filler keeps the batch shape; its row is zeroed after encoding so
                # the pipeline removes the file entirely (no gray-placeholder poison).
                failed.append(i + j)
                print(f"[encode_worker] read error, skipping: {p}", flush=True)
                pil.append(Image.new("RGB", (64, 64), (0, 0, 0)))
            else:
                pil.append(img)
        with torch.no_grad():
            if kind == "hf":
                pv = helper(images=pil, return_tensors="pt")["pixel_values"].to(dev, dt)
                e = _norm(m.get_image_features(pixel_values=pv))
            else:
                prep, _ = helper
                t = torch.stack([prep(x) for x in pil]).to(dev, dt)
                e = _norm(m.encode_image(t))
        out.append(e.cpu().float().numpy())
    embs = np.concatenate(out, axis=0)
    for idx in failed:
        embs[idx] = 0.0   # zero-vector sentinel → grade_pipeline_v2 drops these rows
    if failed:
        print(f"[encode_worker] {len(failed)}/{len(paths)} unreadable → zero-row sentinel", flush=True)
    return embs


def encode_text(kind, m, helper, texts):
    dev = _device()
    with torch.no_grad():
        if kind == "hf":
            inp = helper(text=list(texts), padding="max_length", max_length=64,
                         truncation=True, return_tensors="pt").to(dev)
            e = _norm(m.get_text_features(**inp))
        else:
            _, tok = helper
            e = _norm(m.encode_text(tok(list(texts)).to(dev)))
    return e.cpu().float().numpy()


def main():
    # CRITICAL: ALL exit paths use os._exit (not normal Python shutdown).
    # PyTorch's atexit handler calls cuCtxDestroy/cudaDeviceReset on exit, which
    # triggers NVIDIA driver callbacks in the parent grade-worker process (which
    # has nvcuda.dll loaded from `import torch`). Those callbacks cause an
    # ACCESS_VIOLATION (exit code 0xC0000005) that kills the grade worker.
    # os._exit bypasses atexit entirely. The OS kernel driver cleans up the CUDA
    # context through a different (safe) code path. Any data already written to
    # disk (.npy / crash.log) is preserved because os._exit does not affect the
    # filesystem — it only skips Python-level finalizers and atexit functions.
    import traceback as _tb
    try:
        mode, in_json, out_npy = sys.argv[1], sys.argv[2], sys.argv[3]
        items = json.load(open(in_json, encoding="utf-8"))
        kind, m, helper = _load()
        _batch = max(1, int(os.environ.get("SIGLIP_ENC_BATCH", "8")))
        if mode == "images":
            print(f"[encode_worker] encode batch={_batch}", flush=True)
            embs = encode_images(kind, m, helper, items, batch=_batch)
        else:
            embs = encode_text(kind, m, helper, items)
        np.save(out_npy, embs.astype(np.float32))
        print(f"[encode_worker] {mode}: {embs.shape} -> {out_npy}", flush=True)
        os._exit(0)
    except Exception:
        print(f"[encode_worker] FATAL:\n{_tb.format_exc()}", flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
