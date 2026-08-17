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
# This is a runtime worker, not a setup script — per the project's "no external
# network calls at runtime" rule, weights must already be cached locally
# (scripts/download_detectors.py-style one-time setup owns fetching). Without
# this, open_clip's create_model_and_transforms() still does an HF Hub
# existence/metadata check even when the local cache is already complete, and
# on a degraded connection that check can hang far longer than any request-level
# timeout (observed: 10+ minutes near-idle CPU/GPU, not a slow download).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_profile as _rp                                    # noqa: E402

# Tier-derived values come from run_profile, which is the ONLY place they are
# declared. This module and siglip2_encoder each used to carry their own copy of
# the checkpoint table, with a comment asking whoever edited one to remember the
# other. That is how Balanced ended up emitting 1024-d images and 1536-d text.
# SIGLIP_HF_DIR still overrides, so the setup script can validate a STAGING
# checkpoint through this exact runtime path before promoting it.
_PROFILE = _rp.current()
_TIER   = _PROFILE.tier
_HF_DIR = _PROFILE.hf_dir
_OC     = {t: (_rp.spec_for(t).model_tag, _rp.spec_for(t).oc_cache)
           for t in _rp.TIERS}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── ONNX image encoder (opt-in: FRAMEGRADE_ENCODER=onnx) ─────────────────────
# Profiling showed the encoder's RAM is dominated by the FRAMEWORK, not the
# model: torch 0.36 GB + transformers 1.60 GB, while the weights themselves
# mmap to VRAM for ~0.01 GB. onnxruntime-gpu imports for 0.03 GB and runs the
# same graph, so this path exists to drop ~2 GB of resident RAM per encode.
#
# IMAGES ONLY. Text encoding still uses the torch path: it needs the Gemma
# tokenizer, it runs once per prompt-set (then hits the disk probe cache), and
# it is not where the memory goes. Splitting on mode keeps the bulk image pass
# free of transformers entirely.
_ONNX_VISION = os.path.join(_ROOT, "models", "onnx", "vision.onnx")
_ONNX_TEXT   = os.path.join(_ROOT, "models", "onnx", "text.onnx")
_TOKENIZER   = os.path.join(_HF_DIR, "tokenizer.json")


def _onnx_text_enabled() -> bool:
    """Text encoding via ONNX needs the graph AND a standalone tokenizer.

    This was the last piece still on PyTorch, and it was the single largest
    peak in the whole system: 2.70 GB, versus 1.20 GB for the ONNX image pass —
    enough on its own to push a cull into the pagefile on a busy machine. The
    blocker was tokenization, which seemed to require transformers (+1.60 GB).
    It does not: the `tokenizers` package reads tokenizer.json directly for
    0.069 GB and produces IDENTICAL ids (verified across 60 real prompts —
    note pad_id=0, not 1).

    The exported graph is the 'high' tier's, so it must be gated on the tier
    exactly like the vision graph is. Without that gate a smaller tier produced
    1024-d image embeddings from its own checkpoint and 1536-d text embeddings
    from this graph, and every probe dot-product (embs @ probes.T) is then a
    shape mismatch — the Balanced tier could not work at all. Caught by the
    checkpoint validator's "text width matches" check.
    """
    return _PROFILE.onnx_enabled(text=True)


def encode_text_onnx(texts):
    """Text embeddings without torch or transformers."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(_TOKENIZER)
    tok.enable_truncation(64)
    tok.enable_padding(length=64, pad_id=0)
    ids = np.array([e.ids for e in tok.encode_batch(
        [_canonicalize(t) for t in texts])], dtype=np.int64)
    sess = _onnx_session(_ONNX_TEXT)
    out = []
    for i in range(0, len(ids), 64):
        e = sess.run(None, {"input_ids": ids[i:i + 64]})[0].astype(np.float32)
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _default_batch() -> int:
    """Encode batch, keyed on DEVICE — never on free RAM.

    Sizing the batch from available memory is what made two identical culls
    disagree on 47 of 514 photos: the batch composition changed between runs and
    kernel selection shifted the last bits of each embedding. Keying on the
    device keeps it deterministic — a given machine always gets the same value.

    CPU wants a bigger batch than GPU. MEASURED on the Fast tier, 16 photos,
    no GPU: batch 4 = 1.40 s/img, 8 = 1.11, 16 = 1.05, with peak RSS flat at
    2.83 GB throughout (the model dominates, not the activations). GPU stays at
    8, which is what the 1.20 GB ONNX peak was measured with.
    """
    return _PROFILE.encode_batch


def _gpu_present() -> bool:
    """Is there a GPU to run on? Cached — this is consulted per session."""
    return _PROFILE.gpu


def _onnx_enabled() -> bool:
    """ONNX is the DEFAULT for image encoding once the graph exists.

    Measured in a real grade: 2.70 GB -> 1.18 GB peak for the per-photo work,
    at equal speed. FRAMEGRADE_ENCODER=torch forces the PyTorch path back — the
    escape hatch matters because the two are not bit-identical (fp16 kernel
    differences put embedding cosine at ~0.9997, which moved 2 of 135 borderline
    grades), so any suspicion about a shoot can be A/B'd in one run.
    Only the 'high' tier has an exported graph; other tiers fall through to torch.
    """
    return _PROFILE.onnx_enabled()


def _onnx_session(graph: str = ""):
    """Create the CUDA session. torch ships the CUDA 12 DLLs onnxruntime needs
    but does not put them on the search path, so add them explicitly."""
    from pathlib import Path as _P
    _tl = _P(_ROOT) / "venv" / "Lib" / "site-packages" / "torch" / "lib"
    if _tl.is_dir():
        try:
            os.add_dll_directory(str(_tl))
            os.environ["PATH"] = str(_tl) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 4
    # Provider order is configurable so the same graph runs on other hardware:
    #   CUDA     NVIDIA (this machine)
    #   DML      any DirectX 12 GPU on Windows — AMD, Intel, Qualcomm/ARM64
    #   CoreML   Apple Silicon
    #   ROCm     AMD on Linux
    #   CPU      everywhere, always the last resort
    # Only providers actually present in the installed onnxruntime build are
    # used, so setting this on a machine without that wheel degrades to CPU
    # rather than failing. FRAMEGRADE_ORT_PROVIDERS overrides the order.
    _order = _PROFILE.ort_providers
    _have = ort.get_available_providers()
    prov = [p for p in _order if p in _have] or ["CPUExecutionProvider"]
    _g = graph or _ONNX_VISION
    sess = ort.InferenceSession(_g, so, providers=prov)
    print(f"[encode_worker] ONNX {os.path.basename(_g)} ({sess.get_providers()[0]})", flush=True)
    return sess


def _onnx_preprocess(img):
    """Reproduce SiglipImageProcessorFast EXACTLY.

    Verified to 1e-7 against the HF processor. The order matters: it resizes the
    UINT8 tensor and only then rescales/normalises. Resizing in float instead
    shifts pixels by up to 1/255, which alone moved embedding cosine from
    0.9997 to 0.9987 — enough to change borderline grades.
    Also note resample=2 is BILINEAR, not bicubic.
    """
    import numpy as _np
    import torchvision.transforms.v2.functional as _TF
    a = _np.asarray(img.convert("RGB")).copy()
    t = torch.from_numpy(a).permute(2, 0, 1)               # uint8 CHW
    t = _TF.resize(t, [384, 384],
                   interpolation=_TF.InterpolationMode.BILINEAR, antialias=True)
    return ((t.float() / 255.0 - 0.5) / 0.5).numpy()


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
    # Any tier may use the lean loader once its fp16 checkpoint exists (this was
    # hardcoded to "high", which meant the smaller tiers were stuck on the heavy
    # open_clip fp32 path and therefore cost MORE RAM than the giant did lean).
    if (not _force_oc
            and os.path.exists(os.path.join(_HF_DIR, "config.json"))):
        from transformers import AutoModel, AutoProcessor
        m = AutoModel.from_pretrained(
            _HF_DIR, dtype=torch.float16 if dev == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(dev).eval()
        proc = AutoProcessor.from_pretrained(_HF_DIR, use_fast=True)
        # Report the tier's ACTUAL checkpoint size. This was hardcoded to the
        # giant's 3.5 GB, so a Balanced or Fast run logged a number nearly 5x
        # its real footprint — exactly the sort of misleading output that sends
        # a later RAM investigation the wrong way.
        try:
            from pathlib import Path as _PathSz
            _sz = sum(f.stat().st_size
                      for f in _PathSz(_HF_DIR).glob("*.safetensors")) / 1e9
            _sz_txt = f"{_sz:.1f} GB"
        except Exception:
            _sz_txt = "unknown size"
        _dt_txt = "fp16" if dev == "cuda" else "fp32"   # CPU runs fp32; fp16 is emulated
        print(f"[encode_worker] HF loader ({_TIER}, {dev}) — lean {_sz_txt} {_dt_txt}",
              flush=True)
        return "hf", m, proc
    import open_clip
    tag, cd = _OC.get(_TIER, _OC["high"])
    # Heavy fallback. MEASURED peak 10.3 GB: open_clip builds the giant model in
    # fp32 and loads a 6.97 GB fp32 .bin before casting to fp16. Say so loudly,
    # with the actual free RAM and the exact command that fixes it — otherwise
    # this silently thrashes the pagefile and looks like a hang mid-grade.
    try:
        import psutil
        _free = psutil.virtual_memory().available / 1e9
        _warn = f"free RAM {_free:.1f} GB vs ~10.3 GB needed" if _free < 10.3 else \
                f"free RAM {_free:.1f} GB"
    except Exception:
        _warn = "free RAM unknown"
    print(f"[encode_worker] open_clip fallback ({_TIER}, {dev}) — heavy path, {_warn}. "
          f"Fix: python scripts/setup_siglip2_hf.py  (builds the ~3.5 GB fp16 checkpoint)",
          flush=True)
    m, _, prep = open_clip.create_model_and_transforms(
        tag, pretrained="webli", precision="fp16", cache_dir=cd)
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer(tag)
    return "oc", m, (prep, tok)


def _print_peak() -> None:
    """Report this subprocess's peak working set — the parent cannot see it.

    This is the number that shows which loader ran: the open_clip fallback
    reads a 6.97 GB fp32 checkpoint into RAM (~8 GB peak) while the HF fp16
    path peaks near ~3.5 GB. Never allowed to fail the encode."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", mi.rss)
        print(f"[encode_worker] peak_wset={peak / 1e9:.2f} GB", flush=True)
    except Exception:
        pass


def _norm(e):
    return e / (e.norm(dim=-1, keepdim=True) + 1e-9)


# ── SigLIP text canonicalization ─────────────────────────────────────────────
# SigLIP/SigLIP-2 were trained on CANONICALIZED captions (big_vision's prompt
# engineering: lowercase, punctuation stripped). open_clip does this for us —
# HFTokenizer.__call__ runs `canonicalize_text(basic_clean(text))` before it
# tokenizes — but the HF loader below calls the raw tokenizer, which does not.
#
# The result was a silent quality bug on the HF path only: any prompt with a
# capital letter or a comma tokenized differently and landed somewhere else in
# embedding space. Measured against the open_clip vectors over this project's
# 430 probe prompts: the 335 already-canonical prompts matched at >= 0.99, while
# the 82 with punctuation/capitals fell as low as 0.365 ("Fan Ho Hong Kong …"
# tokenized as proper nouns instead of `fan ho hong kong`). Images were never
# affected. Mirrors open_clip/tokenizer.py basic_clean + canonicalize_text.
_PUNCT_TABLE = str.maketrans("", "", __import__("string").punctuation)


def _canonicalize(text: str) -> str:
    import html as _html
    try:
        import ftfy as _ftfy
        text = _ftfy.fix_text(text)
    except Exception:
        pass                                   # ftfy absent: rest still applies
    text = _html.unescape(_html.unescape(text)).strip()
    text = text.replace("_", " ")
    text = text.translate(_PUNCT_TABLE)
    text = text.lower()
    return " ".join(text.split()).strip()


def encode_images(kind, m, helper, paths, batch=8):
    from PIL import Image
    from raw_support import RAW_EXTS, extract_embedded_preview
    dev = _device(); dt = next(m.parameters()).dtype
    # Results are written straight into one preallocated (N, D) array instead of
    # being collected per batch and np.concatenate'd at the end. The old form
    # held the full list of per-batch arrays AND the concatenated copy at the
    # same instant — a transient double of the whole result set on top of the
    # resident model, right at the end of a bulk encode.
    embs: "np.ndarray | None" = None
    written = 0
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
        _b = e.cpu().float().numpy()
        if embs is None:
            embs = np.zeros((len(paths), _b.shape[1]), dtype=np.float32)
        embs[written:written + len(_b)] = _b
        written += len(_b)
        del _b
    if embs is None:                       # no batches ran (empty input)
        embs = np.zeros((0, 0), dtype=np.float32)
    for idx in failed:
        embs[idx] = 0.0   # zero-vector sentinel → grade_pipeline_v2 drops these rows
    if failed:
        print(f"[encode_worker] {len(failed)}/{len(paths)} unreadable → zero-row sentinel", flush=True)
    return embs


def encode_images_onnx(sess, paths, batch=8):
    """Same contract as encode_images: normalised (N, D) float32, zero-row for
    unreadable files so grade_pipeline_v2 drops them."""
    from PIL import Image
    from raw_support import RAW_EXTS, extract_embedded_preview
    out = None
    written = 0
    failed = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        arrs = []
        for j, p in enumerate(chunk):
            img = None
            try:
                if os.path.splitext(p)[1].lower() in RAW_EXTS:
                    img = extract_embedded_preview(p, "RGB")
                else:
                    im = Image.open(p)
                    try: im.draft("RGB", (512, 512))
                    except Exception: pass
                    img = im.convert("RGB")
            except Exception:
                img = None
            if img is None:
                failed.append(i + j)
                print(f"[encode_worker] read error, skipping: {p}", flush=True)
                img = Image.new("RGB", (64, 64), (0, 0, 0))
            arrs.append(_onnx_preprocess(img))
        x = np.stack(arrs).astype(np.float16)
        e = sess.run(None, {"pixel_values": x})[0].astype(np.float32)
        e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
        if out is None:
            out = np.zeros((len(paths), e.shape[1]), dtype=np.float32)
        out[written:written + len(e)] = e
        written += len(e)
    if out is None:
        out = np.zeros((0, 0), dtype=np.float32)
    for idx in failed:
        out[idx] = 0.0
    if failed:
        print(f"[encode_worker] {len(failed)}/{len(paths)} unreadable -> zero-row sentinel", flush=True)
    return out


def encode_text(kind, m, helper, texts):
    dev = _device()
    with torch.no_grad():
        if kind == "hf":
            # Canonicalize FIRST — the open_clip path does this inside its
            # tokenizer, so without it the two loaders are not the same
            # function of the same text (see _canonicalize).
            inp = helper(text=[_canonicalize(t) for t in texts],
                         padding="max_length", max_length=64,
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

        # ONNX fast path — images only, and only when explicitly enabled.
        if mode == "images" and _onnx_enabled():
            _batch = max(1, int(os.environ.get("SIGLIP_ENC_BATCH",
                                               str(_default_batch()))))
            embs = encode_images_onnx(_onnx_session(), items, batch=_batch)
            np.save(out_npy, embs.astype(np.float32))
            print(f"[encode_worker] images(onnx): {embs.shape} -> {out_npy}", flush=True)
            _print_peak()
            os._exit(0)

        if mode == "text" and _onnx_text_enabled():
            embs = encode_text_onnx(items)
            np.save(out_npy, embs.astype(np.float32))
            print(f"[encode_worker] text(onnx): {embs.shape} -> {out_npy}", flush=True)
            _print_peak()
            os._exit(0)

        kind, m, helper = _load()
        _batch = max(1, int(os.environ.get("SIGLIP_ENC_BATCH",
                                           str(_default_batch()))))
        if mode == "images":
            print(f"[encode_worker] encode batch={_batch}", flush=True)
            embs = encode_images(kind, m, helper, items, batch=_batch)
        else:
            embs = encode_text(kind, m, helper, items)
        np.save(out_npy, embs.astype(np.float32))
        print(f"[encode_worker] {mode}: {embs.shape} -> {out_npy}", flush=True)
        _print_peak()
        os._exit(0)
    except Exception:
        print(f"[encode_worker] FATAL:\n{_tb.format_exc()}", flush=True)
        _print_peak()
        os._exit(1)


if __name__ == "__main__":
    main()
