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
    # Apple Silicon: MPS is opt-in (FIRSTCUT_TORCH_DEVICE=mps) until verified
    # on real hardware — a wrong device answer here silently changes every
    # embedding, and the deterministic-CPU fallback is always correct.
    forced = os.environ.get("FIRSTCUT_TORCH_DEVICE", "")
    if forced and forced != "auto":
        return forced
    # FIRSTCUT_FORCE_CPU (2026-09-15): a crashing NVIDIA driver DLL
    # (nvobjectloader64.dll, 0xC0000005 at model load) killed every worker
    # spawn mid-cull. The ONNX path honors this switch directly; the torch
    # path (this line) must too, because torch.cuda.is_available() itself can
    # survive while the CUDA session dies later in native code.
    if os.environ.get("FIRSTCUT_FORCE_CPU", "").strip() == "1":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── ONNX image encoder (opt-in: FIRSTCUT_ENCODER=onnx) ─────────────────────
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
    at equal speed. FIRSTCUT_ENCODER=torch forces the PyTorch path back — the
    escape hatch matters because the two are not bit-identical (fp16 kernel
    differences put embedding cosine at ~0.9997, which moved 2 of 135 borderline
    grades), so any suspicion about a shoot can be A/B'd in one run.
    Only the 'high' tier has an exported graph; other tiers fall through to torch.
    """
    return _PROFILE.onnx_enabled()


def _free_vram_gb():
    """Gigabytes of free VRAM on device 0, or None when it cannot be measured
    (no nvidia-smi, no NVIDIA GPU, timeout). Measured, not assumed — the
    2026-09-07 warm-up died on 'CUDA failure 2: out of memory' during a
    transient VRAM squeeze even though the card was idle minutes later."""
    try:
        import subprocess as _sp
        _flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
        _out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            creationflags=_flags, text=True, timeout=5,
        )
        return float(_out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


def _onnx_session(graph: str = ""):
    """Create the CUDA session. torch ships the CUDA 12 DLLs onnxruntime needs
    but does not put them on the search path, so add them explicitly."""
    from pathlib import Path as _P
    _tl = _P(_ROOT) / "venv" / "Lib" / "site-packages" / "torch" / "lib"
    # ── cuDNN version preference (2026-09-16) ─────────────────────────────────
    # torch 2.5.1+cu121 bundles cuDNN 9.1 and its lib dir is what ORT ends up
    # loading. onnxruntime-gpu 1.22 is built against a newer cuDNN 9.x, and the
    # old 9.1 DLLs under concurrent allocation pressure reproduce as native
    # 0xC0000005 deaths in the encode worker (crash.log streaks on 2026-09-15).
    # The nvidia-cudnn-cu12 wheel carries a current cuDNN 9 — prefer its bin dir
    # BEFORE torch/lib so ORT resolves cudnn64_9.dll from the wheel. Missing
    # wheel → torch's 9.1 stays, exactly as before.
    _cudnn = _P(_ROOT) / "venv" / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"
    if _cudnn.is_dir():
        try:
            os.add_dll_directory(str(_cudnn))
            os.environ["PATH"] = str(_cudnn) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
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
    # rather than failing. FIRSTCUT_ORT_PROVIDERS overrides the order.
    _order = _PROFILE.ort_providers
    _have = ort.get_available_providers()
    prov = [p for p in _order if p in _have] or ["CPUExecutionProvider"]
    # ── Hard CPU kill-switch (read via os.environ directly) ──────────────────
    # FIRSTCUT_ORT_PROVIDERS goes through setting()/profile, which spawned
    # workers may not see; this one is read straight from the process env so a
    # wedged GPU driver (cudaSetDevice failing with 0 MiB used — observed)
    # can be bypassed with certainty. Set FIRSTCUT_FORCE_CPU=1 in the runner's
    # environment to force pure-CPU ONNX regardless of provider order.
    if os.environ.get("FIRSTCUT_FORCE_CPU", "").strip() == "1":
        prov = ["CPUExecutionProvider"]
        print("[encode_worker] FIRSTCUT_FORCE_CPU=1 — CPU provider forced", flush=True)
    # VRAM gate (2026-09-07): a CUDA session created during a transient VRAM
    # squeeze dies with "CUDA failure 2: out of memory / bad allocation" — four
    # FATALs + respawn churn for one warm attempt. If free VRAM cannot hold the
    # session, skip CUDA DELIBERATELY and use the next provider (DML/CPU).
    if any("CUDA" in p for p in prov):
        _vram = _free_vram_gb()
        # -- Smart VRAM gate (2026-09-13) -----------------------------------
        # The old gate silently fell back to DML/CPU below 1.5 GB. Two
        # problems: the high-tier CPU path is measured unusable (6.5-7.6 GB,
        # 11-14 s/img), and a silent provider switch drifts embeddings (fp16
        # kernel cosine ~0.9997) - borderline grades can flip. A VRAM squeeze
        # is usually TRANSIENT (2026-09-07: warm-up died on 'CUDA failure 2'
        # with the card idle minutes later), so wait for recovery first; only
        # if the squeeze outlasts the wait, refuse with exit 5 so the parent
        # reports a readable VRAM message instead of a native 0xC0000005
        # from a mid-graph allocation failure.
        if _vram is not None and _vram < 1.5 and _gpu_present():
            import time as _t
            _vram_deadline = _t.time() + 60.0
            while _t.time() < _vram_deadline:
                print(f"[encode_worker] VRAM squeeze ({_vram:.2f} GB free) - waiting for recovery", flush=True)
                _t.sleep(3.0)
                _vram = _free_vram_gb()
                if _vram is None or _vram >= 1.5:
                    break
            if _vram is not None and _vram < 1.5:
                print(f"[encode_worker] VRAM refusal: only {_vram:.2f} GB free for the CUDA "
                      f"session (need ~1.5). The card is shared with the desktop, WebView2 "
                      f"and the backend's warm CLIP - close one of them and retry.", flush=True)
                os._exit(5)
    _g = graph or _ONNX_VISION
    # ── VRAM arena cap ────────────────────────────────────────────────────────
    # ORT's CUDA arena defaults to grow-as-needed with big extend steps; on a
    # 6 GB card shared with the desktop, WebView2, and the backend's warm CLIP,
    # the arena's reservation overcommits and session init dies with
    # "bad allocation" even when free VRAM nominally covers the weights.
    # kSameAsRequested grows in exact increments, and gpu_mem_limit bounds the
    # arena so an over-reservation is impossible; overflow falls to the next
    # provider instead of killing the worker. (Passed via the InferenceSession
    # provider_options= kwarg — SessionOptions has no provider_options attr.)
    _prov_opts = [
        {"device_id": 0,
         "arena_extend_strategy": "kSameAsRequested",
         "gpu_mem_limit": int(4.5 * 1024 * 1024 * 1024),
         "cudnn_conv_algo_search": "HEURISTIC"}
        if "CUDA" in p else {}
        for p in prov
    ]
    sess = ort.InferenceSession(_g, so, providers=prov, provider_options=_prov_opts)
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


_DECODE_LAST_ERR = None   # last decode exception message (see _decode_last_err)


def _decode_with_timeout(p, timeout_s=30):
    """Decode one image with a hang guard. Corrupt RAWs can spin forever in
    native preview-extraction code — a try/except never fires because the
    thread never returns. Run the decode in a helper thread and abandon it on
    timeout. Returns the image or None (None → existing failed/zero-row path).
    The abandoned daemon thread dies with the worker process; one leaked spin
    per poison file is acceptable vs. freezing the whole cull."""
    import threading
    global _DECODE_LAST_ERR
    # RAW_EXTS / extract_embedded_preview live only as function-locals in the
    # encode paths — resolve them here or _work NameErrors on EVERY file and
    # the whole encode comes back "unreadable" (all 2756 once, silently).
    from raw_support import RAW_EXTS as _RAW_EXTS, extract_embedded_preview as _extract_preview
    from PIL import Image as _Image
    result = {}

    def _work():
        try:
            if os.path.splitext(p)[1].lower() in _RAW_EXTS:
                result["img"] = _extract_preview(p, "RGB")
            else:
                im = _Image.open(p)
                try: im.draft("RGB", (512, 512))
                except Exception: pass
                result["img"] = im.convert("RGB")
        except Exception as exc:
            result["img"] = None
            # Keep the reason — swallowed errors here produced the silent
            # "2756/2756 unreadable" all-zero encode with no diagnosis.
            result["err"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        print(f"[encode_worker] DECODE TIMEOUT after {timeout_s}s (poison file, skipped): {p}", flush=True)
        return None
    if result.get("img") is None and result.get("err"):
        global _DECODE_LAST_ERR
        _DECODE_LAST_ERR = result["err"]
    return result.get("img")


def _decode_last_err() -> "str | None":
    """Reason from the most recent failed _decode_with_timeout call (the
    per-file exception is otherwise swallowed and every file just "fails"
    with no diagnosis — happened as a real all-2756-unreadable encode)."""
    return _DECODE_LAST_ERR


def _iter_decoded_batches(paths, batch):
    """Yield (start_index, chunk_paths, pil_images, failed_local, first_err).

    PREFETCH PIPELINE (2026-09-11): the GPU used to sit idle while the NEXT
    batch's JPEGs decoded on the CPU — decode and encode ran strictly
    sequentially. A bounded producer thread decodes one batch AHEAD while the
    consumer feeds the current one to the model. Bounded at 2 queued batches
    (~64 draft-downscaled 512px images, ≈130 MB worst case), so prefetch
    adds speed without adding meaningful RAM — important on the 16 GB
    machines this pipeline targets. Reused by BOTH the torch and ONNX encode
    loops so their bookkeeping (failed-index sentinels, first-error capture)
    stays identical.
    """
    from PIL import Image
    import queue as _q
    q: "_q.Queue" = _q.Queue(maxsize=2)
    _SENT = object()

    def _producer():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            pil, failed_local, first_err = [], [], None
            for j, p in enumerate(chunk):
                img = _decode_with_timeout(p)
                if img is None:
                    failed_local.append(j)
                    if first_err is None:
                        first_err = _decode_last_err()
                    img = Image.new("RGB", (64, 64), (0, 0, 0))
                pil.append(img)
            q.put((i, chunk, pil, failed_local, first_err))
        q.put(_SENT)

    import threading as _th
    _th.Thread(target=_producer, daemon=True, name="decode-prefetch").start()
    while True:
        item = q.get()
        if item is _SENT:
            break
        yield item


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
    first_err = None   # first decode exception message, for diagnosis
    for i, chunk, pil, failed_local, batch_first_err in _iter_decoded_batches(paths, batch):
        for j in failed_local:
            # Unreadable / no embedded preview / decode hang → mark for drop.
            # A tiny black filler keeps the batch shape; its row is zeroed
            # after encoding so the pipeline removes the file entirely.
            failed.append(i + j)
            if first_err is None:
                first_err = batch_first_err
                print(f"[encode_worker] read error ({first_err or 'unknown'}), skipping: {chunk[j]}", flush=True)
            elif (i + j) % 500 == 0:
                print(f"[encode_worker] read error #{i+j+1}: {chunk[j]}", flush=True)
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
    if len(failed) == len(paths) and paths:
        # EVERY file failed to decode — that is never a normal cull outcome, it
        # means the card dropped out / a DLL failed under commit pressure /
        # the wrong mount. Returning "success" with all-zero rows silently
        # grades nothing and the pipeline reports an empty cull. Exit non-zero
        # so the parent's retry ladder (gc + backoff + fresh process) runs.
        # The failed paths ride in the message so the parent can tell a
        # DETERMINISTIC per-file failure (corrupt/truncated bytes — respawning
        # can never fix those) from a transient machine state, and name the
        # files to the user. main() exits 3 for this class.
        _fail_list = ", ".join(paths[:8]) + ("…" if len(paths) > 8 else "")
        raise RuntimeError(
            f"all {len(paths)} files unreadable (first reason: {first_err or 'unknown'}) "
            f"— failing this attempt so the caller retries [failed: {_fail_list}]")
    return embs


def encode_images_onnx(sess, paths, batch=8):
    """Same contract as encode_images: normalised (N, D) float32, zero-row for
    unreadable files so grade_pipeline_v2 drops them."""
    from PIL import Image
    from raw_support import RAW_EXTS, extract_embedded_preview
    out = None
    written = 0
    failed = []
    first_err = None
    for i, chunk, pil, failed_local, batch_first_err in _iter_decoded_batches(paths, batch):
        for j in failed_local:
            failed.append(i + j)
            if first_err is None:
                first_err = batch_first_err
                print(f"[encode_worker] read error ({first_err or 'unknown'}), skipping: {chunk[j]}", flush=True)
            elif (i + j) % 500 == 0:
                print(f"[encode_worker] read error #{i+j+1}: {chunk[j]}", flush=True)
        arrs = []
        for img in pil:
            arrs.append(_onnx_preprocess(img))
        x = np.stack(arrs).astype(np.float16)
        try:
            e = sess.run(None, {"pixel_values": x})[0].astype(np.float32)
        except Exception as _oom:
            # -- Mid-encode OOM shrink (2026-09-13) -------------------------
            # A VRAM squeeze that outlasts the pre-flight gate can land HERE
            # as a Python-level CUDA/bad_alloc failure. Catch the catchable
            # form and retry the same chunk ONE image at a time: a batch of 1
            # needs almost no activation memory, and a ViT has no cross-image
            # interaction, so the result is bit-identical to the batched run.
            if not ("memory" in str(_oom).lower() or "alloc" in str(_oom).lower()
                    or "cuda" in str(_oom).lower()):
                raise
            print(f"[encode_worker] OOM on batch of {len(x)} - retrying one image "
                  f"at a time ({str(_oom)[:120]})", flush=True)
            e = None
            for _sub in arrs:
                _e = sess.run(None, {"pixel_values": _sub[None].astype(np.float16)})[0].astype(np.float32)
                _e = _e / (np.linalg.norm(_e, axis=1, keepdims=True) + 1e-9)
                e = _e if e is None else np.concatenate([e, _e], axis=0)
            e = np.asarray(e, dtype=np.float32)
        if e is not None:
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
    if len(failed) == len(paths) and paths:
        # See encode_images: 100% unreadable is a machine/card state, not a
        # cull outcome — fail the attempt so the parent's ladder retries.
        _fail_list = ", ".join(paths[:8]) + ("…" if len(paths) > 8 else "")
        raise RuntimeError(
            f"all {len(paths)} files unreadable (first reason: {first_err or 'unknown'}) "
            f"— failing this attempt so the caller retries [failed: {_fail_list}]")
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


def _serve_singleton_gate(idle_s: float) -> None:
    """Machine-wide single-instance gate for serve mode (2026-09-10).

    Why: multiple owners can prewarm an encoder — the backend server
    (/api/encoder/warm), the grade_runner subprocess, and (observed live)
    a window process whose warm request landed in its own stack. Each
    spawned its own `encode_worker serve`, and two live workers sat on
    ~870 MB while the user saw "2.0 GB free with nothing open". The
    per-process _WARM global cannot see another process's worker.

    Contract: cache/encoder_warm.lock holds {pid, heartbeat_epoch}. A
    serve() instance that finds a holder whose pid is ALIVE and whose
    heartbeat is FRESH exits immediately — one warm worker machine-wide,
    whatever races at startup. A dead pid, or a heartbeat older than 5
    minutes (a live worker heartbeats every 30 s even mid-encode, so a
    stale beat means the process is hung or already dying), is a takeover:
    the old holder is killed if reachable, and this instance takes the
    lock. Never fatal: any gate error just means "proceed" — the idle
    timeout still caps the damage of a duplicate.
    """
    import json as _json
    import threading as _threading
    import time as _time

    try:
        from pathlib import Path as _Path
        _lock = _Path(os.environ.get("FIRSTCUT_DATA_DIR", "")) if os.environ.get("FIRSTCUT_DATA_DIR") else None
        if _lock is None:
            _lock = _Path(__file__).resolve().parent.parent / "cache"
        _lock.mkdir(parents=True, exist_ok=True)
        _lock = _lock / "encoder_warm.lock"
        _stale_after = 300.0
        _now = _time.time()

        def _read_lock():
            try:
                _d = _json.loads(_lock.read_text(encoding="utf-8"))
                return int(_d.get("pid", 0)), float(_d.get("heartbeat", 0))
            except Exception:
                return 0, 0.0

        _pid, _beat = _read_lock()
        if _pid and _pid != os.getpid():
            _alive = False
            try:
                import psutil as _ps
                _name = (_ps.Process(_pid).name() or "").lower()
                _alive = "python" in _name
            except Exception:
                _alive = False
            if _alive and (_now - _heartbeat_of(_lock)) < _stale_after:
                # A healthy warm worker already serves the machine.
                print(f"[encode_worker] warm encoder already active pid={_pid} — exiting", flush=True)
                os._exit(0)
            if _alive:
                # Stale heartbeat: hung or wedged holder — take over.
                try:
                    import psutil as _ps
                    _ps.Process(_pid).kill()
                    print(f"[encode_worker] took over stale warm encoder pid={_pid}", flush=True)
                except Exception:
                    pass

        def _heartbeat_loop():
            while True:
                try:
                    _lock.write_text(_json.dumps(
                        {"pid": os.getpid(), "heartbeat": _time.time()}), encoding="utf-8")
                except Exception:
                    pass
                _time.sleep(30.0)

        _lock.write_text(_json.dumps(
            {"pid": os.getpid(), "heartbeat": _time.time()}), encoding="utf-8")
        _threading.Thread(target=_heartbeat_loop, daemon=True, name="warm-lock-heartbeat").start()
    except Exception:
        pass   # the gate is a guard, not a dependency — never block serve()


def _heartbeat_of(_lock) -> float:
    import json as _json
    try:
        return float(_json.loads(_lock.read_text(encoding="utf-8")).get("heartbeat", 0))
    except Exception:
        return 0.0


def serve():
    """Persistent warm-encoder loop (FIRSTCUT_WARM_ENCODER, default on).

    The one-shot main() above re-imports torch+transformers+sklearn and
    reloads the model on EVERY cull — ~500 MB of imports + a ~1.6 GB model
    load placed at the most memory-hostile moment of the whole pipeline. On
    the 2026-09-07 commit-starved machine that import itself OOM'd ("The
    paging file is too small" / sklearn MemoryError at 0.47 GB RSS) and
    killed grades repeatedly, surviving even 4 retry attempts.

    serve() pays that cost ONCE, whenever this process starts (app boot is
    the freshest the machine ever gets), then loops:

        stdin : one JSON line per job — {"mode", "in", "out", "resp"}
        stdout: "[encode_worker] ..." progress (inherited crash.log)
        resp  : {"ok": true} | {"ok": false, "error": ...} written to the
                path in job["resp"] (file-based, so the parent never needs
                to read our stdout and no pipe can deadlock)

    Exits on stdin EOF (parent died) or after FIRSTCUT_WARM_IDLE_S (default
    600 s) idle, so RAM/VRAM free up between sessions. os._exit everywhere:
    see main()'s CUDA-atexit note.

    A machine-wide singleton gate runs BEFORE anything loads: if another
    healthy warm worker is already serving, this instance exits immediately
    (see _serve_singleton_gate) — one warm worker per machine, no matter how
    many owners race a prewarm.
    """
    import json as _json
    import queue as _queue
    import threading as _threading
    import traceback as _tb

    idle_s = float(os.environ.get("FIRSTCUT_WARM_IDLE_S", "600") or 600)
    _serve_singleton_gate(idle_s)
    lines: "_queue.Queue" = _queue.Queue()

    def _reader():
        for raw in sys.stdin:
            lines.put(raw)
        lines.put(None)          # EOF — parent closed the pipe

    _threading.Thread(target=_reader, daemon=True, name="stdin-reader").start()

    state = {"kind": None, "m": None, "helper": None, "onnx_sess": None}

    def _ensure_loaded():
        if state["m"] is None:
            state["kind"], state["m"], state["helper"] = _load()
            print("[encode_worker] warm load complete", flush=True)

    def _run_job(job) -> None:
        resp_path = job.get("resp") or ""

        def _respond(ok, error=""):
            if resp_path:
                try:
                    with open(resp_path, "w", encoding="utf-8") as f:
                        _json.dump({"ok": bool(ok), "error": error[-800:]}, f)
                except Exception:
                    pass

        try:
            mode = job["mode"]
            in_json, out_npy = job["in"], job["out"]
            items = _json.load(open(in_json, encoding="utf-8"))
            if mode == "images" and _onnx_enabled():
                if state["onnx_sess"] is None:
                    state["onnx_sess"] = _onnx_session()
                embs = encode_images_onnx(state["onnx_sess"], items,
                                          batch=max(1, int(os.environ.get(
                                              "SIGLIP_ENC_BATCH", str(_default_batch())))))
            else:
                _ensure_loaded()
                _batch = max(1, int(os.environ.get("SIGLIP_ENC_BATCH",
                                                   str(_default_batch()))))
                if mode == "images":
                    embs = encode_images(state["kind"], state["m"],
                                         state["helper"], items, batch=_batch)
                else:
                    embs = encode_text(state["kind"], state["m"],
                                       state["helper"], items)
            np.save(out_npy, embs.astype(np.float32))
            print(f"[encode_worker] serve {mode}: {embs.shape} -> {out_npy}", flush=True)
            _respond(True)
        except Exception:
            print(f"[encode_worker] serve job failed:\n{_tb.format_exc()}", flush=True)
            _respond(False, _tb.format_exc())

    print("[encode_worker] serve ready", flush=True)
    # ── Ready marker (2026-09-16) ──────────────────────────────────────────────
    # The parent's _warm_ensure waits for THIS pid to appear here before
    # submitting a job. Closes the respawn race: a fresh serve instance that
    # exits immediately at the singleton gate (another healthy worker already
    # serves machine-wide) never writes the marker, so the parent learns of the
    # loss in seconds instead of burning the 45 s no-CPU watchdog on a job that
    # was eaten. The pid check makes stale markers from dead workers harmless.
    try:
        from pathlib import Path as _Pm
        import time as _tm
        _rd = _Pm(os.environ.get("FIRSTCUT_DATA_DIR", "")) if os.environ.get("FIRSTCUT_DATA_DIR") else None
        if _rd is None:
            _rd = _Pm(__file__).resolve().parent.parent / "cache"
        _rd.mkdir(parents=True, exist_ok=True)
        (_rd / "encode_worker.ready.json").write_text(_json.dumps(
            {"pid": os.getpid(), "ts": _tm.time()}), encoding="utf-8")
    except Exception:
        pass   # marker is an optimization — serve() works without it
    while True:
        try:
            line = lines.get(timeout=idle_s)
        except _queue.Empty:
            print(f"[encode_worker] serve idle {idle_s:.0f}s — exiting to free RAM", flush=True)
            os._exit(0)
        if line is None:
            os._exit(0)          # parent closed stdin — job object/terminate
        line = line.strip()
        if not line:
            continue
        try:
            job = _json.loads(line)
        except Exception:
            continue
        _run_job(job)


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
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()                      # never returns — the loop os._exits
        return
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
    except Exception as _exc:  # noqa: BLE001 — top-level boundary of a child process
        # Classify resource exhaustion (2026-09-07: a transient VRAM squeeze
        # made four warm attempts die with full FATAL tracebacks each — noise
        # that buried the one line that mattered). Known resource failures get
        # ONE classified line; everything else keeps the full traceback. The
        # exit code stays 1 either way: callers fall back on it, not on text.
        _msg = str(_exc)
        # A bare MemoryError() carries an EMPTY message (CPython raises it with
        # no args under commit exhaustion — observed 2026-09-08 at the cull
        # wall). splitlines() on "" is [], and [0] here crashed the error path
        # itself with IndexError, masking the one classified line that matters.
        _first_line = _msg.splitlines()[0] if _msg.splitlines() else type(_exc).__name__
        if ("out of memory" in _msg or "bad allocation" in _msg
                or isinstance(_exc, MemoryError)
                or "CUDA" in _msg and "out of memory" in _msg.lower()):
            print(f"[encode_worker] resource failure (CUDA/RAM exhausted): "
                  f"{_first_line[:160]} — caller falls back", flush=True)
        elif "files unreadable" in _msg:
            # Deterministic per-file decode failure (corrupt/truncated bytes,
            # wrong mount): a fresh process with a fresh model load will read
            # exactly the same bytes and fail exactly the same way. Exit 3 so
            # the parent's ladder can fail fast instead of burning 8 model
            # reloads on bytes that will never decode (2026-09-10: two 128 KB
            # truncated ARWs wedged a grade for 12+ minutes this way).
            print(f"[encode_worker] deterministic unreadable failure: "
                  f"{_first_line[:300]}", flush=True)
            _print_peak()
            os._exit(3)
        elif os.environ.get("FIRSTCUT_ENCODE_TRACEBACK", "") in ("1", "true", "yes"):
            print(f"[encode_worker] FATAL:\n{_tb.format_exc()}", flush=True)
        else:
            print(f"[encode_worker] FATAL: {type(_exc).__name__}: "
                  f"{_first_line[:200]}", flush=True)
        _print_peak()
        os._exit(1)


if __name__ == "__main__":
    main()
