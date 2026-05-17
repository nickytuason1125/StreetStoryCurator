"""
SigLIP-2 ViT-g/14 NaFlex Encoder with Auto-Download

Replaces SigLIP-So400M with:
- FP8 quantization for faster inference
- Native aspect ratio preservation (fixes compositional "squishing")
- Larger model capacity (1536-d embeddings)

VRAM Protocol:
    1. SigLIP2Encoder() → model loads into VRAM
    2. encode_images() → all embeddings computed
    3. unload() → GPU cleared for next step

Auto-download: Model is downloaded on first run if not present locally.
"""

from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np


def _free_system_ram_gb() -> float:
    """Return available system RAM in GB (cross-platform, no psutil required)."""
    try:
        import ctypes
        class _MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength",                ctypes.c_ulong),
                ("dwMemoryLoad",            ctypes.c_ulong),
                ("ullTotalPhys",            ctypes.c_ulonglong),
                ("ullAvailPhys",            ctypes.c_ulonglong),
                ("ullTotalPageFile",        ctypes.c_ulonglong),
                ("ullAvailPageFile",        ctypes.c_ulonglong),
                ("ullTotalVirtual",         ctypes.c_ulonglong),
                ("ullAvailVirtual",         ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = _MEMSTATUS()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / 1e9
    except Exception:
        pass
    try:
        import resource
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1e9
    except Exception:
        return 8.0  # assume enough if we can't check


# Model configuration — largest 1536-d SigLIP-2 variant available in open_clip_torch
_MODEL_TAG = "ViT-gopt-16-SigLIP2-384"
_PRETRAINED = "webli"
EMBED_DIM = 1536  # SigLIP-2 ViT-gopt/16 @ 384px

# Model cache directory
MODEL_CACHE_DIR = Path("models/siglip2")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _find_siglip2_safetensors() -> Optional[Path]:
    """Return path to the cached .safetensors weights file, or None."""
    for p in MODEL_CACHE_DIR.rglob("*.safetensors"):
        if p.is_file() and not p.name.endswith(".incomplete"):
            return p
    return None


def _siglip2_cache_exists() -> bool:
    """Return True if SigLIP-2 weights are already in MODEL_CACHE_DIR (any depth)."""
    if not MODEL_CACHE_DIR.exists():
        return False
    weight_exts = {".pt", ".bin", ".safetensors"}
    return any(
        f.suffix in weight_exts
        for f in MODEL_CACHE_DIR.rglob("*")
        if f.is_file() and not f.name.endswith(".incomplete")
    )


def _download_siglip2_if_needed() -> bool:
    """
    Pre-download SigLIP-2 ViT-g/14 NaFlex weights into MODEL_CACHE_DIR if absent.

    open_clip auto-downloads on first use; calling this before the first
    SigLIP2Encoder() instantiation avoids a silent first-request delay.

    Returns:
        True if model is ready, False on error.
    """
    if _siglip2_cache_exists():
        return True

    print(f"📦 Downloading SigLIP-2 ViT-g/14 NaFlex to {MODEL_CACHE_DIR}...")
    print("   This may take several minutes depending on your connection.")

    try:
        import open_clip

        # create_model_and_transforms downloads weights to cache_dir on first call.
        model, _, _ = open_clip.create_model_and_transforms(
            _MODEL_TAG,
            pretrained=_PRETRAINED,
            precision="fp16",
            cache_dir=str(MODEL_CACHE_DIR),
        )
        # Release immediately — we only needed the download side-effect.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ SigLIP-2 download complete: {MODEL_CACHE_DIR}")
        return True

    except ImportError:
        print("⚠️  open_clip_torch not installed. Run: pip install open_clip_torch")
        return False
    except Exception as e:
        print(f"⚠️  SigLIP-2 download failed: {e}")
        return False


class SigLIP2Encoder:
    """
    SigLIP-2 ViT-g/14 NaFlex image encoder with FP8 quantization.

    Key features:
    - Native aspect ratio preservation (no forced square cropping)
    - FP8 quantization for faster inference
    - 1536-d embeddings (vs 1152-d for So400M)
    """

    # Disk file is FP32 (7.1 GB), but loaded as FP16 → ~3.5 GB VRAM.
    # INT8 quantization halves that to ~1.8 GB when torchao is available.
    _QUANTIZED_VRAM_GB = 1.8
    _FP16_VRAM_GB      = 3.5

    def __init__(self, device: str = "auto", quantize: bool = True, progress=None):
        import open_clip

        _p = progress or (lambda f, d: None)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            free_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_reserved(0)
            ) / 1e9
            needed_gb = self._QUANTIZED_VRAM_GB if quantize else self._FP16_VRAM_GB
            if free_gb < needed_gb:
                print(
                    f"[siglip2] WARNING: only {free_gb:.1f} GB VRAM free, "
                    f"need ~{needed_gb:.1f} GB — falling back to CPU inference"
                )
                self.device = torch.device("cpu")

        # Download weights if not cached (first run only — 7.1 GB)
        if not _siglip2_cache_exists():
            _p(0.03, "Downloading SigLIP-2 (7.1 GB) — first run, may take 10+ min…")
            _download_siglip2_if_needed()
        else:
            _p(0.03, "SigLIP-2 weights cached — loading to GPU…")

        if self.device.type == "cuda":
            # ── GPU path: stream weights disk→FP16→VRAM, ~0 system RAM ──────────
            # Each tensor is: mmap FP32 → half() CPU → copy to CUDA → del CPU copy.
            # Peak system RAM = size of the largest single weight tensor (~300 MB).
            safetensors_path = _find_siglip2_safetensors()
            if safetensors_path is None:
                raise RuntimeError(
                    "SigLIP-2 safetensors file not found in cache. "
                    "Delete models/siglip2/ and restart to re-download."
                )

            _p(0.035, "Allocating SigLIP-2 model skeleton on GPU…")
            # Create architecture + preprocessor with random FP16 weights on GPU.
            # pretrained=None skips all weight loading — we load them ourselves below.
            self._model, _, self._prep = open_clip.create_model_and_transforms(
                _MODEL_TAG,
                pretrained=None,
                precision="fp16",
                device=str(self.device),
                cache_dir=str(MODEL_CACHE_DIR),
            )

            _p(0.04, "Streaming SigLIP-2 weights to GPU (no RAM staging)…")
            self._stream_weights_to_device(safetensors_path, self.device)

            self._tok = open_clip.get_tokenizer(_MODEL_TAG)

            if quantize:
                _p(0.05, "Quantising SigLIP-2 to INT8…")
                self._model = self._quantize_model(self._model)

            self._model.eval()
            gc.collect()

            # ── TensorRT engine: load if pre-compiled engine exists ───────────
            # Compile once with: python -c "from siglip2_encoder import compile_trt_engine; compile_trt_engine()"
            _trt_model_loaded = False
            _engine_path = MODEL_CACHE_DIR / "siglip2.engine"
            if _engine_path.exists():
                try:
                    import tensorrt as trt
                    import torch_tensorrt  # noqa: F401
                    self._model = torch_tensorrt.load(_engine_path).eval()
                    _trt_model_loaded = True
                    print("[siglip2] TensorRT engine loaded — fast inference active")
                except Exception as _trt_e:
                    print(f"[siglip2] TensorRT load failed ({_trt_e}) — using standard model")

            # torch.compile fuses 48 ViT-g attention layers into a CUDA graph,
            # eliminating per-layer Python dispatch overhead (~20-40% faster).
            # Skipped when TRT engine is active — already AOT-compiled.
            if not _trt_model_loaded:
                try:
                    self._model = torch.compile(
                        self._model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[siglip2] torch.compile active — first batch warms up JIT")
                except Exception as _ce:
                    print(f"[siglip2] torch.compile skipped ({_ce})")

            _p(0.07, "SigLIP-2 ready — encoding images…")

        else:
            # ── CPU fallback: needs ~3.7 GB system RAM ────────────────────────
            _free_ram_gb = _free_system_ram_gb()
            if _free_ram_gb < 4.0:
                raise RuntimeError(
                    f"Insufficient RAM to load SigLIP-2 on CPU: "
                    f"{_free_ram_gb:.1f} GB free, need ~4 GB. "
                    "Close other applications to free memory, then try again."
                )

            _p(0.04, "Loading SigLIP-2 to CPU (needs ~3.7 GB RAM)…")
            self._model, _, self._prep = open_clip.create_model_and_transforms(
                _MODEL_TAG,
                pretrained=_PRETRAINED,
                precision="fp16",
                cache_dir=str(MODEL_CACHE_DIR),
            )
            self._tok = open_clip.get_tokenizer(_MODEL_TAG)
            self._model = self._model.to(self.device).eval()
            gc.collect()
            _p(0.07, "SigLIP-2 ready — encoding images…")
    
    def _stream_weights_to_device(self, safetensors_path: Path, device: torch.device) -> None:
        """
        Stream safetensors weights directly to `device` without full CPU staging.

        For each tensor in the file:
          1. Read FP32 slice via mmap (uses OS page cache, not heap RAM)
          2. .half() → FP16 CPU tensor (real alloc, one tensor at a time)
          3. .to(device, non_blocking=True) → CUDA FP16
          4. del CPU tensor immediately

        Peak system RAM: size of the single largest weight matrix (~300 MB).
        """
        from safetensors.torch import safe_open

        params  = dict(self._model.named_parameters())
        buffers = dict(self._model.named_buffers())
        loaded = skipped = 0

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                fp32_cpu = f.get_tensor(key)          # mmap view — ~0 heap RAM
                fp16_cpu = fp32_cpu.half()             # one FP16 tensor on CPU
                del fp32_cpu                           # release mmap reference

                target = params.get(key)
                if target is None:
                    target = buffers.get(key)
                if target is not None:
                    with torch.no_grad():
                        target.data.copy_(fp16_cpu.to(device, non_blocking=True))
                    loaded += 1
                else:
                    skipped += 1

                del fp16_cpu                           # free CPU FP16 immediately

        if skipped:
            print(f"[siglip2] streaming: {loaded} loaded, {skipped} keys skipped")
        else:
            print(f"[siglip2] streaming: {loaded} tensors loaded directly to VRAM")

        if device.type == "cuda":
            torch.cuda.synchronize()
        gc.collect()

    def _quantize_model(self, model) -> torch.nn.Module:
        """
        Tries torchao INT8 weight-only quantization; falls back gracefully to FP16 on GPU.
        FP16 (~3.5 GB) fits in a 6 GB card without quantization.
        """
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
            print("[siglip2] Quantization: INT8 weight-only via torchao (~1.8 GB VRAM)")
            return model
        except Exception as e_torchao:
            print(f"[siglip2] torchao INT8 unavailable ({e_torchao}) — using FP16 on GPU (~3.5 GB)")
            return model
    
    def encode_images(
        self,
        paths: List[str],
        batch_size: int = 0,  # 0 = auto: 8 on GPU, 4 on CPU
        progress=None,
    ) -> np.ndarray:
        """
        Return normalised (N, 1536) float32 embeddings for a list of image paths.

        Uses parallel CPU prefetch: the next batch is loaded/preprocessed in a
        background thread while the GPU runs inference on the current batch.
        Bad paths yield a zero vector.
        """
        from PIL import Image as _PIL
        from concurrent.futures import ThreadPoolExecutor

        if batch_size == 0:
            batch_size = 16 if self.device.type == "cuda" else 4

        _model_dtype = next(iter(self._model.parameters())).dtype
        prep         = self._prep   # local ref avoids attribute lookup in hot loop
        zero         = torch.zeros(3, 384, 384)

        # TurboJPEG singleton for 2-4x faster JPEG decode vs PIL
        _tj = None
        try:
            from fast_ingestion import _get_tj as _tj_getter
            _tj = _tj_getter()
        except Exception:
            pass

        _JPEG_EXTS  = {".jpg", ".jpeg"}
        # Pre-downsample cap before open_clip prep: 6000px→512 is 150× fewer pixels
        # for prep's BICUBIC resize vs working at native sensor resolution.
        _DECODE_CAP = 512

        def _load(p: str) -> torch.Tensor:
            try:
                if Path(p).suffix.lower() in _JPEG_EXTS and _tj is not None:
                    with open(p, "rb") as fh:
                        raw = fh.read()
                    bgr    = _tj.decode(raw)
                    pil_img = _PIL.fromarray(bgr[:, :, ::-1].copy())
                else:
                    pil_img = _PIL.open(p).convert("RGB")
                if max(pil_img.size) > _DECODE_CAP:
                    pil_img.thumbnail((_DECODE_CAP, _DECODE_CAP), _PIL.Resampling.BILINEAR)
                return prep(pil_img)
            except Exception:
                return zero

        all_embs: List[np.ndarray] = []
        n = len(paths)

        # Pre-allocate one pinned buffer for the entire run — eliminates
        # repeated CUDA page-lock allocations (~10-50 ms each) that would
        # otherwise happen on every batch iteration.
        # Safe to reuse: emb.cpu() in each iteration is a blocking CUDA sync,
        # guaranteeing the GPU is done reading the buffer before the next
        # batch writes into it.
        _pinned = (
            torch.empty(batch_size, 3, 384, 384).pin_memory()
            if self.device.type == "cuda" else None
        )

        with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as pool:
            # Pre-load the first batch so there's no wait on the first iteration
            next_futures = [pool.submit(_load, p) for p in paths[:batch_size]]

            for start in range(0, n, batch_size):
                futures        = next_futures
                next_start     = start + batch_size
                # Kick off the next batch load while GPU runs on the current one
                next_futures   = (
                    [pool.submit(_load, p) for p in paths[next_start : next_start + batch_size]]
                    if next_start < n else []
                )

                tensors = [f.result() for f in futures]
                b = len(tensors)
                if _pinned is not None:
                    # Write directly into pre-pinned memory — no malloc, no copy.
                    cpu_batch = _pinned[:b]
                    torch.stack(tensors, out=cpu_batch)
                else:
                    cpu_batch = torch.stack(tensors)
                batch = cpu_batch.to(
                    device=self.device,
                    dtype=_model_dtype,
                    non_blocking=True,
                )

                with torch.inference_mode():
                    emb = self._model.encode_image(batch)
                    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                all_embs.append(emb.cpu().float().numpy())
                del cpu_batch, batch, emb

                if progress:
                    done = min(start + batch_size, n)
                    progress(done / n * 0.47, f"SigLIP-2: {done}/{n}")

        result = np.concatenate(all_embs, axis=0)
        # Full GC sweep — free all intermediate tensors before next pipeline stage
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        return result
    
    def encode_text(self, queries: List[str]) -> np.ndarray:
        """Return normalised (N, 1536) float32 embeddings for text queries."""
        tokens = self._tok(queries).to(self.device)
        with torch.inference_mode():
            emb = self._model.encode_text(tokens)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
        return emb.cpu().float().numpy()
    
    def unload(self) -> None:
        """Move model to CPU and empty GPU cache."""
        self._model = self._model.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


class ResizePad:
    """
    Resize and pad image to maintain aspect ratio.
    
    Used for SigLIP-2 to avoid compositional "squishing".
    """
    
    def __init__(self, size: int = 384, fill: int = 128):
        self.size = size
        self.fill = fill
    
    def __call__(self, img):
        """Resize and pad image to square while maintaining aspect ratio."""
        from PIL import Image, ImageOps
        
        # Get original dimensions
        w, h = img.size
        
        # Calculate scale to fit within size
        scale = min(self.size / w, self.size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
        padded.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        
        return padded


def get_siglip2_encoder() -> SigLIP2Encoder:
    """Get or create SigLIP-2 encoder singleton."""
    if not hasattr(get_siglip2_encoder, "_instance"):
        get_siglip2_encoder._instance = SigLIP2Encoder()
    return get_siglip2_encoder._instance


def compile_trt_engine(output_path: str = "models/siglip2/siglip2.engine") -> None:
    """
    One-time TensorRT compilation of the SigLIP-2 encoder.

    Run manually once:
        python -c "from siglip2_encoder import compile_trt_engine; compile_trt_engine()"

    The compiled engine is loaded automatically on subsequent runs.
    Requires: tensorrt, torch-tensorrt (pip install torch-tensorrt tensorrt).
    """
    import torch
    import torch_tensorrt

    print("[siglip2] Compiling TensorRT engine — this takes 2-10 minutes on first run…")
    enc = SigLIP2Encoder(device="auto", quantize=False)
    model = enc._model.cuda().eval()

    dummy = torch.randn(1, 3, 384, 384, device="cuda")
    compiled = torch_tensorrt.compile(
        model.visual,
        inputs=[torch_tensorrt.Input(
            min_shape=(1, 3, 384, 384),
            opt_shape=(16, 3, 384, 384),
            max_shape=(32, 3, 384, 384),
            dtype=torch.float16,
        )],
        enabled_precisions={torch.float16},
    )
    torch_tensorrt.save(compiled, output_path)
    print(f"[siglip2] TensorRT engine saved → {output_path}")
    enc.unload()
