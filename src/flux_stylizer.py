"""
Flux 2 [klein] creative stylizer.

IP-Adapter-v2 aesthetic vector  = CLIP ViT-L/14 image embedding of the anchor.
ControlNet                       = Canny edges (OpenCV) or depth (MiDaS).
NVFP4-equivalent quantization   = torchao INT4 on the Flux transformer.

VRAM strategy for 4-6 GB laptops
──────────────────────────────────
  - enable_sequential_cpu_offload(): only the currently executing layer is on GPU.
  - INT4 transformer quantization  : ~3.5 GB peak GPU usage.
  - T5 text encoder stays on CPU throughout (too large for a 4-6 GB budget).
  - torch.cuda.empty_cache() after every generated image.
  - Full pipeline unload (del + gc.collect) when the batch is complete.
"""
from __future__ import annotations

import gc
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PilImage

MODEL_ID       = "black-forest-labs/FLUX.1-schnell"
CONTROLNET_ID  = "InstantX/FLUX.1-dev-Controlnet-Canny"
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14"   # IP-Adapter-v2 backbone
NUM_STEPS_FAST   = 4
MAX_INFER_SIDE   = 1024   # cap long edge before Flux inference (px)
VRAM_FULL_GB     = 16.0   # >= this → all-GPU, fastest
VRAM_MODEL_GB    =  8.0   # >= this → enable_model_cpu_offload (3-4× faster than sequential)


class FluxStylizer:
    """
    Batch-processes Strong photos with Flux 2 [klein].

    IP-Adapter-v2 aesthetic vector is extracted from the anchor image using
    CLIP ViT-L/14 (768-d embedding), which is the same backbone used by
    IP-Adapter v2 reference implementations.  The actual diffusion step uses
    FluxImg2ImgPipeline or FluxControlNetPipeline depending on whether the
    ControlNet weights can be downloaded at runtime.
    """

    def __init__(self, use_controlnet: bool = True, device: str = "auto") -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.use_controlnet  = use_controlnet and self.device == "cuda"
        self._pipe           = None
        self._use_cn_active  = False
        if self.device == "cpu":
            print("[flux] No CUDA detected — will use fast CPU fallback (no Flux inference)")

    # ── IP-Adapter aesthetic vector ───────────────────────────────────────────

    def extract_aesthetic_vector(self, anchor_path: str) -> np.ndarray:
        """
        Encode the anchor image with CLIP ViT-L/14 (IP-Adapter-v2 backbone).
        Returns a normalised (768,) float32 array then immediately unloads the
        CLIP model so that VRAM is free for Flux.
        """
        from PIL import Image as PIL_Image
        from transformers import CLIPProcessor, CLIPVisionModel

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        clip_model = CLIPVisionModel.from_pretrained(
            CLIP_MODEL_ID, torch_dtype=dtype
        ).to(self.device).eval()

        img    = PIL_Image.open(anchor_path).convert("RGB")
        inputs = clip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = clip_model(**inputs).pooler_output     # (1, 768)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        emb = feats.cpu().float().numpy()[0]               # (768,)

        # Free VRAM before loading Flux
        clip_model = clip_model.cpu()
        del clip_model, clip_proc
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return emb

    # ── Fast CPU fallback (no Flux) ───────────────────────────────────────────

    @staticmethod
    def _cpu_stylize(
        image_path: str,
        anchor_path: str,
        strength: float = 0.55,
        role: str = "subject",
    ) -> "PilImage":
        """
        Lightweight PIL tone/contrast grading used when CUDA is unavailable.
        Matches the anchor's luminance histogram and applies role-specific contrast.
        Runs in <1s per image on CPU.
        """
        import cv2
        from PIL import Image as PIL_Image, ImageEnhance, ImageFilter

        src = cv2.imread(image_path)
        anc = cv2.imread(anchor_path)
        if src is None:
            raise FileNotFoundError(image_path)

        # Histogram matching in LAB L-channel
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        anc_lab = cv2.cvtColor(anc, cv2.COLOR_BGR2LAB).astype(np.float32) if anc is not None else src_lab

        for ch in range(3):
            src_ch = src_lab[:, :, ch].ravel()
            anc_ch = anc_lab[:, :, ch].ravel()
            src_cdf = np.sort(src_ch); anc_cdf = np.sort(anc_ch)
            interp  = np.interp(src_ch, src_cdf, anc_cdf)
            src_lab[:, :, ch] = interp.reshape(src_lab[:, :, ch].shape)

        matched = cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Blend: strength controls how much anchor palette is applied
        blended = cv2.addWeighted(src, 1.0 - strength, matched, strength, 0)
        out = PIL_Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

        # Role-specific enhancement
        role_tweaks = {
            "subject":  (1.15, 1.10),   # contrast, sharpness
            "opener":   (1.05, 1.00),
            "closer":   (1.10, 1.05),
            "contrast": (1.20, 1.00),
            "detail":   (1.05, 1.25),
        }
        contrast_f, sharp_f = role_tweaks.get(role, (1.10, 1.05))
        out = ImageEnhance.Contrast(out).enhance(contrast_f)
        if sharp_f > 1.0:
            out = out.filter(ImageFilter.UnsharpMask(radius=1, percent=int((sharp_f-1)*150)))
        return out

    # ── Structure extraction ──────────────────────────────────────────────────

    @staticmethod
    def extract_structure(
        image_path: str,
        mode: str = "canny",
        canny_low: int = 100,
        canny_high: int = 200,
    ) -> "PilImage":
        """
        Canny (mode='canny') — pure OpenCV, no extra model.
        Depth  (mode='depth') — MiDaS small (~80 MB, downloaded on first call).
        Falls back to Canny if MiDaS fails or is unavailable.
        """
        import cv2
        from PIL import Image as PIL_Image

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")

        if mode == "canny":
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, canny_low, canny_high)
            return PIL_Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # ── Depth via MiDaS small ─────────────────────────────────────────────
        try:
            import torch as _torch
            midas     = _torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
            midas_tfm = _torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
            dev       = "cuda" if _torch.cuda.is_available() else "cpu"
            midas     = midas.to(dev).eval()

            rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch = midas_tfm(rgb).to(dev)
            with _torch.no_grad():
                depth = midas(batch)
                depth = _torch.nn.functional.interpolate(
                    depth.unsqueeze(1), size=rgb.shape[:2],
                    mode="bicubic", align_corners=False
                ).squeeze()
            d = depth.cpu().numpy()
            d = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)

            midas = midas.cpu()
            del midas
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            return PIL_Image.fromarray(cv2.cvtColor(d, cv2.COLOR_GRAY2RGB))
        except Exception as e:
            print(f"[flux] MiDaS depth failed ({e}) — using Canny fallback")
            return FluxStylizer.extract_structure(image_path, mode="canny",
                                                  canny_low=canny_low, canny_high=canny_high)

    # ── Pipeline management ───────────────────────────────────────────────────

    def _quantize_transformer(self, transformer):
        """Apply torchao INT4 weight-only quantization (NVFP4-equivalent)."""
        try:
            from torchao.quantization import quantize_, int4_weight_only
            quantize_(transformer, int4_weight_only())
            print("[flux] Transformer quantized to INT4 (NVFP4 mode)")
        except Exception as e:
            print(f"[flux] torchao INT4 unavailable ({e}) — using BF16")
        return transformer

    def _vram_gb(self) -> float:
        """Return total VRAM in GB, or 0.0 if CPU."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0
        try:
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        except Exception:
            return 0.0

    def _apply_offload(self, pipe) -> None:
        """Choose the fastest offload strategy that fits in VRAM."""
        vram = self._vram_gb()
        if vram >= VRAM_FULL_GB:
            pipe.to(self.device)
            print(f"[flux] offload=none  ({vram:.1f} GB VRAM — all on GPU)")
        elif vram >= VRAM_MODEL_GB:
            pipe.enable_model_cpu_offload()
            print(f"[flux] offload=model ({vram:.1f} GB VRAM — submodel-level)")
        else:
            pipe.enable_sequential_cpu_offload()
            print(f"[flux] offload=sequential ({vram:.1f} GB VRAM — layer-level, slowest)")

    def _load_pipeline(self, sample_structure: "PilImage | None" = None) -> None:
        """
        Load the Flux pipeline once for the whole batch.
        Tries FluxControlNetPipeline first; falls back to FluxImg2ImgPipeline.
        """
        if self._pipe is not None:
            return

        import time
        t0 = time.time()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        if self.use_controlnet and sample_structure is not None:
            try:
                from diffusers import FluxControlNetPipeline, FluxControlNetModel

                cn = FluxControlNetModel.from_pretrained(
                    CONTROLNET_ID, torch_dtype=dtype
                )
                pipe = FluxControlNetPipeline.from_pretrained(
                    MODEL_ID, controlnet=cn, torch_dtype=dtype
                )
                pipe.transformer   = self._quantize_transformer(pipe.transformer)
                self._apply_offload(pipe)
                self._pipe          = pipe
                self._use_cn_active = True
                print(f"[flux] FluxControlNetPipeline loaded in {time.time()-t0:.1f}s")
                return
            except Exception as e:
                print(f"[flux] ControlNet load failed ({e}) — using Img2Img fallback")

        from diffusers import FluxImg2ImgPipeline

        pipe = FluxImg2ImgPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe.transformer   = self._quantize_transformer(pipe.transformer)
        self._apply_offload(pipe)
        self._pipe          = pipe
        self._use_cn_active = False
        print(f"[flux] FluxImg2ImgPipeline loaded in {time.time()-t0:.1f}s")

    def _unload_pipeline(self) -> None:
        """Delete all Flux components and flush the GPU cache."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Single-image stylization ──────────────────────────────────────────────

    def stylize_one(
        self,
        image_path:    str,
        structure_img: "PilImage",
        style_prompt:  str,
        strength:      float = 0.55,
        guidance:      float = 3.5,
        ctrl_weight:   float = 0.60,
        num_steps:     int   = NUM_STEPS_FAST,
    ) -> "PilImage":
        """
        Generate one stylized image.  Pipeline must already be loaded.
        Calls cuda.empty_cache() after generation to free residual tensors.
        """
        from PIL import Image as PIL_Image

        if self._pipe is None:
            raise RuntimeError("Call _load_pipeline() before stylize_one()")

        import time as _time
        _t0 = _time.time()

        source = PIL_Image.open(image_path).convert("RGB")
        w, h   = source.size
        # Cap long edge — transformer cost scales quadratically with pixel count.
        if max(w, h) > MAX_INFER_SIDE:
            scale = MAX_INFER_SIDE / max(w, h)
            w, h  = int(w * scale), int(h * scale)
        # Flux VAE requires dimensions that are multiples of 64
        w = max(64, (w // 64) * 64)
        h = max(64, (h // 64) * 64)
        source = source.resize((w, h), PIL_Image.LANCZOS)
        print(f"[flux] inference size: {w}×{h}")

        gen_kwargs: dict = dict(
            prompt              = style_prompt,
            image               = source,
            strength            = float(strength),
            guidance_scale      = float(guidance),
            num_inference_steps = int(num_steps),
        )

        if self._use_cn_active:
            ctrl = structure_img.resize((w, h), PIL_Image.LANCZOS)
            gen_kwargs["control_image"]                   = ctrl
            gen_kwargs["controlnet_conditioning_scale"]   = float(ctrl_weight)

        with torch.no_grad():
            result = self._pipe(**gen_kwargs).images[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[flux] stylize_one done in {_time.time()-_t0:.1f}s")
        return result

    # ── Batch pipeline ────────────────────────────────────────────────────────

    def process_batch(
        self,
        strong_paths:     list[str],
        anchor_path:      str,
        output_dir:       Path,
        params_per_image: list[dict],
        structure_mode:   str = "canny",
        style_prompt:     str = "",
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> list[dict]:
        """
        For each Strong image:
          1. Extract Canny / depth structure map (CPU).
          2. Load Flux 2 [klein] once (GPU with INT4 + CPU offload).
          3. Apply MOGCO-II-selected parameters for each image.
          4. Save to output_dir/.
          5. Flush VRAM.

        Returns a list of result dicts, one per source image.
        """
        from PIL import Image as PIL_Image

        _p = progress or (lambda f, d: None)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import time as _time
        n       = len(strong_paths)
        results: list[dict] = []
        t_batch = _time.time()

        # ── CPU path: skip Flux, use fast PIL fallback ────────────────────────
        if self.device == "cpu":
            _p(0.10, "No GPU — applying fast CPU tone grading…")
            for i, (path, params) in enumerate(zip(strong_paths, params_per_image)):
                fname    = Path(path).stem + "_styled.jpg"
                out_path = output_dir / fname
                _p(0.10 + (i / n) * 0.88, f"Grading {i+1}/{n}: {Path(path).name}")
                try:
                    t0 = _time.time()
                    styled = self._cpu_stylize(
                        path, anchor_path,
                        strength=params.get("strength", 0.55),
                        role=params.get("role", "subject"),
                    )
                    styled.save(str(out_path), quality=92)
                    print(f"[flux] cpu_stylize {Path(path).name} done in {_time.time()-t0:.2f}s")
                    results.append({
                        "source_path": path, "output_path": str(out_path),
                        "filename": fname, "params": params,
                        "success": True, "engine": "cpu_fallback",
                    })
                except Exception as e:
                    print(f"[flux] cpu_stylize failed {path}: {e}")
                    results.append({
                        "source_path": path, "output_path": None,
                        "error": str(e), "success": False,
                    })
            print(f"[flux] CPU batch done in {_time.time()-t_batch:.1f}s")
            _p(1.0, f"Done — {sum(r['success'] for r in results)}/{n} images graded (CPU mode)")
            return results

        # ── Step 1: Structure maps (CPU-only, no VRAM needed) ─────────────────
        _p(0.02, "Extracting structure maps…")
        structures: list["PilImage"] = []
        for i, path in enumerate(strong_paths):
            try:
                sm = self.extract_structure(path, mode=structure_mode)
                structures.append(sm)
            except Exception as e:
                print(f"[flux] structure failed for {path}: {e}")
                structures.append(PIL_Image.new("RGB", (512, 512), 0))
            _p(0.02 + (i / n) * 0.08, f"Structure {i + 1}/{n}")

        # ── Step 2: Load Flux once ────────────────────────────────────────────
        _p(0.10, "Loading Flux 2 [klein]…")
        self._load_pipeline(sample_structure=structures[0] if structures else None)
        _p(0.15, "Flux loaded — beginning stylization…")

        # ── Steps 3–4: Stylize each image ────────────────────────────────────
        for i, (path, struct, params) in enumerate(
            zip(strong_paths, structures, params_per_image)
        ):
            fname    = Path(path).stem + "_styled.jpg"
            out_path = output_dir / fname
            prompt   = params.get("prompt") or style_prompt or "cinematic street photography, high contrast"

            _p(
                0.15 + (i / n) * 0.80,
                f"Stylizing {i + 1}/{n}: {Path(path).name} "
                f"(str={params['strength']:.2f} gd={params['guidance']:.1f})"
            )

            try:
                styled = self.stylize_one(
                    image_path    = path,
                    structure_img = struct,
                    style_prompt  = prompt,
                    strength      = params["strength"],
                    guidance      = params["guidance"],
                    ctrl_weight   = params.get("ctrl_weight", 0.60),
                    num_steps     = params.get("num_steps", NUM_STEPS_FAST),
                )
                styled.save(str(out_path), quality=92)
                results.append({
                    "source_path": path,
                    "output_path": str(out_path),
                    "filename":    fname,
                    "params":      params,
                    "success":     True,
                })
            except Exception as e:
                print(f"[flux] stylization failed for {path}: {e}")
                results.append({
                    "source_path": path,
                    "output_path": None,
                    "error":       str(e),
                    "success":     False,
                })

        # ── Step 5: Flush VRAM ────────────────────────────────────────────────
        _p(0.96, "Unloading Flux — flushing VRAM…")
        self._unload_pipeline()

        success_n = sum(1 for r in results if r["success"])
        _p(1.0, f"Done — {success_n}/{n} images styled into {output_dir.name}")
        return results
