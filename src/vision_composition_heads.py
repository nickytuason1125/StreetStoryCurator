"""
Context-aware composition analysis heads.

DepthHead              — relative depth maps (0=near, 255=far)
SegCompositionAnalyzer — person masks + depth-layer categorization;
                         over-the-shoulder portrait → comp_score override 0.85
ChiaroscuroHead        — structural encoder + luminance bimodality; deactivates low-lum penalties
"""
from __future__ import annotations

import gc
import numpy as np
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# DepthHead
# ---------------------------------------------------------------------------

class DepthHead:
    """Depth Anything V2 Small — relative monocular depth, 0=foreground 255=background."""

    _REPO = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self):
        self._pipe  = None
        self._ready = False

    _TRT_ENGINE        = "depth_anything_v2_vits.engine"   # pre-compiled TensorRT FP16 engine
    _DEPTH_BATCH_SIZE  = 8    # GPU mini-batch for HF depth pipeline
    _DEPTH_INPUT_SIZE  = 518  # Depth Anything V2 native input resolution

    def load(self) -> bool:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Prefer a pre-compiled TensorRT FP16 engine (2-4× faster than PyTorch).
        # Compile with: trtexec --onnx=depth_vits.onnx --fp16 --saveEngine=depth_anything_v2_vits.engine
        if Path(self._TRT_ENGINE).exists() and device == "cuda":
            try:
                import tensorrt as trt
                import pycuda.driver as cuda
                import pycuda.autoinit  # noqa: F401
                with open(self._TRT_ENGINE, "rb") as f:
                    engine_data = f.read()
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self._trt_engine  = runtime.deserialize_cuda_engine(engine_data)
                self._trt_context = self._trt_engine.create_execution_context()
                self._trt_mode    = True
                self._ready       = True
                print(f"[DepthHead] TensorRT FP16 engine loaded: {self._TRT_ENGINE}")
                return True
            except Exception as e:
                print(f"[DepthHead] TRT engine load failed ({e}) — PyTorch fallback")

        # PyTorch / HuggingFace transformers path
        try:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline(
                task="depth-estimation",
                model=self._REPO,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self._trt_mode = False
            self._ready    = True
            print(f"[DepthHead] Loaded {self._REPO} on {device}")
            return True
        except Exception as e:
            print(f"[DepthHead] Load failed: {e}")
            return False

    def infer(self, path: str) -> Optional[np.ndarray]:
        """Return (H, W) uint8 depth array where 0=near, 255=far, or None on failure."""
        if not self._ready:
            return None
        if getattr(self, "_trt_mode", False) and self._pipe is None:
            return None   # TRT engine loaded but HF pipe unavailable; caller handles None
        try:
            import torch
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img.thumbnail((self._DEPTH_INPUT_SIZE, self._DEPTH_INPUT_SIZE), Image.Resampling.BILINEAR)
            with torch.inference_mode():
                out = self._pipe(img)
            depth = np.array(out["depth"])
            d_min, d_max = float(depth.min()), float(depth.max())
            span = max(d_max - d_min, 1e-6)
            return np.clip(((depth - d_min) / span) * 255, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[DepthHead] Inference failed for {Path(path).name}: {e}")
            return None

    def infer_batch(self, paths: list[str]) -> dict[str, Optional[np.ndarray]]:
        """
        Depth inference in mini-batches of _DEPTH_BATCH_SIZE images.

        Accumulates decoded PIL images into contiguous blocks before each pipeline
        call so the GPU processes a full Tensor Core batch rather than one image at
        a time.  TRT mode falls back to sequential infer() — the engine has no
        list API.
        """
        if not self._ready or not paths:
            return {p: None for p in paths}

        if getattr(self, "_trt_mode", False):
            return {p: self.infer(p) for p in paths}

        try:
            import torch
            from PIL import Image
            from concurrent.futures import ThreadPoolExecutor as _TPE

            _cap = self._DEPTH_INPUT_SIZE

            def _load_and_cap(p: str):
                try:
                    img = Image.open(p).convert("RGB")
                    img.thumbnail((_cap, _cap), Image.Resampling.BILINEAR)
                    return img
                except Exception:
                    return None

            result: dict[str, Optional[np.ndarray]] = {p: None for p in paths}

            # Stream one batch at a time — only _DEPTH_BATCH_SIZE images in RAM
            # at once instead of the full N-image list.
            n_workers = min(self._DEPTH_BATCH_SIZE, 4)
            with _TPE(max_workers=n_workers) as pool:
                for batch_start in range(0, len(paths), self._DEPTH_BATCH_SIZE):
                    b_paths_all = paths[batch_start : batch_start + self._DEPTH_BATCH_SIZE]
                    b_imgs_raw  = list(pool.map(_load_and_cap, b_paths_all))
                    b_valid = [
                        (p, img) for p, img in zip(b_paths_all, b_imgs_raw)
                        if img is not None
                    ]
                    del b_imgs_raw
                    if not b_valid:
                        continue
                    b_paths_v, b_imgs_v = zip(*b_valid)
                    with torch.inference_mode():
                        outs = self._pipe(list(b_imgs_v))
                    del b_imgs_v
                    for p, out in zip(b_paths_v, outs):
                        depth = np.array(out["depth"])
                        d_min, d_max = float(depth.min()), float(depth.max())
                        span  = max(d_max - d_min, 1e-6)
                        result[p] = np.clip(((depth - d_min) / span) * 255, 0, 255).astype(np.uint8)

            return result

        except Exception as e:
            print(f"[DepthHead] Batch inference failed ({e}) — per-image fallback")
            return {p: self.infer(p) for p in paths}

    def unload(self):
        self._pipe  = None
        self._ready = False
        try:
            import torch, gc as _gc
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            _gc.collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# SegCompositionAnalyzer
# ---------------------------------------------------------------------------

# Depth thresholds for layer assignment (0=near, 255=far)
_FOREGROUND_DEPTH_MAX  = 40    # Z < 40  → immediate foreground
_MIDGROUND_DEPTH_MIN   = 60    # 60 ≤ Z ≤ 160 → midground subject layer
_MIDGROUND_DEPTH_MAX   = 160
_OTS_PORTRAIT_COMP     = 0.85  # over-the-shoulder composition score override

_TV_PERSON_LABEL = 1     # torchvision COCO detection label index (0=background, 1=person)
_SEG_CONF        = 0.35


class SegCompositionResult:
    """Per-image result from SegCompositionAnalyzer."""

    __slots__ = (
        "has_person",
        "foreground_mask",   # bool array H×W — person pixels in fg layer
        "midground_mask",    # bool array H×W — person pixels in midground layer
        "is_ots_portrait",   # over-the-shoulder detection
        "comp_override",     # float or None
        "subject_masks",     # list of bool H×W arrays, one per detected person
    )

    def __init__(self):
        self.has_person      = False
        self.foreground_mask = None
        self.midground_mask  = None
        self.is_ots_portrait = False
        self.comp_override   = None
        self.subject_masks   = []


class SegCompositionAnalyzer:
    """
    Mask R-CNN (torchvision, BSD-3) person instance segmentation +
    depth-layer composition logic — replaces ultralytics YOLO11s-seg
    (AGPL-3.0). This is the one detection site in the app that needs real
    instance-segmentation masks (fg/mid depth-layer union for the
    over-the-shoulder-portrait override), so it can't use the boxes-only
    D-FINE-nano detector shared by the other sites (src/dfine_detector.py).
    """

    def __init__(self):
        self._model  = None
        self._ready  = False
        self._device = "cpu"

    def load(self) -> bool:
        try:
            import torch
            from torchvision.models.detection import (
                maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
                box_score_thresh=_SEG_CONF,
            ).to(device).eval()
            self._device = device
            self._ready  = True
            print(f"[SegComp] Mask R-CNN loaded on {device}")
            return True
        except Exception as e:
            print(f"[SegComp] Load failed: {e}")
            return False

    def analyze(self, path: str, depth_map: Optional[np.ndarray]) -> SegCompositionResult:
        result = SegCompositionResult()
        if not self._ready:
            return result

        try:
            import torch
            from PIL import Image
            import torchvision.transforms.functional as TF

            img = Image.open(path).convert("RGB")
            img_w, img_h = img.size
            canvas_area  = img_h * img_w
            tensor = TF.to_tensor(img).to(self._device)
            with torch.inference_mode():
                out = self._model([tensor])[0]
        except Exception as e:
            print(f"[SegComp] Inference failed for {Path(path).name}: {e}")
            return result

        keep = (out["labels"] == _TV_PERSON_LABEL) & (out["scores"] >= _SEG_CONF)
        for mask_data in out["masks"][keep]:
            # mask_data: (1, H, W) float tensor [0,1], already at source resolution
            mask_np = mask_data[0].cpu().numpy() > 0.5

            mask_area = int(mask_np.sum())
            if mask_area < int(0.0008 * canvas_area):
                continue   # ignore tiny distant figures

            result.has_person = True
            result.subject_masks.append(mask_np)

            if depth_map is not None:
                dmap_resized = depth_map
                if depth_map.shape != (img_h, img_w):
                    dimg = Image.fromarray(depth_map).resize((img_w, img_h), Image.BILINEAR)
                    dmap_resized = np.array(dimg)

                person_depths = dmap_resized[mask_np]
                mean_depth    = float(person_depths.mean()) if len(person_depths) else 128.0

                if mean_depth < _FOREGROUND_DEPTH_MAX:
                    fg = mask_np if result.foreground_mask is None else (result.foreground_mask | mask_np)
                    result.foreground_mask = fg
                elif _MIDGROUND_DEPTH_MIN <= mean_depth <= _MIDGROUND_DEPTH_MAX:
                    mg = mask_np if result.midground_mask is None else (result.midground_mask | mask_np)
                    result.midground_mask = mg

        # Over-the-shoulder portrait: one midground person, one foreground occlusion
        if (
            result.midground_mask is not None
            and result.foreground_mask is not None
        ):
            mid_frac = float(result.midground_mask.sum()) / canvas_area
            fg_frac  = float(result.foreground_mask.sum()) / canvas_area
            # OTS: midground subject + partial foreground person visible at edge
            if 0.05 < mid_frac < 0.45 and 0.03 < fg_frac < 0.25:
                result.is_ots_portrait = True
                result.comp_override   = _OTS_PORTRAIT_COMP

        return result

    def analyze_batch(
        self, paths: list[str], depth_maps: dict[str, Optional[np.ndarray]]
    ) -> dict[str, "SegCompositionResult"]:
        """
        Single Mask R-CNN batched call for all paths — eliminates N-1 Python
        dispatch overheads (torchvision detection models accept a list of
        variable-resolution tensors and batch them internally). Falls back
        to per-image analyze() on any error.
        """
        results: dict[str, SegCompositionResult] = {p: SegCompositionResult() for p in paths}
        if not self._ready or not paths:
            return results

        try:
            import torch
            from PIL import Image as _PIL_img
            import torchvision.transforms.functional as TF

            imgs  = [_PIL_img.open(p).convert("RGB") for p in paths]
            sizes = [img.size for img in imgs]   # (w, h) per image
            tensors = [TF.to_tensor(img).to(self._device) for img in imgs]
            with torch.inference_mode():
                outs = self._model(tensors)
        except Exception as e:
            print(f"[SegComp] Batch inference failed ({e}) — per-image fallback")
            for p in paths:
                results[p] = self.analyze(p, depth_maps.get(p))
            return results

        for path, (img_w, img_h), out in zip(paths, sizes, outs):
            result = results[path]
            canvas_area = img_h * img_w
            keep = (out["labels"] == _TV_PERSON_LABEL) & (out["scores"] >= _SEG_CONF)

            for mask_data in out["masks"][keep]:
                mask_np = mask_data[0].cpu().numpy() > 0.5

                if int(mask_np.sum()) < int(0.0008 * canvas_area):
                    continue

                result.has_person = True
                result.subject_masks.append(mask_np)

                depth_map = depth_maps.get(path)
                if depth_map is not None:
                    dmap_resized = depth_map
                    if depth_map.shape != (img_h, img_w):
                        from PIL import Image as _PIL_img2
                        dmap_resized = np.array(
                            _PIL_img2.fromarray(depth_map).resize((img_w, img_h), _PIL_img2.BILINEAR)
                        )
                    person_depths = dmap_resized[mask_np]
                    mean_depth    = float(person_depths.mean()) if len(person_depths) else 128.0

                    if mean_depth < _FOREGROUND_DEPTH_MAX:
                        result.foreground_mask = (
                            mask_np if result.foreground_mask is None
                            else result.foreground_mask | mask_np
                        )
                    elif _MIDGROUND_DEPTH_MIN <= mean_depth <= _MIDGROUND_DEPTH_MAX:
                        result.midground_mask = (
                            mask_np if result.midground_mask is None
                            else result.midground_mask | mask_np
                        )

            if result.midground_mask is not None and result.foreground_mask is not None:
                mid_frac = float(result.midground_mask.sum()) / canvas_area
                fg_frac  = float(result.foreground_mask.sum()) / canvas_area
                if 0.05 < mid_frac < 0.45 and 0.03 < fg_frac < 0.25:
                    result.is_ots_portrait = True
                    result.comp_override   = _OTS_PORTRAIT_COMP

        return results

    def unload(self):
        self._model = None
        self._ready = False
        try:
            import torch, gc as _gc
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            _gc.collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ChiaroscuroHead
# ---------------------------------------------------------------------------

_STRUCT_HEAD_PATH  = str(Path("models/vision_probe"))
_CHIAROSCURO_LUM   = 45.0   # scene mean luminance below this → candidate
_BIMODAL_STD_MIN   = 28.0   # luminance std above this confirms bimodal shadow/highlight

class ChiaroscuroHead:
    """
    Structural vision encoder + luminance bimodality to detect intentional chiaroscuro lighting.
    When active, deactivates low-luminance penalties for the image.
    """

    def __init__(self):
        self._model     = None
        self._processor = None
        self._ready     = False

    def load(self) -> bool:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(_STRUCT_HEAD_PATH, local_files_only=True)
            self._model     = AutoModel.from_pretrained(_STRUCT_HEAD_PATH, local_files_only=True).to(device).eval()
            self._device    = device
            self._ready     = True
            print(f"[ChiaroscuroHead] Structural encoder loaded on {device}")
            return True
        except Exception as e:
            print(f"[ChiaroscuroHead] Load failed: {e}")
            return False

    def is_chiaroscuro(self, path: str, mean_lum: float, std_lum: float) -> bool:
        """
        Return True when image shows intentional chiaroscuro:
          - Dark scene (mean_lum < 45) with high contrast (std > 28)
          - Structural encoder patch variance confirms strong edge energy
        """
        if not self._ready:
            return False

        # Fast luminance gate
        if mean_lum >= _CHIAROSCURO_LUM:
            return False
        if std_lum < _BIMODAL_STD_MIN:
            return False

        # Edge energy: high spatial variance in patch tokens = strong edges
        try:
            import torch
            from PIL import Image
            img    = Image.open(path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs)
            patch_tokens = out.last_hidden_state[:, 1:, :]   # drop CLS
            patch_var = float(patch_tokens.var(dim=1).mean().cpu())
            return patch_var > 0.15
        except Exception as e:
            print(f"[ChiaroscuroHead] Encoder inference failed for {Path(path).name}: {e}")
            # Fall back to luminance+std heuristic only
            return mean_lum < 35.0 and std_lum > 35.0

    def score_batch(self, paths: list[str], lum_stats: list[tuple[float, float]]) -> list[bool]:
        """Return per-path chiaroscuro flags."""
        return [
            self.is_chiaroscuro(p, m, s)
            for p, (m, s) in zip(paths, lum_stats)
        ]

    def unload(self):
        self._model     = None
        self._processor = None
        self._ready     = False
        try:
            import torch, gc as _gc
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            _gc.collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_composition_analysis(
    paths: list[str],
    lum_stats: list[tuple[float, float]],
    progress=None,
    progress_start: float = 0.84,
    progress_end:   float = 0.95,
    eligible_paths: Optional[set] = None,
) -> dict:
    """
    Run DepthHead → SegCompositionAnalyzer → ChiaroscuroHead sequentially.
    Each is loaded, run, then unloaded before the next loads.

    Returns dict with keys:
        composition_overrides  dict[str, float]  — path → comp score override
        subject_masks          dict[str, list]   — path → list of bool H×W masks
        chiaroscuro_flags      dict[str, bool]   — path → is chiaroscuro
    """
    n = len(paths)
    composition_overrides: dict[str, float] = {}
    subject_masks_out:     dict[str, list]  = {}
    chiaroscuro_flags:     dict[str, bool]  = {}

    if n == 0:
        return {
            "composition_overrides": composition_overrides,
            "subject_masks":         subject_masks_out,
            "chiaroscuro_flags":     chiaroscuro_flags,
        }

    def _prog(frac: float, msg: str):
        if progress:
            try:
                progress(progress_start + frac * (progress_end - progress_start), msg)
            except Exception:
                pass

    # Burst dedup: restrict expensive per-image ops to eligible paths only.
    # Non-eligible cluster members inherit no composition result (no OTS override,
    # no subject masks) — acceptable since burst frames are near-identical.
    comp_paths = paths if eligible_paths is None else [p for p in paths if p in eligible_paths]
    comp_lum   = (
        lum_stats if eligible_paths is None
        else [lum_stats[i] for i, p in enumerate(paths) if p in eligible_paths]
    )
    n_comp = len(comp_paths)
    if n_comp < n:
        print(f"[comp] Burst dedup: {n - n_comp} cluster members skip depth/seg/chiaroscuro")

    # --- Depth maps (batch inference — single HF pipeline call for eligible paths) ---
    depth_maps: dict[str, Optional[np.ndarray]] = {p: None for p in paths}
    _prog(0.0, "Depth map estimation (batch)…")
    dh = DepthHead()
    if dh.load() and comp_paths:
        partial = dh.infer_batch(comp_paths)
        depth_maps.update(partial)
        _prog(0.10, f"Depth batch complete [{n_comp}/{n_comp}]")
    dh.unload()

    # --- Segmentation + composition (single GPU batch call) ---
    _prog(0.12, "Subject segmentation (GPU batch)…")
    sca = SegCompositionAnalyzer()
    if sca.load() and comp_paths:
        batch_results = sca.analyze_batch(comp_paths, depth_maps)
        for p, result in batch_results.items():
            if result.comp_override is not None:
                composition_overrides[p] = result.comp_override
            if result.subject_masks:
                subject_masks_out[p] = result.subject_masks
        _prog(0.72, f"Seg batch complete [{n_comp}/{n_comp}]")
    sca.unload()

    # --- Chiaroscuro (eligible paths only) ---
    _prog(0.75, "Chiaroscuro detection…")
    ch = ChiaroscuroHead()
    if ch.load() and comp_paths:
        flags = ch.score_batch(comp_paths, comp_lum)
        for p, flag in zip(comp_paths, flags):
            chiaroscuro_flags[p] = flag
        _prog(0.95, "Chiaroscuro done")
    ch.unload()

    _prog(1.0, "Composition analysis complete")
    return {
        "composition_overrides": composition_overrides,
        "subject_masks":         subject_masks_out,
        "chiaroscuro_flags":     chiaroscuro_flags,
    }
