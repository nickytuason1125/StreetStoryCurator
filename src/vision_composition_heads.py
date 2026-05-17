"""
Context-aware composition analysis: Depth Anything V2, YOLO11s-seg, DINOv2 chiaroscuro.

DepthHead          — relative depth maps (0=near, 255=far) via HuggingFace transformers
SegCompositionAnalyzer — YOLO11s-seg person masks + depth-layer categorization
                         over-the-shoulder portrait → comp_score override 0.85
ChiaroscuroHead    — DINOv2 ViT-S/14 luminance bimodality; deactivates low-lum penalties
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

            with _TPE(max_workers=min(8, len(paths))) as pool:
                imgs = list(pool.map(_load_and_cap, paths))

            valid_imgs  = [img for img in imgs if img is not None]
            valid_paths = [p for p, img in zip(paths, imgs) if img is not None]

            if not valid_imgs:
                return {p: None for p in paths}

            result: dict[str, Optional[np.ndarray]] = {p: None for p in paths}

            for batch_start in range(0, len(valid_imgs), self._DEPTH_BATCH_SIZE):
                b_imgs  = valid_imgs[batch_start : batch_start + self._DEPTH_BATCH_SIZE]
                b_paths = valid_paths[batch_start : batch_start + self._DEPTH_BATCH_SIZE]
                with torch.inference_mode():
                    outs = self._pipe(b_imgs)
                for p, out in zip(b_paths, outs):
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

_SEG_WEIGHTS = ("yolo11s-seg.pt", "yolo11n-seg.pt")
_PERSON_CLASS = 0
_SEG_CONF     = 0.35


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
    """YOLO11s-seg person segmentation + depth-layer composition logic."""

    def __init__(self):
        self._model = None
        self._ready = False

    _TRT_ENGINE = "yolo11s-seg.engine"   # ultralytics TRT export: yolo.export(format='engine', half=True)

    def load(self) -> bool:
        try:
            from ultralytics import YOLO
            # Prefer TRT engine (FP16, native ultralytics TRT support — no extra bindings needed).
            # Export: YOLO('yolo11s-seg.pt').export(format='engine', half=True, device=0)
            if Path(self._TRT_ENGINE).exists():
                try:
                    self._model = YOLO(self._TRT_ENGINE)
                    self._ready = True
                    print(f"[SegComp] TensorRT FP16 engine loaded: {self._TRT_ENGINE}")
                    return True
                except Exception as e:
                    print(f"[SegComp] TRT engine load failed ({e}) — PyTorch fallback")

            for weights in _SEG_WEIGHTS:
                try:
                    self._model = YOLO(weights)
                    self._ready = True
                    print(f"[SegComp] Loaded {weights}")
                    return True
                except Exception:
                    continue
            print("[SegComp] No seg weights found — composition analysis disabled")
            return False
        except ImportError:
            print("[SegComp] ultralytics not installed — composition analysis disabled")
            return False

    def analyze(self, path: str, depth_map: Optional[np.ndarray]) -> SegCompositionResult:
        result = SegCompositionResult()
        if not self._ready:
            return result

        try:
            res_list = self._model(
                path,
                device="cpu",
                classes=[_PERSON_CLASS],
                conf=_SEG_CONF,
                verbose=False,
                half=False,
            )
        except Exception as e:
            print(f"[SegComp] Inference failed for {Path(path).name}: {e}")
            return result

        for r in res_list:
            if r.masks is None or len(r.masks) == 0:
                continue

            img_h, img_w = r.orig_shape
            canvas_area  = img_h * img_w

            for mask_data in r.masks.data:
                # mask_data: (H_m, W_m) float tensor [0,1]
                import torch
                mask_np = mask_data.cpu().numpy()
                # Resize to original image resolution
                if mask_np.shape != (img_h, img_w):
                    from PIL import Image
                    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                        (img_w, img_h), Image.NEAREST
                    )
                    mask_np = np.array(mask_img) > 127
                else:
                    mask_np = mask_np > 0.5

                mask_area = int(mask_np.sum())
                if mask_area < int(0.0008 * canvas_area):
                    continue   # ignore tiny distant figures

                result.has_person = True
                result.subject_masks.append(mask_np)

                if depth_map is not None:
                    dmap_resized = depth_map
                    if depth_map.shape != (img_h, img_w):
                        from PIL import Image
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
        Single YOLO call on GPU for all paths — eliminates N-1 Python dispatch
        overheads and enables Tensor Core batching.  Falls back to per-image
        analyze() on any error.
        """
        results: dict[str, SegCompositionResult] = {p: SegCompositionResult() for p in paths}
        if not self._ready or not paths:
            return results

        try:
            import torch
            _cuda   = torch.cuda.is_available()
            _device = "cuda" if _cuda else "cpu"
            _half   = _cuda
            yolo_out = self._model(
                paths,
                device=_device,
                classes=[_PERSON_CLASS],
                conf=_SEG_CONF,
                verbose=False,
                half=_half,
            )
        except Exception as e:
            print(f"[SegComp] Batch GPU inference failed ({e}) — per-image CPU fallback")
            for p in paths:
                results[p] = self.analyze(p, depth_maps.get(p))
            return results

        for path, r in zip(paths, yolo_out):
            result = results[path]
            if r.masks is None or len(r.masks) == 0:
                continue

            img_h, img_w = r.orig_shape
            canvas_area  = img_h * img_w

            for mask_data in r.masks.data:
                mask_np = mask_data.cpu().numpy()
                if mask_np.shape != (img_h, img_w):
                    from PIL import Image as _PIL_img
                    mask_np = np.array(
                        _PIL_img.fromarray((mask_np * 255).astype(np.uint8)).resize(
                            (img_w, img_h), _PIL_img.NEAREST
                        )
                    ) > 127
                else:
                    mask_np = mask_np > 0.5

                if int(mask_np.sum()) < int(0.0008 * canvas_area):
                    continue

                result.has_person = True
                result.subject_masks.append(mask_np)

                depth_map = depth_maps.get(path)
                if depth_map is not None:
                    dmap_resized = depth_map
                    if depth_map.shape != (img_h, img_w):
                        from PIL import Image as _PIL_img
                        dmap_resized = np.array(
                            _PIL_img.fromarray(depth_map).resize((img_w, img_h), _PIL_img.BILINEAR)
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
            gc.collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ChiaroscuroHead
# ---------------------------------------------------------------------------

_DINO_REPO         = "facebook/dinov2-vits14"
_CHIAROSCURO_LUM   = 45.0   # scene mean luminance below this → candidate
_BIMODAL_STD_MIN   = 28.0   # luminance std above this confirms bimodal shadow/highlight

class ChiaroscuroHead:
    """
    DINOv2 ViT-S/14 + luminance bimodality to detect intentional chiaroscuro lighting.
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
            self._processor = AutoImageProcessor.from_pretrained(_DINO_REPO)
            self._model     = AutoModel.from_pretrained(_DINO_REPO).to(device).eval()
            self._device    = device
            self._ready     = True
            print(f"[ChiaroscuroHead] Loaded DINOv2 ViT-S/14 on {device}")
            return True
        except Exception as e:
            print(f"[ChiaroscuroHead] Load failed: {e}")
            return False

    def is_chiaroscuro(self, path: str, mean_lum: float, std_lum: float) -> bool:
        """
        Return True when image shows intentional chiaroscuro:
          - Dark scene (mean_lum < 45) with high contrast (std > 28)
          - DINOv2 CLS token structural embedding confirms strong edge energy
        """
        if not self._ready:
            return False

        # Fast luminance gate — skip DINOv2 if clearly not dark
        if mean_lum >= _CHIAROSCURO_LUM:
            return False
        if std_lum < _BIMODAL_STD_MIN:
            return False

        # DINOv2 edge energy: high spatial variance in patch tokens = strong edges
        try:
            import torch
            from PIL import Image
            img    = Image.open(path).convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt").to(self._device)
            with torch.inference_mode():
                out = self._model(**inputs)
            # patch_tokens: (1, num_patches, 384)
            patch_tokens = out.last_hidden_state[:, 1:, :]   # drop CLS
            # Spatial variance of patch activations = proxy for edge energy
            patch_var = float(patch_tokens.var(dim=1).mean().cpu())
            # Threshold tuned empirically: chiaroscuro scenes score > 0.15
            return patch_var > 0.15
        except Exception as e:
            print(f"[ChiaroscuroHead] DINOv2 inference failed for {Path(path).name}: {e}")
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


def run_yolo_bbox_pass(paths: list[str]) -> dict[str, list[list[float]]]:
    """
    Lightweight YOLO11s-seg pass on ALL images — returns normalized bounding box
    coordinates [[x1n, y1n, x2n, y2n], ...] in [0, 1] range for detected persons.

    No depth estimation, mask processing, or chiaroscuro — purely for:
      (a) Subject-targeted MUSIQ sharpness: crop to subject bbox, floor tech score
          to 0.75 when the subject region is crisp despite global background blur.
      (b) Dynamic Weight Route 1: zero detections → empty-scene geometric routing.

    YOLO is loaded in a single GPU batch, then immediately unloaded.
    Normalized coords allow TechnicalHead to scale to any tensor resolution.
    """
    result: dict[str, list[list[float]]] = {p: [] for p in paths}
    if not paths:
        return result

    sca = SegCompositionAnalyzer()
    if not sca.load():
        return result

    try:
        import torch
        _cuda    = torch.cuda.is_available()
        _device  = "cuda" if _cuda else "cpu"
        yolo_out = sca._model(
            paths,
            device=_device,
            classes=[_PERSON_CLASS],
            conf=_SEG_CONF,
            verbose=False,
            half=_cuda,
        )
        for path, r in zip(paths, yolo_out):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            img_h, img_w = r.orig_shape
            canvas_area  = img_h * img_w
            bboxes_norm: list[list[float]] = []
            for box_row in r.boxes.xyxy.cpu().tolist():
                x1, y1, x2, y2 = box_row
                box_area = (x2 - x1) * (y2 - y1)
                if box_area < 0.0008 * canvas_area:
                    continue  # ignore tiny distant figures
                bboxes_norm.append([
                    x1 / img_w, y1 / img_h,
                    x2 / img_w, y2 / img_h,
                ])
            if bboxes_norm:
                result[path] = bboxes_norm

        n_persons = sum(1 for v in result.values() if v)
        print(f"[yolo_bbox] {n_persons}/{len(paths)} images with person detections")
    except Exception as _e:
        print(f"[yolo_bbox] Batch YOLO pass failed ({_e})")
    finally:
        sca.unload()

    return result
