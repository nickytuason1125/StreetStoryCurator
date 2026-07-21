"""
D-FINE-nano person detector — AGPL-free replacement for ultralytics YOLO.

Apache-2.0 (ustc-community/dfine-nano-coco via HuggingFace transformers,
already an installed dependency — zero new pip package for this swap).
Loaded with local_files_only=True from a pre-cached local directory
(models/dfine_nano/, same convention as ChiaroscuroHead in
vision_composition_heads.py) — the one-time download itself lives in
scripts/download_detectors.py, never a runtime path, per this app's
"no external network calls at runtime" rule.

Benchmarked against the YOLO11s-seg reference on 100 real images: 0 missed
persons, 100% recall on subjects occupying <5% of canvas — a strict
superset of YOLO's detections.

GPU + batching: a first attempt at CUDA offload called the model one image
at a time (batch size 1) and measured ~15x SLOWER than CPU (1996ms/img) —
kernel-launch and host<->device transfer overhead completely dominated the
tiny per-call compute, with nothing for the GPU to amortize against. That
was a batching bug, not a fact about the architecture: D-FINE's image
processor resizes every input to a FIXED 640x640, so images stack into one
batched tensor with zero padding logic needed. Measured with real batching:
  CPU,  batch, slow processor: 152ms/img
  CPU,  batch, fast processor: 119ms/img  (use_fast=True — enabled below;
                                            transformers defaults it off
                                            for backward compat, not because
                                            it's worse)
  GPU,  batch, slow processor:  31ms/img
  GPU,  batch, fast processor:  19ms/img  <- ~7x faster than the original
                                            134ms/img unbatched-CPU baseline
Chunked (not one unbounded mega-batch): some call sites (grading's full
import batch via vision_grading_heads.py) can pass thousands of images in
one call — an unchunked batch at 640x640x3xfloat32 would be a multi-GB
input tensor. Chunk size follows the same bounded-batch convention already
used elsewhere in this codebase (DepthHead._DEPTH_BATCH_SIZE).

This module is the shared boxes-only primitive used by every site that only
needs bounding boxes (yolo_auditor.py, creative_director.py's
person_kill_switch, vision_grading_heads.py's _run_yolo_seg). The one site
needing real instance-segmentation masks (SegCompositionAnalyzer in
vision_composition_heads.py) uses torchvision Mask R-CNN instead, since
D-FINE is a detection-only architecture.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_ROOT      = Path(__file__).resolve().parent.parent
_MODEL_DIR = _ROOT / "models" / "dfine_nano"
_CHUNK     = 32   # images per forward pass — bounds peak memory regardless of call size

_model:     Optional[object] = None
_processor: Optional[object] = None
_person_id: Optional[int]    = None
_device:    str = "cpu"


def _load():
    global _model, _processor, _person_id, _device
    if _model is not None:
        return _model, _processor
    if not _MODEL_DIR.exists():
        print(
            "[dfine] models/dfine_nano/ not found — run "
            "scripts/download_detectors.py once (one-time setup, requires "
            "network) — person detection disabled"
        )
        return None, None
    try:
        import torch
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        # use_fast=True: torchvision-based preprocessing instead of PIL —
        # transformers defaults this off only for old-checkpoint backward
        # compatibility, not because it's slower; measured ~25% faster
        # preprocessing on this model regardless of device.
        _processor = AutoImageProcessor.from_pretrained(
            str(_MODEL_DIR), local_files_only=True, use_fast=True,
        )
        _model = AutoModelForObjectDetection.from_pretrained(str(_MODEL_DIR), local_files_only=True)
        _model.to(_device).eval()
        _person_id = next(
            k for k, v in _model.config.id2label.items() if v.lower() == "person"
        )
        print(f"[dfine] D-FINE-nano loaded on {_device}, person label id={_person_id}")
    except Exception as e:
        print(f"[dfine] load failed: {e}")
        _model, _processor, _person_id = None, None, None
    return _model, _processor


def is_available() -> bool:
    """True once the detector has been (successfully) loaded via _load()."""
    return _load()[0] is not None


def unload() -> None:
    global _model, _processor, _person_id
    _model, _processor, _person_id = None, None, None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def detect_persons(paths: list[str], conf: float = 0.35) -> dict[str, list[dict]]:
    """
    Returns path -> list of {"bbox": [x1n, y1n, x2n, y2n] (normalised [0,1]),
    "conf": float} for detected persons at or above `conf`. Empty list per
    path on any failure or when the model is unavailable — callers apply
    their own area/silhouette/confidence filtering on top of this primitive,
    matching each call site's existing distinct contract.

    Processes paths in fixed-size batches (_CHUNK), one real forward pass
    per batch — not one call per image. A single bad image fails only that
    image's chunk-mates fall back to empty, never the whole call.
    """
    import torch
    from PIL import Image

    result: dict[str, list[dict]] = {p: [] for p in paths}
    model, processor = _load()
    if model is None:
        return result

    def _detect_one(p: str) -> list[dict]:
        """Single-image fallback — no recursion, independent try/except per
        call so one corrupt image can never take down its chunk-mates or
        loop forever on repeated decode failure."""
        try:
            img = Image.open(p).convert("RGB")
            iw, ih = img.size
            inputs = processor(images=img, return_tensors="pt").to(_device)
            out = model(**inputs)
            det = processor.post_process_object_detection(
                out, target_sizes=torch.tensor([[ih, iw]]), threshold=conf,
            )[0]
            boxes: list[dict] = []
            for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                if int(label) != _person_id:
                    continue
                x1, y1, x2, y2 = box.tolist()
                boxes.append({
                    "bbox": [x1 / iw, y1 / ih, x2 / iw, y2 / ih],
                    "conf": float(score),
                })
            return boxes
        except Exception as e:
            print(f"[dfine] inference failed for {Path(p).name}: {e}")
            return []

    with torch.inference_mode():
        for start in range(0, len(paths), _CHUNK):
            batch_paths = paths[start:start + _CHUNK]
            try:
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=imgs, return_tensors="pt").to(_device)
                out = model(**inputs)
                sizes = torch.tensor([[im.size[1], im.size[0]] for im in imgs])
                dets = processor.post_process_object_detection(
                    out, target_sizes=sizes, threshold=conf,
                )
                for p, im, det in zip(batch_paths, imgs, dets):
                    iw, ih = im.size
                    boxes: list[dict] = []
                    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                        if int(label) != _person_id:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        boxes.append({
                            "bbox": [x1 / iw, y1 / ih, x2 / iw, y2 / ih],
                            "conf": float(score),
                        })
                    result[p] = boxes
            except Exception as e:
                print(f"[dfine] batch failed ({e}) — falling back to per-image for this chunk")
                for p in batch_paths:
                    result[p] = _detect_one(p)
    return result
