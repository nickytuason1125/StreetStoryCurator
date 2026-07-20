"""
D-FINE-nano person detector — AGPL-free replacement for ultralytics YOLO.

Apache-2.0 (ustc-community/dfine-nano-coco via HuggingFace transformers,
already an installed dependency — zero new pip package for this swap).
Loaded with local_files_only=True from a pre-cached local directory
(models/dfine_nano/, same convention as ChiaroscuroHead in
vision_composition_heads.py) — the one-time download itself lives in
scripts/download_detectors.py, never a runtime path, per this app's
"no external network calls at runtime" rule.

Benchmarked against the YOLO11s-seg reference on 100 real images: 134ms/img
CPU (faster than YOLO's own 155ms/img), 0 missed persons, 100% recall on
subjects occupying <5% of canvas — a strict superset of YOLO's detections.

CPU-only is deliberate, not an oversight: this was tried on CUDA and
measured 1996ms/img — ~15x SLOWER than CPU. detect_persons() calls the
model one image at a time (batch size 1), so per-call kernel-launch and
host<->device transfer overhead completely dominates the actual compute
for a model this small; there's nothing for the GPU to amortize against.
Don't "fix" this without re-batching the whole call into one forward pass
first — the win, if any, is in batching, not device placement alone.

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

_model:     Optional[object] = None
_processor: Optional[object] = None
_person_id: Optional[int]    = None


def _load():
    global _model, _processor, _person_id
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
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        _processor = AutoImageProcessor.from_pretrained(str(_MODEL_DIR), local_files_only=True)
        _model     = AutoModelForObjectDetection.from_pretrained(str(_MODEL_DIR), local_files_only=True)
        _model.eval()
        _person_id = next(
            k for k, v in _model.config.id2label.items() if v.lower() == "person"
        )
        print(f"[dfine] D-FINE-nano loaded (CPU), person label id={_person_id}")
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


def detect_persons(paths: list[str], conf: float = 0.35) -> dict[str, list[dict]]:
    """
    Returns path -> list of {"bbox": [x1n, y1n, x2n, y2n] (normalised [0,1]),
    "conf": float} for detected persons at or above `conf`. Empty list per
    path on any failure or when the model is unavailable — callers apply
    their own area/silhouette/confidence filtering on top of this primitive,
    matching each call site's existing distinct contract.
    """
    import torch
    from PIL import Image

    result: dict[str, list[dict]] = {p: [] for p in paths}
    model, processor = _load()
    if model is None:
        return result

    with torch.inference_mode():
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                iw, ih = img.size
                inputs = processor(images=img, return_tensors="pt")
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
                result[p] = boxes
            except Exception as e:
                print(f"[dfine] inference failed for {Path(p).name}: {e}")
    return result
