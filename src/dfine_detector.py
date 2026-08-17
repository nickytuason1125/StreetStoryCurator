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
_CHUNK     = 16   # FIXED images per forward pass — see _chunk_size (determinism)


def _chunk_size(n: int) -> int:
    """Images per forward pass — FIXED, deliberately not RAM-derived.

    This was briefly sized from free RAM (the batch used to hold 32 fully
    decoded frames, ~2.3 GB). That made the batch COMPOSITION vary run to run,
    and detection is sensitive to it at the margin: two identical culls of the
    same 514 photos disagreed on 47 of them, because chunk=8 one run and
    chunk=10 the next flipped borderline person detections, which changed the
    scoring formula each photo routed through. Grades must be reproducible.

    The memory reason is gone: draft-decoding to 640px cut each image from
    ~24 MB to ~1.2 MB, so a fixed batch of 16 costs ~20 MB — nothing worth
    trading determinism for. FRAMEGRADE_DFINE_CHUNK still overrides.
    """
    import os as _os
    _env = _os.environ.get("FRAMEGRADE_DFINE_CHUNK")
    if _env:
        try:
            return max(1, min(int(_env), max(n, 1)))
        except ValueError:
            pass
    return max(1, min(_CHUNK, max(n, 1)))


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



# Draft-decode is revertible: FRAMEGRADE_DFINE_DRAFT=0 restores full-resolution
# decoding. It is 3.8x faster (690 -> 182 ms/img) because the processor resizes
# everything to 640x640 anyway, but it does change the pixels the detector sees,
# which moved 3 of 135 borderline grades in an A/B. Flag exists so that can be
# re-tested on any shoot without a code change.
def _draft_enabled() -> bool:
    import os as _os
    return _os.environ.get("FRAMEGRADE_DFINE_DRAFT", "1").strip() != "0"


def _open_rgb(path: str):
    """Decode ANY supported file to RGB, RAW included.

    RAW was silently unsupported here: PIL cannot open .RW2/.ARW/.DNG, so every
    RAW image raised, fell through to an empty detection, and was recorded as
    person_detected=False. That routes it to the empty-scene scoring formula and
    skips the quality model — i.e. an entire RAW library was graded as if it
    contained no people. raw_support.extract_embedded_preview is the same
    embedded-JPEG path encode_worker already uses (never demosaics, memory-safe).
    """
    from PIL import Image
    import os as _os
    try:
        from raw_support import RAW_EXTS, load_rgb
    except Exception:
        RAW_EXTS, load_rgb = frozenset(), None
    if _os.path.splitext(path)[1].lower() in RAW_EXTS:
        if load_rgb is None:
            return None
        # load_rgb falls back to a demosaic when the body embeds no preview (or
        # only a navigation-sized one). Using extract_embedded_preview alone
        # meant those RAWs returned None here and were recorded as
        # person_detected=False — the same silent "no people" bug this function
        # was written to fix, just for a different subset of cameras.
        img, src = load_rgb(path, "RGB")
        if img is None:
            print(f"[dfine] unreadable RAW, skipping: {_os.path.basename(path)}")
        return img
    src = Image.open(path)
    if _draft_enabled():
        try: src.draft("RGB", (640, 640))
        except Exception: pass
    img = src.convert("RGB")
    try: src.close()
    except Exception: pass
    return img


def detect_persons_from_arrays(items, conf: float = 0.35) -> dict:
    """Detect persons in ALREADY-DECODED images: [(key, HWC uint8 array), ...].

    Exists so the pipeline can decode each photo ONCE and feed both this
    detector and the quality model from the same buffer. Previously every photo
    was decoded three times per cull (gate at full res for blur variance,
    detection at 640px, quality at 512px), and decode is the single largest cost
    in a cull — 214-241 ms/img, against 5 ms for the quality model itself.

    The detector's processor resizes everything to 640x640 anyway, so accepting
    a 512px buffer costs a little input resolution. That can shift a borderline
    detection, which only changes WHICH SCORING FORMULA a photo routes through —
    it never touches the quality score itself. That is the deliberate trade:
    routing may move slightly, the technical score cannot.
    """
    import torch
    result: dict = {k: [] for k, _ in items}
    model, processor = _load()
    if model is None or not items:
        return result
    chunk = _chunk_size(len(items))
    with torch.inference_mode():
        for start in range(0, len(items), chunk):
            batch = items[start:start + chunk]
            try:
                imgs = [a for _, a in batch]
                wh = [(a.shape[1], a.shape[0]) for a in imgs]      # (w, h)
                inputs = processor(images=imgs, return_tensors="pt").to(_device)
                out = model(**inputs)
                sizes = torch.tensor([[h, w] for (w, h) in wh])
                dets = processor.post_process_object_detection(
                    out, target_sizes=sizes, threshold=conf)
                for (key, _), (iw, ih), det in zip(batch, wh, dets):
                    boxes = []
                    for box, score, label in zip(det["boxes"], det["scores"], det["labels"]):
                        if int(label) != _person_id:
                            continue
                        x1, y1, x2, y2 = box.tolist()
                        boxes.append({"bbox": [x1 / iw, y1 / ih, x2 / iw, y2 / ih],
                                      "conf": float(score)})
                    result[key] = boxes
                del inputs, out, dets, sizes
            except Exception as e:
                print(f"[dfine] array batch failed ({e}) — chunk left empty")
    return result


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
            img = _open_rgb(p)
            if img is None:
                return []
            iw, ih = img.size
            inputs = processor(images=img, return_tensors="pt").to(_device)
            img.close()          # free the decoded frame before the forward pass
            del img
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

    _chunk = _chunk_size(len(paths))
    print(f"[dfine] detecting over {len(paths)} images, chunk={_chunk}")
    with torch.inference_mode():
        for start in range(0, len(paths), _chunk):
            batch_paths = paths[start:start + _chunk]
            try:
                # `with Image.open(...)` closes the file handle as soon as the
                # RGB copy exists — the old `Image.open(p).convert("RGB")` left
                # one dangling handle per image until GC.
                imgs, batch_paths_ok = [], []
                for p in batch_paths:
                    im = _open_rgb(p)          # handles RAW + draft decode
                    if im is not None:
                        imgs.append(im); batch_paths_ok.append(p)
                if not imgs:
                    continue
                batch_paths = batch_paths_ok
                wh = [im.size for im in imgs]      # (w, h) — needed after imgs is freed
                inputs = processor(images=imgs, return_tensors="pt").to(_device)
                # The decoded frames exist only to build `inputs` (the processor
                # resizes to the model's fixed input size) and to read their
                # dimensions, which `wh` now holds. Release them BEFORE the
                # forward pass so the batch's full-resolution RGB buffers and the
                # model's activations are never resident at the same time.
                for _im in imgs:
                    try: _im.close()
                    except Exception: pass
                del imgs
                out = model(**inputs)
                sizes = torch.tensor([[h, w] for (w, h) in wh])
                dets = processor.post_process_object_detection(
                    out, target_sizes=sizes, threshold=conf,
                )
                for p, (iw, ih), det in zip(batch_paths, wh, dets):
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
                # Drop this chunk's tensors before the next window is decoded,
                # so peak stays O(chunk) rather than creeping across chunks.
                del inputs, out, dets, sizes, wh
            except Exception as e:
                print(f"[dfine] batch failed ({e}) — falling back to per-image for this chunk")
                for p in batch_paths:
                    result[p] = _detect_one(p)
    return result
