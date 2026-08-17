"""
Vision IQA Head — TOPIQ NR quality backbone.

UniQAHead: pyiqa 'topiq_nr' metric (CLIP-based no-reference IQA).
'uniqa' does not exist in pyiqa 0.1.x — topiq_nr is the correct replacement.

D-FINE-nano routing (src/dfine_detector.py, Apache-2.0 — replaces the former
ultralytics YOLO11s-seg here, AGPL-3.0):
  Route 1 (empty scene): 0 humans detected → score = sqrt(composition * lighting)
                          from SpecVLM aspect scores. Decouples Human/Culture penalty.
  Route 2 (layered frame): human in midground (bbox center_y 33–67%) + blurred
                            foreground (low Laplacian variance, bottom third of image)
                            → TOPIQ NR on subject crop only.
  Standard: TOPIQ NR on full resized image.

Speed design:
  - Images loaded in parallel (TurboJPEG via fast_ingestion.decode_one).
  - TOPIQ NR: GPU batch inference at 512×512, mini-batches of 8.
  - Route 2: per-image crop inference (variable crop size prevents batching).
  - D-FINE: CPU inference, per-image; returns per-image route decisions.
  - VRAM: single model — no sequential load/unload needed.
"""

from __future__ import annotations

import gc
import numpy as np
from pathlib import Path
from typing import List, Optional
import torch
import torchvision.transforms.functional as TF

_uniqa_singleton: Optional["UniQAHead"] = None


def release_iqa_models() -> None:
    """Evict UniQA singleton to free VRAM before Creative Mode loads LLMs."""
    global _uniqa_singleton
    if _uniqa_singleton is not None:
        try:
            _uniqa_singleton.unload()
        except Exception:
            pass
        _uniqa_singleton = None
    print("[vision_heads] IQA singleton released — VRAM freed")


def _purge_vram() -> None:
    gc.collect()
    # Guard on is_initialized() (NOT is_available()): empty_cache/ipc_collect will
    # INITIALIZE a CUDA context if one doesn't exist yet. This module now runs inside
    # the isolated iqa_worker subprocess (where CUDA is legitimately initialized), so
    # this must never create a context in a CUDA-free caller. See vram_manager.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# TOPIQ NR operating point on this install: across two audited sessions
# (135 + 25 photos, 2026-06) raw outputs sat in 0.31-0.46 for everything from
# sharp daylight frames to motion-blurred night shots — centered far below the
# 0.5-neutral the fusion gates assume (low_tech_clutter < 0.42, tech ceilings
# < 0.50), so "technical" was a near-constant penalty with little discrimination.
# Fixed affine recalibration (same constants every run — absolute, NOT batch-
# relative): centre the observed operating point (0.40) at 0.5 and widen the
# spread 2.5×. Retune _TOPIQ_CENTER if the camera/lens profile changes.
_TOPIQ_CENTER = 0.40
_TOPIQ_GAIN   = 2.5


def _batch_normalize(scores: np.ndarray) -> np.ndarray:
    """Fixed affine recalibration of raw TOPIQ NR scores (see constants above).

    Min-max batch stretching was removed earlier — it produced relative rather
    than absolute technical scores. This is NOT that: the mapping is identical
    for every batch, it only re-centres TOPIQ's compressed output range.
    """
    s = scores.astype(np.float32)
    return np.clip(0.50 + (s - _TOPIQ_CENTER) * _TOPIQ_GAIN, 0.05, 0.95)


def _load_images_parallel(
    image_paths: List[str],
    n_workers: int = 8,
    max_size: int = 512,
    pin: bool = False,
) -> List[Optional[torch.Tensor]]:
    """
    Decode images in parallel via TurboJPEG (JPEG) / PIL (other formats).
    Returns (C, H, W) float32 tensors capped at max_size on the long edge.

    pin=False by default: pinned (page-locked) memory cannot be paged out, so a
    wide list of pinned decodes is the worst thing to hold under RAM pressure.
    The streamed IQA path holds only one chunk at a time, so the tiny H2D-copy
    speedup pinning buys is not worth the non-swappable footprint. Callers that
    genuinely need pinned tensors can opt back in.
    """
    from fast_ingestion import decode_one
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _load_one(p: str) -> Optional[torch.Tensor]:
        t = decode_one(p, target_hw=None, pin=pin)
        if t is None:
            return None
        _, h, w = t.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            t = TF.resize(t, [int(h * scale), int(w * scale)], antialias=True)
        return t

    with _TPE(max_workers=min(n_workers, len(image_paths) or 1)) as pool:
        return list(pool.map(_load_one, image_paths))


def _iqa_chunk_size(n: int) -> int:
    """Decode-window size — FIXED, deliberately not RAM-derived.

    Deriving this from free RAM made the DECODE WINDOW vary run to run
    (chunk=226 one cull, chunk=187 the next), which changes TOPIQ's batch
    composition and therefore the float reduction order on the GPU. Two
    identical culls of the same 514 photos then disagreed on 76 of them —
    mostly noise (mean 0.019) but 5 crossed a grade threshold. A grading tool
    has to be reproducible, so the window is now constant.

    Memory is still bounded: peak decode RAM is O(chunk), and 256 images at
    ~5 MB each (512px float32) is ~1.3 GB worst case, which the streamed design
    already handled. FRAMEGRADE_IQA_CHUNK overrides for a memory-tight machine —
    at the cost of reproducibility across machines with different free RAM.
    """
    import os as _os
    _env = _os.environ.get("FRAMEGRADE_IQA_CHUNK")
    if _env:
        try:
            return max(16, min(int(_env), max(n, 1)))
        except ValueError:
            pass
    return max(16, min(256, max(n, 1)))


def _run_yolo_seg(
    image_paths: List[str],
) -> tuple:
    """
    Run D-FINE-nano (person class only) on all images — replaces ultralytics
    YOLO11s-seg (AGPL-3.0) with the Apache-2.0 shared detector in
    dfine_detector.py. Boxes-only (no masks — this function's name is a
    holdover; it never consumed segmentation masks, only boxes).

    Falls back gracefully to person_detected=True for all images when the
    detector is unavailable.

    Returns:
        person_detected_dict  path → bool
        subject_bboxes_dict   path → list[[x1n, y1n, x2n, y2n]] (normalised [0,1])
    """
    import dfine_detector

    person_detected: dict = {}
    subject_bboxes:  dict = {}

    if image_paths and not dfine_detector.is_available():
        print("[uniqa_head] D-FINE unavailable — all images route to standard UniQA")
        for p in image_paths:
            person_detected[p] = True   # safe default: treat as person present
        return person_detected, subject_bboxes

    detections = dfine_detector.detect_persons(image_paths, conf=0.55)

    for path in image_paths:
        boxes = detections.get(path, [])
        bboxes_norm = [
            det["bbox"] for det in boxes
            if (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]) >= 0.005
        ]
        if bboxes_norm:
            person_detected[path] = True
            subject_bboxes[path]  = bboxes_norm
        else:
            person_detected[path] = False

    n_person = sum(1 for v in person_detected.values() if v)
    print(f"[uniqa_head] D-FINE: {n_person}/{len(image_paths)} images with person detected")
    return person_detected, subject_bboxes


def _is_layered_frame(
    tensor: torch.Tensor,
    bboxes_norm: list,
) -> bool:
    """
    True when: person is in midground (bbox center_y 33–67%) AND foreground is blurred
    (Laplacian variance in the bottom third of the image < 500).

    OOF/bokeh foreground typically yields variance < 200; sharp foreground > 2000.
    """
    import torch.nn.functional as F

    midground = any(0.33 <= (b[1] + b[3]) / 2 <= 0.67 for b in bboxes_norm)
    if not midground:
        return False

    _, H, W  = tensor.shape
    fg_start = int(H * 0.67)
    fg_slice = tensor[:, fg_start:, :]
    weights  = torch.tensor([0.299, 0.587, 0.114], dtype=tensor.dtype)
    gray     = (fg_slice.cpu() * weights.view(3, 1, 1)).sum(0)
    lap_k    = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=tensor.dtype)
    lap      = F.conv2d(gray.unsqueeze(0).unsqueeze(0), lap_k.view(1, 1, 3, 3), padding=1).squeeze()
    return float(lap.var().item()) < 500.0


_OTS_PORTRAIT_COMP = 0.85   # over-the-shoulder composition score override


def _detect_framing_obstruction(bboxes_norm: list) -> tuple:
    """
    Identify a Framing Obstruction: a large, off-centre person dominating the frame
    edge while a smaller, more central subject person is visible behind them.

    Obstruction criteria (must satisfy BOTH):
      1. Largest detected person occupies > 30% of total frame area.
      2. Their horizontal centre is off-centre (center_x < 0.35 OR > 0.65).

    When an obstruction is found the remaining (smaller, central) persons are
    returned as the true subject bounding boxes.  UniQA will crop to these for
    sharpness evaluation — isolating the subject from the occluding body.

    Returns:
        (True,  subject_bboxes)   obstruction found; subject_bboxes to crop to
        (False, bboxes_norm)      no obstruction; caller should use original bboxes
    """
    if len(bboxes_norm) < 2:
        return False, bboxes_norm

    areas     = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes_norm]
    centers_x = [(b[0] + b[2]) / 2              for b in bboxes_norm]

    # Find the most dominant off-centre person
    obstruction_idx = None
    for i, (area, cx) in enumerate(zip(areas, centers_x)):
        if area > 0.30 and (cx < 0.35 or cx > 0.65):
            if obstruction_idx is None or area > areas[obstruction_idx]:
                obstruction_idx = i

    if obstruction_idx is None:
        return False, bboxes_norm

    subject_bbs = [b for i, b in enumerate(bboxes_norm) if i != obstruction_idx]
    if not subject_bbs:
        return False, bboxes_norm   # obstruction was the only person

    return True, subject_bbs


def _derive_ots_from_bboxes(
    image_paths: List[str],
    subject_bboxes: dict,
) -> dict:
    """
    Detect over-the-shoulder portraits from YOLO bboxes — no DepthHead required.

    Replaces SegCompositionAnalyzer + DepthHead OTS logic:
      Foreground proxy: largest bbox by area AND (area > 12% OR near top/bottom edge).
      Midground proxy: any other detected person with center_y ∈ [0.30, 0.70].
      OTS fires when: fg_area ∈ (0.03, 0.25) AND mg_area ∈ (0.05, 0.45) —
      same fractions as SegCompositionAnalyzer to preserve behaviour.
    """
    overrides: dict = {}
    for path in image_paths:
        bboxes = subject_bboxes.get(path, [])
        if len(bboxes) < 2:
            continue

        areas   = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        max_idx = int(np.argmax(areas))
        max_area = areas[max_idx]
        fg_bbox  = bboxes[max_idx]
        fg_cy    = (fg_bbox[1] + fg_bbox[3]) / 2

        # Foreground proxy: large person OR near image boundary (top/bottom)
        if not (max_area > 0.12 or fg_cy < 0.22 or fg_cy > 0.78):
            continue

        mg_area = sum(
            areas[i]
            for i, b in enumerate(bboxes)
            if i != max_idx and 0.30 <= (b[1] + b[3]) / 2 <= 0.70
        )
        if mg_area < 0.001:
            continue

        if 0.03 < max_area < 0.25 and 0.05 < mg_area < 0.45:
            overrides[path] = _OTS_PORTRAIT_COMP

    return overrides


class UniQAHead:
    """
    TOPIQ NR quality backbone (pyiqa 'topiq_nr').

    CLIP-based no-reference IQA covering both technical quality and aesthetic appeal.
    Replaces the invalid 'uniqa' metric name (not present in pyiqa 0.1.x).

    YOLO11s-seg routing selects the input region per image:
      Route 1 — empty scene (0 humans detected): geometric blend of SpecVLM aspect scores.
      Route 2 — layered frame: TOPIQ NR on YOLO subject crop.
      Standard — everything else: TOPIQ NR on full resized image (batched).
    """

    _METRIC_NAME = "topiq_nr"
    _BATCH_SIZE  = 16   # was 8 — doubled; 130 images = 9 batches instead of 17
    _INPUT_SIZE  = 384  # was 512 — TOPIQ NR trains at 224; 384 saves ~44% compute

    def __init__(self) -> None:
        self._model  = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _timed_create(metric_name: str, device: str, timeout: int = 120):
        import os as _os, pyiqa
        # In the isolated iqa_worker subprocess, load in the MAIN thread. Creating a
        # CUDA model in a background thread and then running inference from the main
        # thread faults with 0xC0000005 in a fresh process (CUDA primary-context
        # thread affinity). The subprocess has its own outer timeout (subprocess.run),
        # so the threaded load-watchdog is unnecessary there. In-process callers keep
        # the watchdog (no env var set).
        if _os.environ.get("IQA_MAIN_THREAD_LOAD") == "1":
            try:
                return pyiqa.create_metric(metric_name, device=device)
            except Exception as e:
                print(f"[uniqa_head] {metric_name} error: {e} — skip")
                return None

        import threading
        _result: list = [None]
        _err:    list = [None]

        def _worker():
            try:
                _result[0] = pyiqa.create_metric(metric_name, device=device)
            except Exception as e:
                _err[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            print(f"[uniqa_head] {metric_name} timed out after {timeout}s — skip")
            return None
        if _err[0] is not None:
            print(f"[uniqa_head] {metric_name} error: {_err[0]} — skip")
            return None
        return _result[0]

    def load(self) -> None:
        self._model = self._timed_create(self._METRIC_NAME, self._device)
        if self._model is not None:
            # torch.compile uses the inductor backend, which REQUIRES Triton.
            # Triton has no Windows wheels, and torch.compile is lazy — it compiles
            # on the first forward pass, so a missing-Triton failure escapes a
            # try/except placed here and instead raises on EVERY image at inference
            # time (BackendCompilerFailed), collapsing all IQA scores to 0.5 and
            # stalling the grade. Only compile when Triton is importable; otherwise
            # run eager (fully functional, just not graph-optimised).
            import importlib.util as _ilu
            _has_triton = _ilu.find_spec("triton") is not None
            if self._device == "cuda" and _has_triton:
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    print(f"[uniqa_head] {self._METRIC_NAME} compiled (reduce-overhead)")
                except Exception as _ce:
                    print(f"[uniqa_head] torch.compile skipped: {_ce}")
            else:
                # Global safety net: if anything else triggers dynamo, fall back to
                # eager instead of hard-failing.
                try:
                    import torch._dynamo as _td
                    _td.config.suppress_errors = True
                except Exception:
                    pass
                if self._device == "cuda":
                    print(f"[uniqa_head] torch.compile skipped — Triton unavailable; running eager")
            print(f"[uniqa_head] {self._METRIC_NAME} loaded on {self._device}")
        else:
            print(f"[uniqa_head] {self._METRIC_NAME} unavailable")

    def score_all(
        self,
        image_paths:             List[str],
        vlm_breakdowns:          Optional[List[dict]] = None,
        progress=None,
        progress_start:          float = 0.60,
        progress_end:            float = 0.83,
        person_detected_in:      Optional[dict] = None,
        subject_bboxes_in:       Optional[dict] = None,
        framing_obstruction_in:  Optional[dict] = None,
    ) -> tuple:
        """
        Score images via YOLO routing + UniQA, decoding in RAM-bounded chunks.

        Route 1 images (empty scene) use SpecVLM aspect scores — no model inference.
        Route 2 images (layered frame) run UniQA on the YOLO subject crop.
        Standard images run UniQA on the full resized image (batched on GPU).

        Memory: images are decoded one chunk at a time (see _iqa_chunk_size) and
        the chunk's tensors are freed before the next window, so peak RAM is
        O(chunk) — a 10 000-image import uses the same working set as 200. The
        old path materialised ALL decodes up front (O(N)) and OOM'd on large
        imports. Scores are identical: _batch_normalize is a fixed affine, so a
        photo's score does not depend on which images share its chunk.

        person_detected_in / subject_bboxes_in: pre-computed YOLO results from
        run_vision_heads(); when provided, the internal _run_yolo_seg() call is
        skipped, eliminating the duplicate YOLO pass.

        Returns:
            quality_norm     np.ndarray (N,)  batch-normalised [0,1]
            person_detected  dict  path → bool
            subject_bboxes   dict  path → list[bbox]
        """
        _p = progress or (lambda f, d: None)
        n  = len(image_paths)
        device = torch.device(self._device)
        S      = self._INPUT_SIZE

        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError("TOPIQ NR (topiq_nr) unavailable — run model prefetch or check pyiqa install")

        # ── YOLO routing detections (paths only — never materialise all tensors) ─
        if person_detected_in is not None and subject_bboxes_in is not None:
            person_detected = person_detected_in
            subject_bboxes  = subject_bboxes_in
            print(f"[uniqa_head] YOLO skipped — pre-computed detections reused ({n} images)")
        else:
            _p(progress_start, f"YOLO routing — {n} images…")
            person_detected, subject_bboxes = _run_yolo_seg(image_paths)

        _fo_map     = framing_obstruction_in or {}
        quality_raw = [0.5] * n
        routes      = [0] * n   # 0=standard, 1=empty-scene, 2=layered-frame/FO
        n_failed    = 0

        # ── Streamed decode: process images in RAM-bounded windows so peak
        #    memory is O(chunk), not O(N). Route 1 (no GPU inference), Route 0
        #    (full-frame TOPIQ) and Route 2 (subject-crop TOPIQ) all run within
        #    the chunk on the same GPU mini-batch size, then the chunk's decoded
        #    tensors are freed before the next window. ────────────────────────
        chunk = _iqa_chunk_size(n)
        print(f"[uniqa_head] Streamed IQA: {n} images, chunk={chunk} "
              f"(peak decode RAM bounded, not O(N))")
        # Time decode vs inference separately: the IQA stage is 58% of a cull's
        # runtime, and nothing recorded whether that is image decoding or GPU
        # work. Optimising the wrong one wastes the effort.
        import time as _tprof
        _t_decode = 0.0
        _t_infer  = 0.0
        done = 0
        for c0 in range(0, n, chunk):
            c_idx     = list(range(c0, min(c0 + chunk, n)))
            c_paths   = [image_paths[i] for i in c_idx]
            _td0 = _tprof.monotonic()
            c_tensors = _load_images_parallel(c_paths, pin=False)
            _t_decode += _tprof.monotonic() - _td0

            # Routing + Route-1 fill for this chunk.
            for pos, i in enumerate(c_idx):
                t = c_tensors[pos]
                if t is None:
                    n_failed += 1
                path       = image_paths[i]
                has_person = person_detected.get(path, True)
                bboxes     = subject_bboxes.get(path, [])
                is_fo      = bool(_fo_map.get(path, False))
                if not has_person:
                    routes[i] = 1
                    bd    = vlm_breakdowns[i] if vlm_breakdowns else {}
                    comp  = float(bd.get("Composition", 0.5))
                    light = float(bd.get("Lighting",    0.5))
                    quality_raw[i] = float(np.sqrt(max(comp, 0.01) * max(light, 0.01)))
                elif is_fo or (bboxes and t is not None and _is_layered_frame(t, bboxes)):
                    routes[i] = 2
                else:
                    routes[i] = 0

            # ── Route 0 (standard): batch UniQA on full images ────────────────
            std_pos = [pos for pos, i in enumerate(c_idx)
                       if routes[i] == 0 and c_tensors[pos] is not None]
            for bstart in range(0, len(std_pos), self._BATCH_SIZE):
                b_pos = std_pos[bstart : bstart + self._BATCH_SIZE]
                if not b_pos:
                    break
                resized = [TF.resize(c_tensors[pos], [S, S], antialias=True) for pos in b_pos]
                batch_t = torch.stack(resized).to(device, non_blocking=True)
                _ti0 = _tprof.monotonic()
                try:
                    with torch.inference_mode():
                        out = self._model(batch_t)
                    out = out.squeeze(-1) if out.dim() > 1 else out
                    for j, pos in enumerate(b_pos):
                        quality_raw[c_idx[pos]] = float(out[j].item())
                    _t_infer += _tprof.monotonic() - _ti0
                    del batch_t
                except Exception as e:
                    print(f"[uniqa_head] Std batch failed ({e}) — per-image fallback")
                    del batch_t
                    for pos in b_pos:
                        try:
                            inp = TF.resize(c_tensors[pos], [S, S], antialias=True).unsqueeze(0).to(device)
                            with torch.inference_mode():
                                s = self._model(inp)
                            quality_raw[c_idx[pos]] = float(s.item() if hasattr(s, "item") else float(s))
                        except Exception:
                            import traceback as _tb_pimg
                            print(f"[uniqa_head] Per-image fallback FULL TRACEBACK (i={c_idx[pos]}):")
                            _tb_pimg.print_exc()
                            quality_raw[c_idx[pos]] = 0.5

            # ── Route 2 (layered frame): batched UniQA on subject crops ──────
            # Crops resize to the same S×S as the standard route, so they batch
            # identically. Crop geometry (2026-06-11): tight person crops upscaled
            # to S×S read as mush and TOPIQ floored them; pad the subject box 15%
            # with context and blend 50/50 with the full-frame score so a sharp
            # frame with a small subject is not technically slandered by its crop.
            r2_pos = [pos for pos, i in enumerate(c_idx)
                      if routes[i] == 2 and c_tensors[pos] is not None]
            r2_crops: list = []
            r2_full:  list = []
            for pos in r2_pos:
                t      = c_tensors[pos]
                bboxes = subject_bboxes[image_paths[c_idx[pos]]]
                _, H, W = t.shape
                x1 = min(b[0] for b in bboxes); y1 = min(b[1] for b in bboxes)
                x2 = max(b[2] for b in bboxes); y2 = max(b[3] for b in bboxes)
                _pad_x = 0.15 * (x2 - x1); _pad_y = 0.15 * (y2 - y1)
                x1 = max(0, int((x1 - _pad_x) * W)); y1 = max(0, int((y1 - _pad_y) * H))
                x2 = min(W, int((x2 + _pad_x) * W)); y2 = min(H, int((y2 + _pad_y) * H))
                crop = t[:, y1:y2, x1:x2] if (y2 - y1) >= 32 and (x2 - x1) >= 32 else t
                r2_crops.append(TF.resize(crop, [S, S], antialias=True))
                r2_full.append(TF.resize(t,    [S, S], antialias=True))

            for rb in range(0, len(r2_pos), self._BATCH_SIZE):
                cpos   = r2_pos[rb : rb + self._BATCH_SIZE]
                ccrops = r2_crops[rb : rb + self._BATCH_SIZE]
                cfull  = r2_full[rb : rb + self._BATCH_SIZE]
                if not cpos:
                    break
                batch_t = torch.stack(ccrops + cfull).to(device, non_blocking=True)
                try:
                    with torch.inference_mode():
                        out = self._model(batch_t)
                    out = out.squeeze(-1) if out.dim() > 1 else out
                    _nc = len(cpos)
                    for j, pos in enumerate(cpos):
                        # 50/50 subject-crop × full-frame blend (see comment above)
                        quality_raw[c_idx[pos]] = 0.5 * float(out[j].item()) + 0.5 * float(out[_nc + j].item())
                except Exception as e:
                    print(f"[uniqa_head] Route2 batch failed ({e}) — per-crop fallback")
                    for j, pos in enumerate(cpos):
                        try:
                            with torch.inference_mode():
                                s_c = self._model(ccrops[j].unsqueeze(0).to(device))
                                s_f = self._model(cfull[j].unsqueeze(0).to(device))
                            quality_raw[c_idx[pos]] = 0.5 * float(s_c.item()) + 0.5 * float(s_f.item())
                        except Exception:
                            import traceback as _tb_r2
                            print(f"[uniqa_head] Route2 crop FULL TRACEBACK "
                                  f"({Path(image_paths[c_idx[pos]]).name}):")
                            _tb_r2.print_exc()
                            quality_raw[c_idx[pos]] = 0.5
                finally:
                    del batch_t

            # Free this chunk's decoded tensors before the next window — this is
            # what keeps peak RAM flat regardless of total image count.
            done += len(c_idx)
            del c_tensors, r2_crops, r2_full
            gc.collect()
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            _p(progress_start + (progress_end - progress_start) * done / max(n, 1),
               f"UniQA {done}/{n} images…")

        print(f"[uniqa_head] TIME  decode {_t_decode:6.1f}s   TOPIQ inference {_t_infer:6.1f}s"
              f"   ({_t_decode/max(n,1)*1000:.0f} + {_t_infer/max(n,1)*1000:.0f} ms/img)", flush=True)
        n_r1 = routes.count(1); n_r2 = routes.count(2); n_std = routes.count(0)
        print(f"[uniqa_head] Routes: {n_r1} empty-scene, {n_r2} layered-frame, {n_std} standard")
        if n_failed:
            print(f"[uniqa_head] {n_failed}/{n} images failed to decode — 0.5 fallback")

        # Recalibrate ONLY genuine TOPIQ outputs (routes 0/2). Route-1 values
        # are VLM aspect blends already on the 0-1 grading scale, and exact-0.5
        # entries are failure fallbacks — both must stay untouched (a real
        # TOPIQ score of exactly 0.500000 is practically impossible).
        quality_arr  = np.array(quality_raw, dtype=np.float32)
        _topiq_mask  = np.array([r != 1 for r in routes], dtype=bool) & (quality_arr != 0.5)
        quality_norm = quality_arr.copy()
        quality_norm[_topiq_mask] = _batch_normalize(quality_arr[_topiq_mask])
        print(
            f"[uniqa_head] UniQA: min={quality_norm.min():.3f}  "
            f"max={quality_norm.max():.3f}  mean={quality_norm.mean():.3f}  "
            f"(TOPIQ-calibrated: {int(_topiq_mask.sum())}/{n})"
        )
        return quality_norm, person_detected, subject_bboxes


    def score_tensors(self, paths, tensors, vlm_breakdowns=None,
                      person_detected=None, subject_bboxes=None,
                      framing_obstruction=None):
        """Score ALREADY-DECODED images. Returns raw (uncalibrated) quality.

        Same routing and maths as score_all's inner loop, but it never touches
        the filesystem — the caller owns the decode. That is the whole point of
        the streaming path: one decode feeds both the detector and this model
        instead of each re-reading every photo.
        """
        device = torch.device(self._device)
        S = self._INPUT_SIZE
        if self._model is None:
            self.load()
        n = len(paths)
        quality_raw = [0.5] * n
        routes = [0] * n
        person_detected = person_detected or {}
        subject_bboxes = subject_bboxes or {}
        _fo = framing_obstruction or {}

        for i, path in enumerate(paths):
            t = tensors[i]
            has_person = person_detected.get(path, True)
            bboxes = subject_bboxes.get(path, [])
            if not has_person:
                routes[i] = 1
                bd = vlm_breakdowns[i] if vlm_breakdowns else {}
                comp = float(bd.get("Composition", 0.5))
                light = float(bd.get("Lighting", 0.5))
                quality_raw[i] = float(np.sqrt(max(comp, 0.01) * max(light, 0.01)))
            elif bool(_fo.get(path, False)) or (bboxes and t is not None
                                                and _is_layered_frame(t, bboxes)):
                routes[i] = 2
            else:
                routes[i] = 0

        std_pos = [i for i in range(n) if routes[i] == 0 and tensors[i] is not None]
        for b0 in range(0, len(std_pos), self._BATCH_SIZE):
            bp = std_pos[b0:b0 + self._BATCH_SIZE]
            if not bp:
                break
            batch_t = torch.stack([TF.resize(tensors[i], [S, S], antialias=True)
                                   for i in bp]).to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    out = self._model(batch_t)
                out = out.squeeze(-1) if out.dim() > 1 else out
                for j, i in enumerate(bp):
                    quality_raw[i] = float(out[j].item())
            except Exception as e:
                print(f"[uniqa_head] streaming std batch failed ({e})")
            finally:
                del batch_t

        r2 = [i for i in range(n) if routes[i] == 2 and tensors[i] is not None]
        for b0 in range(0, len(r2), self._BATCH_SIZE):
            bp = r2[b0:b0 + self._BATCH_SIZE]
            crops, fulls = [], []
            for i in bp:
                t = tensors[i]
                bb = subject_bboxes.get(paths[i], [])
                _, H, W = t.shape
                x1 = min(b[0] for b in bb); y1 = min(b[1] for b in bb)
                x2 = max(b[2] for b in bb); y2 = max(b[3] for b in bb)
                px, py = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
                X1 = max(0, int((x1 - px) * W)); Y1 = max(0, int((y1 - py) * H))
                X2 = min(W, int((x2 + px) * W)); Y2 = min(H, int((y2 + py) * H))
                crop = t[:, Y1:Y2, X1:X2] if (Y2 - Y1) >= 32 and (X2 - X1) >= 32 else t
                crops.append(TF.resize(crop, [S, S], antialias=True))
                fulls.append(TF.resize(t, [S, S], antialias=True))
            if not crops:
                break
            batch_t = torch.stack(crops + fulls).to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    out = self._model(batch_t)
                out = out.squeeze(-1) if out.dim() > 1 else out
                nc = len(bp)
                for j, i in enumerate(bp):
                    quality_raw[i] = 0.5 * float(out[j].item()) + 0.5 * float(out[nc + j].item())
            except Exception as e:
                print(f"[uniqa_head] streaming route2 batch failed ({e})")
            finally:
                del batch_t
        return quality_raw, routes

    def unload(self) -> None:
        self._model = None
        _purge_vram()


def _run_vision_heads_streaming(
    image_paths, image_embeddings, prompt_embedding, clip_scores,
    genre_ref_embs=None, lum_stats=None, progress=None,
    comp_eligible_paths=None, vlm_breakdowns=None,
) -> dict:
    """Streaming IQA: decode each photo ONCE, feed both models from that buffer.

    The batched path runs two full passes over the folder — detection decodes
    every photo, then scoring decodes every photo again — on top of the
    early-exit gate's own full-resolution decode. Three decodes per photo, and
    decode is the dominant cost of a cull (214-241 ms/img versus 5 ms for the
    quality model). This processes a chunk at a time: decode -> detect -> route
    -> score -> discard, so peak memory is O(chunk) and each photo is read once.

    TOPIQ scores are UNCHANGED: the decode is the same 512px path score_all
    already used, so the quality model sees identical pixels. Only the detector
    changes — it now sees that 512px buffer instead of its own 640px read, which
    can move a borderline detection and therefore which formula a photo routes
    through. Enable with FRAMEGRADE_SHARED_DECODE=1.
    """
    import os as _os
    import time as _t
    import dfine_detector as _dfine
    _p = progress or (lambda f, d: None)
    n = len(image_paths)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return {"quality": empty, "tech": empty, "aesthetic": empty, "breakdowns": [],
                "composition_overrides": {}, "chiaroscuro_flags": {},
                "person_detected": {}, "framing_obstruction": {}, "subject_bboxes": {}}

    try:
        chunk = int(_os.environ.get("FRAMEGRADE_STREAM_CHUNK", "128"))
    except ValueError:
        chunk = 128
    chunk = max(8, min(chunk, n))

    comp_lum = lum_stats if lum_stats else [(128.0, 64.0)] * n
    _comp_eligible = comp_eligible_paths or set(image_paths)

    global _uniqa_singleton
    if _uniqa_singleton is None or _uniqa_singleton._model is None:
        _p(0.70, "Loading quality model…")
        _uniqa_singleton = UniQAHead()
        _uniqa_singleton.load()

    quality_raw: list = [0.5] * n
    routes: list = [0] * n
    person_detected: dict = {}
    subject_bboxes: dict = {}
    framing_obstruction: dict = {}
    composition_overrides: dict = {}
    chiaroscuro_flags: dict = {}
    t_decode = t_detect = t_score = 0.0

    for c0 in range(0, n, chunk):
        c1 = min(c0 + chunk, n)
        c_paths = image_paths[c0:c1]

        _td = _t.monotonic()
        tensors = _load_images_parallel(c_paths, pin=False)      # THE ONLY DECODE
        t_decode += _t.monotonic() - _td

        # detector consumes the same buffer (uint8 HWC as the processor expects)
        _tt = _t.monotonic()
        arrays = [(p, (t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy())
                  for p, t in zip(c_paths, tensors) if t is not None]
        det = _dfine.detect_persons_from_arrays(arrays, conf=0.55)
        t_detect += _t.monotonic() - _tt

        for p in c_paths:
            boxes = [d["bbox"] for d in det.get(p, [])
                     if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) >= 0.005]
            person_detected[p] = bool(boxes)
            if boxes:
                subject_bboxes[p] = boxes

        for p in c_paths:
            bb = subject_bboxes.get(p, [])
            if p in _comp_eligible and len(bb) >= 2:
                is_fo, subj = _detect_framing_obstruction(bb)
                if is_fo:
                    framing_obstruction[p] = True
                    subject_bboxes[p] = subj
        composition_overrides.update(
            _derive_ots_from_bboxes([p for p in c_paths if p in _comp_eligible],
                                    subject_bboxes))

        dark = [(p, comp_lum[c0 + i]) for i, p in enumerate(c_paths)
                if comp_lum[c0 + i][0] < 50.0 and p in _comp_eligible]
        if dark:
            try:
                from vision_composition_heads import ChiaroscuroHead as _Ch
                _ch = _Ch()
                if _ch.load():
                    for p, flag in zip([d[0] for d in dark],
                                       _ch.score_batch([d[0] for d in dark],
                                                       [d[1] for d in dark])):
                        chiaroscuro_flags[p] = flag
                else:
                    for p, (lm, ls) in dark:
                        chiaroscuro_flags[p] = (lm < 55.0) and (ls > 48.0)
                _ch.unload()
            except Exception as e:
                print(f"[vision_heads] chiaroscuro skipped ({e})")

        _ts = _t.monotonic()
        q, r = _uniqa_singleton.score_tensors(
            c_paths, tensors,
            vlm_breakdowns=vlm_breakdowns[c0:c1] if vlm_breakdowns else None,
            person_detected=person_detected, subject_bboxes=subject_bboxes,
            framing_obstruction=framing_obstruction)
        t_score += _t.monotonic() - _ts
        quality_raw[c0:c1] = q
        routes[c0:c1] = r

        del tensors, arrays
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        _p(0.70 + 0.13 * (c1 / n), f"Scoring image quality — {c1}/{n} photos…")

    print(f"[vision_heads] STREAMING TIME  decode {t_decode:6.1f}s  detect {t_detect:6.1f}s  "
          f"score {t_score:6.1f}s  ({t_decode/max(n,1)*1000:.0f} + "
          f"{t_detect/max(n,1)*1000:.0f} + {t_score/max(n,1)*1000:.0f} ms/img)", flush=True)

    # Same fixed-affine recalibration as the batched path (chunk-independent).
    q_arr = np.array(quality_raw, dtype=np.float32)
    mask = np.array([r != 1 for r in routes], dtype=bool) & (q_arr != 0.5)
    q_norm = q_arr.copy()
    q_norm[mask] = _batch_normalize(q_arr[mask])
    print(f"[vision_heads] Routes: {routes.count(1)} empty-scene, "
          f"{routes.count(2)} layered-frame, {routes.count(0)} standard")
    return {
        "quality": q_norm, "tech": q_norm, "aesthetic": q_norm,
        "breakdowns": [{"Technical": round(float(q_norm[i]), 3)} for i in range(n)],
        "composition_overrides": composition_overrides,
        "chiaroscuro_flags": chiaroscuro_flags,
        "person_detected": person_detected,
        "framing_obstruction": framing_obstruction,
        "subject_bboxes": subject_bboxes,
    }


def run_vision_heads(
    image_paths:         List[str],
    image_embeddings:    np.ndarray,
    prompt_embedding:    Optional[np.ndarray],
    clip_scores:         np.ndarray,
    genre_ref_embs:      Optional[np.ndarray] = None,
    lum_stats:           Optional[List[tuple]] = None,
    progress=None,
    comp_eligible_paths: Optional[set] = None,
    vlm_breakdowns:      Optional[List[dict]] = None,
) -> dict:
    """
    Run composition analysis then UniQAHead (single unified quality model).

    UniQA replaces TOPIQ NR, MUSIQ, and Aesthetic Predictor V2.5.
    YOLO11s-seg routing selects inference mode per image:
      Route 1 (empty scene): geometric blend of SpecVLM composition × lighting.
      Route 2 (layered frame): UniQA on subject crop.
      Standard: UniQA on full image (batched).

    vlm_breakdowns: list of per-image SpecVLM aspect dicts (same order as image_paths),
                    used by Route 1 to read Composition and Lighting scores.

    Returns:
        quality              np.ndarray (N,)  UniQA quality score [0,1]
        tech                 np.ndarray (N,)  alias of quality (pipeline compat)
        aesthetic            np.ndarray (N,)  alias of quality (pipeline compat)
        breakdowns           list[dict]       per-photo aspect scores
        composition_overrides dict[str,float] path → comp score override
        chiaroscuro_flags    dict[str,bool]   path → True when chiaroscuro
        person_detected      dict[str,bool]   path → True when person detected
    """
    import os as _os_sd
    if _os_sd.environ.get("FRAMEGRADE_SHARED_DECODE", "").strip() == "1":
        return _run_vision_heads_streaming(
            image_paths, image_embeddings, prompt_embedding, clip_scores,
            genre_ref_embs=genre_ref_embs, lum_stats=lum_stats, progress=progress,
            comp_eligible_paths=comp_eligible_paths, vlm_breakdowns=vlm_breakdowns)

    _p = progress or (lambda f, d: None)
    n  = len(image_paths)

    if n == 0:
        empty = np.array([], dtype=np.float32)
        return {
            "quality":                empty,
            "tech":                   empty,
            "aesthetic":              empty,
            "breakdowns":             [],
            "composition_overrides":  {},
            "chiaroscuro_flags":      {},
            "person_detected":        {},
            "framing_obstruction":    {},
        }

    composition_overrides: dict = {}
    chiaroscuro_flags:     dict = {}

    comp_lum     = lum_stats if lum_stats else [(128.0, 64.0)] * n
    dark_indices = [i for i, (m, _) in enumerate(comp_lum) if m < 50.0]
    dark_paths   = [image_paths[i] for i in dark_indices]
    dark_lum     = [comp_lum[i]    for i in dark_indices]

    # ── Single YOLO pass — paths only, no tensors in memory yet ─────────────
    # Eliminates the duplicate YOLO call that used to happen inside
    # SegCompositionAnalyzer.analyze_batch() AND inside UniQAHead.score_all().
    # IQA fractions live in 0.66 → 0.83: vision grading (Qwen) ends at 0.65,
    # so anything lower makes the bar jump backwards and look stuck.
    _p(0.66, f"YOLO person detection — {n} images…")
    import time as _tvh
    _t_dfine0 = _tvh.monotonic()
    person_detected_dict, subject_bboxes_dict = _run_yolo_seg(image_paths)
    _t_dfine = _tvh.monotonic() - _t_dfine0
    print(f"[vision_heads] TIME  person detection {_t_dfine:6.1f}s "
          f"({_t_dfine/max(n,1)*1000:.0f} ms/img)", flush=True)

    _comp_eligible = comp_eligible_paths or set(image_paths)
    _ots_paths     = [p for p in image_paths if p in _comp_eligible]

    # ── Framing Obstruction detection (before OTS, higher priority) ───────────
    # Large off-centre person (>30% area, center_x <0.35 or >0.65) = obstruction.
    # subject_bboxes_dict is updated in-place to point to the secondary subject,
    # so UniQA Route 2 crops to the real subject, not the occluding body.
    framing_obstruction_dict: dict = {}
    for _fo_path in _ots_paths:
        _fo_bbs = subject_bboxes_dict.get(_fo_path, [])
        if len(_fo_bbs) >= 2:
            _is_fo, _subj_bbs = _detect_framing_obstruction(_fo_bbs)
            if _is_fo:
                framing_obstruction_dict[_fo_path] = True
                subject_bboxes_dict[_fo_path]      = _subj_bbs   # crop to subject
    if framing_obstruction_dict:
        print(f"[vision_heads] Framing Obstruction: {len(framing_obstruction_dict)} images detected "
              f"(large off-centre occluder → cropping to subject for UniQA)")

    # ── OTS composition overrides (bbox-based, no DepthHead) ─────────────────
    composition_overrides = _derive_ots_from_bboxes(_ots_paths, subject_bboxes_dict)
    if composition_overrides:
        print(f"[vision_heads] OTS portraits: {len(composition_overrides)} overrides (bbox-derived)")

    # ── ChiaroscuroHead (dark images only; DepthHead + SegComp removed) ───────
    if dark_paths:
        _p(0.68, f"Chiaroscuro detection — {len(dark_paths)}/{n} dark images…")
        try:
            from vision_composition_heads import ChiaroscuroHead as _ChHead
            _elig_dark_pairs = [
                (dark_paths[j], dark_lum[j])
                for j in range(len(dark_paths))
                if dark_paths[j] in _comp_eligible
            ]
            if _elig_dark_pairs:
                _elig_dp, _elig_dl = zip(*_elig_dark_pairs)
                _ch = _ChHead()
                if _ch.load():
                    _flags = _ch.score_batch(list(_elig_dp), list(_elig_dl))
                    for p, flag in zip(_elig_dp, _flags):
                        chiaroscuro_flags[p] = flag
                else:
                    # Probe model absent (models/vision_probe has no
                    # preprocessor_config.json on this install) — fall back to
                    # a luminance heuristic so dark intentional work is not
                    # left unprotected: chiaroscuro = deep shadows (low mean)
                    # WITH strong tonal separation (high std). A muddy
                    # underexposure is dark but flat and stays unflagged.
                    for p, (_lm, _ls) in zip(_elig_dp, _elig_dl):
                        chiaroscuro_flags[p] = (_lm < 55.0) and (_ls > 48.0)
                    _n_h = sum(1 for v in chiaroscuro_flags.values() if v)
                    print(f"[vision_heads] ChiaroscuroHead unavailable — luminance "
                          f"heuristic flagged {_n_h}/{len(_elig_dp)} dark images")
                _ch.unload()
                _n_ch = sum(1 for v in chiaroscuro_flags.values() if v)
                print(f"[vision_heads] Chiaroscuro: {_n_ch}/{len(dark_paths)} dark images flagged")
        except Exception as e:
            print(f"[vision_heads] ChiaroscuroHead failed ({e}) — skipping")
    else:
        _p(0.68, f"Chiaroscuro skipped — all {n} images well-lit")
        print(f"[vision_heads] Chiaroscuro skipped — all {n} images well-lit")

    # ── UniQA Head (pre-computed YOLO passed in — no second YOLO pass) ────────
    # Images are NOT pre-loaded here: score_all decodes in RAM-bounded chunks
    # (peak memory O(chunk), not O(N)), so a 5-10k-image import no longer OOMs
    # on the up-front full-batch decode that used to hold every frame at once.
    global _uniqa_singleton
    if _uniqa_singleton is None or _uniqa_singleton._model is None:
        _p(0.70, "Loading UniQA…")
        _uniqa_singleton = UniQAHead()
        _uniqa_singleton.load()
        print("[vision_heads] UniQAHead loaded — cached as singleton")
    else:
        _p(0.70, "UniQA cached — scoring directly…")
        print("[vision_heads] UniQAHead singleton reused — no reload")

    _t_uniqa0 = _tvh.monotonic()
    try:
        quality_scores, _, _ = _uniqa_singleton.score_all(
            image_paths             = image_paths,
            vlm_breakdowns          = vlm_breakdowns,
            progress                = _p,
            progress_start          = 0.70,
            progress_end            = 0.83,
            person_detected_in      = person_detected_dict,
            subject_bboxes_in       = subject_bboxes_dict,
            framing_obstruction_in  = framing_obstruction_dict,
        )
        print(f"[vision_heads] TIME  quality head total {_tvh.monotonic()-_t_uniqa0:6.1f}s", flush=True)
        print(
            f"[vision_heads] UniQA: min={quality_scores.min():.3f}  "
            f"max={quality_scores.max():.3f}  mean={quality_scores.mean():.3f}"
        )
    except Exception as e:
        import traceback as _tb_uniqa
        print(f"[vision_heads] UniQAHead FATAL — full traceback follows:")
        _tb_uniqa.print_exc()
        raise

    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    breakdowns = [
        {"Technical": round(float(quality_scores[i]), 3)}
        for i in range(n)
    ]

    return {
        "quality":                quality_scores,
        "tech":                   quality_scores,    # pipeline compat
        "aesthetic":              quality_scores,    # pipeline compat
        "breakdowns":             breakdowns,
        "composition_overrides":  composition_overrides,
        "chiaroscuro_flags":      chiaroscuro_flags,
        "person_detected":        person_detected_dict,
        "framing_obstruction":    framing_obstruction_dict,
        "subject_bboxes":         subject_bboxes_dict,
    }
