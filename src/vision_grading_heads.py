"""
Vision IQA Head — UniQA unified quality backbone.

UniQAHead: pyiqa 'uniqa' metric replacing TOPIQ NR, MUSIQ, and Aesthetic V2.5.

YOLO11s-seg routing:
  Route 1 (empty scene): 0 humans detected → score = sqrt(composition * lighting)
                          from SpecVLM aspect scores. Decouples Human/Culture penalty.
  Route 2 (layered frame): human in midground (bbox center_y 33–67%) + blurred
                            foreground (low Laplacian variance, bottom third of image)
                            → UniQA on subject crop only.
  Standard: UniQA on full resized image.

Speed design:
  - Images loaded in parallel (TurboJPEG via fast_ingestion.decode_one).
  - UniQA: GPU batch inference at 512×512, mini-batches of 8.
  - Route 2: per-image crop inference (variable crop size prevents batching).
  - YOLO: .engine → .pt → n-variant fallback; returns per-image route decisions.
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _batch_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-Max scaling with a floor of 0.20: batch_min→0.20, batch_max→1.0.

    The 0.20 floor prevents artistic motion-blur or high-ISO grain — intentional
    street photography choices — from being zeroed out purely because they happen
    to be the worst-technical photo in a given batch.
    """
    if len(scores) < 2:
        return np.clip(scores.astype(np.float32), 0.20, 1.0)
    lo   = float(np.min(scores))
    hi   = float(np.max(scores))
    span = max(hi - lo, 1e-4)
    return np.clip((scores - lo) / span * 0.80 + 0.20, 0.20, 1.0).astype(np.float32)


def _load_images_parallel(
    image_paths: List[str],
    n_workers: int = 8,
    max_size: int = 768,
) -> List[Optional[torch.Tensor]]:
    """
    Decode images in parallel via TurboJPEG (JPEG) / PIL (other formats).
    Returns (C, H, W) float32 pin_memory tensors capped at max_size on the long edge.
    """
    from fast_ingestion import decode_one
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _load_one(p: str) -> Optional[torch.Tensor]:
        t = decode_one(p, target_hw=None, pin=True)
        if t is None:
            return None
        _, h, w = t.shape
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            t = TF.resize(t, [int(h * scale), int(w * scale)], antialias=True)
        return t

    with _TPE(max_workers=min(n_workers, len(image_paths) or 1)) as pool:
        return list(pool.map(_load_one, image_paths))


def _run_yolo_seg(
    image_paths: List[str],
    tensors: List[Optional[torch.Tensor]],
) -> tuple:
    """
    Run YOLO11s-seg (person class only) on all images.

    YOLO model search order: yolo11s-seg.engine → yolo11s-seg.pt → yolo11n-seg.pt.
    Falls back gracefully to person_detected=True for all images when unavailable.

    Returns:
        person_detected_dict  path → bool
        subject_bboxes_dict   path → list[[x1n, y1n, x2n, y2n]] (normalised [0,1])
    """
    person_detected: dict = {}
    subject_bboxes:  dict = {}

    candidates = [
        Path("models") / "yolo11s-seg.engine",
        Path("models") / "yolo11s-seg.pt",
        Path("models") / "yolo11n-seg.pt",
    ]
    yolo = None
    for c in candidates:
        if c.exists():
            try:
                from ultralytics import YOLO
                yolo = YOLO(str(c))
                print(f"[uniqa_head] YOLO loaded: {c.name}")
                break
            except Exception as e:
                print(f"[uniqa_head] YOLO {c.name} failed: {e}")

    if yolo is None:
        print("[uniqa_head] YOLO unavailable — all images route to standard UniQA")
        for p in image_paths:
            person_detected[p] = True   # safe default: treat as person present
        return person_detected, subject_bboxes

    for path, t in zip(image_paths, tensors):
        if t is None:
            person_detected[path] = False
            continue
        try:
            img_np = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            results = yolo(img_np, verbose=False, classes=[0])   # 0 = person
            boxes   = results[0].boxes if results else None
            if boxes is None or len(boxes) == 0:
                person_detected[path] = False
            else:
                _, H, W = t.shape
                bboxes_norm: list = []
                for box in boxes.xyxy.cpu().numpy():
                    x1n = float(box[0]) / W
                    y1n = float(box[1]) / H
                    x2n = float(box[2]) / W
                    y2n = float(box[3]) / H
                    # Skip sub-0.5% area detections — filters noise artefacts,
                    # handrail shadows, and distant background pixels that YOLO
                    # occasionally classifies as class 0.
                    if (x2n - x1n) * (y2n - y1n) < 0.005:
                        continue
                    bboxes_norm.append([x1n, y1n, x2n, y2n])
                if bboxes_norm:
                    person_detected[path] = True
                    subject_bboxes[path]  = bboxes_norm
                else:
                    person_detected[path] = False
        except Exception as e:
            print(f"[uniqa_head] YOLO failed for {Path(path).name}: {e}")
            person_detected[path] = True   # safe default

    n_person = sum(1 for v in person_detected.values() if v)
    print(f"[uniqa_head] YOLO: {n_person}/{len(image_paths)} images with person detected")
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
    Unified Image Quality Assessment backbone (pyiqa 'uniqa').

    Replaces TOPIQ NR + MUSIQ (technical) and Aesthetic Predictor V2.5 (aesthetic)
    with a single model covering both technical quality and aesthetic appeal.

    YOLO11s-seg routing selects the input region per image:
      Route 1 — empty scene (0 humans detected): geometric blend of SpecVLM aspect scores.
      Route 2 — layered frame: UniQA on YOLO subject crop.
      Standard — everything else: UniQA on full resized image (batched).
    """

    _METRIC_NAME = "uniqa"
    _BATCH_SIZE  = 8
    _INPUT_SIZE  = 512

    def __init__(self) -> None:
        self._model  = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _timed_create(device: str, timeout: int = 120):
        import pyiqa, threading
        _result: list = [None]
        _err:    list = [None]

        def _worker():
            try:
                _result[0] = pyiqa.create_metric("uniqa", device=device)
            except Exception as e:
                _err[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            print(f"[uniqa_head] uniqa timed out after {timeout}s — skip")
            return None
        if _err[0] is not None:
            print(f"[uniqa_head] uniqa error: {_err[0]} — skip")
            return None
        return _result[0]

    def load(self) -> None:
        self._model = self._timed_create(self._device)
        if self._model is not None:
            print(f"[uniqa_head] UniQA loaded on {self._device}")
        else:
            print("[uniqa_head] UniQA unavailable")

    def score_all(
        self,
        tensors:                 List[Optional[torch.Tensor]],
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
        Score images via YOLO routing + UniQA.

        Route 1 images (empty scene) use SpecVLM aspect scores — no model inference.
        Route 2 images (layered frame) run UniQA on the YOLO subject crop.
        Standard images run UniQA on the full resized image (batched on GPU).

        person_detected_in / subject_bboxes_in: pre-computed YOLO results from
        run_vision_heads(); when provided, the internal _run_yolo_seg() call is
        skipped, eliminating the duplicate YOLO pass.

        Returns:
            quality_norm     np.ndarray (N,)  batch-normalised [0,1]
            person_detected  dict  path → bool
            subject_bboxes   dict  path → list[bbox]
        """
        _p = progress or (lambda f, d: None)
        n  = len(tensors)
        device = torch.device(self._device)
        S      = self._INPUT_SIZE

        if self._model is None:
            self.load()
        if self._model is None:
            raise RuntimeError("UniQA unavailable — weights not cached")

        # ── YOLO routing ─────────────────────────────────────────────────────
        if person_detected_in is not None and subject_bboxes_in is not None:
            person_detected = person_detected_in
            subject_bboxes  = subject_bboxes_in
            print(f"[uniqa_head] YOLO skipped — pre-computed detections reused ({n} images)")
        else:
            _p(progress_start, f"YOLO routing — {n} images…")
            person_detected, subject_bboxes = _run_yolo_seg(image_paths, tensors)

        _fo_map = framing_obstruction_in or {}
        routes: List[int] = []   # 0=standard, 1=empty-scene, 2=layered-frame/FO
        for path, t in zip(image_paths, tensors):
            has_person = person_detected.get(path, True)
            bboxes     = subject_bboxes.get(path, [])
            is_fo      = bool(_fo_map.get(path, False))
            if not has_person:
                routes.append(1)
            elif is_fo or (bboxes and t is not None and _is_layered_frame(t, bboxes)):
                routes.append(2)
            else:
                routes.append(0)

        n_r1  = routes.count(1)
        n_r2  = routes.count(2)
        n_std = routes.count(0)
        print(f"[uniqa_head] Routes: {n_r1} empty-scene, {n_r2} layered-frame, {n_std} standard")

        quality_raw = [0.5] * n

        # ── Route 1: geometric blend from VLM aspect scores (no GPU inference) ─
        for i, route in enumerate(routes):
            if route != 1:
                continue
            bd    = vlm_breakdowns[i] if vlm_breakdowns else {}
            comp  = float(bd.get("Composition", 0.5))
            light = float(bd.get("Lighting",    0.5))
            quality_raw[i] = float(np.sqrt(max(comp, 0.01) * max(light, 0.01)))

        # ── Route 0 (standard): batch UniQA on full images ────────────────────
        std_indices = [i for i, r in enumerate(routes) if r == 0 and tensors[i] is not None]
        n_std_valid  = len(std_indices)
        n_std_batches = max(1, (n_std_valid + self._BATCH_SIZE - 1) // self._BATCH_SIZE) if n_std_valid else 1

        for b_idx, batch_start in enumerate(range(0, max(n_std_valid, 1), self._BATCH_SIZE)):
            batch_i = std_indices[batch_start : batch_start + self._BATCH_SIZE]
            if not batch_i:
                break
            resized = [TF.resize(tensors[i], [S, S], antialias=True) for i in batch_i]
            batch_t = torch.stack(resized).to(device, non_blocking=True)
            try:
                with torch.inference_mode():
                    out = self._model(batch_t)
                out = out.squeeze(-1) if out.dim() > 1 else out
                for j, gl_i in enumerate(batch_i):
                    quality_raw[gl_i] = float(out[j].item())
                del batch_t
            except Exception as e:
                print(f"[uniqa_head] Batch {b_idx} failed ({e}) — per-image fallback")
                del batch_t
                for gl_i in batch_i:
                    if tensors[gl_i] is None:
                        continue
                    try:
                        inp = TF.resize(tensors[gl_i], [S, S], antialias=True).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            s = self._model(inp)
                        quality_raw[gl_i] = float(s.item() if hasattr(s, "item") else float(s))
                    except Exception:
                        quality_raw[gl_i] = 0.5

            done_so_far = min(batch_start + self._BATCH_SIZE, n_std_valid)
            _p(
                progress_start + (progress_end - progress_start) * 0.70 * (b_idx + 1) / n_std_batches,
                f"UniQA {done_so_far}/{n_std_valid} standard images…",
            )

        # ── Route 2 (layered frame): per-image UniQA on subject crop ──────────
        r2_indices = [i for i, r in enumerate(routes) if r == 2]
        for r2_cnt, i in enumerate(r2_indices):
            t      = tensors[i]
            path   = image_paths[i]
            bboxes = subject_bboxes[path]
            _, H, W = t.shape
            x1 = max(0, int(min(b[0] for b in bboxes) * W))
            y1 = max(0, int(min(b[1] for b in bboxes) * H))
            x2 = min(W, int(max(b[2] for b in bboxes) * W))
            y2 = min(H, int(max(b[3] for b in bboxes) * H))
            crop = t[:, y1:y2, x1:x2] if (y2 - y1) >= 32 and (x2 - x1) >= 32 else t
            inp  = TF.resize(crop, [S, S], antialias=True).unsqueeze(0).to(device)
            try:
                with torch.inference_mode():
                    s = self._model(inp)
                quality_raw[i] = float(s.item() if hasattr(s, "item") else float(s))
            except Exception as e:
                print(f"[uniqa_head] Route2 crop failed for {Path(path).name}: {e}")
                quality_raw[i] = 0.5
            _p(
                progress_start + (progress_end - progress_start) * (
                    0.70 + 0.30 * (r2_cnt + 1) / max(len(r2_indices), 1)
                ),
                f"UniQA crop {r2_cnt + 1}/{len(r2_indices)} layered-frame…",
            )

        quality_norm = _batch_normalize(np.array(quality_raw, dtype=np.float32))
        print(
            f"[uniqa_head] UniQA: min={quality_norm.min():.3f}  "
            f"max={quality_norm.max():.3f}  mean={quality_norm.mean():.3f}"
        )
        return quality_norm, person_detected, subject_bboxes

    def unload(self) -> None:
        self._model = None
        _purge_vram()


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

    # ── Load images early — shared by YOLO pass and UniQA ────────────────────
    _p(0.55, f"Loading {n} images for IQA…")
    tensors  = _load_images_parallel(image_paths)
    n_failed = sum(1 for t in tensors if t is None)
    if n_failed:
        print(f"[vision_heads] {n_failed}/{n} images failed to load — using 0.5 fallback")

    # ── Single YOLO pass — for OTS detection AND UniQA routing ───────────────
    # Eliminates the duplicate YOLO call that used to happen inside
    # SegCompositionAnalyzer.analyze_batch() AND inside UniQAHead.score_all().
    _p(0.57, f"YOLO person detection — {n} images…")
    person_detected_dict, subject_bboxes_dict = _run_yolo_seg(image_paths, tensors)

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
        _p(0.59, f"Chiaroscuro detection — {len(dark_paths)}/{n} dark images…")
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
                _ch.unload()
                _n_ch = sum(1 for v in chiaroscuro_flags.values() if v)
                print(f"[vision_heads] Chiaroscuro: {_n_ch}/{len(dark_paths)} dark images flagged")
        except Exception as e:
            print(f"[vision_heads] ChiaroscuroHead failed ({e}) — skipping")
    else:
        _p(0.59, f"Chiaroscuro skipped — all {n} images well-lit")
        print(f"[vision_heads] Chiaroscuro skipped — all {n} images well-lit")

    # ── UniQA Head (pre-computed YOLO passed in — no second YOLO pass) ────────
    global _uniqa_singleton
    if _uniqa_singleton is None or _uniqa_singleton._model is None:
        _p(0.60, "Loading UniQA…")
        _uniqa_singleton = UniQAHead()
        _uniqa_singleton.load()
        print("[vision_heads] UniQAHead loaded — cached as singleton")
    else:
        _p(0.60, "UniQA cached — scoring directly…")
        print("[vision_heads] UniQAHead singleton reused — no reload")

    try:
        quality_scores, _, _ = _uniqa_singleton.score_all(
            tensors                 = tensors,
            image_paths             = image_paths,
            vlm_breakdowns          = vlm_breakdowns,
            progress                = _p,
            progress_start          = 0.60,
            progress_end            = 0.83,
            person_detected_in      = person_detected_dict,
            subject_bboxes_in       = subject_bboxes_dict,
            framing_obstruction_in  = framing_obstruction_dict,
        )
        print(
            f"[vision_heads] UniQA: min={quality_scores.min():.3f}  "
            f"max={quality_scores.max():.3f}  mean={quality_scores.mean():.3f}"
        )
    except Exception as e:
        print(f"[vision_heads] UniQAHead failed ({e}) — CLIP fallback")
        quality_scores       = clip_scores.copy()
        person_detected_dict = {}
        _uniqa_singleton     = None

    del tensors
    gc.collect()

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
    }
