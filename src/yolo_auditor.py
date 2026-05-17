"""
YOLO person auditor — short-circuit gate for empty/liminal scene briefs.

Loads yolo11s.pt; falls back to yolo11n, yolo26s, yolo26n if weights absent.
Only activates when the active Creative Direction brief implies empty scenes.
Returns the set of image paths where a person was detected at conf >= threshold.

Short-circuit rule: detected images are immediately assigned score 0.00
and skip the entire IQA pass (Step 4b), advancing directly to LanceDB.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

_CONF_THRESH        = 0.35    # strict YOLO person detection threshold
_PERSON_CLASS       = 0       # COCO class 0 = person
_MIN_AREA_FRACTION  = 0.0008  # ignore detections < 0.08% of canvas (distant background figures)

_yolo_model: Optional[object] = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        for weights in ("yolo11s.pt", "yolo11n.pt", "yolo26s.pt", "yolo26n.pt"):
            try:
                _yolo_model = YOLO(weights)
                print(f"[yolo_auditor] Loaded {weights}")
                return _yolo_model
            except Exception:
                continue
        print("[yolo_auditor] No YOLO weights found — person audit disabled")
    except ImportError:
        print("[yolo_auditor] ultralytics not installed — person audit disabled")
    return None


def audit_paths(
    paths: list[str],
    conf: float = _CONF_THRESH,
) -> tuple[set[str], set[str]]:
    """
    Run YOLO person detection on each path (CPU, sequential).

    Returns:
        hard_disqualified  paths where a real non-silhouette person was detected.
                           → score set to 0.00, IQA skipped.
        soft_penalized     paths where only a dark-scene silhouette was found:
                           full-image mean luminance < 35 AND bounding-box
                           luminance std < 25 (uniform dark region = silhouette).
                           → -0.15 score penalty applied; Anchor Floor can still
                           protect the image if overall aesthetic quality is high.
    """
    model = _load_yolo()
    if model is None:
        return set(), set()

    hard_disqualified: set[str] = set()
    soft_penalized:    set[str] = set()

    for path in paths:
        try:
            # Load image once per path — reused for full-image and per-box luminance.
            try:
                from PIL import Image as _PILImg
                _img_arr     = np.array(_PILImg.open(path).convert("RGB"))
                _Y_full      = (0.299 * _img_arr[:, :, 0]
                              + 0.587 * _img_arr[:, :, 1]
                              + 0.114 * _img_arr[:, :, 2])
                _full_mean_lum = float(_Y_full.mean())
            except Exception:
                _img_arr       = None
                _full_mean_lum = 128.0   # neutral — silhouette gating disabled

            results = model(
                path,
                device="cpu",
                classes=[_PERSON_CLASS],
                conf=conf,
                verbose=False,
                half=False,
            )
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                img_h, img_w = r.orig_shape
                canvas_area  = img_h * img_w
                area_thresh  = _MIN_AREA_FRACTION * canvas_area
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < area_thresh:
                        print(
                            f"[yolo_auditor] Ignored distant figure in "
                            f"{Path(path).name} — box_area={box_area:.0f}px "
                            f"< {area_thresh:.0f}px ({_MIN_AREA_FRACTION*100:.3f}% canvas)"
                        )
                        continue

                    if _img_arr is not None:
                        _px1  = max(0, int(x1));              _py1 = max(0, int(y1))
                        _px2  = min(_img_arr.shape[1], int(x2)); _py2 = min(_img_arr.shape[0], int(y2))
                        _patch = _img_arr[_py1:_py2, _px1:_px2]
                        if _patch.size > 0:
                            _y_lum    = (0.299 * _patch[:, :, 0]
                                       + 0.587 * _patch[:, :, 1]
                                       + 0.114 * _patch[:, :, 2])
                            _mean_lum = float(_y_lum.mean())
                            _std_lum  = float(_y_lum.std())

                            # Absolute shadow artifact — very dark crop mean
                            if _mean_lum < 15.0:
                                print(
                                    f"[yolo_auditor] Shadow artifact in "
                                    f"{Path(path).name} — Y={_mean_lum:.1f} < 15 "
                                    f"(dark silhouette, ignored)"
                                )
                                continue

                            # Dark-scene silhouette: low full-image luminance AND
                            # very uniform crop (low std) = intentional silhouette subject.
                            # Soft penalty instead of hard disqualification — the Anchor
                            # Floor can still protect if aesthetic quality is high.
                            if _full_mean_lum < 35.0 and _std_lum < 25.0:
                                print(
                                    f"[yolo_auditor] Dark silhouette in "
                                    f"{Path(path).name} — scene_Y={_full_mean_lum:.1f} < 35, "
                                    f"crop_std={_std_lum:.1f} < 25 → soft penalty (-0.15)"
                                )
                                soft_penalized.add(path)
                                break

                    # Clear non-silhouette person → hard disqualification
                    hard_disqualified.add(path)
                    break
        except Exception as e:
            print(f"[yolo_auditor] Inference failed for {Path(path).name}: {e}")

    return hard_disqualified, soft_penalized


def unload() -> None:
    global _yolo_model
    _yolo_model = None
    try:
        import gc
        gc.collect()
    except Exception:
        pass
