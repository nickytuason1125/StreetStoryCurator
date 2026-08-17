"""
face_signals.py — face detection and subject-focus measurement (YuNet).

Why this exists
---------------
Grading here scores aesthetics. It has no notion of whether the person in the
frame is actually SHARP, which for most photographers is the first cull pass and
is not a matter of taste: a beautifully composed portrait with the focus on the
ear behind the eye is a reject, and no aesthetic model reliably catches it.

What is honestly measurable with the model we ship
--------------------------------------------------
YuNet (models/face_detection_yunet_2023mar.onnx, 232 KB) returns a box, a
confidence, and five landmarks — right eye, left eye, nose tip, and both mouth
corners. Landmarks are POSITIONS. They do not encode eye state.

So this module provides:
  * face detection      — how many faces, how large, where
  * subject focus       — sharpness inside the face box measured against the
                          sharpness of the frame as a whole
  * face-region quality — enough to tell "soft subject" from "soft photo"

It deliberately does NOT provide an eyes-open/closed verdict. That needs a
trained eye-state classifier; inferring it from landmark geometry is a
coin-flip dressed up as a measurement, and a wrong "eyes closed" silently
deletes a keeper. See ``eye_state_available()``.

Scores are REPORTED, not folded into the grade. Nothing here changes an existing
score — the signal is surfaced so it can be displayed and filtered on.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL = str(Path(__file__).resolve().parent.parent
             / "models" / "face_detection_yunet_2023mar.onnx")

# Detection resolution. MEASURED on 40 real street frames (5168px originals):
#   side=320  -> 0 faces found at conf 0.75. A face occupying 100px of a 5168px
#               frame is ~6px after that downscale, well under YuNet's floor.
#   side=640  -> 3 photos with faces
#   side=1024 -> 11 photos with faces, and ZERO faces on frames where the person
#               detector found nobody (i.e. no false positives at conf 0.75)
# 320 is YuNet's training size and is fine for webcam-style framing; it is the
# wrong choice for full-frame photographs.
_DET_SIDE = 1024
_CONF = 0.70
_NMS = 0.3

# Below these the focus measurement is not trustworthy. Faces are still
# REPORTED; only the focus verdict is withheld.
#
# The pixel floor alone was not enough. Skin is LOW-texture, so a small face
# against a detailed background scores low Laplacian variance even in perfect
# focus — the ratio conflates "soft" with "smooth". MEASURED across 108 real
# frames, by face area as a fraction of the frame:
#
#     <1%    n=33  median ratio 1.06   39% flagged soft   <- artifact
#     1-2%   n=23  median ratio 1.94    4% flagged
#     2-5%   n=37  median ratio 1.98   14% flagged
#     >5%    n=15  median ratio 2.59    0% flagged
#
# The verdict is only meaningful once the face is a real part of the frame, so
# it is withheld below 1%. That makes this a portrait/event signal, which is
# what it is for — in wide street work most faces are simply too small to judge.
_MIN_FOCUS_FACE_PX = 64
_MIN_FOCUS_FACE_FRAC = 0.01

# Decoding a 5168px frame to measure a face costs ~1 s/photo and buys nothing:
# detection happens at _DET_SIDE anyway. JPEG can be decoded straight to a
# smaller size in the DCT domain, which is nearly free.
_LOAD_SIDE = 1600

_detector = None          # cv2 detector, created once per process
_last_size = None


def available() -> bool:
    """True when face signals can be computed at all."""
    try:
        import cv2
        return os.path.exists(_MODEL) and hasattr(cv2, "FaceDetectorYN")
    except Exception:
        return False


def eye_state_available() -> bool:
    """False: no eyes-open/closed model is shipped.

    Kept as an explicit function so callers ask rather than assume. YuNet's eye
    landmarks are coordinates, not state — a closed eye still has a landmark.
    """
    return False


def _get_detector(w: int, h: int):
    global _detector, _last_size
    import cv2
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            _MODEL, "", (w, h), _CONF, _NMS, 5000)
        _last_size = (w, h)
    elif _last_size != (w, h):
        _detector.setInputSize((w, h))
        _last_size = (w, h)
    return _detector


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard focus measure.

    Scale-dependent by nature, which is exactly why it is only ever used here as
    a RATIO between two regions of the same image, never as an absolute
    threshold. An absolute cutoff would reject every soft-by-intent frame and
    every low-contrast scene.
    """
    import cv2
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_faces(bgr: np.ndarray, conf: float = _CONF) -> list:
    """Detect faces in a BGR array. Returns [] when there are none or on error.

    Never raises: a detector failure must not take down a cull.
    """
    if bgr is None or not available():
        return []
    try:
        import cv2
        h, w = bgr.shape[:2]
        if h < 20 or w < 20:
            return []
        scale = _DET_SIDE / max(h, w)
        if scale < 1.0:
            small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = bgr, 1.0
        sh, sw = small.shape[:2]
        det = _get_detector(sw, sh)
        det.setScoreThreshold(float(conf))
        _n, faces = det.detect(small)
        if faces is None:
            return []
        out = []
        for f in faces:
            x, y, fw, fh = (float(v) / scale for v in f[:4])
            pts = [(float(f[4 + 2 * i]) / scale, float(f[5 + 2 * i]) / scale)
                   for i in range(5)]
            out.append({
                "box": [max(0.0, x), max(0.0, y), fw, fh],
                "confidence": float(f[-1]),
                # order is fixed by YuNet's output layout
                "landmarks": {"right_eye": pts[0], "left_eye": pts[1],
                              "nose": pts[2], "right_mouth": pts[3],
                              "left_mouth": pts[4]},
                "area_frac": float(fw * fh) / float(w * h) if w and h else 0.0,
            })
        return out
    except Exception:
        return []


def face_metrics(img, conf: float = _CONF) -> dict:
    """Face + focus signals for one image (PIL Image or BGR ndarray).

    focus_ratio is the subject's sharpness divided by the frame's. Above 1 the
    face is sharper than the scene around it — a deliberate portrait. Well below
    1 the face is the softest thing in the frame, which is the missed-focus
    case: the camera locked onto something else.
    """
    empty = {"faces_detected": 0, "largest_face_frac": 0.0,
             "face_sharpness": 0.0, "global_sharpness": 0.0,
             "focus_ratio": None, "subject_in_focus": None,
             "eye_state_supported": False, "faces": []}
    if img is None or not available():
        return empty
    try:
        import cv2
        if not isinstance(img, np.ndarray):
            bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        else:
            bgr = img
        faces = detect_faces(bgr, conf)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g_sharp = _sharpness(gray)
        if not faces:
            out = dict(empty)
            out["global_sharpness"] = g_sharp
            return out

        biggest = max(faces, key=lambda f: f["area_frac"])
        x, y, fw, fh = (int(round(v)) for v in biggest["box"])
        H, W = gray.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + max(1, fw)), min(H, y + max(1, fh))
        f_sharp = _sharpness(gray[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else 0.0

        # A tiny face gives a Laplacian dominated by resampling noise, so the
        # ratio would be authoritative-looking nonsense. Report the face, not a
        # focus verdict.
        too_small = (min(fw, fh) < _MIN_FOCUS_FACE_PX
                     or biggest["area_frac"] < _MIN_FOCUS_FACE_FRAC)
        ratio = None if (too_small or g_sharp <= 1e-6) else (f_sharp / g_sharp)
        return {
            "faces_detected": len(faces),
            "largest_face_frac": round(biggest["area_frac"], 5),
            "face_sharpness": round(f_sharp, 2),
            "global_sharpness": round(g_sharp, 2),
            "focus_ratio": round(ratio, 3) if ratio is not None else None,
            # Threshold is deliberately generous: this flags "the subject is
            # clearly the softest thing here", not "slightly less sharp than
            # the background", which is normal for a wide aperture.
            "subject_in_focus": (None if ratio is None else bool(ratio >= 0.6)),
            "eye_state_supported": eye_state_available(),
            "faces": faces,
        }
    except Exception:
        return empty


def _load_small(path: str):
    """Load at roughly _LOAD_SIDE, as cheaply as the format allows.

    For JPEG, PIL's draft() decodes at 1/2, 1/4 or 1/8 scale directly in the DCT
    domain — a 5168px frame becomes 1292px without ever materialising the full
    image. RAW goes through load_rgb, whose embedded preview is already small.
    """
    from PIL import Image
    from raw_support import RAW_EXTS, load_rgb
    if Path(path).suffix.lower() in RAW_EXTS:
        return load_rgb(path, "RGB")
    try:
        im = Image.open(path)
        try:
            im.draft("RGB", (_LOAD_SIDE, _LOAD_SIDE))
        except Exception:
            pass
        im = im.convert("RGB")
        if max(im.size) > _LOAD_SIDE:
            k = _LOAD_SIDE / max(im.size)
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))),
                           Image.BILINEAR)
        return im, "pil"
    except Exception:
        return None, "unreadable"


def metrics_for_path(path: str, conf: float = _CONF) -> dict:
    """face_metrics for a file path, RAW included."""
    try:
        img, src = _load_small(path)
        if img is None:
            out = dict(face_metrics(None))
            out["unreadable"] = True
            return out
        return face_metrics(img, conf)
    except Exception:
        return face_metrics(None)
