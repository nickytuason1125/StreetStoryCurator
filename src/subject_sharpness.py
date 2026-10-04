"""
Subject sharpness — is the SUBJECT sharp, regardless of the background?

Why this exists (2026-10-03): motion-blurred misses graded Mid. The Technical
axis had Spearman +0.06 with measured sharpness on a 445-photo SD card, and
the only blur rule (early_exit Laplacian var < 4.0) fires on near-featureless
frames only. Whole-frame sharpness cannot be the fix: shallow-depth-of-field
frames (sharp cat, soft background) and pans (sharp subject, streaked
background) measure as "blurry" on whole-frame metrics while being the
photographer's best shots.

The question that separates the three cases is local, not global:

    pan            subject sharp,   background streaked   → no penalty
    shallow focus  subject sharp,   background soft       → no penalty
    missed shot    subject smeared                        → Weak

Subject = D-FINE boxes (all COCO classes — people, boats, cats, bikes, cars),
falling back to the sharpest region of the frame when nothing is detected.
Pure CPU/numpy; never touches torch, so it is safe in any process.
"""
from __future__ import annotations

import numpy as np

# Long edge the sharpness is measured at. Motion blur of a few pixels at
# 6000 px vanishes at 512 px (the IQA buffer), so this needs its own decode;
# RAW embedded previews are ~1600 px and cost almost nothing to extract.
MEASURE_EDGE = 1600

# Tile grid for the local-sharpness map.
_TILE = 32


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]
    s = MEASURE_EDGE / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _tiles(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    return a[: th * _TILE, : tw * _TILE].reshape(th, _TILE, tw, _TILE).sum(axis=(1, 3))


def sharpness_map(gray: np.ndarray) -> np.ndarray:
    return _direction_maps(gray)[0]


def _direction_maps(gray: np.ndarray):
    """(sharpness, (rx, ry)) per tile. sharpness = min of the two directions;
    rx / ry are each direction's own sharpness, NaN where that direction has
    too few edges to judge (stripes, horizons) — used by the pan test."""
    """Per-tile sharpness in [0, 1] — the re-blur test (Crété-Roffet 2007).

    Blur the tile a bit more and measure how much neighbour-to-neighbour
    variation it loses. A sharp tile loses most of it; an already-blurred tile
    barely changes. Being a ratio, it is contrast-independent: a dark or
    low-contrast subject that is in focus still reads sharp.

    Measured separately along x and y and the MINIMUM kept — motion blur
    smears one direction and leaves the other crisp, which a combined measure
    averages away. Flat tiles (sky, wall) carry no evidence and are NaN.
    """
    import cv2
    g = gray.astype(np.float32)
    h, w = g.shape
    th, tw = h // _TILE, w // _TILE
    if th == 0 or tw == 0:
        nan = np.full((1, 1), np.nan, np.float32)
        return nan, (nan, nan)
    ratios, sos, total = [], [], 0.0
    for axis, ksize in ((1, (9, 1)), (0, (1, 9))):
        b = cv2.blur(g, ksize)
        d_o = np.abs(np.diff(g, axis=axis))
        d_b = np.abs(np.diff(b, axis=axis))
        lost = np.maximum(0.0, d_o - d_b)
        pad = ((0, 0), (0, 1)) if axis == 1 else ((0, 1), (0, 0))
        so = _tiles(np.pad(d_o, pad), th, tw)
        sl = _tiles(np.pad(lost, pad), th, tw)
        ratios.append(sl / (so + 1e-3 * _TILE * _TILE))
        sos.append(so)
        total = total + so
    # Texture is judged on BOTH directions together. Gating each direction on
    # its own let motion blur escape: the smear flattens the blur direction
    # below the gate, that direction became NaN, and the min fell back to the
    # still-crisp other direction — a smeared tile read as sharp. A tile with
    # edges in one direction only (stripes, a horizon) now reads LOW, which is
    # harmless: regions are summarised by their SHARPEST tiles (see _peak).
    textured = total / (2 * _TILE * _TILE) > 1.5    # mean |diff| in grey levels
    sharp = np.where(textured, np.minimum(ratios[0], ratios[1]), np.nan).astype(np.float32)
    # Per-direction maps for the pan test: a direction is judged only where it
    # has its own edges, so a horizon or a hull (edges in y only) cannot read
    # as "smeared in x".
    gate = 1.0 * _TILE * _TILE
    rx = np.where(sos[0] > gate, ratios[0], np.nan).astype(np.float32)
    ry = np.where(sos[1] > gate, ratios[1], np.nan).astype(np.float32)
    return sharp, (rx, ry)


def _box_tiles(m: np.ndarray, box) -> np.ndarray:
    """Tiles FULLY inside the box. A tile straddling the edge is part sharp
    background, and since regions are summarised by their sharpest tiles, a
    smeared subject in front of a sharp street read as sharp. Tiny boxes with
    no interior tile fall back to every tile they touch."""
    th, tw = m.shape
    x1, y1, x2, y2 = box
    r1, r2 = int(np.ceil(y1 * th)), int(np.floor(y2 * th))
    c1, c2 = int(np.ceil(x1 * tw)), int(np.floor(x2 * tw))
    if r2 - r1 >= 1 and c2 - c1 >= 1:
        return m[r1:r2, c1:c2]
    r1, r2 = int(np.floor(y1 * th)), int(np.ceil(y2 * th))
    c1, c2 = int(np.floor(x1 * tw)), int(np.ceil(x2 * tw))
    return m[r1:max(r2, r1 + 1), c1:max(c2, c1 + 1)]


def _peak(v: np.ndarray) -> float | None:
    """Sharpest part of a region — 90th percentile of its textured tiles.

    A subject only needs SOME sharp detail (eyes, a hull edge) to be in focus;
    the mean would punish a sharp face on a soft-edged body.
    """
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    return float(np.percentile(v, 90))


def measure(img: np.ndarray, boxes: list | None = None) -> dict:
    """Subject vs background sharpness for one decoded image (HWC RGB or gray).

    boxes: normalised [x1, y1, x2, y2] subject boxes, most important first.
    Returns {"subject": float|None, "background": float|None,
             "source": "box"|"peak", "ratio": float|None}.
    """
    m, (rx, ry) = _direction_maps(_gray(img))
    th, tw = m.shape
    mask = np.zeros(m.shape, bool)
    cands = []                              # (peak, box) for the top boxes, in order
    for b in (boxes or [])[:3]:
        mask[int(b[1] * th):int(np.ceil(b[3] * th)), int(b[0] * tw):int(np.ceil(b[2] * tw))] = True
        p = _peak(_box_tiles(m, b))
        if p is not None:
            cands.append((p, b))
    bg = _peak(m[~mask]) if mask.any() else None

    # Pan test on the background: the CRISPEST background detail in each
    # direction. A static scene has crisp detail both ways (masts, windows);
    # a pan smears even its crispest detail along the pan while the other
    # direction stays crisp; defocus softens both. streak = gap between the
    # two directions' peaks.
    streak = None
    if mask.any():
        px, py = _peak(rx[~mask]), _peak(ry[~mask])
        if px is not None and py is not None:
            streak = abs(px - py)

    # WHICH box is the subject depends on whether the frame is a pan:
    #  - static frame: the MAIN subject (first box — callers sort biggest x
    #    most confident first). Taking the sharpest box let a sharp bystander
    #    rescue a smeared main subject: DSC09273's person in the doorway reads
    #    0.40, a smaller person behind 0.56.
    #  - pan: the SHARPEST box. The biggest object in a pan is often a
    #    background ship smeared by the camera move; the tracked subject is the
    #    sharp one (main-subject-only capped 5 good pans, e.g. DSC08943).
    subj, source, chosen = None, "peak", None
    if cands:
        if streak is not None and streak >= PAN_STREAK_MIN:
            subj, chosen = max(cands, key=lambda c: c[0])
        else:
            subj, chosen = cands[0]
        source = "box"
    if chosen is not None:
        sx, sy = [_box_tiles(rx, chosen).ravel()], [_box_tiles(ry, chosen).ravel()]
    else:
        # No detectable subject: the sharpest region of the frame stands in.
        subj = _peak(m)
        sx, sy = [rx.ravel()], [ry.ravel()]
    # Subject streak: gap between the subject's crispest detail along x and
    # along y. Motion blur / shake smears one direction (gap 0.16-0.34 on the
    # 2026-10-03 card); defocus and soft manual-focus or vintage rendering are
    # soft evenly (gap <= 0.03).
    subj_streak = None
    qx, qy = _peak(np.concatenate(sx)), _peak(np.concatenate(sy))
    if qx is not None and qy is not None:
        subj_streak = abs(qx - qy)
    ratio = (subj / bg) if (subj is not None and bg) else None
    return {"subject": subj, "background": bg, "source": source, "ratio": ratio,
            "streak": streak, "subject_streak": subj_streak}


# ── Grading rule ─────────────────────────────────────────────────────────────
# Absolute, never batch-relative. Calibrated 2026-10-03 on 445 Sony ARW
# previews (F:/DCIM/100MSDCF), every frame below 0.63 checked crop-by-crop:
#   < 0.54        all smeared or visibly soft (0.29 0.45 0.46 0.49 0.50 0.53)
#   0.55 - 0.62   MIXED: moderate smears (0.56 0.57 0.59 0.61 0.62) sit next
#                 to sharp low-texture subjects — a white van and a bollard
#                 (graded Strong) at 0.56
#   sharp cat / portraits ≥ 0.61; 34 edited JPEG finals minimum 0.59
# 0.52, not 0.54: DSC08831 (0.534, close portrait in a dim car) looked soft on
# the 1600 px preview but is sharp at full resolution — crisp catchlight,
# lashes, brow hairs. A close face is mostly smooth skin with little fine
# detail, so portraits read low. Under 0.52 on this card: 0.29 0.45 0.46 0.49
# 0.50, all smeared at full size. Moderate smears in the mixed band are
# deliberately left alone: dropping a sharp photo to Weak is a worse error
# than leaving a soft frame in Mid.
SMEARED_BELOW = 0.52
# A smeared subject cannot be better than this — just under the Weak line
# (0.41). A cap, not a deduction: an already-Weak photo is not pushed lower.
SMEARED_CAP = 0.40


# Pan: sharp subject + background smeared along one direction only.
# Calibrated 2026-10-03 on the same card, background crops checked by eye:
# streak >= 0.29 was a real pan in nearly every frame; 0.23-0.28 mixed pans with
# static ships whose long hulls carry edges in one direction only. 34 edited
# JPEG finals: max streak 0.05. 21 frames qualify on that card, all were Mid.
# The boost is additive, not a floor — a weak pan stays below Strong.
PAN_STREAK_MIN = 0.28
PAN_SUBJECT_MIN = 0.60
PAN_BOOST = 0.10
# A GOOD pan keeps the tracked subject sharp in BOTH directions; a failed pan
# smears it along the pan. Tracked-subject smear on the 21 pans of the
# 2026-10-03 card, checked crop by crop: <= 0.16 sharp, 0.18 borderline,
# 0.23 and 0.34 visibly smeared (DSC08947, DSC08906 — the user: "should be
# weak"). Before this check DSC08906 was boosted Mid -> Strong.
PAN_SUBJECT_STREAK_MAX = 0.17      # boost only below this
FAILED_PAN_STREAK = 0.20           # streaked background + subject smeared this much -> Weak


def _pan_background(m: dict) -> bool:
    k = m.get("streak")
    return k is not None and k >= PAN_STREAK_MIN


def is_pan(m: dict | None) -> bool:
    """A GOOD pan: streaked background, sharp tracked subject."""
    if not m or not _pan_background(m):
        return False
    s, g = m.get("subject"), m.get("subject_streak")
    return (s is not None and g is not None
            and s >= PAN_SUBJECT_MIN and g < PAN_SUBJECT_STREAK_MAX)


def is_failed_pan(m: dict | None) -> bool:
    """Camera moved with the subject but the subject smeared anyway."""
    if not m or not _pan_background(m):
        return False
    g = m.get("subject_streak")
    return g is not None and g >= FAILED_PAN_STREAK


# Softness that is EVEN in every direction is not treated as a miss: a manual-
# focus lens wide open, vintage glow, or a slightly soft portrait is a look,
# and the Soft-Focus Gate exists precisely so such frames are not sunk by
# pixel-sharpness. Only DIRECTIONAL smear (motion, shake) is capped at
# SMEARED_BELOW; even softness is capped only when hopeless (< DEFOCUS_BELOW,
# e.g. DSC08829 at 0.29 — nothing in focus anywhere).
# Motion smears measured 0.115-0.34 on the main subject (DSC09032 0.115);
# even softness <= 0.03 (DSC08831 0.003). 0.08 keeps a wide margin for soft
# manual-focus portraits.
MOTION_GAP = 0.08
# Directional smear this strong is Weak at ANY sharpness (checked by eye on
# 24 candidates 2026-10-04: >= 0.20 smeared in nearly all; 08944 at 0.29 the
# one borderline). Equals FAILED_PAN_STREAK — a failed pan is the same defect.
SHAKE_STREAK = 0.20
DEFOCUS_BELOW = 0.40


def is_smeared(m: dict | None) -> bool:
    if not m:
        return False
    s = m.get("subject")
    if s is None:
        return False
    if s < DEFOCUS_BELOW or is_failed_pan(m):
        return True
    # Strong one-direction smear is shake/motion even when some detail
    # survives (2026-10-04: boat-deck shake at 0.53-0.63 sharpness, smeared
    # 0.20-0.32, graded Strong). Good pans never get here: their tracked
    # subject smears < PAN_SUBJECT_STREAK_MAX.
    g0 = m.get("subject_streak")
    if g0 is not None and g0 >= SHAKE_STREAK and not is_pan(m):
        return True
    g = m.get("subject_streak")
    return s < SMEARED_BELOW and g is not None and g >= MOTION_GAP


def apply_cap(score: float, m: dict | None) -> float:
    return min(score, SMEARED_CAP) if is_smeared(m) else score


# ── Batch scoring (runs inside the isolated IQA subprocess) ─────────────────

# A person or animal among the detected subjects — a taste-learner input
# (master_judge.EXTRA "_living"), not a grading rule: penalising scenes with
# no living subject LOWERED agreement with the photographer (2026-10-03).
_LIVING = {"person", "cat", "dog", "bird", "horse", "cow", "sheep"}

def _decode(path: str):
    """RGB uint8 at ~MEASURE_EDGE. RAW → embedded preview (raw_support, same
    path every other stage uses); others → PIL draft decode."""
    import os
    try:
        from raw_support import RAW_EXTS, load_rgb
    except Exception:
        RAW_EXTS, load_rgb = frozenset(), None
    if os.path.splitext(path)[1].lower() in RAW_EXTS:
        if load_rgb is None:
            return None
        img, _ = load_rgb(path, "RGB")
        # Older bodies embed tiny previews (Canon CRW: 640x480). Sharpness is
        # calibrated at ~MEASURE_EDGE and a 640 px preview hides blur, so
        # decode the sensor at half size instead (small sensors: ~1100 px).
        if img is not None and max(img.size) < 1200:
            try:
                import rawpy
                from PIL import Image
                from raw_support import RAWPY_LOCK
                with RAWPY_LOCK, rawpy.imread(path) as r:   # decoded from a thread pool
                    arr = r.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                if max(arr.shape[:2]) > max(img.size):
                    img = Image.fromarray(arr)
            except Exception:
                pass                          # keep the preview — better than nothing
    else:
        from raw_support import jpeg_preview
        img = jpeg_preview(path, MEASURE_EDGE)   # camera JPEG: built-in preview
        if img is None:
            from PIL import Image
            src = Image.open(path)
            try:
                src.draft("RGB", (MEASURE_EDGE, MEASURE_EDGE))
            except Exception:
                pass
            img = src.convert("RGB")
            src.close()
    if img is None:
        return None
    w, h = img.size
    s = MEASURE_EDGE / max(w, h)
    # Within 2% of MEASURE_EDGE, measure as-is (2026-10-04): a LANCZOS shrink
    # of a 1616 px camera preview to 1600 px cost ~34 ms of CPU per photo for
    # a 1% scale change the tile-based sharpness map cannot see.
    import os as _os
    # OFF by default (2026-10-04): skipping the LANCZOS shrink changed 39 of
    # 600 smeared-subject flags — the thresholds are calibrated on resized
    # pixels. Grades must not drift for speed.
    _near = 0.98 if _os.environ.get("FIRSTCUT_SHARP_NEAR_RESIZE_SKIP", "0").strip() == "1" else 1.0
    if s < _near:
        from PIL import Image
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return np.asarray(img)


def score_paths(paths: list[str], progress=None, chunk: int = 16) -> dict:
    """path -> measure() dict (+ "labels"). Memory is O(chunk): decode, detect,
    measure, discard. Any per-photo failure leaves that photo out — the grade
    then simply has no subject-sharpness signal for it (no cap applied)."""
    import dfine_detector as _dfine
    out: dict = {}
    n = len(paths)

    def _decode_retry(p):
        for _try in (1, 2):               # one retry: a RAM squeeze or card I/O hiccup
            try:
                return _decode(p)
            except Exception as e:
                if _try == 2:
                    print(f"[subject_sharp] decode failed {p}: {e}")
        return None

    # Decode each chunk on a thread pool (PIL/rawpy release the GIL). This was
    # one photo at a time — ~45 s per 200 photos on 61 MP JPEGs — while the
    # detector itself costs ~19 ms/img. map() keeps input order, so the
    # detection batches and every result are identical to the serial loop.
    from concurrent.futures import ThreadPoolExecutor
    _threads = getattr(_dfine, "decode_threads", None)
    with ThreadPoolExecutor(max_workers=_threads() if _threads else 4,
                            thread_name_prefix="sharp-decode") as pool:
        for c0 in range(0, n, chunk):
            part = paths[c0:c0 + chunk]
            items = [(p, a) for p, a in zip(part, pool.map(_decode_retry, part))
                     if a is not None]
            _measure_chunk(items, _dfine, out, pool)
            del items
            if progress:
                progress(min(c0 + chunk, n), n)
    flagged = sum(1 for m in out.values() if is_smeared(m))
    pans = sum(1 for m in out.values() if is_pan(m))
    print(f"[subject_sharp] measured {len(out)}/{n}, smeared subject: {flagged}, "
          f"pans: {pans}", flush=True)
    return out


def score_paths_cached(paths: list[str]) -> tuple:
    """(results, misses): measure ONLY photos whose subjects this process's
    person pass already detected (dfine_detector.cached_subjects).

    CPU-only by construction — it never calls the detector — so iqa_worker can
    run it on a side thread while the quality model holds the GPU on the main
    thread (CUDA work stays on the main thread; see IQA_MAIN_THREAD_LOAD).
    `misses` go through score_paths() on the main thread afterwards.
    """
    import dfine_detector as _dfine
    _cached = getattr(_dfine, "cached_subjects", None)
    import os as _os
    if _os.environ.get("FIRSTCUT_DFINE_SHARED_PASS", "0").strip() != "1":
        _cached = None
    hit = [p for p in paths if _cached and _cached(p) is not None]
    hit_set = set(hit)
    misses = [p for p in paths if p not in hit_set]
    return (score_paths(hit) if hit else {}), misses


def _measure_chunk(items, _dfine, out: dict, pool=None) -> None:
    """Detect subjects for one decoded chunk and measure each photo into `out`.

    Photos the person pass already detected in this process reuse its
    all-class boxes (dfine_detector.cached_subjects) instead of a second
    D-FINE pass; only the rest are detected here."""
    import os as _os
    _cached = getattr(_dfine, "cached_subjects", None)
    if _os.environ.get("FIRSTCUT_DFINE_SHARED_PASS", "0").strip() != "1":
        _cached = None                     # kill switch: always run our own pass
    det = {}
    todo = []
    for p, a in items:
        c = _cached(p) if _cached else None
        if c is None:
            todo.append((p, a))
        else:
            det[p] = c
    if todo:
        det.update(_dfine.detect_subjects_from_arrays(todo, conf=0.5))
    def _one(item):
        p, a = item
        try:
            boxes = [d for d in det.get(p, [])
                     if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) >= 0.005]
            # Most important subject first: big and confident.
            boxes.sort(key=lambda d: -((d["bbox"][2] - d["bbox"][0])
                                       * (d["bbox"][3] - d["bbox"][1]) * d["conf"]))
            m = measure(a, [d["bbox"] for d in boxes])
            m["labels"] = [d["label"] for d in boxes[:3]]
            m["living"] = any(d["label"] in _LIVING for d in boxes)
            return p, m
        except Exception as e:
            print(f"[subject_sharp] measure failed {p}: {e}")
            return p, None

    # measure() is pure per-photo numpy/cv2 work — on the pool when given
    # (2026-10-04), in input order, so `out` fills exactly as the loop did.
    for p, m in (pool.map(_one, items) if pool is not None else map(_one, items)):
        if m is not None:
            out[p] = m
