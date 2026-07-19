"""
Cascaded early-exit gate — runs before any GPU model loads.

Evaluation order (fastest first):
  Step 1: CPU Laplacian blur check  — pure NumPy, parallel across cores
  Step 2: YOLO11-Seg person gate    — brief-conditional, CPU inference

Images that fail either step are assigned score 0.00 and skipped in all
downstream GPU stages (SigLIP-2, Depth, Aesthetic Predictor, etc.).

Blur threshold (_BLUR_VAR_MIN = 4.0) is intentionally very permissive:
  - Vintage lens softness          ≈ Laplacian var 30–150
  - Intentional motion blur        ≈ var 10–50
  - Atmospheric/foggy fine art     ≈ var 8–40
  - Truly unrecoverable blur/OOF   ≈ var 0–4   ← only these are failed
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import os as _os

# "Forgiving Bouncer": only CATASTROPHIC blur is hard-rejected here. The floor is
# intentionally very low (Laplacian variance < 4.0) so that intentional motion
# blur, vintage soft-focus, film grain, and atmospheric/foggy fine-art shots all
# PASS this gate and are graded downstream — where the SigLIP zero-shot grader's
# "intentional soft focus / vintage" positive probe gives them a fair score
# (intent-aware amnesty). Do NOT raise this to catch soft photos; that is the
# grader's job, not the bouncer's. Tunable via FRAMEGRADE_BLUR_MIN for the field.
try:
    _BLUR_VAR_MIN = float(_os.environ.get("FRAMEGRADE_BLUR_MIN", "4.0"))
except (TypeError, ValueError):
    _BLUR_VAR_MIN = 4.0
_CENTER_CROP_PCT = 0.60  # evaluate centre 60% of frame; avoids vignette/border bias

# Flat-block technical floor: a grey/black/white/empty frame (half-written, locked,
# or corrupt decode) has pixel std-dev near zero. Below this it is a TECHNICAL
# FAILURE — the AI models would otherwise hallucinate a composition on a void.
# Tunable via FRAMEGRADE_FLAT_STD.
try:
    _FLAT_STD_MIN = float(_os.environ.get("FRAMEGRADE_FLAT_STD", "2.0"))
except (TypeError, ValueError):
    _FLAT_STD_MIN = 2.0


def _file_ready(path: str, tries: int = 3, backoff: float = 0.10) -> bool:
    """Phase 1 — file lock & completeness guard.

    Returns True only when the file exists, is non-empty, its size is STABLE
    across a short settle window (i.e. not still being written/flushed), and it
    can be opened for reading (not exclusively locked by another writer). Retries
    up to `tries` times with `backoff` between attempts to ride out a RAW export /
    copy still flushing to disk. Returns False if it never becomes ready — the
    caller then marks it a technical failure instead of feeding a half-written
    buffer to the decoder. Cheap on the happy path (existing files pass on try 1)."""
    import time as _t
    for _ in range(max(1, tries)):
        try:
            s1 = _os.path.getsize(path)
            if s1 <= 0:
                raise OSError("zero-size / not yet flushed")
            _t.sleep(0.02)                         # settle window to catch active writes
            s2 = _os.path.getsize(path)
            if s1 == s2:
                with open(path, "rb") as _fh:      # catches sharing/lock violations
                    _fh.read(1)
                return True
        except (OSError, PermissionError, IOError):
            pass
        _t.sleep(backoff)
    return False


def _technical_inspect(path: str) -> dict:
    """Multi-layer technical inspection performed BEFORE any AI model touches the
    pixels. Returns one of:
        {"reason": None,         "blur": <var>}   passed — real photographic content
        {"reason": "unreadable", "blur": 0.0}     not ready / decode failed / None / empty
        {"reason": "flat",       "blur": 0.0}     std < floor (grey/black/white/blank block)

    RAW POLICY (2026-07-04): camera RAW gets the Phase-1 readiness check ONLY — its
    pixels are NOT decoded here. Two hard-won reasons:
      1. Crash: decoding every RAW's embedded preview via libraw across this 8-thread
         gate is a native-crash risk in the memory-tight windowed grade worker
         (libraw is not reliably thread-safe and allocates real memory per file) —
         it killed the worker with 0xC0000005 right after the gate.
      2. Over-reject: the Laplacian blur floor is calibrated on full JPEGs; a RAW's
         embedded preview has a different variance scale, so the floor misfired and
         hard-rejected ~6% of a real RW2 shoot before the SigLIP grader's soft-focus
         amnesty could even see them.
    Corrupt RAWs are still dropped downstream by the encoder (zero-vector sentinel),
    and a flat RAW simply scores low via SigLIP — no model hallucinates on a void.
    Non-RAW formats (JPG/PNG/…) still get the full flat/void + blur inspection with a
    fast, thread-safe PIL decode — that's where the half-written / grey-square export
    race actually happens."""
    # Phase 1 — file readiness / lock / completeness (cheap, no image decode).
    if not _file_ready(path):
        return {"reason": "unreadable", "blur": 0.0}
    try:
        _ext = _os.path.splitext(path)[1].lower()
        from raw_support import RAW_EXTS
    except Exception:
        _ext, RAW_EXTS = "", frozenset()
    # RAW: readiness-only — never decode libraw in this parallel gate.
    if _ext in RAW_EXTS:
        return {"reason": None, "blur": 999.0}
    # Non-RAW: full flat/void + blur inspection (fast, thread-safe PIL decode).
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        if img is None:
            return {"reason": "unreadable", "blur": 0.0}
        arr = np.asarray(img, dtype=np.float32)
    except Exception:
        return {"reason": "unreadable", "blur": 0.0}
    if arr.size == 0:
        return {"reason": "unreadable", "blur": 0.0}
    # Phase 2 — flat-block check: std over the whole frame catches grey/black/white voids.
    if float(arr.std()) < _FLAT_STD_MIN:
        return {"reason": "flat", "blur": 0.0}
    # Blur — Laplacian variance on the centre crop (reuse the already-loaded array).
    h, w = arr.shape[:2]
    cw = int(w * _CENTER_CROP_PCT); ch = int(h * _CENTER_CROP_PCT)
    x0 = (w - cw) // 2; y0 = (h - ch) // 2
    crop = arr[y0:y0 + ch, x0:x0 + cw]
    lap  = np.diff(crop, n=2, axis=0)
    return {"reason": None, "blur": float(lap.var()) if lap.size else 999.0}


def run_early_exit_gate(
    paths: list[str],
    n_workers: int = 8,
    run_yolo: bool = False,    # True only when CD brief implies empty/liminal scenes
    yolo_conf: float = 0.35,
) -> tuple[list[str], set[str], set[str], set[str], dict[str, str]]:
    """
    Technical inspection (readiness → flat/void → blur) then optional YOLO gate.

    Returns:
        survivors          list[str]       paths that passed all checks, original order
        blur_disqualified  set[str]        catastrophically blurry → score 0.00
        yolo_hard          set[str]        person in empty-scene brief → score 0.00
        yolo_soft          set[str]        dark silhouette → -0.15 penalty (IQA still runs)
        technical_disq     dict[str,str]   path → "flat" | "unreadable" (score 0.00, models skipped)
    """
    if not paths:
        return [], set(), set(), set(), {}

    # ── Step 1: technical inspection (readiness + flat/void + blur), parallel ──
    with ThreadPoolExecutor(max_workers=min(n_workers, len(paths))) as pool:
        results = list(pool.map(_technical_inspect, paths))

    technical_disq:    dict[str, str] = {}
    blur_disqualified: set[str] = set()
    post_blur: list[str] = []
    for p, r in zip(paths, results):
        _reason = r.get("reason")
        if _reason:   # "flat" or "unreadable" — never reaches any model
            technical_disq[p] = _reason
            print(f"[early_exit] Technical fail ({_reason}): {Path(p).name}")
            continue
        _v = r.get("blur", 999.0)
        if _v < _BLUR_VAR_MIN:
            blur_disqualified.add(p)
            print(
                f"[early_exit] Blur fail: {Path(p).name} "
                f"(Laplacian var={_v:.2f} < {_BLUR_VAR_MIN})"
            )
        else:
            post_blur.append(p)

    if technical_disq:
        _nflat = sum(1 for v in technical_disq.values() if v == "flat")
        print(
            f"[early_exit] Technical gate: {len(technical_disq)}/{len(paths)} failed "
            f"({_nflat} flat/blank, {len(technical_disq) - _nflat} unreadable/locked)"
            f" -> score 0.00, all models skipped"
        )
    if blur_disqualified:
        print(
            f"[early_exit] Blur gate: {len(blur_disqualified)}/{len(paths)} failed "
            f"(Laplacian var < {_BLUR_VAR_MIN})"
        )

    # ── Step 2: YOLO11-Seg person gate (brief-conditional) ────────────────
    yolo_hard: set[str] = set()
    yolo_soft: set[str] = set()

    if run_yolo and post_blur:
        try:
            from yolo_auditor import audit_paths, unload as _yu
            yolo_hard, yolo_soft = audit_paths(post_blur, conf=yolo_conf)
            _yu()
            if yolo_hard:
                print(f"[early_exit] YOLO gate: {len(yolo_hard)} hard-disqualified "
                      f"(person in empty-scene brief)")
            if yolo_soft:
                print(f"[early_exit] YOLO gate: {len(yolo_soft)} soft-penalized "
                      f"(dark silhouette, -0.15)")
        except Exception as exc:
            print(f"[early_exit] YOLO gate failed ({exc}) — skipping")

    survivors = [p for p in post_blur if p not in yolo_hard]
    return survivors, blur_disqualified, yolo_hard, yolo_soft, technical_disq
