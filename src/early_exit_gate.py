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

_BLUR_VAR_MIN    = 4.0   # only catastrophic blur fails (Laplacian variance)
_CENTER_CROP_PCT = 0.60  # evaluate centre 60% of frame; avoids vignette/border bias


def _blur_score(path: str) -> float:
    """
    Laplacian variance of the centre crop (greyscale).
    Higher = sharper.  Returns 999.0 on load failure (give benefit of doubt).
    """
    try:
        from PIL import Image
        img  = Image.open(path).convert("L")
        w, h = img.size
        cw   = int(w * _CENTER_CROP_PCT)
        ch   = int(h * _CENTER_CROP_PCT)
        x0   = (w - cw) // 2
        y0   = (h - ch) // 2
        arr  = np.array(img.crop((x0, y0, x0 + cw, y0 + ch)), dtype=np.float32)
        # Second-order difference (discrete Laplacian) in both axes
        lap  = np.diff(arr, n=2, axis=0)
        return float(lap.var())
    except Exception:
        return 999.0   # load error → treat as non-blurry, downstream handles failure


def run_early_exit_gate(
    paths: list[str],
    n_workers: int = 8,
    run_yolo: bool = False,    # True only when CD brief implies empty/liminal scenes
    yolo_conf: float = 0.35,
) -> tuple[list[str], set[str], set[str], set[str]]:
    """
    Run Laplacian blur check then (optionally) YOLO person gate.

    Returns:
        survivors          list[str]  paths that passed all checks, original order
        blur_disqualified  set[str]   catastrophically blurry → score 0.00
        yolo_hard          set[str]   person in empty-scene brief → score 0.00
        yolo_soft          set[str]   dark silhouette → -0.15 penalty (IQA still runs)
    """
    if not paths:
        return [], set(), set(), set()

    # ── Step 1: CPU Laplacian blur (parallel I/O + NumPy) ─────────────────
    with ThreadPoolExecutor(max_workers=min(n_workers, len(paths))) as pool:
        blur_scores = list(pool.map(_blur_score, paths))

    blur_disqualified: set[str] = set()
    post_blur: list[str] = []
    for p, v in zip(paths, blur_scores):
        if v < _BLUR_VAR_MIN:
            blur_disqualified.add(p)
            print(
                f"[early_exit] Blur fail: {Path(p).name} "
                f"(Laplacian var={v:.2f} < {_BLUR_VAR_MIN})"
            )
        else:
            post_blur.append(p)

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
    return survivors, blur_disqualified, yolo_hard, yolo_soft
