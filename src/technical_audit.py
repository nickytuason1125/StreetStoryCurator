"""
Per-image technical audit — blur classification, highlight detection, horizon tilt.

Single CPU pass (PIL + NumPy only, no GPU). Runs after the VLM / IQA stack but
before post-fusion grade gates so results can hard-cap scores.

Blur types
----------
  sharp   — centre and overall Laplacian variance both high
  bokeh   — centre significantly sharper than edge corners → intentional depth-of-field
  panning — strong directional gradient + overall soft → intentional motion blur
  shake   — soft overall, not directional, not bokeh → camera shake / missed focus
  severe  — Laplacian var < 4.0 (catastrophic; early_exit_gate would have caught it,
            but may still appear via fast-scan path)

Stored keys in per_photo_breakdowns
-------------------------------------
  blur_type          str    'sharp' | 'bokeh' | 'panning' | 'shake' | 'severe'
  highlight_clip     float  fraction of pixels with any channel > 250
  highlight_spread   bool   True when blown region > 8% of frame (distracting)
  shadow_clip        float  fraction of pixels with all channels < 15
  has_horizon        bool   True when a dominant near-horizontal edge spans > 40% width
  horizon_tilt_deg   float  deviation from level (degrees, 0 = perfectly level)
"""
from __future__ import annotations

import numpy as np
from concurrent.futures import ThreadPoolExecutor


# ── Thresholds ─────────────────────────────────────────────────────────────────
_SEVERE_VAR          = 4.0    # catastrophic blur (matches early_exit_gate)
_SHAKE_VAR           = 30.0   # soft overall → shake / missed focus
_BOKEH_RATIO_MIN     = 2.5    # centre / edge sharpness ratio for bokeh
_BOKEH_CTR_MIN       = 40.0   # centre must itself be sharp for bokeh diagnosis
_PANNING_DIR_RATIO   = 3.0    # horizontal-vs-vertical gradient energy ratio
_PANNING_MAX_VAR     = 120.0  # stop diagnosing panning on genuinely sharp frames
_HIGHLIGHT_DISTRACT  = 0.08   # >8% blown → distracting, not just a bright lamp
_HORIZON_COL_FRAC    = 0.40   # horizon must span this fraction of width
_HORIZON_IQR_FRAC    = 0.15   # max allowed row-variation across columns (as fraction of H)
_HORIZON_GEO_DEG     = 3.0    # tilt threshold for geo/architectural shots
_HORIZON_ANY_DEG     = 10.0   # tilt threshold for any shot (excessive Dutch angle)


# ── Blur classification ────────────────────────────────────────────────────────

def _classify_blur(path: str) -> dict:
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        # Centre zone — inner 40% (subject region)
        cy, cx = h // 2, w // 2
        rh, rw = max(1, int(h * 0.20)), max(1, int(w * 0.20))
        centre = arr[cy - rh : cy + rh, cx - rw : cx + rw]

        # Corner samples — outer 18% each corner (background / edge zone)
        ch, cw = max(1, int(h * 0.18)), max(1, int(w * 0.18))
        corners = [
            arr[:ch, :cw], arr[:ch, -cw:],
            arr[-ch:, :cw], arr[-ch:, -cw:],
        ]

        def _lv(a: np.ndarray) -> float:
            return float(np.diff(a, n=2, axis=0).var()) if a.shape[0] > 2 and a.shape[1] > 1 else 999.0

        full_var   = _lv(arr)
        centre_var = _lv(centre)
        edge_vars  = [_lv(c) for c in corners]
        edge_var   = float(np.mean(edge_vars)) if edge_vars else centre_var
        bokeh_ratio = centre_var / (edge_var + 1e-6)

        # Directional energy: compare horizontal vs vertical first-derivative mean-square
        gh = float(np.mean(np.diff(arr, axis=1) ** 2))
        gv = float(np.mean(np.diff(arr, axis=0) ** 2))
        dir_ratio = max(gh, gv) / (min(gh, gv) + 1e-6)

        # Classification — ordered by severity
        if full_var < _SEVERE_VAR or centre_var < _SEVERE_VAR:
            bt = 'severe'
        elif (dir_ratio >= _PANNING_DIR_RATIO
              and full_var < _PANNING_MAX_VAR
              and bokeh_ratio >= 1.5):
            # Directional + overall soft + centre-biased → panning / intentional motion
            bt = 'panning'
        elif bokeh_ratio >= _BOKEH_RATIO_MIN and centre_var >= _BOKEH_CTR_MIN:
            # Centre much sharper than edges → depth-of-field / bokeh
            bt = 'bokeh'
        elif full_var < _SHAKE_VAR and centre_var < _SHAKE_VAR:
            # Uniformly soft — neither directional nor bokeh → shake / missed focus
            bt = 'shake'
        else:
            bt = 'sharp'

        return {
            'blur_type':   bt,
            'centre_lap':  round(centre_var, 2),
            'edge_lap':    round(edge_var,   2),
            'bokeh_ratio': round(bokeh_ratio, 2),
            'full_lap':    round(full_var,   2),
        }

    except Exception:
        return {
            'blur_type': 'sharp', 'centre_lap': 999.0,
            'edge_lap': 999.0, 'bokeh_ratio': 1.0, 'full_lap': 999.0,
        }


# ── Highlight + shadow audit ───────────────────────────────────────────────────

def _audit_highlights(path: str) -> dict:
    try:
        from PIL import Image
        img  = Image.open(path).convert("RGB")
        arr  = np.array(img, dtype=np.uint8)
        tot  = arr.shape[0] * arr.shape[1]

        blown   = np.any(arr > 250, axis=2)
        crushed = np.all(arr < 15,  axis=2)

        hc = float(blown.sum() / tot)
        sc = float(crushed.sum() / tot)

        return {
            'highlight_clip':   round(hc, 4),
            'highlight_spread': hc > _HIGHLIGHT_DISTRACT,
            'shadow_clip':      round(sc, 4),
        }

    except Exception:
        return {'highlight_clip': 0.0, 'highlight_spread': False, 'shadow_clip': 0.0}


# ── Horizon tilt detection ─────────────────────────────────────────────────────

def _horizon_audit(path: str) -> dict:
    """
    Detect whether a clear horizon exists and measure its tilt.

    Strategy: compute vertical Sobel (detects horizontal edges); for each column
    find the row with the strongest horizontal edge. If those peak rows are
    consistent across >= 40% of the frame width, a horizon is present. Linear
    regression over peak-row vs column gives the tilt slope → degrees.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        # Downsample for speed; 400×300 is sufficient for line detection
        img = img.resize((400, 300), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        # Vertical gradient (absolute): detects horizontal edges
        Gy = np.abs(np.diff(arr, axis=0))   # (H-1, W)

        # Per-column: row of strongest horizontal edge and its strength
        col_peak_row = np.argmax(Gy, axis=0).astype(np.float32)   # (W,)
        col_peak_val = Gy.max(axis=0)                               # (W,)

        # Only consider columns whose edge is in the top-15% magnitude
        strong_thresh = float(np.percentile(Gy, 85))
        good_cols     = col_peak_val > strong_thresh
        n_good        = int(good_cols.sum())

        if n_good < int(w * _HORIZON_COL_FRAC):
            return {'has_horizon': False, 'horizon_tilt_deg': 0.0}

        peak_rows = col_peak_row[good_cols]
        row_iqr   = float(np.percentile(peak_rows, 75) - np.percentile(peak_rows, 25))

        if row_iqr > h * _HORIZON_IQR_FRAC:
            # Peak rows vary too much — multiple competing edges, not a single horizon
            return {'has_horizon': False, 'horizon_tilt_deg': 0.0}

        # Linear regression to measure tilt
        good_col_idx = np.where(good_cols)[0].astype(np.float32)
        slope = float(np.polyfit(good_col_idx, peak_rows, 1)[0])

        # slope is in image-pixels/image-pixel (rows per column).
        # With a 400×300 downsample the aspect ratio is already baked in.
        tilt_deg = abs(float(np.degrees(np.arctan(slope))))

        return {'has_horizon': True, 'horizon_tilt_deg': round(tilt_deg, 1)}

    except Exception:
        return {'has_horizon': False, 'horizon_tilt_deg': 0.0}


# ── Public entry point ─────────────────────────────────────────────────────────

def _audit_single(path: str) -> dict:
    return {**_classify_blur(path), **_audit_highlights(path), **_horizon_audit(path)}


def run_technical_audit(paths: list[str], n_workers: int = 8) -> list[dict]:
    """
    Run all three audits for every path in parallel (CPU threads).
    Returns list[dict] in the same order as input.
    """
    if not paths:
        return []

    with ThreadPoolExecutor(max_workers=min(n_workers, len(paths))) as pool:
        results = list(pool.map(_audit_single, paths))

    blur_counts: dict[str, int] = {}
    for r in results:
        k = r.get('blur_type', 'sharp')
        blur_counts[k] = blur_counts.get(k, 0) + 1
    n_blown  = sum(1 for r in results if r.get('highlight_spread'))
    n_horiz  = sum(1 for r in results if r.get('has_horizon'))
    n_tilted = sum(1 for r in results if r.get('has_horizon') and r.get('horizon_tilt_deg', 0) > _HORIZON_GEO_DEG)

    print(
        f"[tech_audit] blur={blur_counts}  "
        f"blown={n_blown}/{len(paths)}  "
        f"horizon={n_horiz} (tilted>{_HORIZON_GEO_DEG}°: {n_tilted})"
    )
    return results
