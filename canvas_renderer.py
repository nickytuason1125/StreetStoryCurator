#!/usr/bin/env python3
"""
canvas_renderer.py -- Eye Feature Overlay Engine

Converts Qwen 2.5-VL bounding boxes (0-1000 normalized) into pixel-accurate
pen-drawn overlays for the frontend Eye Feature button.

Three structural layers on every overlay
-----------------------------------------
  Crimson box   (4 px, RGBA 235,52,52,180)   subject_bbox + area_pct label
  Cyan axis     (3 px, RGBA 52,186,235,150)  horizontal center-line of
                                              compositional_axis_box
  Yellow arrow  (2 px, RGBA 255,200,0,210)   h_gap measurement rule,
                                              labeled with pixel distance

Optional calibration panel (top-right corner)
----------------------------------------------
  Loaded from calibration_telemetry.json when present.
  Shows per-slot calibrated mean, delta, and gate pass/fail.

Public API
----------
draw_validation_overlay(src, coordinates, output_path, area_pct, h_gap,
                        thresholds=None, slot_name="") -> str

render_story_overlays(slot_results, output_dir) -> list[dict]
    Batch critique overlays from live SlotResult objects (Stage 4a).

render_validation_overlays(manifest_path, output_dir) -> list[dict]
    Batch overlays from final_story_manifest.json.
    Writes eye_overlay_url back into the manifest atomically.

Output: ./static/eye_feature_overlays/verified_{image_id}.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths and palette
# ---------------------------------------------------------------------------

_ROOT        = Path(__file__).parent
_OVERLAY_DIR = _ROOT / "static" / "eye_feature_overlays"
_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

_TELEMETRY_PATH = _ROOT / "calibration_telemetry.json"

# Spec-accurate RGBA values
_CRIMSON_RGBA = (235,  52,  52, 180)   # subject bounding box
_CYAN_RGBA    = ( 52, 186, 235, 150)   # compositional axis line
_YELLOW_RGBA  = (255, 200,   0, 210)   # h_gap measurement arrow
_TEXT_RGBA    = (255, 255, 255, 240)
_SHADOW_RGBA  = (  0,   0,   0, 200)
_PANEL_BG     = ( 10,  10,  10, 195)
_GREEN_RGBA   = ( 80, 220, 100, 230)   # gate PASS
_AMBER_RGBA   = (255, 165,   0, 230)   # gate WARN
_RED_RGBA     = (220,  70,  70, 230)   # gate FAIL / OUT
_MAGENTA_RGBA = (220,  60, 200, 220)   # horizon tilt line

# Fallback bbox emitted when Qwen did not produce real coordinates
_FALLBACK_BBOX = [0, 0, 1000, 1000]


# =============================================================================
# Calibration telemetry
# =============================================================================

def load_calibration_telemetry(path: str | Path = _TELEMETRY_PATH) -> dict:
    """Return calibration_telemetry.json content, or {} if absent."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[canvas] WARNING: could not load telemetry ({exc})")
        return {}


def _gate_status(
    value: float,
    mean:  Optional[float],
    std:   Optional[float],
) -> tuple[str, tuple]:
    """
    z-score gate: |z|<=1 -> PASS/green, |z|<=2 -> WARN/amber, else OUT/red.
    Falls back to N/A / amber when no calibration data.
    """
    if mean is None or std is None:
        return "N/A", _AMBER_RGBA
    z = (value - mean) / max(std, 0.5)
    if abs(z) <= 1.0:
        return "PASS", _GREEN_RGBA
    if abs(z) <= 2.0:
        return "WARN", _AMBER_RGBA
    return "OUT", _RED_RGBA


# =============================================================================
# Helpers
# =============================================================================

def _has_spatial_data(entry: dict) -> bool:
    """True when the entry has real Qwen-derived bboxes (not full-frame fallback)."""
    facts = entry.get("spatial_facts")
    if not facts:
        return False
    bbox = entry.get("subject_bbox", _FALLBACK_BBOX)
    return bbox != _FALLBACK_BBOX


def _load_font(size: int = 14):
    from PIL import ImageFont
    candidates = [
        _ROOT / "static" / "fonts" / "DejaVuSans.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/verdana.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                continue
    return ImageFont.load_default()


def _scale_bbox(bbox: list[int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """[ymin, xmin, ymax, xmax] in 0-1000 → (x0, y0, x1, y1) in pixels."""
    ymin, xmin, ymax, xmax = bbox
    x0 = max(0, min(int(round(xmin / 1000.0 * img_w)), img_w - 1))
    y0 = max(0, min(int(round(ymin / 1000.0 * img_h)), img_h - 1))
    x1 = max(0, min(int(round(xmax / 1000.0 * img_w)), img_w - 1))
    y1 = max(0, min(int(round(ymax / 1000.0 * img_h)), img_h - 1))
    return x0, y0, x1, y1


def _text_size(font, text: str) -> tuple[int, int]:
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return len(text) * 7, 12


def _draw_label(
    draw, text: str, x: int, y: int, font,
    color: tuple = _TEXT_RGBA,
    img_w: int = 9999,
    img_h: int = 9999,
) -> None:
    tw, th = _text_size(font, text)
    x = max(2, min(x, img_w - tw - 4))
    y = max(2, min(y, img_h - th - 4))
    draw.text((x + 1, y + 1), text, font=font, fill=_SHADOW_RGBA)
    draw.text((x,     y    ), text, font=font, fill=color)


def _draw_pen_rect(draw, x0, y0, x1, y1, color: tuple, width: int) -> None:
    """Draw a rectangle using four lines (no solid fill)."""
    draw.line([(x0, y0), (x1, y0)], fill=color, width=width)
    draw.line([(x1, y0), (x1, y1)], fill=color, width=width)
    draw.line([(x1, y1), (x0, y1)], fill=color, width=width)
    draw.line([(x0, y1), (x0, y0)], fill=color, width=width)


def _draw_dimension_arrow(
    draw,
    x0: int, y: int, x1: int,
    color: tuple, width: int,
    img_w: int = 9999,
    img_h: int = 9999,
) -> None:
    """
    Horizontal dimensional arrow from (x0, y) to (x1, y).
    Arrowhead at x1, origin tick at x0.
    Suppressed when |x1 - x0| < 6 px.
    """
    if abs(x1 - x0) < 6:
        return
    x0 = max(0, min(x0, img_w - 1))
    x1 = max(0, min(x1, img_w - 1))
    y  = max(0, min(y,  img_h - 1))

    draw.line([(x0, y), (x1, y)], fill=color, width=width)

    direction = 1 if x1 > x0 else -1
    head_len  = max(10, width * 4)
    head_half = max(5,  width * 2)
    tip   = (x1, y)
    base1 = (x1 - direction * head_len, y - head_half)
    base2 = (x1 - direction * head_len, y + head_half)
    draw.polygon([tip, base1, base2], fill=color)

    # Origin tick mark
    draw.line([(x0, y - head_half), (x0, y + head_half)],
              fill=color, width=max(1, width - 1))


def _save_png_atomic(img, out_path: Path) -> None:
    """Write to a .tmp file, then os.replace() to final path. Preserves RGBA for transparent overlays."""
    tmp = out_path.with_suffix(".tmp.png")
    out_img = img if img.mode in ("RGBA", "LA") else img.convert("RGB")
    out_img.save(str(tmp), format="PNG", optimize=True)
    os.replace(str(tmp), str(out_path))


# =============================================================================
# Calibration panel
# =============================================================================

def _draw_calibration_panel(
    draw,
    img_w:     int,
    img_h:     int,
    font_xs,
    slot_name: str,
    area_pct:  float,
    h_gap:     int,
    telemetry: dict,
) -> None:
    """Semi-transparent panel in the top-right corner showing gate results."""
    slot_stats  = telemetry.get("slot_stats", {})
    cal_thresh  = telemetry.get("calibrated_thresholds", {})
    this_slot   = slot_stats.get(slot_name, {})
    area_stats  = this_slot.get("area_pct", {})
    hgap_stats  = this_slot.get("h_gap",    {})

    dom_thresh  = cal_thresh.get("DOMINANT_AREA_PCT",  20.0)
    h_thresh    = cal_thresh.get("H_CONTRAST_THRESH", 161.0)

    area_mean   = area_stats.get("mean")
    area_std    = area_stats.get("std")
    hgap_mean   = hgap_stats.get("mean")
    hgap_std    = hgap_stats.get("std")

    area_status, area_col = _gate_status(area_pct, area_mean, area_std)
    hgap_status, hgap_col = _gate_status(
        abs(h_gap),
        abs(hgap_mean) if hgap_mean is not None else None,
        hgap_std,
    )

    dom_pass  = area_pct < dom_thresh
    dom_label = "DOM GATE: PASS" if dom_pass else "DOM GATE: FAIL"
    dom_color = _GREEN_RGBA if dom_pass else _RED_RGBA

    h_strong  = abs(h_gap) > h_thresh
    h_label   = "CONTRAST: STRONG" if h_strong else "CONTRAST: SOFT"
    h_color   = _YELLOW_RGBA if h_strong else (180, 180, 180, 200)

    lines: list[tuple[str, tuple]] = [
        (f"SLOT  {slot_name}",                   _TEXT_RGBA),
        (f"AREA  {area_pct:.1f}%",               _TEXT_RGBA),
    ]
    if area_mean is not None:
        lines.append((
            f"      CAL {area_mean:.1f}  D{area_pct - area_mean:+.1f}",
            area_col,
        ))
    lines.append((f"      {area_status}  |  {dom_label}", dom_color if not dom_pass else area_col))
    lines.append((f"HGAP  {h_gap:+d}",               _TEXT_RGBA))
    if hgap_mean is not None:
        lines.append((
            f"      CAL {hgap_mean:+.0f}  D{abs(h_gap) - abs(hgap_mean):+.0f}",
            hgap_col,
        ))
    lines.append((f"      THR {h_thresh:.0f}  |  {h_label}", h_color))

    padding  = 8
    line_gap = 3
    _, char_h = _text_size(font_xs, "W")
    line_h   = char_h + line_gap
    max_tw   = max(_text_size(font_xs, t)[0] for t, _ in lines)
    panel_w  = max_tw + padding * 2
    panel_h  = len(lines) * line_h + padding * 2

    px0 = img_w - panel_w - 8
    py0 = 8
    px1 = img_w - 8
    py1 = py0 + panel_h

    draw.rectangle([(px0, py0), (px1, py1)], fill=_PANEL_BG)
    draw.rectangle([(px0, py0), (px1, py1)], outline=(255, 255, 255, 55), width=1)

    tx, ty = px0 + padding, py0 + padding
    for text, color in lines:
        draw.text((tx + 1, ty + 1), text, font=font_xs, fill=_SHADOW_RGBA)
        draw.text((tx,     ty    ), text, font=font_xs, fill=color)
        ty += line_h


# =============================================================================
# Technical audit overlay layers (blur / highlights / horizon)
# =============================================================================

def _draw_technical_audit_layers(
    draw,
    overlay,            # PIL Image RGBA — modified in-place via alpha_composite
    src_path: str,
    img_w: int,
    img_h: int,
    font_sm,
    font_xs,
) -> None:
    """
    Adds three optional visual layers derived from technical_audit.py:

      Layer 5 — Blur type
          shake  → red corner hatching + "CAMERA SHAKE" label (bottom-centre)
          bokeh  → green dashed centre rectangle + "BOKEH ✓"
          panning→ blue horizontal motion bars + "PANNING ✓"
          severe → red diagonal cross + "SEVERE BLUR"
          sharp  → no mark

      Layer 6 — Highlight clipping
          If highlight_spread: amber pixel mask over blown regions + "BLOWN N%"

      Layer 7 — Horizon tilt
          If has_horizon and tilt > 3°: magenta tilted line + level reference +
          "HORIZON +N.N°" label.  Green "LEVEL ✓" when tilt ≤ 3°.
    """
    import math
    import numpy as np
    from PIL import Image, ImageDraw

    try:
        import sys, os
        _src_dir = os.path.join(os.path.dirname(__file__), "src")
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)
        from technical_audit import _audit_single
        audit = _audit_single(src_path)
    except Exception as _e:
        print(f"[canvas] tech audit skipped: {_e}")
        return

    blur_type  = audit.get("blur_type",         "sharp")
    hl_spread  = audit.get("highlight_spread",   False)
    hl_clip    = audit.get("highlight_clip",     0.0)
    has_horiz  = audit.get("has_horizon",        False)
    tilt_deg   = audit.get("horizon_tilt_deg",   0.0)

    # ── Layer 5: Blur type ─────────────────────────────────────────────────────
    if blur_type in ("shake", "severe"):
        # Red diagonal hatching in all four corners
        corner_w = img_w // 5
        corner_h = img_h // 5
        _hatch_color = (220, 50, 50, 140)
        hatch_step = max(8, corner_w // 8)
        for cx0, cy0, cx1, cy1 in [
            (0,             0,              corner_w,       corner_h),
            (img_w-corner_w, 0,             img_w,          corner_h),
            (0,             img_h-corner_h, corner_w,       img_h),
            (img_w-corner_w, img_h-corner_h, img_w,         img_h),
        ]:
            for off in range(-max(corner_w, corner_h), max(corner_w, corner_h), hatch_step):
                x0c = max(cx0, cx0 + off)
                y0c = max(cy0, cy0)
                x1c = min(cx1, cx0 + off + max(corner_w, corner_h))
                y1c = min(cy1, cy0 + max(corner_w, corner_h))
                draw.line([(x0c, cy0), (cx0 + off, cy0 + max(corner_w, corner_h))],
                          fill=_hatch_color, width=2)

        lbl = "SEVERE BLUR" if blur_type == "severe" else "⚠ CAMERA SHAKE"
        lw, lh = _text_size(font_sm, lbl)
        _draw_label(draw, lbl, (img_w - lw) // 2, img_h - lh - 14,
                    font_sm, color=_RED_RGBA, img_w=img_w, img_h=img_h)

    elif blur_type == "bokeh":
        # Green dashed rectangle around the centre zone (inner 40%)
        bx0 = int(img_w * 0.30)
        by0 = int(img_h * 0.30)
        bx1 = int(img_w * 0.70)
        by1 = int(img_h * 0.70)
        dash, gap = 14, 8
        for axis in ["top", "bottom", "left", "right"]:
            if axis == "top":
                pts = [(x, by0) for x in range(bx0, bx1, dash + gap)]
                for px in pts:
                    draw.line([px, (min(px[0] + dash, bx1), by0)],
                              fill=_GREEN_RGBA, width=2)
            elif axis == "bottom":
                pts = [(x, by1) for x in range(bx0, bx1, dash + gap)]
                for px in pts:
                    draw.line([px, (min(px[0] + dash, bx1), by1)],
                              fill=_GREEN_RGBA, width=2)
            elif axis == "left":
                pts = [(bx0, y) for y in range(by0, by1, dash + gap)]
                for py in pts:
                    draw.line([py, (bx0, min(py[1] + dash, by1))],
                              fill=_GREEN_RGBA, width=2)
            elif axis == "right":
                pts = [(bx1, y) for y in range(by0, by1, dash + gap)]
                for py in pts:
                    draw.line([py, (bx1, min(py[1] + dash, by1))],
                              fill=_GREEN_RGBA, width=2)
        _draw_label(draw, "BOKEH ✓", bx0 + 4, by0 - _text_size(font_xs, "W")[1] - 6,
                    font_xs, color=_GREEN_RGBA, img_w=img_w, img_h=img_h)

    elif blur_type == "panning":
        # Three horizontal blue motion-bar stripes in the vertical centre band
        bar_y_base = img_h // 2
        for offset, alpha in [(-20, 80), (0, 130), (20, 80)]:
            y = bar_y_base + offset
            bar_color = (52, 186, 235, alpha)
            draw.line([(0, y), (img_w, y)], fill=bar_color, width=3)
        _draw_label(draw, "→ PANNING ✓", 12, bar_y_base - 20,
                    font_xs, color=_CYAN_RGBA, img_w=img_w, img_h=img_h)

    # ── Layer 6: Highlight clipping mask ──────────────────────────────────────
    if hl_spread:
        try:
            with Image.open(src_path) as _orig:
                orig_np = np.array(_orig.convert("RGB"), dtype=np.uint8)
            blown = np.any(orig_np > 250, axis=2)               # H×W boolean
            # Resize blown mask to overlay dimensions if needed
            if blown.shape != (img_h, img_w):
                from PIL import Image as _PILI
                blown_img = _PILI.fromarray(blown.astype(np.uint8) * 255, mode="L")
                blown_img = blown_img.resize((img_w, img_h), _PILI.NEAREST)
                blown = np.array(blown_img) > 127
            hl_arr          = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            hl_arr[blown]   = [255, 165, 0, 110]                # amber, 43% opacity
            hl_layer        = Image.fromarray(hl_arr, mode="RGBA")
            overlay.alpha_composite(hl_layer)
            # Re-bind draw to the updated overlay
            draw._image = overlay
        except Exception as _he:
            print(f"[canvas] highlight mask failed: {_he}")

        lbl_h = f"BLOWN HIGHLIGHTS {round(hl_clip * 100)}%"
        lw, lh = _text_size(font_xs, lbl_h)
        # Place in top-left, below any existing corner labels
        _draw_label(draw, lbl_h, 10, 10,
                    font_xs, color=_AMBER_RGBA, img_w=img_w, img_h=img_h)

    # ── Layer 7: Horizon tilt line ─────────────────────────────────────────────
    if has_horiz:
        cy = img_h // 2
        tilt_rad = math.radians(tilt_deg)
        dy = int((img_w / 2) * math.tan(tilt_rad))

        if tilt_deg > 3.0:
            # Magenta tilted horizon line
            draw.line([(0, cy + dy), (img_w, cy - dy)],
                      fill=_MAGENTA_RGBA, width=2)
            # Thin white reference (what "level" looks like)
            draw.line([(0, cy), (img_w, cy)],
                      fill=(255, 255, 255, 60), width=1)
            lbl_horiz = f"HORIZON +{tilt_deg}° TILT"
            _draw_label(draw, lbl_horiz, 10, cy + dy + 6,
                        font_xs, color=_MAGENTA_RGBA, img_w=img_w, img_h=img_h)
        else:
            # Level horizon — green confirmation tick
            draw.line([(0, cy), (img_w, cy)],
                      fill=_GREEN_RGBA[:3] + (80,), width=1)
            _draw_label(draw, "LEVEL ✓", 10, cy + 4,
                        font_xs, color=_GREEN_RGBA, img_w=img_w, img_h=img_h)


# =============================================================================
# Core overlay renderer
# =============================================================================

def draw_validation_overlay(
    src_image_path: str,
    coordinates:    dict,
    output_path:    str,
    area_pct:       float,
    h_gap:          int,
    thresholds:     Optional[dict] = None,
    slot_name:      str            = "",
) -> str:
    """
    Render three structural layers onto a photograph and save as PNG.

    Layer 1 — Crimson box (4 px, RGBA 235,52,52,180)
        subject_bbox outline.  Label inside top-left corner: "N.N% frame".
        Label above top edge: "[subject_label]".

    Layer 2 — Cyan vector line (3 px, RGBA 52,186,235,150)
        Horizontal center-line of compositional_axis_box from left to right
        edge, with tick marks at both ends.
        Label above: "[anchor_label]".

    Layer 3 — Yellow dimension arrow (2 px, RGBA 255,200,0,210)
        Drawn between the subject box edge and the axis line edge, spanning
        the h_gap.  Label: "h_gap: {px}px" centered above the shaft.
        Suppressed when |h_gap| < 10 units (negligible gap).

    Layer 4 (optional) — Calibration panel (top-right corner)
        Rendered when `thresholds` is the calibration_telemetry dict.

    Parameters
    ----------
    src_image_path : absolute path to the source photograph.
    coordinates    : dict with keys:
                       subject_bbox           [ymin,xmin,ymax,xmax] 0-1000
                       subject_label          str
                       compositional_axis_box [ymin,xmin,ymax,xmax] 0-1000
                           (falls back to anchor_bbox if absent)
                       anchor_label           str
    output_path    : destination PNG file path.
    area_pct       : subject frame-area % from spatial_facts.
    h_gap          : horizontal gap in 0-1000 units from spatial_facts.
    thresholds     : calibration_telemetry dict, or None to skip the panel.
    slot_name      : story slot name for per-slot calibration lookup.

    Returns the resolved output path as a string.
    """
    from PIL import Image, ImageDraw

    src = Path(src_image_path)
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")

    with Image.open(src) as raw:
        img = raw.convert("RGBA")
    img_w, img_h = img.size

    # Resolve bbox keys — support both manifest field names
    subject_bbox  = coordinates.get("subject_bbox",   _FALLBACK_BBOX)
    subject_label = coordinates.get("subject_label",  "Subject")
    axis_bbox     = coordinates.get("compositional_axis_box",
                     coordinates.get("anchor_bbox",   _FALLBACK_BBOX))
    anchor_label  = coordinates.get("anchor_label",   "Axis")

    # Denormalize to pixel coordinates
    sx0, sy0, sx1, sy1 = _scale_bbox(subject_bbox, img_w, img_h)
    ax0, ay0, ax1, ay1 = _scale_bbox(axis_bbox,    img_w, img_h)

    # h_gap in pixels (0-1000 → image width)
    h_gap_px = int(round(abs(h_gap) / 1000.0 * img_w))

    print(
        f"[canvas] {src.name}  {img_w}x{img_h}px\n"
        f"  subject  0-1000={subject_bbox} -> px=({sx0},{sy0})-({sx1},{sy1})"
        f"  area={area_pct:.1f}%\n"
        f"  axis     0-1000={axis_bbox}    -> px=({ax0},{ay0})-({ax1},{ay1})\n"
        f"  h_gap    0-1000={h_gap} -> {h_gap_px}px"
    )

    font_sm = _load_font(size=max(11, img_w // 100))
    font_xs = _load_font(size=max( 9, img_w // 130))

    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay, "RGBA")

    # ── Layer 1: Crimson subject box ─────────────────────────────────────────
    _draw_pen_rect(draw, sx0, sy0, sx1, sy1, color=_CRIMSON_RGBA, width=4)

    area_label = f"{area_pct:.1f}% frame"
    _draw_label(draw, area_label, sx0 + 4, sy0 + 4, font_sm,
                color=_CRIMSON_RGBA[:3] + (230,), img_w=img_w, img_h=img_h)

    subj_tag = f"[{subject_label[:40]}]"
    _, lh    = _text_size(font_sm, subj_tag)
    _draw_label(draw, subj_tag, sx0 + 2, max(sy0 - lh - 4, 2), font_sm,
                color=_CRIMSON_RGBA[:3] + (230,), img_w=img_w, img_h=img_h)

    # ── Layer 2: Cyan compositional axis line ─────────────────────────────────
    axis_cy = (ay0 + ay1) // 2
    draw.line([(ax0, axis_cy), (ax1, axis_cy)], fill=_CYAN_RGBA, width=3)

    tick_h = max(6, img_h // 80)
    draw.line([(ax0, axis_cy - tick_h), (ax0, axis_cy + tick_h)],
              fill=_CYAN_RGBA, width=2)
    draw.line([(ax1, axis_cy - tick_h), (ax1, axis_cy + tick_h)],
              fill=_CYAN_RGBA, width=2)

    axis_tag = f"[{anchor_label[:40]}]"
    _, alh   = _text_size(font_xs, axis_tag)
    _draw_label(draw, axis_tag, ax0 + 2, max(axis_cy - alh - 3, 2), font_xs,
                color=_CYAN_RGBA[:3] + (230,), img_w=img_w, img_h=img_h)

    # ── Layer 3: Yellow h_gap dimension arrow ────────────────────────────────
    arrow_y = (min(sy0, ay0) + max(sy1, ay1)) // 2

    if abs(h_gap) >= 10:
        # Arrow runs from the near edge of one box to the near edge of the other
        arr_x0 = ax1 if h_gap > 0 else sx1   # start at the trailing edge
        arr_x1 = sx0 if h_gap > 0 else ax0   # end at the leading edge

        _draw_dimension_arrow(draw, arr_x0, arrow_y, arr_x1,
                              color=_YELLOW_RGBA, width=2,
                              img_w=img_w, img_h=img_h)

        gap_lbl = f"h_gap: {h_gap_px}px"
        gtw, gth = _text_size(font_xs, gap_lbl)
        mid_x    = (arr_x0 + arr_x1) // 2
        _draw_label(draw, gap_lbl, mid_x - gtw // 2,
                    max(arrow_y - gth - 4, 2), font_xs,
                    color=_YELLOW_RGBA[:3] + (230,), img_w=img_w, img_h=img_h)
    else:
        # Boxes overlap — draw a zero-gap marker
        marker_x = (sx0 + sx1 + ax0 + ax1) // 4
        draw.line([(marker_x - 8, arrow_y), (marker_x + 8, arrow_y)],
                  fill=_YELLOW_RGBA, width=2)
        _draw_label(draw, "h_gap ~0", marker_x + 10, arrow_y - 6, font_xs,
                    color=_YELLOW_RGBA[:3] + (180,), img_w=img_w, img_h=img_h)

    # ── Layer 5-7: Technical audit (blur / highlights / horizon) ─────────────
    _draw_technical_audit_layers(
        draw, overlay, str(src), img_w, img_h, font_sm, font_xs,
    )
    # Re-bind draw after potential alpha_composite in highlight layer
    draw = ImageDraw.Draw(overlay, "RGBA")

    # ── Layer 8: Calibration panel (optional) ────────────────────────────────
    if thresholds:
        _draw_calibration_panel(
            draw, img_w, img_h, font_xs,
            slot_name, area_pct, h_gap, thresholds,
        )

    # ── Save transparent overlay only (photo stays behind in the browser) ────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_png_atomic(overlay, out)
    print(f"[canvas] saved -> {out}")
    return str(out)


# =============================================================================
# Batch render — Stage 4a (live pipeline, SlotResult objects)
# =============================================================================

def render_story_overlays(
    slot_results: list,
    output_dir:   str = str(_OVERLAY_DIR),
) -> list[dict]:
    """
    Render pen-box critique overlays for live SlotResult objects.
    Saves critique_{image_id}.png per entry.
    Returns one result dict per slot.
    """
    from PIL import Image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []

    for r in slot_results:
        image_id = Path(r.image_path).stem
        out_path = str(out_dir / f"critique_{image_id}.png")
        coordinates = {
            "subject_bbox":           r.subject_bbox,
            "subject_label":          r.subject_label,
            "compositional_axis_box": r.anchor_bbox,
            "anchor_label":           r.anchor_label,
        }
        from dataclasses import asdict
        facts    = {}
        try:
            from vision_story_mode import _derive_spatial_facts
            facts = _derive_spatial_facts(r)
        except Exception:
            pass
        area_pct = float(facts.get("subject_area_pct", 0.0))
        h_gap    = int(facts.get("h_gap", 0))

        try:
            draw_validation_overlay(
                src_image_path = r.image_path,
                coordinates    = coordinates,
                output_path    = out_path,
                area_pct       = area_pct,
                h_gap          = h_gap,
                slot_name      = r.slot_name,
            )
            with Image.open(out_path) as vi:
                vw, vh = vi.size
                sx0, sy0, sx1, sy1 = _scale_bbox(r.subject_bbox, vw, vh)
                pixel = vi.getpixel(((sx0 + sx1) // 2, (sy0 + sy1) // 2))[:3]
            pixel_ok = any(c > 10 for c in pixel)
            rendered.append({
                "image_id":    image_id,
                "slot":        r.slot_name,
                "overlay_url": f"/static/eye_feature_overlays/critique_{image_id}.png",
                "pixel_check": "PASS" if pixel_ok else "WARN: dark center pixel",
            })
        except Exception as exc:
            print(f"[canvas] ERROR {image_id}: {exc}")
            rendered.append({
                "image_id":    image_id,
                "slot":        r.slot_name,
                "overlay_url": None,
                "pixel_check": f"FAIL: {exc}",
            })

    passed = sum(1 for r in rendered if r["pixel_check"] == "PASS")
    print(f"[canvas] Stage 4a: {passed}/{len(rendered)} overlays OK")
    return rendered


# =============================================================================
# Batch render — from manifest (main CLI path)
# =============================================================================

def render_validation_overlays(
    manifest_path:  str = "final_story_manifest.json",
    output_dir:     str = str(_OVERLAY_DIR),
    telemetry_path: str = str(_TELEMETRY_PATH),
) -> list[dict]:
    """
    Read final_story_manifest.json, render structural overlays for every entry
    that has real spatial data, then atomically patch the manifest with the
    new eye_overlay_url values.

    Entries whose subject_bbox is the full-frame fallback [0,0,1000,1000]
    (meaning Qwen produced no real coordinates) are skipped gracefully.

    Output:
        {output_dir}/verified_{image_id}.png  for each processed entry
        manifest patched in-place with "eye_overlay_url": "/static/..."

    Returns a list of result dicts, one per sequence entry.
    """
    from PIL import Image

    mpath = Path(manifest_path)
    if not mpath.exists():
        raise FileNotFoundError(f"Manifest not found: {mpath}")

    manifest_data = json.loads(mpath.read_text(encoding="utf-8"))
    sequence      = manifest_data.get("sequence", [])

    if not sequence:
        print("[canvas] No sequence entries in manifest.")
        return []

    # Load calibration telemetry for the optional panel
    telemetry = load_calibration_telemetry(telemetry_path)
    if telemetry:
        ct = telemetry.get("calibrated_thresholds", {})
        print(
            f"[canvas] Telemetry loaded — "
            f"DOM={ct.get('DOMINANT_AREA_PCT','?')}  "
            f"H_THRESH={ct.get('H_CONTRAST_THRESH','?')}  "
            f"OPENER_CAP={ct.get('OPENER_MAX_AREA','?')}"
        )
    else:
        print("[canvas] No calibration_telemetry.json — calibration panel skipped.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []

    for entry in sequence:
        image_id  = Path(entry["image_path"]).stem
        slot_name = entry.get("assigned_slot", "")

        if not _has_spatial_data(entry):
            print(f"[canvas] SKIP {image_id} — no real spatial data (fallback bbox)")
            rendered.append({
                "image_id":    image_id,
                "slot":        slot_name,
                "overlay_url": None,
                "pixel_check": "SKIP: no spatial data",
                "area_pct":    0.0,
                "h_gap":       0,
            })
            continue

        facts    = entry.get("spatial_facts") or {}
        area_pct = float(facts.get("subject_area_pct", 0.0))
        h_gap    = int(facts.get("h_gap", 0))

        coordinates = {
            "subject_bbox":           entry.get("subject_bbox",  _FALLBACK_BBOX),
            "subject_label":          entry.get("subject_label", "Subject"),
            "compositional_axis_box": entry.get("anchor_bbox",   _FALLBACK_BBOX),
            "anchor_label":           entry.get("anchor_label",  "Axis"),
        }

        out_path = out_dir / f"verified_{image_id}.png"

        try:
            draw_validation_overlay(
                src_image_path = entry["image_path"],
                coordinates    = coordinates,
                output_path    = str(out_path),
                area_pct       = area_pct,
                h_gap          = h_gap,
                thresholds     = telemetry if telemetry else None,
                slot_name      = slot_name,
            )

            # Pixel-verify center of subject box
            with Image.open(str(out_path)) as vimg:
                vw, vh = vimg.size
                sx0, sy0, sx1, sy1 = _scale_bbox(coordinates["subject_bbox"], vw, vh)
                pixel = vimg.getpixel(((sx0 + sx1) // 2, (sy0 + sy1) // 2))[:3]
            pixel_ok    = any(c > 10 for c in pixel)
            overlay_url = f"/static/eye_feature_overlays/verified_{image_id}.png"
            entry["eye_overlay_url"] = overlay_url

            rendered.append({
                "image_id":    image_id,
                "slot":        slot_name,
                "overlay_url": overlay_url,
                "pixel_check": "PASS" if pixel_ok else "WARN: dark center pixel",
                "area_pct":    area_pct,
                "h_gap":       h_gap,
            })

        except Exception as exc:
            print(f"[canvas] ERROR {image_id}: {exc}")
            rendered.append({
                "image_id":    image_id,
                "slot":        slot_name,
                "overlay_url": None,
                "pixel_check": f"FAIL: {exc}",
                "area_pct":    area_pct,
                "h_gap":       h_gap,
            })

    # Atomically update the manifest with new eye_overlay_url values
    manifest_data["sequence"] = sequence
    payload   = json.dumps(manifest_data, indent=2, ensure_ascii=False)
    tmp_mpath = mpath.with_suffix(".tmp")
    tmp_mpath.write_text(payload, encoding="utf-8")
    os.replace(str(tmp_mpath), str(mpath))
    print(f"[canvas] Manifest updated atomically -> {mpath.name}")

    passed  = sum(1 for r in rendered if r["pixel_check"] == "PASS")
    skipped = sum(1 for r in rendered if r["pixel_check"].startswith("SKIP"))
    print(f"[canvas] {passed}/{len(rendered) - skipped} rendered OK  ({skipped} skipped)")
    return rendered


# =============================================================================
# VLM educational bbox layer (Layer 9)
# =============================================================================

def _draw_vlm_education_boxes(
    draw,
    overlay,
    vlm_bboxes: list,
    img_w: int,
    img_h: int,
    font_label,
    font_note,
) -> None:
    """
    Draw color-coded educational region annotations from vlm_bboxes.

    Color coding
    ------------
      Green  (80, 220, 100)  anchor_subject, composition_anchor  → STRENGTH
      Red    (220, 85,  85)  focal_point_miss, motion_blur       → ISSUE
      Amber  (245, 166,  35) blown_highlight, crushed_shadow,    → CAUTION
                              light_leak

    bbox_2d is [x1, y1, x2, y2] in absolute pixel coordinates.
    Each region gets: low-opacity fill, corner L-marks, and a label pill.
    """
    if not vlm_bboxes:
        return

    from PIL import Image as _PILI, ImageDraw as _PILID

    _POSITIVE = {"anchor_subject", "composition_anchor"}
    _NEGATIVE = {"focal_point_miss", "motion_blur"}
    _WARNING  = {"blown_highlight", "crushed_shadow", "light_leak"}

    for entry in vlm_bboxes:
        lbl  = entry.get("label", "")
        bbox = entry.get("bbox_2d") or []
        just = entry.get("justification", "")
        if len(bbox) < 4:
            continue

        x1 = max(0, min(int(bbox[0]), img_w - 1))
        y1 = max(0, min(int(bbox[1]), img_h - 1))
        x2 = max(0, min(int(bbox[2]), img_w - 1))
        y2 = max(0, min(int(bbox[3]), img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        if lbl in _POSITIVE:
            col, cat = (80, 220, 100), "STRENGTH"
        elif lbl in _NEGATIVE:
            col, cat = (220, 85, 85),  "ISSUE"
        elif lbl in _WARNING:
            col, cat = (245, 166, 35), "CAUTION"
        else:
            col, cat = (180, 180, 180), "NOTE"

        # Semi-transparent region fill
        fill_layer = _PILI.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        _fd = _PILID.Draw(fill_layer, "RGBA")
        _fd.rectangle([x1, y1, x2, y2], fill=col + (22,))
        overlay.alpha_composite(fill_layer)

        # Corner L-marks
        cm   = max(12, min(img_w // 45, 32))
        sw   = max(2, img_w // 300)
        col_a = col + (210,)
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                  (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            draw.line([(px, py + dy * cm), (px, py), (px + dx * cm, py)],
                      fill=col_a, width=sw)

        # Label pill
        lbl_str  = f"{cat}: {lbl.replace('_', ' ').upper()}"
        just_str = (just[:42] + "…" if len(just) > 42 else just) if just else ""
        lw, lh   = _text_size(font_label, lbl_str)
        nw, nh   = (_text_size(font_note, just_str) if just_str else (0, 0))
        box_w    = max(lw, nw) + 16
        box_h    = lh + (nh + 4 if just_str else 0) + 12
        pill_x   = max(4, min(x1, img_w - box_w - 4))
        pill_y   = (y1 - box_h - sw * 2) if y1 >= box_h + sw * 4 else (y1 + sw * 2)
        pill_y   = max(4, min(pill_y, img_h - box_h - 4))

        try:
            draw.rounded_rectangle(
                [pill_x, pill_y, pill_x + box_w, pill_y + box_h],
                radius=4, fill=(0, 0, 0, 185),
            )
        except AttributeError:
            draw.rectangle(
                [pill_x, pill_y, pill_x + box_w, pill_y + box_h],
                fill=(0, 0, 0, 185),
            )
        draw.rectangle([pill_x, pill_y, pill_x + 3, pill_y + box_h],
                       fill=col + (220,))
        tx = pill_x + 9
        draw.text((tx + 1, pill_y + 6 + 1), lbl_str, font=font_label,
                  fill=(0, 0, 0, 180))
        draw.text((tx,     pill_y + 6    ), lbl_str, font=font_label,
                  fill=col + (240,))
        if just_str:
            draw.text((tx + 1, pill_y + 6 + lh + 4 + 1), just_str, font=font_note,
                      fill=(0, 0, 0, 160))
            draw.text((tx,     pill_y + 6 + lh + 4    ), just_str, font=font_note,
                      fill=(240, 240, 240, 210))


# =============================================================================
# Japanese-spec hexagon radar chart (Layer 10)
# =============================================================================

def _draw_hex_spec(
    draw,
    overlay,
    breakdown: dict,
    score: float,
    grade: str,
    img_w: int,
    img_h: int,
    font_label,
    font_note,
    font_score,
) -> None:
    """
    Render a Japanese-character-spec hexagonal radar chart in the lower-right
    corner of the overlay.

    Vertices (clockwise from top, pointy-top hexagon):
      0 top         Composition   90°
      1 upper-right Lighting      30°
      2 lower-right Technical    -30°
      3 bottom      Human/Culture -90°
      4 lower-left  Narrative    -150°
      5 upper-left  Atmosphere   150°

    Each vertex shows a short label and score %. The center shows the overall
    score numeral in the grade colour. Three concentric guide rings at 33%,
    67%, 100% plus axis spokes. Score polygon is filled with translucent grade
    colour.
    """
    import math
    from PIL import Image as _PILI, ImageDraw as _PILID

    _SPECS = [
        ("Composition",   "COMP",    90),
        ("Lighting",      "LIGHT",   30),
        ("Technical",     "TECH",   -30),
        ("Human/Culture", "HUMAN",  -90),
        ("Narrative",     "NARR",  -150),
        ("Atmosphere",    "ATMO",   150),
    ]
    N = len(_SPECS)

    # ── Sizing & position ────────────────────────────────────────────────────
    R       = min(img_w, img_h) * 0.195
    margin  = R * 1.60
    cx      = img_w - margin
    cy      = img_h - margin

    if "Strong" in grade:
        gc = (80, 220, 100)
    elif "Mid" in grade:
        gc = (245, 166, 35)
    else:
        gc = (220, 85, 85)

    def pt(r, angle_deg):
        a = math.radians(angle_deg)
        return (cx + r * math.cos(a), cy - r * math.sin(a))

    # ── Background disc ──────────────────────────────────────────────────────
    bg_r   = R * 1.52
    bg_lay = _PILI.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    _bd    = _PILID.Draw(bg_lay, "RGBA")
    _bd.ellipse([cx - bg_r, cy - bg_r, cx + bg_r, cy + bg_r],
                fill=(4, 4, 14, 178))
    overlay.alpha_composite(bg_lay)

    # ── Axis spokes ──────────────────────────────────────────────────────────
    for _, _, ang in _SPECS:
        vx, vy = pt(R, ang)
        draw.line([(cx, cy), (vx, vy)], fill=(255, 255, 255, 28), width=1)

    # ── Concentric rings ─────────────────────────────────────────────────────
    for frac, alpha, lw in [(0.333, 22, 1), (0.667, 32, 1), (1.0, 75, 2)]:
        ring = [pt(R * frac, a) for _, _, a in _SPECS]
        ring.append(ring[0])
        draw.line(ring, fill=(255, 255, 255, alpha), width=lw)

    # ── Score polygon ─────────────────────────────────────────────────────────
    vals = [max(0.04, min(1.0, float(breakdown.get(k, 0.5))))
            for k, _, _ in _SPECS]
    score_pts = [pt(R * v, a) for v, (_, _, a) in zip(vals, _SPECS)]

    fill_lay = _PILI.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    _fd      = _PILID.Draw(fill_lay, "RGBA")
    _fd.polygon(score_pts, fill=gc + (52,))
    overlay.alpha_composite(fill_lay)

    stroke_lw = max(1, img_w // 480)
    draw.line(score_pts + [score_pts[0]], fill=gc + (235,), width=stroke_lw)

    # ── Vertex dots ───────────────────────────────────────────────────────────
    dot_r = max(3, img_w // 270)
    for p in score_pts:
        draw.ellipse([p[0]-dot_r, p[1]-dot_r, p[0]+dot_r, p[1]+dot_r],
                     fill=gc + (255,))

    # ── Vertex labels ─────────────────────────────────────────────────────────
    # (ox_side, oy_side): -1 = left/above, 0 = center, +1 = right/below
    _OFFSETS = [(0,-1), (+1,-1), (+1,+1), (0,+1), (-1,+1), (-1,-1)]
    label_r  = R * 1.31
    gap      = max(4, int(R * 0.07))

    for i, ((key, short, ang), v, (ox_s, oy_s)) in enumerate(
            zip(_SPECS, vals, _OFFSETS)):
        lx, ly   = pt(label_r, ang)
        pct_str  = f"{int(round(v * 100))}%"
        sw, sh   = _text_size(font_label, short)
        pw, ph   = _text_size(font_note,  pct_str)
        col_w    = max(sw, pw)

        tx = (int(lx) - col_w // 2          if ox_s == 0
              else int(lx) + gap             if ox_s > 0
              else int(lx) - col_w - gap)
        ty = (int(ly) - sh - ph - gap * 2   if oy_s < 0
              else int(ly) + gap)

        # Clamp to image bounds
        tx = max(2, min(tx, img_w - col_w - 2))
        ty = max(2, min(ty, img_h - sh - ph - 4))

        # Aspect short name (white, shadowed)
        draw.text((tx+1, ty+1), short, font=font_label, fill=(0,0,0,185))
        draw.text((tx,   ty  ), short, font=font_label, fill=(255,255,255,225))
        # Score % in grade colour
        draw.text((tx+1, ty+sh+2+1), pct_str, font=font_note, fill=(0,0,0,155))
        draw.text((tx,   ty+sh+2  ), pct_str, font=font_note, fill=gc+(215,))

    # ── Center: overall score numeral ─────────────────────────────────────────
    ctr_str = str(int(round(score * 100)))
    cw, ch  = _text_size(font_score, ctr_str)
    draw.text((int(cx-cw//2)+1, int(cy-ch//2)+1), ctr_str,
              font=font_score, fill=(0,0,0,200))
    draw.text((int(cx-cw//2),   int(cy-ch//2)  ), ctr_str,
              font=font_score, fill=gc+(245,))


# =============================================================================
# Reasoning overlay — photographer contact-sheet annotations
# =============================================================================

def render_reasoning_overlay(
    image_path:    str,
    reasoning_log: str,
    breakdown:     dict,
    score:         float,
    grade:         str,
) -> str:
    """
    Draw photographer-style reasoning annotations directly on the image using PIL.

    Produces a JPEG in static/eye_feature_overlays/reasoning_{stem}.jpg and
    returns the server-relative URL  /static/eye_feature_overlays/reasoning_{stem}.jpg.

    Layout
    ------
    • Layer 9  — VLM bbox educational regions (green/red/amber fills + pills)
    • Layer 10 — Japanese-spec hexagon radar chart, lower-right corner
                 Six axes: COMP · LIGHT · TECH · HUMAN · NARR · ATMO
                 Score polygon filled in grade colour; overall score at centre
    • Score badge — bottom-left, grade colour, large numeral + tier word
    """
    from PIL import Image, ImageDraw

    img    = Image.open(image_path).convert("RGBA")
    W, H   = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw   = ImageDraw.Draw(overlay)

    # ── Fonts ──────────────────────────────────────────────────────────────
    f_label = _load_font(max(13, W // 90))   # aspect label (bold look)
    f_note  = _load_font(max(11, W // 110))  # observation sentence
    f_score = _load_font(max(28, W // 40))   # large score numeral
    f_tier  = _load_font(max(12, W // 90))   # tier word

    # ── Grade colour ───────────────────────────────────────────────────────
    if "Strong" in grade:
        gc = (80, 220, 100)
    elif "Mid" in grade:
        gc = (245, 166, 35)
    else:
        gc = (220, 85, 85)

    # ── Parse reasoning_log (header/verdict used for score badge) ─────────────
    rlines    = reasoning_log.split("\n")
    header    = rlines[0] if rlines else ""
    verdict   = rlines[1] if len(rlines) > 1 else ""
    tier_word = header.split()[0].upper() if header else ""
    pct       = int(round(score * 100))

    # ── Layer 9: VLM educational bbox annotations (background) ────────────────
    _vlm_bboxes = breakdown.get("vlm_bboxes") or []
    if _vlm_bboxes:
        _draw_vlm_education_boxes(
            draw, overlay, _vlm_bboxes, W, H, f_label, f_note,
        )
        from PIL import ImageDraw as _ID_rebind
        draw = _ID_rebind.Draw(overlay)

    # ── Layer 10: Hexagon radar spec ──────────────────────────────────────────
    _draw_hex_spec(
        draw, overlay, breakdown, score, grade, W, H,
        f_label, f_note, f_score,
    )
    from PIL import ImageDraw as _ID_rebind2
    draw = _ID_rebind2.Draw(overlay)

    # ── Score badge — bottom-left ───────────────────────────────────────────
    pad     = max(16, H // 55)
    score_txt = str(pct)
    sw, sh    = _text_size(f_score, score_txt)
    tw, _     = _text_size(f_tier,  tier_word)
    badge_w   = sw + tw + pad * 3 + 12
    badge_h   = sh + pad
    bx0 = pad;  by0 = H - pad - badge_h
    bx1 = bx0 + badge_w;  by1 = H - pad

    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=6,
                            fill=(0, 0, 0, 185))
    # Score numeral
    nx = bx0 + pad
    ny = by0 + (badge_h - sh) // 2
    draw.text((nx + 2, ny + 2), score_txt, font=f_score, fill=(0, 0, 0, 180))
    draw.text((nx,     ny    ), score_txt, font=f_score, fill=gc + (245,))
    # Tier word
    tx2 = nx + sw + 10
    ty2 = ny + sh - _text_size(f_tier, tier_word)[1] - 2
    draw.text((tx2 + 1, ty2 + 1), tier_word, font=f_tier, fill=(0, 0, 0, 160))
    draw.text((tx2,     ty2    ), tier_word, font=f_tier, fill=gc + (230,))

    # Verdict text below badge
    if verdict:
        verd_short = verdict if len(verdict) <= 60 else verdict[:58] + "…"
        f_verd = _load_font(max(10, W // 110))
        draw.text((bx0, by1 + 5), verd_short, font=f_verd,
                  fill=(220, 220, 220, 160))

    # ── Composite + save ───────────────────────────────────────────────────
    result = Image.alpha_composite(img, overlay).convert("RGB")
    stem   = Path(image_path).stem
    out_p  = _OVERLAY_DIR / f"reasoning_{stem}.jpg"
    tmp    = out_p.with_suffix(".tmp.jpg")
    result.save(str(tmp), "JPEG", quality=88, optimize=True)
    os.replace(str(tmp), str(out_p))
    return f"/static/eye_feature_overlays/reasoning_{stem}.jpg"


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    # python canvas_renderer.py                            -> validation (default manifest)
    # python canvas_renderer.py path/to/manifest.json     -> explicit manifest path

    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("final_story_manifest.json")

    if not manifest.exists():
        print(f"[canvas] Manifest not found: {manifest}")
        print("  Run vision_story_mode.py first, then: python canvas_renderer.py")
        sys.exit(1)

    print(f"[canvas] Processing manifest: {manifest}")
    print("=" * 62)

    results = render_validation_overlays(str(manifest))

    print()
    print("=" * 62)
    print(" Eye Feature Overlay Summary")
    print("=" * 62)
    for r in results:
        if r["pixel_check"].startswith("SKIP"):
            mark = "[SKIP]"
        elif r["pixel_check"] == "PASS":
            mark = "[OK]  "
        else:
            mark = "[FAIL]"
        print(f"  {mark} {r['slot']:<26} {r['image_id']}")
        if not r["pixel_check"].startswith("SKIP"):
            print(f"         area={r['area_pct']:.1f}%  h_gap={r['h_gap']:+d}")
        if r["overlay_url"]:
            print(f"         -> {r['overlay_url']}")
    ok_n  = sum(1 for r in results if r["pixel_check"] == "PASS")
    skp_n = sum(1 for r in results if r["pixel_check"].startswith("SKIP"))
    print(f"\n  {ok_n}/{len(results) - skp_n} overlays rendered  ({skp_n} skipped)")
    print("=" * 62)
