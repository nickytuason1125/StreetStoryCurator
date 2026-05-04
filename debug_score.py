"""
Diagnostic script — prints every intermediate value for one photo.
Usage (from the street-story-curator directory, with venv active):
    python debug_score.py "path/to/photo.jpg"
"""
import sys, cv2, numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python debug_score.py <image_path>")
    sys.exit(1)

path = sys.argv[1]
buf  = np.fromfile(path, dtype=np.uint8)
img  = cv2.imdecode(buf, cv2.IMREAD_COLOR)
if img is None:
    print("ERROR: could not load image"); sys.exit(1)

ih, iw = img.shape[:2]
if max(ih, iw) > 1080:
    scale = 1080 / max(ih, iw)
    img   = cv2.resize(img, (int(iw*scale), int(ih*scale)), interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"\n── Image ──────────────────────────────")
print(f"  File        : {path}")
print(f"  Dimensions  : {w}×{h}  (after resize)")

# Histogram-based metrics
hist         = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
total        = float(hist.sum()) or 1.0
blown        = float(hist[245:].sum()) / total
blocked      = float(hist[:8].sum())   / total
shadow_dark  = float(hist[:35].sum())  / total
midtone_frac = float(hist[50:210].sum()) / total

print(f"\n── Histogram ──────────────────────────")
print(f"  blocked     (hist[:8])  : {blocked:.4f}  (old chiaroscuro trigger > 0.12)")
print(f"  shadow_dark (hist[:35]) : {shadow_dark:.4f}  (new trigger > 0.18)")
print(f"  blown       (hist[245:]) : {blown:.4f}")
print(f"  midtone_frac             : {midtone_frac:.4f}")

# Centre brightness
cy_c1, cy_c2 = h // 4, 3 * h // 4
cx_c1, cx_c2 = w // 4, 3 * w // 4
center_mean  = float(gray[cy_c1:cy_c2, cx_c1:cx_c2].mean())
global_mean  = float(gray.mean())

print(f"\n── Brightness ─────────────────────────")
print(f"  global mean  : {global_mean:.1f}")
print(f"  centre mean  : {center_mean:.1f}")
print(f"  centre - global : {center_mean - global_mean:.1f}  (need > 18 for chiaroscuro)")

chiaroscuro = (
    shadow_dark  > 0.18
    and center_mean > 65
    and blown      < 0.12
    and (center_mean - global_mean) > 18
)
print(f"\n── Chiaroscuro detection ──────────────")
print(f"  shadow_dark > 0.18  : {shadow_dark > 0.18}")
print(f"  center_mean > 65    : {center_mean > 65}")
print(f"  blown < 0.12        : {blown < 0.12}")
print(f"  centre-global > 18  : {center_mean - global_mean > 18}")
print(f"  *** chiaroscuro = {chiaroscuro} ***")

# Sharpness
lap_var      = float(cv2.Laplacian(gray, cv2.CV_64F).var())
cell_h, cell_w = max(h // 3, 1), max(w // 3, 1)
region_vars  = []
for ri in range(3):
    for ci in range(3):
        patch = gray[ri*cell_h:(ri+1)*cell_h, ci*cell_w:(ci+1)*cell_w]
        if patch.size > 0:
            region_vars.append(float(cv2.Laplacian(patch, cv2.CV_64F).var()))
best_sharp = max([lap_var] + region_vars)

rv       = np.array(region_vars, dtype=np.float32)
blur_cv  = float(rv.std() / (rv.mean() + 1e-6))

blur5       = cv2.GaussianBlur(gray, (5, 5), 0)
noise_map   = np.abs(gray.astype(np.float32) - blur5.astype(np.float32))
shadow_mask = gray < 55
noise_level = float(noise_map[shadow_mask].mean()) if shadow_mask.any() else 0.0
noise_pen   = min(noise_level / 12.0, 0.50)

intentional_soft = (
    best_sharp < 250
    and blur_cv   < 0.55
    and noise_level > 1.5
    and best_sharp > 8
)
sharpness_score = min(best_sharp / 400.0, 1.0)
if best_sharp < 60:
    if intentional_soft:
        sharpness_score = max(sharpness_score, 0.38)
    else:
        sharpness_score = min(sharpness_score, 0.18)

if chiaroscuro:
    exp_qual     = float(np.clip(1.0 - abs(center_mean - 110) / 120.0, 0.0, 1.0))
    exposure_pen = min(blown * 2.0, 0.15)
else:
    exposure_pen = min(blown * 4.0, 0.60) + min(blocked * 3.0, 0.40)
    exposure_pen = min(exposure_pen, 0.70)
    exp_qual = None  # computed later in lighting

tech = float(np.clip(
    sharpness_score
    * (1.0 - exposure_pen * 0.55)
    * (1.0 - noise_pen   * 0.35)
    * (0.80 + 0.20 * min(midtone_frac / 0.65, 1.0)),
    0.0, 1.0
))

print(f"\n── Technical ──────────────────────────")
print(f"  best_sharp      : {best_sharp:.1f}")
print(f"  blur_cv         : {blur_cv:.3f}")
print(f"  noise_level     : {noise_level:.2f}")
print(f"  noise_pen       : {noise_pen:.3f}")
print(f"  sharpness_score : {sharpness_score:.3f}")
print(f"  exposure_pen    : {exposure_pen:.3f}  (chiaroscuro branch: {chiaroscuro})")
print(f"  intentional_soft: {intentional_soft}")
print(f"  tech score      : {tech:.3f}")

# Lighting
mean_bright = global_mean
if chiaroscuro:
    pass  # exp_qual already set
elif mean_bright < 40:
    exp_qual = (mean_bright / 40.0) * 0.45
elif mean_bright > 210:
    exp_qual = max(0.0, (255 - mean_bright) / 45.0) * 0.45
else:
    exp_qual = 1.0 - abs(mean_bright - 118) / 170.0

contrast_score = min(float(np.std(gray)) / 52.0, 1.0)
ps = max(h // 10, 16)
rh, rw = (h // ps) * ps, (w // ps) * ps
if rh > 0 and rw > 0:
    blocks = gray[:rh, :rw].reshape(rh // ps, ps, rw // ps, ps)
    local_contrast = min(float(blocks.std(axis=(1, 3)).mean()) / 38.0, 1.0)
else:
    local_contrast = 0.4

def color_mood_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h2, s, v = cv2.split(hsv)
    warmth = np.mean(h2[h2 < 30]) / 180.0 if np.any(h2 < 30) else 0.0
    saturation_harmony = 1.0 - np.std(s) / 128.0
    contrast_mood = np.sqrt(np.var(v)) / 128.0
    return 0.4*warmth + 0.3*saturation_harmony + 0.3*contrast_mood

mood  = float(np.clip(color_mood_score(img), 0.0, 1.0))
light = float(np.clip(
    0.30 * exp_qual + 0.28 * contrast_score + 0.22 * local_contrast + 0.20 * mood,
    0.0, 1.0
))

print(f"\n── Lighting ───────────────────────────")
print(f"  exp_qual        : {exp_qual:.3f}")
print(f"  contrast_score  : {contrast_score:.3f}")
print(f"  local_contrast  : {local_contrast:.3f}")
print(f"  mood            : {mood:.3f}")
print(f"  light score     : {light:.3f}")

# Hard overrides
print(f"\n── Hard overrides ─────────────────────")
raw_base = 0.50   # placeholder; actual depends on DINOv2 which needs ONNX
print(f"  blown > 0.25                       : {blown > 0.25}")
print(f"  blocked > 0.35 and not chiaroscuro : {blocked > 0.35 and not chiaroscuro}")
print(f"  best_sharp < 60 and not soft        : {best_sharp < 60 and not intentional_soft}")

print(f"\n── Nice-but-empty penalty check ───────")
print(f"  Would need: auth<0.38, human<0.32, comp<0.48, light<0.52")
print(f"  light={light:.3f} (< 0.52: {light < 0.52})")
print(f"  chiaroscuro exempt: {chiaroscuro}  → penalty {'SKIPPED' if chiaroscuro else 'may fire'}")

print(f"\nDone. Re-run after server restart to see grade change.\n")
