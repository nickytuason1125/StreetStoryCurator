import cv2
import numpy as np
import base64


def _build_ryg_lut() -> np.ndarray:
    """BGR lookup table: index 0 (low variance/blurry) → Red, index 255 (sharp) → Green."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            g = int(i / 127.0 * 255)
            lut[i] = (0, g, 255)       # BGR: R=255, G rises, B=0
        else:
            r = int((1.0 - (i - 128) / 127.0) * 255)
            lut[i] = (0, 255, r)       # BGR: R falls, G=255, B=0
    return lut


_RYG_LUT = _build_ryg_lut()


def generate_technical_heatmap(image_path: str, grid_size: int = 32) -> str:
    """
    Divide the image into grid_size×grid_size blocks, compute Laplacian variance
    per block (higher = sharper), and return a Base64-encoded RGBA PNG overlay.

    Color encoding: red = blurry, green = sharp.  Alpha ≈ 40%.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows = max(1, h // grid_size)
    cols = max(1, w // grid_size)

    var_grid = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            y0 = r * grid_size
            y1 = min(h, (r + 1) * grid_size)
            x0 = c * grid_size
            x1 = min(w, (c + 1) * grid_size)
            block = gray[y0:y1, x0:x1]
            var_grid[r, c] = cv2.Laplacian(block, cv2.CV_64F).var()

    vmin, vmax = var_grid.min(), var_grid.max()
    if vmax > vmin:
        norm_grid = ((var_grid - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    else:
        norm_grid = np.zeros((rows, cols), dtype=np.uint8)

    # Smooth upscale back to original resolution
    heatmap_mono = cv2.resize(norm_grid, (w, h), interpolation=cv2.INTER_CUBIC)

    # Map scalar values to BGR colors via LUT
    colored = _RYG_LUT[heatmap_mono]  # shape: (h, w, 3)

    alpha = np.full((h, w, 1), int(255 * 0.40), dtype=np.uint8)
    bgra = np.concatenate([colored, alpha], axis=2)

    _, buf = cv2.imencode(".png", bgra)
    return base64.b64encode(buf.tobytes()).decode("ascii")
