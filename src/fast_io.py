"""
High-performance image I/O and ONNX input preparation.

Two entry points:
  load_image_optimized(path, target_size)  — file → (C,H,W) float32 [0,1]
      Uses libvips JPEG shrink-on-load when available (fastest cold-read path).
      Falls back to unicode-safe cv2 transparently.

  bgr_to_chw(bgr, target_size)            — loaded BGR uint8 → (C,H,W) float32 [0,1]
      Used inside _analyze where the image is already in RAM.
      Avoids re-reading from disk; contiguous layout for ONNX throughput.
"""

import numpy as np

try:
    import pyvips
    _HAS_VIPS = True
except ImportError:
    _HAS_VIPS = False

# ImageNet normalisation constants — module-level, allocated once
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_image_optimized(path: str, target_size: int = 224) -> np.ndarray | None:
    """
    Load and resize an image for direct ONNX input.

    pyvips path: JPEG shrink-on-load decodes at native 1/8 resolution — avoids
    reading the full pixel buffer for large RAW/JPEG files.
    cv2 fallback: np.fromfile + imdecode for unicode-safe Windows paths.

    Returns: (C, H, W) float32 [0.0, 1.0], or None on failure.
    Pyvips Image and memory buffer go out of scope here — no leak.
    """
    try:
        if _HAS_VIPS:
            vimg = pyvips.Image.thumbnail(
                path, target_size, height=target_size, crop="centre"
            )
            if vimg.bands == 1:
                vimg = vimg.colourspace("srgb")
            elif vimg.bands == 4:
                vimg = vimg.flatten()               # drop alpha channel
            arr = np.frombuffer(vimg.write_to_memory(), dtype=np.uint8).reshape(
                vimg.height, vimg.width, 3          # always 3 bands after above guards
            ).copy()                                # own the buffer before vimg is freed
        else:
            import cv2
            buf = np.fromfile(path, dtype=np.uint8)   # unicode-safe
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("imdecode returned None")
            arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr = cv2.resize(arr, (target_size, target_size), interpolation=cv2.INTER_AREA)

        return np.ascontiguousarray(
            np.transpose(arr, (2, 0, 1)), dtype=np.float32
        ) / 255.0

    except Exception:
        return None


def bgr_to_chw(bgr: np.ndarray, target_size: int) -> np.ndarray:
    """
    Convert an already-loaded BGR uint8 array to (C, H, W) float32 [0.0, 1.0].

    Used in _analyze where the image is already in memory — no disk re-read.
    np.ascontiguousarray ensures the layout is cache-friendly for ONNX kernels.
    """
    import cv2
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(
        np.transpose(resized, (2, 0, 1)), dtype=np.float32
    ) / 255.0


def normalize_imagenet(chw: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalisation in-place. Returns the same array."""
    chw -= IMAGENET_MEAN
    chw /= IMAGENET_STD
    return chw
