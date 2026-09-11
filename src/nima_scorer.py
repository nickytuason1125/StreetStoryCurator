"""
nima_scorer.py — AVA-trained NIMA aesthetic scores through the exported ONNX.

The anti-mush add-on (2026-09-11): the Fast/CLIP path scores aesthetics with
SigLIP probes, which squash everything into a 0.4–0.6 band — a real cull
measured 59% of photos inside 0.5–0.6 with a median of 0.56. NIMA is a
MobileNetV2 trained on ~250k human aesthetic ratings (AVA, idealo weights);
its scores carry genuine spread, and at 224×224 on CPU it costs ~40 ms per
photo — affordable even in a scan pass.

Contract (set by nima_setup.py): models/onnx/nima.onnx takes `pixel_values`
(N, 3, 224, 224) float32 RGB in [0, 1] and outputs a (N, 10) rating
distribution over 1–10. We report the mean rating mapped to [0, 1]
deterministically per photo — no batch statistics, so the absolute-
calibration invariance guarantee holds.

None is returned whenever the model is absent or broken: callers degrade to
the CLIP aesthetic and the cull never dies because of this add-on.
"""
from pathlib import Path


_ONNX_PATH = Path(__file__).resolve().parent.parent / "models" / "onnx" / "nima.onnx"


def nima_scores(paths, progress=None) -> "object":
    """(N,) float32 aesthetic scores in [0,1] for image paths, or None.

    Deterministic per photo: same file → same score in any batch.
    """
    import numpy as np

    if not _ONNX_PATH.exists():
        return None
    try:
        import onnxruntime as ort
        from PIL import Image
        from raw_support import RAW_EXTS, extract_embedded_preview
    except Exception as e:
        print(f"[nima] scorer unavailable ({e}) — keeping CLIP aesthetic")
        return None

    try:
        sess = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[nima] ONNX session failed ({e}) — keeping CLIP aesthetic")
        return None

    n = len(paths)
    out = np.zeros(n, dtype=np.float32)
    batch = 16
    for i in range(0, n, batch):
        arrs = []
        for p in paths[i:i + batch]:
            try:
                img = Image.open(p)
                try:
                    # JPEG decode downscale in DCT domain — 224 is the target.
                    img.draft("RGB", (256, 256))
                except Exception:
                    pass
                img = img.convert("RGB")
            except Exception:
                try:
                    img = extract_embedded_preview(p)
                except Exception:
                    arrs.append(np.zeros((3, 224, 224), np.float32))   # zero → mean-ish
                    continue
            arrs.append(
                np.asarray(img.resize((224, 224), Image.BILINEAR), np.float32
                           ).transpose(2, 0, 1) / 255.0)
        x = np.stack(arrs).astype(np.float32)
        ratings = sess.run(None, {"pixel_values": x})[0].astype(np.float64)
        mean10 = (ratings * np.arange(1, 11)).sum(axis=1)
        out[i:i + len(mean10)] = ((mean10 - 1.0) / 9.0).astype(np.float32)
        if progress:
            progress(0.0, f"NIMA aesthetic {min(i + batch, n)}/{n}…")
    print(f"[nima] aesthetic: min={out.min():.3f}  max={out.max():.3f}  "
          f"mean={out.mean():.3f}  spread(p75-p25)={float(np.percentile(out, 75) - np.percentile(out, 25)):.3f}")
    return out
