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


def nima_scores(paths, progress=None, threads: int = 0) -> "object":
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
        _so = ort.SessionOptions()
        if threads:
            # Capped when it runs alongside the GPU encode (prefetch_nima) so
            # it never starves the encoder's decode/preprocess thread.
            _so.intra_op_num_threads = int(threads)
        sess = ort.InferenceSession(str(_ONNX_PATH), _so, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[nima] ONNX session failed ({e}) — keeping CLIP aesthetic")
        return None

    n = len(paths)
    out = np.zeros(n, dtype=np.float32)
    batch = 16
    def _decode(p):
        """One photo -> (3, 224, 224) float32, exactly as the serial loop did.
        Pure, so batches decode on a thread pool (2026-10-04: ~30 ms/photo
        serial was 18 s of a 600-RAW cull); map() keeps input order."""
        try:
            from raw_support import jpeg_preview
            img = jpeg_preview(p, 256)          # camera JPEG: built-in preview
            if img is None:
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
                img = None
            if img is None:
                return np.zeros((3, 224, 224), np.float32)   # zero → mean-ish
        return (np.asarray(img.resize((224, 224), Image.BILINEAR), np.float32
                           ).transpose(2, 0, 1) / 255.0)

    from concurrent.futures import ThreadPoolExecutor
    import os as _os_n
    _pool = ThreadPoolExecutor(max_workers=int(threads) or min(8, _os_n.cpu_count() or 4),
                               thread_name_prefix="nima-decode")
    for i in range(0, n, batch):
        arrs = list(_pool.map(_decode, paths[i:i + batch]))
        x = np.stack(arrs).astype(np.float32)
        ratings = sess.run(None, {"pixel_values": x})[0].astype(np.float64)
        mean10 = (ratings * np.arange(1, 11)).sum(axis=1)
        out[i:i + len(mean10)] = ((mean10 - 1.0) / 9.0).astype(np.float32)
        if progress:
            progress(0.0, f"NIMA aesthetic {min(i + batch, n)}/{n}…")
    _pool.shutdown(wait=False)
    print(f"[nima] aesthetic: min={out.min():.3f}  max={out.max():.3f}  "
          f"mean={out.mean():.3f}  spread(p75-p25)={float(np.percentile(out, 75) - np.percentile(out, 25)):.3f}")
    return out


def prefetch_nima(paths, threads: int = 2):
    """Start NIMA on a background thread; returns an object with .result()
    -> {path: score} (empty when NIMA is absent/failed).

    2026-10-04: NIMA is CPU-only and independent of the image encoder, but ran
    after IQA with the GPU idle (~9 s per 600 RAWs). Started before the GPU
    encode, it finishes in that stage's shadow. Scores are per-photo
    deterministic (see nima_scores), so the result is identical.
    """
    import threading
    box = {"map": {}}

    def _run():
        try:
            import numpy as _np
            sc = nima_scores(list(paths), threads=threads)
            if sc is not None:
                box["map"] = {p: float(v) for p, v in zip(paths, _np.asarray(sc))}
        except Exception as e:
            print(f"[nima] prefetch failed ({e}) — will score after IQA", flush=True)

    t = threading.Thread(target=_run, daemon=True, name="nima-prefetch")
    t.start()

    class _Handle:
        def result(self):
            t.join()
            return box["map"]
    return _Handle()
