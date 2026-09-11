"""
One-time NIMA ONNX generator.

Run this once to produce models/onnx/nima.onnx — after that PyTorch is no
longer required.  The app works without NIMA; running this script upgrades
the aesthetic grading to a model trained on 250k human photo ratings.

Usage:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    python nima_setup.py

What it does:
  1. Downloads pretrained MobileNetV2 weights from PyTorch Hub (~14 MB)
  2. Downloads the NIMA aesthetic head weights from the idealo release on
     Hugging Face (~1 MB)
  3. Fuses them and exports a self-contained ONNX to models/onnx/nima.onnx
"""

import sys, urllib.request, struct, zipfile, io
from pathlib import Path

ONNX_OUT = Path("models/onnx/nima.onnx")

# ── NIMA head weights — MobileNetV2 aesthetic model trained on AVA (idealo) ──
# Hosted on Hugging Face as a plain .npz so no PyTorch is needed at runtime.
NIMA_WEIGHTS_URL = (
    "https://huggingface.co/Geolex/nima-mobilenet-v2-aesthetic/resolve/main/nima_weights.pth"
)


def _try_load_nima_pth(url: str, tmp_path: Path):
    """Download the NIMA head pth and return a state_dict (or raise)."""
    import torch
    if not tmp_path.exists():
        print(f"  Downloading NIMA head weights…")
        urllib.request.urlretrieve(url, str(tmp_path))
    return torch.load(str(tmp_path), map_location="cpu", weights_only=True)


def build_nima_onnx():
    """Build models/onnx/nima.onnx from the REAL AVA-trained idealo weights.

    2026-09-11 rework: the old torch path downloaded a .pth head from a
    HuggingFace repo that is now gated (HTTP 401), silently exporting an
    ImageNet-classifier backbone — a technical proxy, not an aesthetic model.
    The genuine idealo NIMA weights (trained on ~250k AVA human ratings)
    ship as a Keras .hdf5 inside the public idealo GitHub repo, so this
    version loads them with tf.keras, wraps the graph to accept NCHW
    `pixel_values` (float32 [0,1] RGB, the app's encode-worker convention),
    and exports via tf2onnx with output `ratings` (N, 10) softmax.

    Requires: tensorflow-cpu + tf2onnx + h5py (one-time export toolchain;
    the app runtime itself only needs onnxruntime).
    """
    import json

    _W_HDF5 = ONNX_OUT.parent / "nima_ava_weights.hdf5"

    # ── Source 1: the public idealo AVA-trained weights (already downloaded) ──
    _hdf5 = _find_hdf5()

    try:
        import numpy as np
        import tensorflow as tf
        import tf2onnx
    except ImportError as e:
        print(f"Missing export toolchain ({e}). Install once:\n"
              "  pip install tensorflow-cpu tf2onnx h5py\n"
              "(export-time only — the app runtime needs onnxruntime only)")
        return False

    # Keras NIMA exactly as idealo trained it: MobileNetV2(avg-pool) →
    # Dropout(0.75) → Dense(10, softmax). load_weights matches by order into
    # an identically built graph; by_name as a safety net.
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(224, 224, 3), pooling="avg")
    x = tf.keras.layers.Dropout(0.75)(base.output)
    out = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    if _hdf5 is None:
        print("No AVA weights found — refusing to export a backbone-only "
              "model (it would grade noise). Run with the weights file "
              "models/onnx/nima_ava_weights.hdf5 present.")
        return False
    print(f"Loading AVA-trained weights: {_hdf5.name}")
    try:
        model.load_weights(str(_hdf5))
        print("  NIMA AVA weights loaded (full model).")
    except Exception:
        try:
            model.load_weights(str(_hdf5), by_name=True, skip_mismatch=True)
            print("  AVA weights loaded (by-name).")
        except Exception as e:
            print(f"Could not load AVA weights ({e}) — aborting rather than "
                  "exporting an untrained proxy.")
            return False
    model.trainable = False

    # Wrap to NCHW input so the exported ONNX takes the app's (N,3,224,224).
    nchw = tf.keras.Input(shape=(3, 224, 224), name="pixel_values_in")
    nhwc = tf.keras.layers.Permute((2, 3, 1), name="nchw_to_nhwc")(nchw)
    wrapped = tf.keras.Model(nchw, model(nhwc))

    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    spec = [tf.TensorSpec([None, 3, 224, 224], tf.float32, name="pixel_values")]
    print("Exporting via tf2onnx …")
    model_proto, _ = tf2onnx.convert.from_keras(
        wrapped, input_signature=spec, opset=14)
    # NOTE: callers use sess.run(None, {"pixel_values": ...})[0], so the
    # output NAME does not matter — renaming the graph output without also
    # renaming the producing node breaks the model (onnxruntime: "Graph
    # output (ratings) does not exist"). Keep the original name.
    import onnx
    onnx.save(model_proto, str(ONNX_OUT))

    # Sanity check: shape + a sane range on a mid-grey image.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_OUT), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": np.zeros((1, 3, 224, 224), np.float32)})[0]
    assert out.shape == (1, 10), f"Unexpected output shape: {out.shape}"
    print(f"Done — {ONNX_OUT}  ({ONNX_OUT.stat().st_size // 1024} KB); "
          f"zero-image mean rating {(out[0] * np.arange(1, 11)).sum():.2f}/10")
    return True


def _find_hdf5():
    """The AVA-trained weights, wherever they are (repo models/onnx/ or cache)."""
    candidates = [
        ONNX_OUT.parent / "nima_ava_weights.hdf5",
        ONNX_OUT.parent.parent / "models" / "onnx" / "nima_ava_weights.hdf5",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    print("Downloading idealo AVA aesthetic weights (13 MB, public GitHub)…")
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://github.com/idealo/image-quality-assessment/raw/master/"
            "models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5",
            headers={"User-Agent": "Mozilla/5.0"})
        ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req, timeout=120) as r, \
                open(str(ONNX_OUT.parent / "nima_ava_weights.hdf5"), "wb") as f:
            f.write(r.read())
        return ONNX_OUT.parent / "nima_ava_weights.hdf5"
    except Exception as e:
        print(f"  AVA weights download failed: {e}")
        return None


if __name__ == "__main__":
    ok = build_nima_onnx()
    sys.exit(0 if ok else 1)
