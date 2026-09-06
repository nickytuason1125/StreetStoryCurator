"""Reproduce the detached-process ONNX session init (the WinError 50/6 trap)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

out = []
try:
    import numpy as np
    from encode_worker import _onnx_session, _onnx_preprocess
    out.append("importing onnxruntime + building session …")
    sess = _onnx_session()
    out.append("session OK: " + str(sess.get_providers()[0]))
    from PIL import Image
    img = Image.new("RGB", (384, 384), (128, 96, 64))
    x = np.stack([_onnx_preprocess(img)]).astype(np.float16)
    e = sess.run(None, {"pixel_values": x})[0].astype(np.float32)
    out.append(f"run OK: shape={e.shape}")
except Exception:
    out.append("FAILED:\n" + traceback.format_exc())

Path(ROOT / "onnx_detached_result.txt").write_text("\n".join(out), encoding="utf-8")
