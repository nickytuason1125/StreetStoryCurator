"""ONNX encode: preprocessing runs on the prefetch thread, off the GPU loop.

2026-10-04, measured on 64 ARWs: ViT-g forward 87 ms/img, _onnx_preprocess
37 ms/img — and the preprocess ran serially between sess.run calls, so the
GPU idled ~30% of the encode. Same function, same order: identical output.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import encode_worker as ew  # noqa: E402


def test_preprocess_happens_off_the_consumer_thread(monkeypatch):
    consumer = threading.get_ident()
    prep_threads = set()

    def fake_decode_chunk(chunk):
        return [Image.new("RGB", (32, 32), (i * 10 % 255, 0, 0)) for i, _ in enumerate(chunk)], [], None

    def fake_prep(img):
        prep_threads.add(threading.get_ident())
        return np.full((3, 4, 4), img.getpixel((0, 0))[0], np.float32)

    monkeypatch.setattr(ew, "_decode_chunk", fake_decode_chunk)
    monkeypatch.setattr(ew, "_onnx_preprocess", fake_prep)

    class _Sess:
        def run(self, _o, feed):
            x = feed["pixel_values"]
            return [x.reshape(len(x), -1)[:, :4].astype(np.float32) + 1.0]

    out = ew.encode_images_onnx(_Sess(), [f"{i}.jpg" for i in range(20)], batch=8)
    assert out.shape == (20, 4)
    assert prep_threads and consumer not in prep_threads, "preprocess ran on the GPU loop thread"
