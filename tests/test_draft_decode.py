"""Contract tests for scaled ("draft") JPEG decode.

A cull's dominant cost was decoding full-resolution JPEGs and immediately
shrinking them: measured 2026-08-28, the IQA stage was 64% of a cull and 99% of
that stage was decode (279 ms/img) against 3 ms for the quality model itself.
decode_one(draft_hint=N) asks libjpeg to downscale in the DCT domain instead.

These tests lock the properties that make that safe:
  1. the decode still covers the hint (never smaller — that would lose detail)
  2. the hint never UPscales a small image
  3. FRAMEGRADE_DRAFT_DECODE=0 restores the exact previous pixels
  4. non-JPEG paths are unaffected
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


@pytest.fixture(scope="module")
def big_jpeg(tmp_path_factory):
    from PIL import Image
    p = tmp_path_factory.mktemp("draft") / "big.jpg"
    rng = np.random.default_rng(0)
    # Structured, not pure noise: JPEG destroys noise, and a flat image would
    # make every scale look identical and the test vacuous.
    y, x = np.mgrid[0:2048, 0:3072]
    arr = ((np.sin(x / 40.0) + np.cos(y / 37.0)) * 60 + 128).astype(np.uint8)
    rgb = np.dstack([arr, np.roll(arr, 7, 1), np.roll(arr, 13, 0)])
    rgb = (rgb.astype(np.int16) + rng.integers(-4, 5, rgb.shape)).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(p, quality=92)
    return str(p)


def test_scale_den_never_goes_below_the_hint():
    from fast_ingestion import _scale_den
    for w, h in ((6000, 4000), (3072, 2048), (1024, 768), (600, 400)):
        d = _scale_den(w, h, 512)
        assert min(w, h) // d >= 512 or d == 1, (w, h, d)


def test_scale_den_does_not_shrink_small_images():
    from fast_ingestion import _scale_den
    assert _scale_den(700, 500, 512) == 1     # 500//2 = 250 < 512 → must not scale


def test_draft_decode_still_covers_the_hint(big_jpeg):
    from fast_ingestion import decode_one
    t = decode_one(big_jpeg, pin=False, draft_hint=512)
    assert t is not None
    _, h, w = t.shape
    assert min(h, w) >= 512, f"decoded {w}x{h}, below the 512 hint"


def test_draft_decode_is_smaller_than_full(big_jpeg):
    """The whole point: fewer pixels come out of the decoder."""
    from fast_ingestion import decode_one
    full = decode_one(big_jpeg, pin=False)
    draft = decode_one(big_jpeg, pin=False, draft_hint=512)
    assert full is not None and draft is not None
    assert draft.numel() < full.numel(), "draft_hint did not reduce the decode"


def test_env_kill_switch_restores_full_decode(big_jpeg, monkeypatch):
    """FRAMEGRADE_DRAFT_DECODE=0 must give back byte-identical old behaviour."""
    from fast_ingestion import decode_one
    full = decode_one(big_jpeg, pin=False)
    monkeypatch.setenv("FRAMEGRADE_DRAFT_DECODE", "0")
    off = decode_one(big_jpeg, pin=False, draft_hint=512)
    assert off is not None and full is not None
    assert off.shape == full.shape
    assert np.array_equal(off.numpy(), full.numpy()), "kill switch did not restore full decode"


def test_png_ignores_draft_hint(tmp_path):
    """draft() is a JPEG/DCT feature; other formats must decode unchanged."""
    from PIL import Image
    from fast_ingestion import decode_one
    p = tmp_path / "x.png"
    Image.fromarray(np.full((900, 1200, 3), 90, dtype=np.uint8)).save(p)
    a = decode_one(str(p), pin=False)
    b = decode_one(str(p), pin=False, draft_hint=512)
    assert a is not None and b is not None
    assert a.shape == b.shape
    assert np.array_equal(a.numpy(), b.numpy())
