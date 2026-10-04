"""The ONNX image encode batch never follows free RAM (2026-10-04).

Measured: the same photos encoded at batch 6 or 2 instead of 8 differ by up to
5e-3 per embedding value (batch 8 twice: bit-identical), which flipped 5 of 600
grade buckets between two identical culls taken at different free RAM — the
RAM planner rewrote SIGLIP_ENC_BATCH and planned 200- vs 350-photo chunks
(350 = 43 x 8 + 6, so the last batch of every chunk changed shape).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import run_profile  # noqa: E402


def test_onnx_batch_ignores_ram_rewrites(monkeypatch):
    monkeypatch.delenv("FIRSTCUT_ONNX_ENC_BATCH", raising=False)
    base = run_profile.onnx_image_batch()
    monkeypatch.setenv("SIGLIP_ENC_BATCH", "2")          # what memory_plan writes when tight
    assert run_profile.onnx_image_batch() == base
    assert base in (8, 16)


def test_explicit_override_still_works(monkeypatch):
    monkeypatch.setenv("FIRSTCUT_ONNX_ENC_BATCH", "4")
    assert run_profile.onnx_image_batch() == 4


def test_chunks_align_to_the_batch():
    import siglip2_encoder as se
    assert se._align_chunk(350, 8) == 344
    assert se._align_chunk(200, 8) == 200
    assert se._align_chunk(5, 8) == 8                      # never below one batch
