"""
The low-RAM streaming converter rewrites model weights by hand, so a silent
error here would not crash — it would produce a checkpoint that loads fine and
embeds subtly wrong vectors. That is the worst failure shape this repo has, so
the properties below are checked numerically rather than structurally.

Covers the two paths that matter:
  * small tensors  — read whole, cast, write
  * large tensors  — filled a row-block at a time (the 256000x1024 token
                     embedding is why the streaming path exists at all)
Both must produce bit-identical fp16 to a plain .astype(float16).

Run:  venv\\Scripts\\python.exe -m pytest tests/test_lean_converter.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

build_lean_checkpoint = pytest.importorskip("build_lean_checkpoint")
save_file = pytest.importorskip("safetensors.numpy").save_file
load_file = pytest.importorskip("safetensors.numpy").load_file

B = build_lean_checkpoint


def _src(tmp_path, tensors, cfg=None):
    """Write a fake fp32 source checkpoint the way HF ships one."""
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "model.safetensors"))
    (d / "config.json").write_text(
        json.dumps(cfg or {"model_type": "siglip2", "torch_dtype": "float32"}),
        encoding="utf-8")
    (d / "preprocessor_config.json").write_text('{"size": 384}', encoding="utf-8")
    return d


def _roundtrip(tmp_path, tensors):
    src = _src(tmp_path, tensors)
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    B._stage2_convert_streaming(str(src), staging)
    out = {}
    for shard in staging.glob("model-*.safetensors"):
        out.update(load_file(str(shard)))
    return out, staging


def test_small_tensors_are_bit_identical(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {f"layer.{i}.weight": rng.standard_normal((32, 64), dtype=np.float32)
               for i in range(8)}
    out, _ = _roundtrip(tmp_path, tensors)
    assert set(out) == set(tensors)
    for k, v in tensors.items():
        assert out[k].dtype == np.float16
        np.testing.assert_array_equal(out[k], v.astype(np.float16))


def test_large_tensor_row_chunked_path_is_bit_identical(monkeypatch, tmp_path):
    """Force the row-block branch on a tensor small enough to test quickly."""
    monkeypatch.setattr(B, "_ROW_CHUNK_BYTES", 4096)   # ~16 rows of 64 floats
    rng = np.random.default_rng(1)
    big = rng.standard_normal((501, 64), dtype=np.float32)   # not a block multiple
    out, _ = _roundtrip(tmp_path, {"text.embed.weight": big})
    np.testing.assert_array_equal(out["text.embed.weight"], big.astype(np.float16))


def test_row_chunking_handles_exact_and_ragged_last_block(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "_ROW_CHUNK_BYTES", 512)    # 2 rows of 64 floats
    rng = np.random.default_rng(2)
    for rows in (4, 5, 1, 2, 7):
        t = rng.standard_normal((rows, 64), dtype=np.float32)
        out, _ = _roundtrip(tmp_path / f"r{rows}", {"w": t})
        np.testing.assert_array_equal(out["w"], t.astype(np.float16))


def test_shards_split_and_index_maps_every_tensor(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "_SHARD_BYTES", 8192)
    rng = np.random.default_rng(3)
    tensors = {f"w{i}": rng.standard_normal((64, 64), dtype=np.float32)
               for i in range(12)}
    out, staging = _roundtrip(tmp_path, tensors)

    shards = sorted(staging.glob("model-*.safetensors"))
    assert len(shards) > 1, "expected the shard budget to force a split"

    idx = json.loads((staging / "model.safetensors.index.json")
                     .read_text(encoding="utf-8"))
    assert set(idx["weight_map"]) == set(tensors)
    for name in idx["weight_map"].values():
        assert (staging / name).exists()
    assert idx["metadata"]["total_size"] == sum(v.size * 2 for v in tensors.values())
    assert set(out) == set(tensors)


def test_config_is_marked_fp16_and_sidecars_copied(tmp_path):
    _, staging = _roundtrip(tmp_path, {"w": np.ones((4, 4), dtype=np.float32)})
    cfg = json.loads((staging / "config.json").read_text(encoding="utf-8"))
    assert cfg["torch_dtype"] == "float16" and cfg["dtype"] == "float16"
    assert cfg["model_type"] == "siglip2", "unrelated config keys must survive"
    assert (staging / "preprocessor_config.json").exists()
    assert not list(staging.glob("model.safetensors")), "fp32 source must not be copied"


def test_scalar_and_1d_tensors_survive(tmp_path):
    tensors = {"logit_scale": np.array(4.5, dtype=np.float32),
               "bias": np.arange(10, dtype=np.float32)}
    out, _ = _roundtrip(tmp_path, tensors)
    assert out["logit_scale"].shape == ()
    np.testing.assert_array_equal(out["bias"], np.arange(10, dtype=np.float16))


def test_streaming_is_chosen_when_ram_is_short(monkeypatch, tmp_path):
    """The path decision must be made on measurement, not by catching an error:
    the low-RAM failure is a segfault, which cannot be caught."""
    # Tested through the pure decision function rather than _stage2_convert, so
    # the assertion does not depend on importing transformers/torch.
    src = _src(tmp_path, {"w": np.ones((256, 256), dtype=np.float32)})
    src_gb = (src / "model.safetensors").stat().st_size / 1e9

    stream, free, need = B._should_stream(str(src), free_gb=0.0)
    assert stream, "should stream when there is no RAM"
    assert need == pytest.approx(src_gb * 1.4)

    stream, _, _ = B._should_stream(str(src), free_gb=64.0)
    assert not stream, "plenty of RAM should use the normal loader"

    # The real case that motivated this: a 3.5 GB source on a machine with
    # 0.3 GB free, which is where from_pretrained segfaults.
    stream, _, _ = B._should_stream(str(src), free_gb=src_gb * 1.4 - 1e-9)
    assert stream, "just under the threshold must still stream"


def test_bf16_source_is_widened_correctly(tmp_path):
    """bf16 has no numpy dtype; a wrong shift would corrupt every weight."""
    vals = np.array([1.0, -2.5, 0.0, 65504.0], dtype=np.float32)
    bf16_bits = (vals.view(np.uint32) >> 16).astype("<u2")
    src = tmp_path / "src"; src.mkdir()
    save_file({"w": bf16_bits}, str(src / "model.safetensors"))

    hdr, base = B._read_header(src / "model.safetensors")
    hdr["w"]["dtype"] = "BF16"          # reinterpret the same bytes as bf16
    with open(src / "model.safetensors", "rb") as fh:
        got = B._load_tensor_fp16(fh, base, hdr["w"])
    expected = (bf16_bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    np.testing.assert_array_equal(got, expected)
