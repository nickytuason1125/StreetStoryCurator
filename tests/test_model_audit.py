"""
The model-weight audit must not produce false positives.

A false positive here means quarantining a weight the app actually loads, so
these tests pin the properties that prevent that: a file opened at runtime is
never a candidate, a file named in source is never a candidate, a checkpoint
shard is kept alive by a reference to its directory, and a git-tracked file is
never a candidate regardless of what the scans say.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_model_audit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import audit_model_refs as amr          # noqa: E402


def test_static_scan_finds_a_named_weight(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    (models / "used.onnx").write_bytes(b"x" * 16)
    (models / "unused.onnx").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "loader.py").write_text("PATH = 'models/used.onnx'\n", encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert (models / "used.onnx") in refs
    assert (models / "unused.onnx") not in refs


def test_static_scan_matches_by_parent_dir(tmp_path):
    """Checkpoint dirs are referenced by directory name, not shard by shard.

    Matching only filenames would call every shard of a live checkpoint dead.
    """
    models = tmp_path / "models"
    ckpt = models / "siglip2_hf_fp16"; ckpt.mkdir(parents=True)
    (ckpt / "model-00001.safetensors").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "enc.py").write_text("DIR = 'models/siglip2_hf_fp16'\n", encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert (ckpt / "model-00001.safetensors") in refs


def test_static_scan_skips_excluded_dirs(tmp_path):
    """A reference from deprecated/ must NOT keep a weight alive.

    Otherwise retired code pins weights forever and nothing is reclaimable.
    """
    models = tmp_path / "models"; models.mkdir()
    (models / "old.onnx").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    dep = src / "deprecated"; dep.mkdir()
    (dep / "legacy.py").write_text("'models/old.onnx'\n", encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert (models / "old.onnx") not in refs


def test_trace_catches_a_runtime_open(tmp_path):
    """The whole point: a file opened at runtime but named nowhere in source."""
    models = tmp_path / "models"; models.mkdir()
    target = models / "runtime_only.bin"
    target.write_bytes(b"x" * (4 * 1024 * 1024))

    reader = tmp_path / "reader.py"
    reader.write_text(
        "import time,sys\n"
        "f=open(sys.argv[1],'rb')\n"
        "f.read()\n"
        "time.sleep(2.0)\n"
        "f.close()\n",
        encoding="utf-8")

    opened = amr.trace_opens([sys.executable, str(reader), str(target)],
                             models, poll_s=0.1)
    assert target in opened


def test_candidates_exclude_everything_referenced(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    (models / "keep.onnx").write_bytes(b"x" * 16)
    (models / "drop.onnx").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "a.py").write_text("'models/keep.onnx'\n", encoding="utf-8")

    result = amr.audit(models, search_roots=[src], traced=set(), tracked=set())
    assert (models / "drop.onnx") in result["candidates"]
    assert (models / "keep.onnx") not in result["candidates"]
    assert result["bytes_reclaimable"] == 16


def test_git_tracked_files_are_never_candidates(tmp_path):
    """Backstop against the scans misjudging a small tracked config.

    models/ is mostly gitignored weights, but a handful of configs ARE tracked.
    Quarantining one would show up as a deletion in git status, and a config a
    loader finds by convention may be named nowhere in source.
    """
    models = tmp_path / "models"; models.mkdir()
    cfg = models / "dfine_nano_config.json"
    cfg.write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "a.py").write_text("nothing referenced here\n", encoding="utf-8")

    without = amr.audit(models, search_roots=[src], traced=set(), tracked=set())
    assert cfg in without["candidates"], "fixture must be a candidate absent the guard"

    with_guard = amr.audit(models, search_roots=[src], traced=set(), tracked={cfg})
    assert cfg not in with_guard["candidates"]
    assert with_guard["bytes_reclaimable"] == 0
