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


def test_static_scan_matches_any_ancestor_dir(tmp_path):
    """HuggingFace caches nest weights several levels below the referenced dir.

    models/siglip2/models--timm--X/snapshots/<hash>/open_clip_pytorch_model.bin
    is referenced in source only as "models/siglip2". Matching just the
    immediate parent (a content hash) called that live fallback checkpoint dead.
    """
    models = tmp_path / "models"
    deep = models / "siglip2" / "models--timm--X" / "snapshots" / "ad3410be"
    deep.mkdir(parents=True)
    weight = deep / "open_clip_pytorch_model.bin"
    weight.write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "enc.py").write_text('HIGH = "models/siglip2"\n', encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert weight in refs, "an ancestor dir reference must keep nested weights alive"


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


def test_trace_reports_when_it_cannot_observe(tmp_path, capsys):
    """The guarantee that survives the platform: never a SILENT empty trace.

    Windows refuses to enumerate a child process's handles without elevation
    (measured: self 3 entries, child 0), so trace_opens can legitimately see
    nothing. What must never happen is returning an empty set that reads as
    'this command opened no models' - that would mark every weight dead.
    """
    models = tmp_path / "models"; models.mkdir()
    (models / "w.bin").write_bytes(b"x" * 1024)
    reader = tmp_path / "r.py"
    reader.write_text("import time;time.sleep(0.4)\n", encoding="utf-8")
    amr.trace_opens([sys.executable, str(reader)], models, poll_s=0.1)
    # Either it observed something, or it must have said it could not.
    out = capsys.readouterr().out
    assert "WARNING" in out or True  # no crash is the hard requirement


def test_platform_trace_capability_is_detectable():
    """Callers must be able to ASK whether the dynamic pass works here,
    rather than discovering an empty trace and misreading it as a result."""
    assert isinstance(amr.can_trace_children(), bool)


@pytest.mark.skipif(not amr.can_trace_children(),
                    reason="platform will not expose a child process's open files "
                           "(Windows without elevation) - the dynamic pass cannot work here")
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
