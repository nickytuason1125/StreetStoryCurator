"""models/qwen3_vl is a REJECTED tournament candidate, not an install dependency.

The 2026-06-14 bake-off rejected Qwen3-VL-4B (10.93 s/img against the
incumbent's 8.79, and a score spread of std 0.076 against 0.145). Its 8.3 GB of
weights then sat in the working tree looking like a half-finished integration,
because the dispatch and the scale anchors for it are still present — they are
leftovers from the bake-off, and they are inert without weights.

These lock that inertness, so the weights can be archived off-disk and a later
change cannot quietly make them required again without failing here first.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_the_model_registry_does_not_list_qwen3_vl():
    """The downloader fetches what the registry lists. It must not list this."""
    reg = _read("src/model_registry.py")
    assert "qwen3_vl" not in reg, (
        "model_registry names qwen3_vl — a rejected model would be downloaded "
        "on every fresh install"
    )


def test_no_module_hardcodes_the_qwen3_vl_weights_path():
    """Only the bake-off harness may name this directory.

    'Never name a weight file in code — ask model_registry' is a standing rule
    here (CLAUDE.md); five modules once hardcoded a deepseek filename and a
    correct install then reported models missing.
    """
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        if py.name == "qwen_vlm_grader.py":
            continue          # dispatches on config model_type, not on a path
        if "models/qwen3_vl" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(py.relative_to(ROOT).as_posix())
    assert offenders == [], f"these modules hardcode the archived path: {offenders}"


def test_the_grader_dispatches_on_config_not_on_disk_layout():
    """The dispatch may stay: it reads model_type from whatever config is
    loaded, so it costs nothing when the weights are absent."""
    src = _read("src/qwen_vlm_grader.py")
    assert 'if _model_type == "qwen3_vl"' in src
    # Line 34 is a doc comment naming models/qwen3_vl as an EXAMPLE of the
    # per-generation layout. That is documentation and stays; what must not
    # exist is the path baked into executable code.
    code_only = "\n".join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "models/qwen3_vl" not in code_only, (
        "the grader hardcodes the archived directory instead of taking a path"
    )


def test_the_tournament_verdict_is_still_on_record():
    """The reason the weights are archived must remain readable."""
    verdict = json.loads((ROOT / "_ab_tournament.json").read_text(encoding="utf-8"))
    assert "rejected" in verdict["verdict"].lower()
    assert "qwen3_4b" in verdict["summary"]
