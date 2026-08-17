"""
The default paths must not require Ollama.

FrameGrade shipped with server.py returning 503 for every non-scan grade when
http://localhost:11434 did not answer. Nothing in the project installed Ollama —
not Setup.ps1, not requirements.txt — and no user-facing document mentioned it,
while CLAUDE.md rule 5 promised a fully offline app. A correct, complete install
therefore could not grade a single photograph.

These tests lock the repair. They are deliberately source-level, in the style of
tests/test_process_boundary.py: the property being protected is "this dependency
does not come back", and that is a statement about the code, not about one run.

Ollama is still permitted in two places, both opt-in and both off by default:
Step 4e (FRAMEGRADE_STEP4E=1) and its helpers in qwen_vlm_grader/critique_engine.
The line is that nothing on a DEFAULT path may reach for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_OLLAMA_MARKERS = ("11434", "ollama")

# Opt-in only. Step 4e is gated on FRAMEGRADE_STEP4E, which defaults to "0", and
# critique_engine keeps the helpers that pass reaches for.
_ALLOWED = {"qwen_vlm_grader.py", "critique_engine.py", "grade_pipeline_v2.py"}


def _sources():
    """Every module on a default path, minus the opt-in Step 4e machinery."""
    for f in sorted(SRC.glob("*.py")):
        if f.name not in _ALLOWED:
            yield f
    yield ROOT / "server.py"


def _offenders(path: Path):
    """Ollama references that are CODE, via the AST.

    Scanning raw text was the obvious approach and it was wrong: it flagged the
    docstrings explaining why Ollama had been removed, which would have made the
    test demand the deletion of its own rationale. Comments and docstrings are
    documentation; only a string literal the program uses, or an identifier it
    calls, is a dependency.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            low = node.value.lower()
            if any(m in low for m in _OLLAMA_MARKERS):
                yield node.lineno, f"string literal {node.value[:60]!r}"
        elif isinstance(node, ast.Name) and any(m in node.id.lower() for m in _OLLAMA_MARKERS):
            yield node.lineno, f"identifier {node.id}"
        elif isinstance(node, ast.Attribute) and any(m in node.attr.lower() for m in _OLLAMA_MARKERS):
            yield node.lineno, f"attribute .{node.attr}"


# Two references survive, both in server.py, both tracked and both unreachable
# from the UI today:
#
#   /api/ollama/status   the route PATH. The handler no longer speaks to Ollama —
#                        it reports local model files — but renaming the path is a
#                        frontend change (App.tsx polls it every 15 s), so the name
#                        outlives the dependency for one more step.
#   /api/models/pull     still proxies Ollama's pull stream. Unreachable from the
#                        UI now that /api/health/engine returns only .gguf names,
#                        and replaced outright when the model downloader lands.
#
# They are listed rather than allowlisted by pattern: this set may only shrink.
_KNOWN_REMAINING = {
    "server.py:/api/ollama/status",
    "server.py:http://localhost:11434/api/pull",
}


def test_no_default_path_reaches_for_ollama():
    found = []
    for f in _sources():
        if not f.exists():
            continue
        for lineno, what in _offenders(f):
            key = None
            if what.startswith("string literal "):
                key = f"{f.name}:{what[len('string literal '):].strip(chr(39))}"
            if key in _KNOWN_REMAINING:
                continue
            found.append(f"{f.relative_to(ROOT)}:{lineno}: {what}")
    assert not found, (
        "Ollama reappeared on a default code path:\n  "
        + "\n  ".join(found)
        + "\n\nUse src/local_llm.py (text) or critique_engine.describe_image "
          "(vision) — both run locally with no external service."
    )


def test_grade_endpoint_has_no_ollama_health_gate():
    """The specific defect: a 503 before the grading thread was even spawned."""
    src = (ROOT / "server.py").read_text(encoding="utf-8", errors="replace")
    assert "_check_ollama_available" not in src, (
        "server.py must not gate grading on an Ollama health check"
    )


def test_local_llm_degrades_instead_of_raising():
    """A missing model is an answer, not an exception.

    Every caller has a defined behaviour without an LLM — Story Mode ranks by
    score, pixel_inspector returns "", pdf_rag falls back to its own extractor.
    Raising here would turn a degraded feature into a failed grade, which is the
    class of bug this whole change is undoing.
    """
    import local_llm
    original = local_llm.model_path
    try:
        local_llm.model_path = lambda: Path("does") / "not" / "exist.gguf"
        local_llm._llm = None
        local_llm._load_attempted = False
        assert local_llm.available() is False
        assert local_llm.generate("anything") is None
    finally:
        local_llm.model_path = original
        local_llm._llm = None
        local_llm._load_attempted = False


def test_local_llm_never_touches_torch_cuda_to_pick_a_device():
    """It runs in the server process, the ancestor of the CUDA grade subprocess."""
    src = (SRC / "local_llm.py").read_text(encoding="utf-8")
    body = src.split("def _load(")[1].split("\ndef ")[0]
    assert "torch" not in body, (
        "local_llm._load must ask tier_select.has_gpu() — a cached subprocess "
        "probe — not torch.cuda, which initialises a context in the parent and "
        "makes it fault 0xC0000005 when a GPU child exits"
    )


@pytest.mark.parametrize("module,attr", [
    ("local_llm", "generate"),
    ("local_llm", "available"),
    ("local_llm", "unload"),
    ("critique_engine", "describe_image"),
    ("critique_engine", "vision_available"),
])
def test_replacement_surface_exists(module, attr):
    """The API the rewired callers depend on."""
    mod = __import__(module)
    assert callable(getattr(mod, attr)), f"{module}.{attr} must be callable"
