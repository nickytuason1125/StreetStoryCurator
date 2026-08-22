r"""
Book-derived phrases must not enter a prompt unless the user opts in.

cache/rag_concepts.json holds phrases EXTRACTED from copyrighted PDFs the user
uploaded. Six code paths injected them into prompts: the Qwen scoring prompt,
three places in grade_pipeline_v2, and the Story text rerank. The source books
were already moved off the repo in August over exactly this concern, and a
prior audit found roughly 25 of 62 phrases were biography prose rather than
photographic criteria -- so the store was carrying both a licensing question
and a quality one.

This does not delete the feature. It makes it OFF by default at the one place
every consumer already passes through, so nothing derived from a book reaches a
prompt in a shipped build, while a user who wants it locally can still turn it
on.

Run:  venv\Scripts\python.exe -m pytest tests/test_rag_off_by_default.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pdf_rag  # noqa: E402
import run_profile  # noqa: E402


def test_setting_is_declared():
    assert "FRAMEGRADE_USE_RAG_CONCEPTS" in run_profile.SETTINGS


def test_off_by_default(monkeypatch):
    """The default build injects nothing book-derived."""
    monkeypatch.delenv("FRAMEGRADE_USE_RAG_CONCEPTS", raising=False)
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts() == []


def test_opt_in_restores_them(monkeypatch):
    monkeypatch.setenv("FRAMEGRADE_USE_RAG_CONCEPTS", "1")
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts() == ["a phrase from a book"]


def test_raw_read_still_works_for_display(monkeypatch):
    """The UI must still be able to SHOW what is stored -- hiding it would make
    the store invisible rather than unused."""
    monkeypatch.delenv("FRAMEGRADE_USE_RAG_CONCEPTS", raising=False)
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts(for_display=True) == ["a phrase from a book"]
