r"""
Book-derived phrase injection policy.

cache/rag_concepts.json holds phrases EXTRACTED from reference PDFs the user
uploaded on their own machine. History: an August 2026 audit found ~25 of 62
phrases were biography prose rather than photographic criteria, so the feature
was made OPT-IN (off by default) as a quality/licence precaution.

Policy reversed 2026-08-23 by maintainer direction: RAG now powers grading
rubrics AND Story/Competition selection, so it is ON by default.
FRAMEGRADE_USE_RAG_CONCEPTS=0 restores the old silent behaviour, and
load_concepts(for_display=True) always shows what is stored regardless.

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


def test_on_by_default(monkeypatch):
    """The default build feeds uploaded book phrases into selection."""
    monkeypatch.delenv("FRAMEGRADE_USE_RAG_CONCEPTS", raising=False)
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts() == ["a phrase from a book"]


def test_opt_out_disables(monkeypatch):
    """Users can still silence every book-derived injection with one env var."""
    monkeypatch.setenv("FRAMEGRADE_USE_RAG_CONCEPTS", "0")
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts() == []


def test_raw_read_still_works_for_display(monkeypatch):
    """The UI must always be able to SHOW what is stored -- hiding it would make
    the store invisible rather than unused."""
    monkeypatch.delenv("FRAMEGRADE_USE_RAG_CONCEPTS", raising=False)
    monkeypatch.setattr(pdf_rag, "_read_concepts_file", lambda: ["a phrase from a book"])
    assert pdf_rag.load_concepts(for_display=True) == ["a phrase from a book"]
