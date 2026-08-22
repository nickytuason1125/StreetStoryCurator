"""
pdf_rag.py — Photography Reference PDF ingestion pipeline.

Flow
────
  Upload PDF → extract text (PyMuPDF) → LLM extracts concept phrases
  → saved to cache/rag_concepts.json

At grade time (grade_pipeline_v2.py), concepts are embedded by SigLIP-2
and blended into the semantic anchor alongside the CD brief.

Public API
──────────
  ingest_pdf(pdf_path)        → list[str]  concept phrases
  load_concepts()             → list[str]  all saved phrases
  clear_concepts()            → None
  list_pdfs()                 → list[dict] {"name": str, "pages": int, "phrases": int}
"""
from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Optional

_CACHE_DIR   = Path(__file__).resolve().parent.parent / "cache"
_PDF_DIR     = _CACHE_DIR / "rag_pdfs"
_CONCEPTS_FILE = _CACHE_DIR / "rag_concepts.json"
_PDF_DIR.mkdir(parents=True, exist_ok=True)


# ── Concept store ─────────────────────────────────────────────────────────────

def load_concepts() -> list[str]:
    """Return all saved concept phrases across all uploaded PDFs."""
    if not _CONCEPTS_FILE.exists():
        return []
    try:
        data = json.loads(_CONCEPTS_FILE.read_text(encoding="utf-8"))
        return data.get("phrases", [])
    except Exception:
        return []


def _save_concepts(phrases: list[str], pdf_meta: list[dict]) -> None:
    data = {
        "phrases": phrases,
        "pdfs":    pdf_meta,
    }
    tmp = _CONCEPTS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(_CONCEPTS_FILE))


def clear_concepts() -> None:
    if _CONCEPTS_FILE.exists():
        _CONCEPTS_FILE.unlink()
    for p in _PDF_DIR.glob("*.pdf"):
        p.unlink(missing_ok=True)


def list_pdfs() -> list[dict]:
    if not _CONCEPTS_FILE.exists():
        return []
    try:
        data = json.loads(_CONCEPTS_FILE.read_text(encoding="utf-8"))
        return data.get("pdfs", [])
    except Exception:
        return []


# ── PDF text extraction ───────────────────────────────────────────────────────

def _extract_text(pdf_path: str | Path, max_chars: int = 24_000) -> tuple[str, int]:
    """
    Extract plain text from a PDF using pypdfium2 (Apache-2.0 — replaces
    PyMuPDF, which is AGPL-3.0/commercial-licensed).
    Returns (text, page_count).  Truncates at max_chars to fit LLM context.
    """
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf_path))
    pages = len(doc)
    chunks: list[str] = []
    total = 0
    for page in doc:
        textpage = page.get_textpage()
        chunk = textpage.get_text_range().strip()
        textpage.close()
        page.close()
        if not chunk:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        total += len(chunk)
    doc.close()
    return "\n\n".join(chunks), pages


# ── LLM concept extraction ────────────────────────────────────────────────────

_CONCEPT_PROMPT = """\
You are a photography critic and educator.
Read the following text from a photography book or reference document.
Extract 15–25 specific, concrete phrases that describe WHAT A STRONG PHOTOGRAPH
LOOKS LIKE — qualities visible in the image itself:
  • what makes a photograph aesthetically strong
  • lighting qualities and how light should behave
  • compositional rules, geometry, and framing principles
  • narrative moment, gesture, decisive moment
  • human presence, emotion, and cultural storytelling
  • technical craft standards (focus, exposure, clarity)

Every phrase must be checkable by looking at a single photograph.
Do NOT include career advice, work habits, biography, or instructions to the
photographer (no "work hard", "study the masters", "travel more").
If the text contains no visual criteria, output fewer phrases — never pad.

Write ONLY the phrases, one per line.
Each phrase should be 5–15 words, descriptive enough to match against a photograph.
No numbering, no headers, no explanation — just the phrases.

TEXT:
{text}

PHRASES:
"""

def _extract_concepts_gguf(text: str) -> list[str]:
    """
    Run concept extraction using the DeepSeek-R1-8B GGUF model (CPU/GPU).
    Returns a list of concept phrases.
    """
    from pathlib import Path as _P
    # Registry, not a filename. A third copy of the same hardcoded path is how
    # the model swap left three modules reaching for weights that had moved.
    try:
        import model_registry as _mr
        gguf = _mr.text_gguf_path()
    except Exception:
        gguf = _P(__file__).resolve().parent.parent / "models" / "unknown.gguf"
    if not gguf.exists():
        # Fallback: try Phi-4 mini
        gguf_alt = list((_P(__file__).resolve().parent.parent / "models").glob("*.gguf"))
        if not gguf_alt:
            raise FileNotFoundError("No GGUF model found for concept extraction")
        gguf = gguf_alt[0]

    from llama_cpp import Llama
    llm = Llama(
        model_path     = str(gguf),
        n_ctx          = 4096,
        n_gpu_layers   = 0,   # CPU only — avoid competing with SigLIP-2 VRAM
        verbose        = False,
    )
    prompt = _CONCEPT_PROMPT.format(text=text[:8000])
    out = llm(
        prompt,
        max_tokens  = 600,
        temperature = 0.2,
        stop        = ["\n\n\n"],
    )
    raw = out["choices"][0]["text"].strip()
    del llm
    return _parse_phrases(raw)


def _parse_phrases(raw: str) -> list[str]:
    """Parse LLM output into clean concept phrases.

    Strips <think>…</think> reasoning blocks (DeepSeek-R1 emits them before
    the answer; without stripping, thinking sentences pass the word-count
    filter and pollute the rubric)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.split("<think>")[0] if "<think>" in raw else raw
    _META = re.compile(
        r"^(here (are|is)\b|these (are|phrases)\b|the following\b|i (have|will)\b"
        r"|based on (the|this)\b|phrases? (extracted|describing)\b|sure[,!]|okay[,!])",
        re.IGNORECASE,
    )
    phrases = []
    for line in raw.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\-•\*\d\.\)]+\s*", "", line)
        if _META.match(line) or line.endswith(":"):
            continue   # LLM preamble/header, not a rubric phrase
        if 4 <= len(line.split()) <= 20 and len(line) > 10:
            phrases.append(line)
    return phrases[:25]


def _extract_concepts_llm(text: str) -> list[str]:
    """Extract rubric phrases with the shared local text model.

    This replaced an Ollama HTTP call that tried gemma3:4b, then qwen2.5:7b, then
    llama3.2 — three model tags nothing in the installer ever pulled, so on any
    machine but the developer's it returned [] immediately and every PDF fell
    through to the slow path below.
    """
    try:
        import local_llm
    except Exception:
        return []
    if not local_llm.available():
        return []
    out = local_llm.generate(
        _CONCEPT_PROMPT.format(text=text[:8000]),
        max_tokens=800,
        temperature=0.2,
    )
    if not out:
        return []
    phrases = _parse_phrases(out)
    if len(phrases) >= 5:
        print(f"[pdf_rag] local model: {len(phrases)} phrases")
        return phrases
    return []


def _extract_concepts(text: str) -> list[str]:
    """Shared local model first, then this module's own GGUF path.

    Both are now local; the difference is that the first is the warm singleton
    every text feature shares, and the second builds its own. Ingest is a rare,
    user-initiated action, so the slow path staying as a backstop costs nothing.
    """
    phrases = _extract_concepts_llm(text)
    if phrases:
        return phrases
    print("[pdf_rag] shared model unavailable/empty — trying the local GGUF…")
    return _extract_concepts_gguf(text)


# ── Public ingest ─────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str | Path, pdf_name: Optional[str] = None) -> list[str]:
    """
    Process one PDF: extract text → LLM concept phrases → append to store.

    Returns the list of new phrases extracted from this PDF.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[pdf_rag] Extracting text from {pdf_path.name}…")
    text, pages = _extract_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No extractable text in {pdf_path.name} (scanned PDF?)")

    print(f"[pdf_rag] {pages} pages, {len(text)} chars → extracting concepts…")
    new_phrases = _extract_concepts(text)
    print(f"[pdf_rag] Extracted {len(new_phrases)} concept phrases")

    # Merge with existing store (deduplicate by lowercase)
    existing = load_concepts()
    existing_lc = {p.lower() for p in existing}
    for p in new_phrases:
        if p.lower() not in existing_lc:
            existing.append(p)
            existing_lc.add(p.lower())

    # Update PDF metadata
    existing_pdfs = list_pdfs()
    name = pdf_name or pdf_path.name
    existing_pdfs = [p for p in existing_pdfs if p.get("name") != name]
    existing_pdfs.append({"name": name, "pages": pages, "phrases": len(new_phrases)})

    _save_concepts(existing, existing_pdfs)
    print(f"[pdf_rag] Store now has {len(existing)} total phrases from {len(existing_pdfs)} PDFs")
    return new_phrases
