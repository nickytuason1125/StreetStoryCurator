"""
ingest_pdfs_multimodal.py — RAG concept phrase ingestion from photography PDFs.

Extracts 15-25 concept phrases per PDF using LLM (GGUF or Ollama fallback),
stores them in cache/rag_concepts.json.  Phrases are injected into the
Qwen2.5-VL grading prompt at grade time as a style/rubric reference.

Usage
-----
  # Ingest a single PDF
  venv\\Scripts\\python.exe scripts/ingest_pdfs_multimodal.py book.pdf

  # Ingest all PDFs in a folder
  venv\\Scripts\\python.exe scripts/ingest_pdfs_multimodal.py path/to/pdf_folder/

  # List currently ingested PDFs and phrase count
  venv\\Scripts\\python.exe scripts/ingest_pdfs_multimodal.py --list

  # Clear all ingested concepts
  venv\\Scripts\\python.exe scripts/ingest_pdfs_multimodal.py --clear

Notes
-----
- Concept phrases are stored in cache/rag_concepts.json (plain JSON, no ChromaDB).
- Up to 8 phrases per grade run are injected into the VLM system prompt as a
  rubric block.  More PDFs = richer style guidance for the grader.
- Requires PyMuPDF (fitz) for text extraction:  pip install pymupdf
- LLM extraction priority: local GGUF → Ollama qwen2.5:7b → error
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project src/ is importable when called from project root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _list_concepts() -> None:
    from pdf_rag import list_pdfs, load_concepts
    pdfs     = list_pdfs()
    concepts = load_concepts()
    if not pdfs:
        print("  No PDFs ingested yet.")
        print("  Run:  python scripts/ingest_pdfs_multimodal.py your_book.pdf")
        return
    print(f"  {len(concepts)} total concept phrases from {len(pdfs)} PDF(s):\n")
    for p in pdfs:
        print(f"    [{p['phrases']:3d} phrases]  {p['name']}  ({p['pages']} pages)")
    print()
    print("  Sample phrases:")
    for phrase in concepts[:8]:
        print(f"    • {phrase}")
    if len(concepts) > 8:
        print(f"    … and {len(concepts) - 8} more")


def _clear_concepts() -> None:
    from pdf_rag import clear_concepts
    clear_concepts()
    print("  All ingested PDF concepts cleared.")


def _ingest(paths: list[Path]) -> None:
    from pdf_rag import ingest_pdf

    total_new = 0
    for p in paths:
        if not p.exists():
            print(f"  SKIP: {p} not found")
            continue
        if p.suffix.lower() != ".pdf":
            print(f"  SKIP: {p.name} is not a PDF")
            continue

        print(f"\n  Ingesting: {p.name}")
        t0 = time.time()
        try:
            new_phrases = ingest_pdf(str(p))
            elapsed     = time.time() - t0
            print(f"  ✓ {len(new_phrases)} new phrases extracted in {elapsed:.1f}s")
            total_new += len(new_phrases)
        except FileNotFoundError:
            print(f"  ✗ PDF not found: {p}")
        except ValueError as e:
            print(f"  ✗ {e}")
        except RuntimeError as e:
            print(f"  ✗ LLM extraction failed: {e}")
            print("    Ensure Ollama is running (ollama pull qwen2.5:7b) or place a")
            print("    GGUF model in models/ for offline extraction.")

    if total_new:
        print(f"\n  Done — {total_new} new phrases added to RAG store.")
        print("  These will be injected into the VLM grading prompt on next run.")
    else:
        print("\n  No new phrases extracted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest photography PDFs into the RAG concept store",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF file(s) or folder(s) containing PDFs to ingest",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List currently ingested PDFs and phrase samples",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove all ingested concepts from the store",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  Street Story Curator — RAG PDF Ingestion")
    print("=" * 62)

    if args.clear:
        _clear_concepts()
        return

    if args.list or not args.paths:
        _list_concepts()
        return

    # Resolve all supplied paths to individual PDF files
    pdf_files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.pdf")) + sorted(p.glob("*.PDF"))
            if not found:
                print(f"  No PDFs found in {p}")
            else:
                pdf_files.extend(found)
                print(f"  Found {len(found)} PDF(s) in {p}")
        else:
            pdf_files.append(p)

    if not pdf_files:
        print("  Nothing to ingest. Supply a PDF path or folder.")
        sys.exit(1)

    _ingest(pdf_files)
    print()
    _list_concepts()


if __name__ == "__main__":
    main()
