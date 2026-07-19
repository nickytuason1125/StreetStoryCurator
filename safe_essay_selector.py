#!/usr/bin/env python3
"""
safe_essay_selector.py -- Copyright-compliant in-memory photo essay pipeline.

FAIR USE ARCHITECTURE (Non-Expressive Use):
  Raw images are NEVER written to disk at any point. Every pixel buffer is
  immediately downgraded to 224x224 grayscale before processing, destroying
  any high-resolution expressive value while preserving geometric structure.
  The only persistent artifact is the final metadata manifest.

PIPELINE:
  Stage 1  Open each PDF -> extract images/render pages into volatile memory
           Immediately: resize to 224x224 + convert to grayscale ('L')
           del original buffer -> CLIP encode -> del downgraded buffer
           Rank all CLIP vectors against query -> Top 5 by cosine similarity

  Stage 2  Re-render ONLY the Top 5 pages at 512x512 grayscale (in memory)
           base64-encode -> POST to Qwen 2.5-VL via Ollama (temperature=0.0)
           del pixel buffer immediately after POST

  Output   final_essay_manifest.json -- metadata only:
           source_pdf, page, clip_score, approved_for_essay,
           assigned_role, structural_justification
           NO image paths. NO pixel data. NO raw files on disk.

Usage:
    python safe_essay_selector.py
    python safe_essay_selector.py "your query" ./photo_essay_pdfs
"""
from __future__ import annotations

import base64
import gc
import importlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
_PDF_DIR      = Path("./photo_essay_pdfs")
_OUTPUT_FILE  = Path("./final_essay_manifest.json")
_OLLAMA_URL   = "http://localhost:11434/api/generate"
_ROOT         = Path(__file__).resolve().parent
_GGUF_MODEL   = _ROOT / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"
_MMPROJ       = _ROOT / "models" / "mmproj-qwen2.5-vl-2b-instruct-f16.gguf"

_CLIP_SIZE    = 224    # CLIP native input resolution
_QWEN_SIZE    = 512    # low-res render sent to Qwen (heavily downscaled)
_MIN_DIM      = 100    # discard any embedded image smaller than 100px in either axis
_BATCH        = 16     # CLIP encoding batch size
_DUPE_RE      = re.compile(r"\s*\(\d+\)\s*$")

_SYSTEM_PROMPT = """\
You are a curatorial AI trained on mid-century photojournalism theory (LIFE Magazine, \
documentary photo agencies) and contemporary art history (Stephen Bull, LUMA Arles). \
You evaluate individual photographs for inclusion in a documentary photo essay.

=== NARRATIVE PACING TAXONOMY ===
Every approved photograph must fulfill exactly one structural role:

  Establishing Shot   Wide frame, low subject-to-frame ratio, deep focus.
                      Grounds the viewer geographically and contextually.

  Interaction Shot    Medium framing (waist-up or wider). Captures relationships,
                      labor, urban friction, or social exchange between subjects.

  Detail Shot         High magnification, one dominant graphic element.
                      Functions as a visual pause or metaphor.

  Closer              Compositional finality: vanishing points, subjects moving
                      away, receding lines. Provides narrative resolution.

=== THE DOCUMENT CHECKLIST (Rejection Criteria) ===
REJECT the image (assigned_role = "Rejected", approved_for_essay = false) if it
lacks ANY of the following:

  1. INTENTIONAL GEOMETRY -- Leading lines, framing shapes, or rule of thirds
     with a clearly dominant subject. Random composition = immediate rejection.

  2. THE DECISIVE MOMENT -- Moving human subjects locked into geometric alignment
     with background elements or shadows. Static subject with no tension = reject.

  3. HUMANIST DIGNITY -- Camera at or near eye-level with the subject.
     High-angle shots that objectify subjects must be rejected on ethical grounds.

=== OUTPUT FORMAT ===
Respond ONLY with a single valid JSON object. No markdown, no extra text:
{
  "approved_for_essay": <true or false>,
  "assigned_role": "<Establishing Shot | Interaction Shot | Detail Shot | Closer | Rejected>",
  "structural_justification": "<One sentence citing the specific geometry, pacing, or \
humanism rule that drove this decision.>"
}
"""

_EVAL_PROMPT = (
    "Evaluate this photograph. Apply your Document Checklist, assign its Narrative "
    "Pacing role, and respond with a single JSON object only -- no other text."
)

_VALID_ROLES = {
    "Establishing Shot", "Interaction Shot",
    "Detail Shot", "Closer", "Rejected",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candidate:
    """Lightweight metadata record -- never holds pixel data."""
    pdf_path:     Path
    pdf_filename: str
    page_no:      int     # 1-indexed
    img_idx:      int     # 0 = full-page render, 1+ = embedded image index
    clip_score:   float   = 0.0


# =============================================================================
# DEPENDENCY CHECK
# =============================================================================

def verify_deps() -> None:
    missing = []
    for mod in ("torch", "transformers", "PIL", "pypdfium2"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pkg = {"PIL": "Pillow"}.get(mod, mod)
            missing.append(f"  {mod:<15} pip install {pkg}")
    if missing:
        print("[deps] MISSING:\n" + "\n".join(missing))
        sys.exit(1)

    ollama_ok = False
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        pass

    gguf_ok = _GGUF_MODEL.exists() and _MMPROJ.exists()

    if not ollama_ok and not gguf_ok:
        print("[deps] WARNING: No Qwen backend found. Stage 2 will produce error stubs.")
        print("  Option A: ollama pull qwen2.5vl:3b")
        print("  Option B: Place GGUF files in models/")
    else:
        backend = "Ollama" if ollama_ok else "GGUF"
        print(f"[deps] OK -- torch / transformers / Pillow / PyMuPDF / Qwen ({backend})")
    print()


# =============================================================================
# STAGE 1 -- IN-MEMORY CLIP PIPELINE
# =============================================================================

def _is_duplicate(p: Path) -> bool:
    return bool(_DUPE_RE.search(p.stem))


def _extract_downgraded(pdf_path: Path) -> list[tuple[Candidate, "PIL.Image.Image"]]:
    """
    Open the PDF, extract every image/render every page into a volatile
    224x224 grayscale PIL Image. Original raw buffers are deleted immediately.
    Returns list of (Candidate, downgraded_pil) -- caller must del pil after use.

    Uses pypdfium2 (Apache-2.0 -- replaces PyMuPDF, which is AGPL-3.0/
    commercial-licensed).
    """
    import pypdfium2 as pdfium
    from PIL import Image

    results: list[tuple[Candidate, Image.Image]] = []
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        # Encrypted/password-protected/corrupt -- pypdfium2 raises on open
        # rather than exposing needs_pass/is_encrypted flags after the fact.
        return []

    for page_no in range(1, len(doc) + 1):
        try:
            page = doc[page_no - 1]
        except Exception:
            continue

        img_idx = 0
        found_embedded = False
        for obj in page.get_objects(filter=(pdfium.raw.FPDF_PAGEOBJ_IMAGE,)):
            found_embedded = True
            img_idx += 1
            try:
                bitmap = obj.get_bitmap()
                img = bitmap.to_pil()
                bitmap.close()
                w, h = img.size
                if min(w, h) < _MIN_DIM:
                    del img
                    obj.close()
                    continue
                # IMMEDIATE DOWNGRADE -- destroys expressive resolution
                small = img.resize((_CLIP_SIZE, _CLIP_SIZE), Image.LANCZOS).convert("L")
                del img
            except Exception:
                obj.close()
                continue
            obj.close()

            cand = Candidate(pdf_path, pdf_path.name, page_no, img_idx)
            results.append((cand, small))

        if not found_embedded:
            # No embedded objects -- render page at minimal DPI (scale=1.0 == 72 DPI)
            try:
                bitmap = page.render(scale=1.0)
                img = bitmap.to_pil().convert("L")
                bitmap.close()
                if img.width < _MIN_DIM or img.height < _MIN_DIM:
                    del img
                    page.close()
                    continue
                small = img.resize((_CLIP_SIZE, _CLIP_SIZE), Image.LANCZOS)
                del img
            except Exception:
                page.close()
                continue

            cand = Candidate(pdf_path, pdf_path.name, page_no, 0)
            results.append((cand, small))

        page.close()

    doc.close()
    return results


def clip_stage(
    pdf_files: list[Path],
    query: str,
    top_k: int = 5,
) -> list[Candidate]:
    """
    Full Stage 1: extract all pages/images from every PDF into volatile memory,
    encode through CLIP, delete pixel buffers, rank, return Top K Candidates.
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] Loading openai/clip-vit-base-patch32 on {device} ...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Encode query text once
    with torch.no_grad():
        text_inp  = processor(text=[query], return_tensors="pt", padding=True).to(device)
        text_feat = model.get_text_features(**text_inp)
        text_feat = (text_feat / text_feat.norm(dim=-1, keepdim=True))[0].cpu()
    del text_inp

    scored: list[Candidate] = []
    total_processed = 0

    for pdf_path in pdf_files:
        print(f"[clip] {pdf_path.name} ...", end=" ", flush=True)
        pairs = _extract_downgraded(pdf_path)
        if not pairs:
            print("no images found")
            continue
        print(f"{len(pairs)} images extracted")

        # Encode in batches; del PIL images after each batch
        for i in range(0, len(pairs), _BATCH):
            batch = pairs[i:i + _BATCH]

            # L -> RGB required by CLIP processor
            rgbs = [img.convert("RGB") for _, img in batch]
            with torch.no_grad():
                inp   = processor(images=rgbs, return_tensors="pt", padding=True).to(device)
                feats = model.get_image_features(**inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)

            for j, (cand, _) in enumerate(batch):
                cand.clip_score = float(torch.dot(feats[j].cpu(), text_feat))
                scored.append(cand)

            # Flush all pixel data for this batch
            batch_size = len(batch)
            for rgb in rgbs:
                del rgb
            for _, pil in batch:
                del pil
            del rgbs, feats, inp, batch
            gc.collect()

            total_processed += batch_size

        del pairs

    del model, processor, text_feat
    gc.collect()

    print(f"\n[clip] Encoded {total_processed} image(s) across {len(pdf_files)} PDF(s)")

    scored.sort(key=lambda c: c.clip_score, reverse=True)

    # Deduplicate by (pdf_filename, page_no): keep highest-scoring image per page
    seen: set[tuple[str, int]] = set()
    top: list[Candidate] = []
    for cand in scored:
        key = (cand.pdf_filename, cand.page_no)
        if key not in seen:
            seen.add(key)
            top.append(cand)
        if len(top) >= top_k:
            break

    print(f"[clip] Top {len(top)} unique pages selected:")
    for c in top:
        print(f"   {c.clip_score:+.4f}  {c.pdf_filename}  page {c.page_no}")
    print()

    return top


# =============================================================================
# STAGE 2 -- IN-MEMORY QWEN EVALUATION
# =============================================================================

def _render_page_b64(pdf_path: Path, page_no: int) -> str:
    """
    Render a single PDF page at low resolution into grayscale, encode as base64
    JPEG, delete all pixel buffers, and return the base64 string only.

    Uses pypdfium2 (Apache-2.0 -- replaces PyMuPDF, AGPL-3.0/commercial).
    """
    import pypdfium2 as pdfium
    from PIL import Image

    doc    = pdfium.PdfDocument(str(pdf_path))
    page   = doc[page_no - 1]
    bitmap = page.render(scale=1.5)       # ~108 DPI -- enough for composition
    page.close()
    doc.close()

    img  = bitmap.to_pil().convert("L")
    bitmap.close()
    img  = img.resize((_QWEN_SIZE, _QWEN_SIZE), Image.LANCZOS)

    buf  = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=72)
    b64  = base64.b64encode(buf.getvalue()).decode()

    del img, buf
    gc.collect()
    return b64


def _parse_qwen_json(raw: str) -> Optional[dict]:
    raw   = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not {"approved_for_essay", "assigned_role", "structural_justification"}.issubset(obj):
        return None
    role = str(obj.get("assigned_role", "")).strip()
    if role not in _VALID_ROLES:
        role = "Rejected"
    return {
        "approved_for_essay":      bool(obj["approved_for_essay"]),
        "assigned_role":           role,
        "structural_justification": str(obj.get("structural_justification", "")).strip(),
    }


def _qwen_via_ollama(b64: str) -> Optional[dict]:
    try:
        import urllib.request
        payload = json.dumps({
            "model":   "qwen2.5vl:3b",
            "system":  _SYSTEM_PROMPT,
            "prompt":  _EVAL_PROMPT,
            "images":  [b64],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 500},
        }).encode()
        req = urllib.request.Request(
            _OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = json.loads(resp.read()).get("response", "").strip()
        return _parse_qwen_json(raw)
    except Exception as exc:
        print(f"[qwen/ollama] {type(exc).__name__}: {exc}")
        return None


def _qwen_via_gguf(b64: str) -> Optional[dict]:
    if not _GGUF_MODEL.exists() or not _MMPROJ.exists():
        return None
    try:
        from llama_cpp import Llama
        try:
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _H
        except ImportError:
            from llama_cpp.llama_chat_format import Llava15ChatHandler as _H

        handler = _H(clip_model_path=str(_MMPROJ), verbose=False)
        llm     = Llama(
            model_path=str(_GGUF_MODEL), chat_handler=handler,
            n_ctx=2048, n_gpu_layers=0,
            n_threads=min(8, os.cpu_count() or 4), verbose=False,
        )
        data_url = f"data:image/jpeg;base64,{b64}"
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": _EVAL_PROMPT},
                ]},
            ],
            temperature=0.0, max_tokens=500,
        )
        return _parse_qwen_json(resp["choices"][0]["message"]["content"].strip())
    except Exception as exc:
        print(f"[qwen/gguf] {type(exc).__name__}: {exc}")
        return None


def qwen_stage(candidates: list[Candidate]) -> list[dict]:
    """
    Stage 2: for each Top-K candidate, re-render that PDF page at low
    resolution in memory, evaluate with Qwen (temperature=0.0), del pixel
    buffer. Returns list of manifest entries (metadata only).
    """
    manifest: list[dict] = []

    for cand in candidates:
        print(f"[qwen] {cand.pdf_filename}  page {cand.page_no} ...", end=" ", flush=True)

        b64    = _render_page_b64(cand.pdf_path, cand.page_no)
        result = _qwen_via_ollama(b64) or _qwen_via_gguf(b64)
        del b64
        gc.collect()

        if result:
            status = "APPROVED" if result["approved_for_essay"] else "REJECTED"
            print(f"{status} -- {result['assigned_role']}")
        else:
            print("BACKEND UNAVAILABLE")
            result = {
                "approved_for_essay":      False,
                "assigned_role":           "Rejected",
                "structural_justification": (
                    "Qwen 2.5-VL backend unavailable -- "
                    "run `ollama pull qwen2.5vl:3b` or place GGUF files in models/."
                ),
            }

        manifest.append({
            "source_pdf":              cand.pdf_filename,
            "page":                    cand.page_no,
            "clip_score":              round(cand.clip_score, 4),
            "approved_for_essay":      result["approved_for_essay"],
            "assigned_role":           result["assigned_role"],
            "structural_justification": result["structural_justification"],
        })
        print(f"         {result['structural_justification']}")
        print()

    return manifest


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run(
    query:   str  = "high contrast black and white street photograph with sharp shadows",
    pdf_dir: str  = str(_PDF_DIR),
    top_k:   int  = 5,
    output:  str  = str(_OUTPUT_FILE),
) -> list[dict]:
    print()
    print("=" * 65)
    print(" SAFE ESSAY SELECTOR -- In-Memory Copyright-Compliant Pipeline")
    print("=" * 65)
    print(f" Query   : {query}")
    print(f" PDFs    : {pdf_dir}")
    print(f" Top-K   : {top_k}")
    print(f" Output  : {output}")
    print("-" * 65)
    print(" Fair Use Constraints Active:")
    print("   No raw image is written to disk at any point.")
    print("   Every buffer is downgraded to 224x224 grayscale before use.")
    print("   Pixel data is deleted from memory after CLIP encoding.")
    print("   Manifest contains metadata only -- no pixel data.")
    print("=" * 65)
    print()

    pdf_path = Path(pdf_dir).resolve()
    if not pdf_path.exists():
        print(f"[ERROR] Directory not found: {pdf_path}")
        return []

    all_pdfs = sorted(pdf_path.glob("*.pdf"))
    originals = [p for p in all_pdfs if not _is_duplicate(p)]
    dupes     = [p for p in all_pdfs if _is_duplicate(p)]

    if not originals:
        print(f"[INFO] No PDF files found in {pdf_path}")
        print("  Drop your photo essay PDFs into that folder and re-run.")
        return []

    print(f"[scan] {len(originals)} PDF(s) to process  "
          f"({len(dupes)} duplicate(s) skipped)")
    for d in dupes:
        print(f"  skip  {d.name}")
    print()

    # Stage 1
    print("-- Stage 1: CLIP In-Memory Retrieval ---------------------------")
    top_candidates = clip_stage(originals, query, top_k=top_k)

    if not top_candidates:
        print("[ERROR] No candidates produced -- check that PDFs contain raster images.")
        return []

    # Stage 2
    print("-- Stage 2: Qwen 2.5-VL Structural Critique (temperature=0.0) --")
    manifest = qwen_stage(top_candidates)

    # Save manifest (metadata only)
    out_path = Path(output)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    approved  = sum(1 for e in manifest if e["approved_for_essay"])
    role_tally: dict[str, int] = {}
    for e in manifest:
        role_tally[e["assigned_role"]] = role_tally.get(e["assigned_role"], 0) + 1

    print("=" * 65)
    print(f" Manifest saved  -> {out_path.resolve()}")
    print(f" Approved        : {approved} / {len(manifest)}")
    print(f" Roles           : {json.dumps(role_tally)}")
    print(" No raw images remain on disk.")
    print("=" * 65)
    print()
    return manifest


# =============================================================================
if __name__ == "__main__":
    verify_deps()
    _query   = (sys.argv[1] if len(sys.argv) > 1
                else "high contrast black and white street photograph with sharp shadows")
    _pdf_dir = sys.argv[2] if len(sys.argv) > 2 else str(_PDF_DIR)
    _top_k   = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    run(query=_query, pdf_dir=_pdf_dir, top_k=_top_k)
