#!/usr/bin/env python3
"""
essay_selector.py -- Deterministic two-stage photo essay selection pipeline.

Stage 1  CLIP (openai/clip-vit-base-patch32)
         Scans ./dataset_images, computes cosine similarity against a literal
         narrative query, returns the Top 5 absolute paths.

Stage 2  Qwen 2.5-VL (temperature=0.0 -- fully deterministic)
         Each image is evaluated against hardcoded curatorial theory drawn
         from mid-century photojournalism and contemporary
         art history (Stephen Bull, LUMA Arles).

Output   final_essay_manifest.json -- JSON array, one entry per image.

Usage:
    python essay_selector.py
    python essay_selector.py "urban documentary sequence at dusk" ./dataset_images
    python essay_selector.py "your query" ./images 8
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# 1. DEPENDENCY VERIFICATION
# =============================================================================

def _verify_deps() -> None:
    """Abort early with a clear install command if any required package is missing."""
    required = {
        "torch":        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
        "transformers": "pip install transformers",
        "PIL":          "pip install Pillow",
    }
    missing = []
    for module, install_cmd in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append((module, install_cmd))

    if missing:
        print("[deps] MISSING DEPENDENCIES:")
        for mod, cmd in missing:
            print(f"  {mod:15}  ->  {cmd}")
        sys.exit(1)

    # Qwen backend -- Ollama or llama_cpp (either is sufficient)
    ollama_ok = False
    gguf_ok   = False

    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = True
    except Exception:
        pass

    try:
        importlib.import_module("llama_cpp")
        gguf_ok = True
    except ImportError:
        pass

    if not ollama_ok and not gguf_ok:
        print("[deps] WARNING: No Qwen backend found.")
        print("  Option A: Install Ollama and run `ollama pull qwen2.5vl:3b`")
        print("  Option B: Place qwen2.5-vl-2b-instruct-q4_k_m.gguf in models/")
        print("  Stage 2 will produce error stubs -- Stage 1 (CLIP) will still run.")

    print("[deps] OK  torch / transformers / Pillow verified")
    if ollama_ok:
        print("[deps] OK  Qwen backend: Ollama")
    elif gguf_ok:
        print("[deps] OK  Qwen backend: llama_cpp GGUF")
    print()


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
_OLLAMA_URL  = "http://localhost:11434/api/generate"
_ROOT        = Path(__file__).resolve().parent
_GGUF_MODEL  = _ROOT / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"
_MMPROJ      = _ROOT / "models" / "mmproj-qwen2.5-vl-2b-instruct-f16.gguf"
_OUTPUT_FILE = "final_essay_manifest.json"

# Curatorial theory injected verbatim into Qwen's system layer.
# Synthesised from LIFE editorial practice, Stephen Bull
# "Photography" (Routledge, 2009), and LUMA Arles programming doctrine.
_SYSTEM_PROMPT = """\
You are a curatorial AI trained on mid-century photojournalism theory (LIFE Magazine, \
documentary photo agencies) and contemporary art history (Stephen Bull, LUMA Arles). \
You evaluate individual photographs for inclusion in a documentary photo essay.

=== NARRATIVE PACING TAXONOMY ===
Every approved photograph must fulfill exactly one structural role:

  Establishing Shot   Wide frame, low subject-to-frame ratio, deep focus. \
Grounds the viewer geographically and contextually. \
Appears early in a sequence to orient the reader.

  Interaction Shot    Medium framing (waist-up or wider). Captures relationships, \
labor, urban friction, or social exchange between two or more subjects. \
The decisive moment must show reciprocal awareness or action.

  Detail Shot         High magnification, shallow depth of field, one dominant \
graphic element (texture, hand, signage, shadow). \
Functions as a visual pause, metaphor, or punctuation within the sequence.

  Closer              Compositional finality: vanishing points, subjects moving \
away, backs to camera, or receding lines. \
Provides narrative resolution and editorial closure.

=== THE DOCUMENT CHECKLIST (Rejection Criteria) ===
REJECT the image (assigned_role = "Rejected", approved_for_essay = false) if it \
lacks ANY of the following:

  1. INTENTIONAL GEOMETRY
     The composition must contain at least one of:
     - Leading lines drawing the eye to a subject or vanishing point.
     - Framing shapes (archways, doorways, shadows as frames).
     - Adherence to rule of thirds with a clearly dominant subject.
     Random or accidental composition is grounds for immediate rejection.

  2. THE DECISIVE MOMENT
     Moving human subjects must lock into geometric alignment with background \
elements, shadows, or architectural lines at the precise frame captured. \
A static subject with no geometric tension is insufficient.

  3. HUMANIST DIGNITY
     The camera must be at or near eye level with the subject. \
High-angle shots that reduce subjects to objects, or voyeuristic \
framings that strip agency, must be rejected on ethical grounds.

=== OUTPUT FORMAT ===
You MUST respond with ONLY a single valid JSON object. \
No markdown, no explanation, no text outside the JSON:
{
  "approved_for_essay": <true or false>,
  "assigned_role": "<Establishing Shot | Interaction Shot | Detail Shot | Closer | Rejected>",
  "structural_justification": "<Exactly one sentence that explicitly cites which \
geometry rule, pacing taxonomy, or humanist criterion drove this decision.>"
}
"""

_EVAL_PROMPT = (
    "Evaluate this photograph. Apply your Document Checklist, assign its Narrative "
    "Pacing role, and respond with a single JSON object only -- no other text."
)


# =============================================================================
# 3. STAGE 1 -- CLIP LITERAL RETRIEVAL
# =============================================================================

def clip_retrieve(query: str, image_dir: str, top_k: int = 5) -> list[str]:
    """
    Embed all images in image_dir and return the top_k absolute paths
    ranked by cosine similarity against the text query.
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    image_dir_path = Path(image_dir).resolve()
    if not image_dir_path.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir_path}\n"
            "Create the directory and populate it with JPEG/PNG images."
        )

    paths = sorted([
        p for p in image_dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ])
    if not paths:
        raise ValueError(
            f"No images found in {image_dir_path}.\n"
            "Add JPEG or PNG files and re-run."
        )

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] Loading openai/clip-vit-base-patch32 on {device} ...")
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print(f"[clip] Embedding {len(paths)} image(s) ...")
    feat_list   : list = []
    valid_paths : list[Path] = []

    with torch.no_grad():
        for p in paths:
            try:
                img  = Image.open(p).convert("RGB")
                inp  = processor(images=img, return_tensors="pt").to(device)
                feat = model.get_image_features(**inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat_list.append(feat)
                valid_paths.append(p)
            except Exception as exc:
                print(f"[clip]   skipped {p.name}: {exc}")

    if not feat_list:
        raise ValueError("No images could be embedded -- check file integrity.")

    import torch as _torch
    text_inp  = processor(text=[query], return_tensors="pt", padding=True).to(device)
    text_feat = model.get_text_features(**text_inp)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    all_feat = _torch.cat(feat_list, dim=0)
    sims     = (all_feat @ text_feat.T).squeeze(1).cpu().tolist()
    ranked   = sorted(zip(sims, valid_paths), key=lambda x: x[0], reverse=True)
    top      = ranked[: min(top_k, len(ranked))]

    print(f"\n[clip] Top {len(top)} matches for: \"{query}\"")
    for score, p in top:
        print(f"   {score:+.4f}  {p.name}")
    print()

    return [str(p.resolve()) for _, p in top]


# =============================================================================
# 4. STAGE 2 -- QWEN 2.5-VL DETERMINISTIC EVALUATION
# =============================================================================

def _b64_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _extract_json(raw: str) -> Optional[dict]:
    """
    Strip <think> blocks, locate the first {...} in the response,
    parse it, and validate the required schema fields.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None

    required = {"approved_for_essay", "assigned_role", "structural_justification"}
    if not required.issubset(obj.keys()):
        return None

    valid_roles = {
        "Establishing Shot", "Interaction Shot",
        "Detail Shot", "Closer", "Rejected",
    }
    role = str(obj.get("assigned_role", "")).strip()
    if role not in valid_roles:
        role = "Rejected"

    return {
        "approved_for_essay":      bool(obj["approved_for_essay"]),
        "assigned_role":           role,
        "structural_justification": str(obj.get("structural_justification", "")).strip(),
    }


def _qwen_ollama(image_path: str) -> Optional[dict]:
    """Evaluate via local Ollama qwen2.5vl:3b (no subprocess, no CUDA flash)."""
    try:
        import urllib.request
        payload = json.dumps({
            "model":   "qwen2.5vl:3b",
            "system":  _SYSTEM_PROMPT,
            "prompt":  _EVAL_PROMPT,
            "images":  [_b64_image(image_path)],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 500},
        }).encode()
        req = urllib.request.Request(
            _OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "").strip()
        return _extract_json(raw)
    except Exception as exc:
        print(f"[qwen/ollama] {type(exc).__name__}: {exc}")
        return None


def _qwen_gguf(image_path: str) -> Optional[dict]:
    """Evaluate via local GGUF (CPU-only -- avoids CUDA subprocess flash)."""
    if not _GGUF_MODEL.exists() or not _MMPROJ.exists():
        return None
    try:
        from llama_cpp import Llama
        try:
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _Handler
        except ImportError:
            from llama_cpp.llama_chat_format import Llava15ChatHandler as _Handler

        from PIL import Image as _PIL
        chat_handler = _Handler(clip_model_path=str(_MMPROJ), verbose=False)
        llm = Llama(
            model_path=str(_GGUF_MODEL),
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=0,
            n_threads=min(8, os.cpu_count() or 4),
            verbose=False,
        )

        img = _PIL.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        data_url = (
            "data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode()
        )
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": _EVAL_PROMPT},
                ]},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        return _extract_json(raw)
    except Exception as exc:
        print(f"[qwen/gguf] {type(exc).__name__}: {exc}")
        return None


def evaluate_image(image_path: str) -> dict:
    """
    Run Stage 2 on a single image.
    Priority: Ollama -> GGUF -> error stub.
    Returns a dict with all four manifest fields.
    """
    name = Path(image_path).name
    print(f"[qwen] {name} ...", end=" ", flush=True)

    result = _qwen_ollama(image_path) or _qwen_gguf(image_path)

    if result:
        icon = "APPROVED" if result["approved_for_essay"] else "REJECTED"
        print(f"{icon} -- {result['assigned_role']}")
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

    return {
        "image_path":              image_path,
        "approved_for_essay":      result["approved_for_essay"],
        "assigned_role":           result["assigned_role"],
        "structural_justification": result["structural_justification"],
    }


# =============================================================================
# 5. PIPELINE ORCHESTRATION
# =============================================================================

def run_essay_selector(
    query:     str = "high contrast black and white street photograph with sharp shadows",
    image_dir: str = "./dataset_images",
    top_k:     int = 5,
    output:    str = _OUTPUT_FILE,
) -> list[dict]:
    print()
    print("=" * 65)
    print(" ESSAY SELECTOR -- Deterministic Two-Stage Pipeline")
    print("=" * 65)
    print(f" Query     : {query}")
    print(f" Directory : {image_dir}")
    print(f" Top-K     : {top_k}")
    print(f" Output    : {output}")
    print("=" * 65)
    print()

    # --- Stage 1 ---
    print(">> Stage 1: CLIP Literal Retrieval")
    print("-" * 45)
    top_paths = clip_retrieve(query, image_dir, top_k=top_k)

    # --- Stage 2 ---
    print(">> Stage 2: Qwen 2.5-VL Structural Critique  (temperature=0.0)")
    print("-" * 45)
    manifest: list[dict] = []
    for path in top_paths:
        entry = evaluate_image(path)
        manifest.append(entry)
        print(f"   Role      : {entry['assigned_role']}")
        print(f"   Rationale : {entry['structural_justification']}")
        print()

    # --- Save ---
    out_path = Path(output)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    approved = sum(1 for e in manifest if e["approved_for_essay"])
    roles    = {}
    for e in manifest:
        roles[e["assigned_role"]] = roles.get(e["assigned_role"], 0) + 1

    print("=" * 65)
    print(f" Manifest saved  -> {out_path.resolve()}")
    print(f" Approved        : {approved} / {len(manifest)}")
    print(f" Role breakdown  : {json.dumps(roles)}")
    print("=" * 65)
    print()
    return manifest


# =============================================================================
# 6. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    _verify_deps()

    _query     = (
        sys.argv[1] if len(sys.argv) > 1
        else "high contrast black and white street photograph with sharp shadows"
    )
    _image_dir = sys.argv[2] if len(sys.argv) > 2 else "./dataset_images"
    _top_k     = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    run_essay_selector(query=_query, image_dir=_image_dir, top_k=_top_k)
