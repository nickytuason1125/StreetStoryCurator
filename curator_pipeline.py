#!/usr/bin/env python3
"""
curator_pipeline.py -- Two-stage visual evaluation pipeline.

Phase 1  CLIP literal retrieval (openai/clip-vit-base-patch32)
         -> cosine similarity against text prompt -> Top 5 image paths

Phase 2  Qwen 2.5-VL curatorial evaluation
         -> editorial pacing / decisive moment / humanist framing
         -> strict JSON per image, saved to essay_selection_report.json

Usage:
    python curator_pipeline.py
    python curator_pipeline.py "your prompt" ./path/to/images
    python curator_pipeline.py "your prompt" ./path/to/images 10
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---- config -----------------------------------------------------------------
_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
_OLLAMA_URL  = "http://localhost:11434/api/generate"
_ROOT        = Path(__file__).resolve().parent
_GGUF_MODEL  = _ROOT / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"
_MMPROJ      = _ROOT / "models" / "mmproj-qwen2.5-vl-2b-instruct-f16.gguf"
_OUTPUT_FILE = "essay_selection_report.json"

_SYSTEM_PROMPT = (
    "You are a master photography editor who has curated landmark documentary photo essays "
    "for leading documentary publications and photo agencies. "
    "Evaluate each photograph against three curatorial criteria:\n\n"
    "1. EDITORIAL PACING -- Does this image function as an establishing shot (wide, "
    "contextual, scene-setting) or a detail shot (tight, intimate, specific)? "
    "A coherent essay requires both types in deliberate sequence.\n\n"
    "2. DECISIVE MOMENT -- Is there geometric alignment between the composition and a "
    "spontaneous, unrepeatable human action or emotional beat? Or is it merely a "
    "competent record of something that was present?\n\n"
    "3. HUMANIST FRAMING -- Does the framing convey dignity and emotional proximity to the "
    "subject? Or does it treat the subject as spectacle or document?\n\n"
    "Respond ONLY with a single valid JSON object. No markdown fences, no explanation, "
    "no text before or after the JSON:\n"
    '{"selected": <true|false>, "role": "<string>", "justification": "<one sentence>"}\n\n'
    "Role must be one of: establishing shot, detail shot, emotional anchor, interaction, "
    "transitional, portrait, or contextual."
)

_EVAL_PROMPT = (
    "Evaluate this photograph against your three curatorial criteria. "
    "Respond with a single JSON object only -- no other text."
)


# =============================================================================
# Phase 1 -- CLIP retrieval
# =============================================================================

def clip_retrieve(prompt: str, image_dir: str, top_k: int = 5) -> list[str]:
    """Embed all images in image_dir and return top_k paths by cosine similarity."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    image_dir_path = Path(image_dir).resolve()
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")

    image_paths = sorted([
        p for p in image_dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ])
    if not image_paths:
        raise ValueError(
            f"No images found in {image_dir_path}.\n"
            "Place JPEG/PNG images in that directory and re-run."
        )

    print("[clip] Loading openai/clip-vit-base-patch32 ...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print(f"[clip] Embedding {len(image_paths)} image(s) on {device} ...")
    feat_list   : list = []
    valid_paths : list[Path] = []

    with torch.no_grad():
        for p in image_paths:
            try:
                img   = Image.open(p).convert("RGB")
                inp   = processor(images=img, return_tensors="pt").to(device)
                feats = model.get_image_features(**inp)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feat_list.append(feats)
                valid_paths.append(p)
            except Exception as e:
                print(f"[clip]   skipped {p.name}: {e}")

    if not feat_list:
        raise ValueError("No images could be embedded.")

    text_inp   = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    text_feats = model.get_text_features(**text_inp)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    all_feats = torch.cat(feat_list, dim=0)           # (N, 512)
    sims      = (all_feats @ text_feats.T).squeeze(1) # (N,)
    scores    = sims.cpu().tolist()

    ranked = sorted(zip(scores, valid_paths), key=lambda x: x[0], reverse=True)
    top    = ranked[: min(top_k, len(ranked))]

    print(f"\n[clip] Top {len(top)} matches for: \"{prompt}\"")
    for score, path in top:
        print(f"   {score:+.4f}  {path.name}")

    return [str(p.resolve()) for _, p in top]


# =============================================================================
# Phase 2 -- Qwen 2.5-VL evaluation
# =============================================================================

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _parse_json(raw: str) -> Optional[dict]:
    """Strip <think> blocks then extract and validate the JSON object."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not all(k in obj for k in ("selected", "role", "justification")):
        return None
    return {
        "selected":      bool(obj["selected"]),
        "role":          str(obj.get("role", "")).strip(),
        "justification": str(obj.get("justification", "")).strip(),
    }


def _via_ollama(image_path: str) -> Optional[dict]:
    """POST to local Ollama qwen2.5vl:3b with the image encoded as base64."""
    try:
        import urllib.request
        payload = json.dumps({
            "model":   "qwen2.5vl:3b",
            "system":  _SYSTEM_PROMPT,
            "prompt":  _EVAL_PROMPT,
            "images":  [_b64(image_path)],
            "stream":  False,
            "options": {"temperature": 0.0, "num_predict": 450},
        }).encode()
        req = urllib.request.Request(
            _OLLAMA_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "").strip()
        return _parse_json(raw)
    except Exception as e:
        print(f"[qwen/ollama] {type(e).__name__}: {e}")
        return None


def _via_gguf(image_path: str) -> Optional[dict]:
    """Run GGUF Qwen2.5-VL locally via llama_cpp (CPU-only)."""
    if not _GGUF_MODEL.exists() or not _MMPROJ.exists():
        return None
    try:
        from llama_cpp import Llama
        try:
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _Handler
        except ImportError:
            from llama_cpp.llama_chat_format import Llava15ChatHandler as _Handler

        import io
        from PIL import Image

        chat_handler = _Handler(clip_model_path=str(_MMPROJ), verbose=False)
        llm = Llama(
            model_path=str(_GGUF_MODEL),
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=0,
            n_threads=min(8, os.cpu_count() or 4),
            verbose=False,
        )
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        image_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text",      "text": _EVAL_PROMPT},
                ]},
            ],
            temperature=0.0,
            max_tokens=450,
        )
        raw = resp["choices"][0]["message"]["content"].strip()
        return _parse_json(raw)
    except Exception as e:
        print(f"[qwen/gguf] {type(e).__name__}: {e}")
        return None


def qwen_evaluate(image_path: str) -> dict:
    """Evaluate a single image. Priority: Ollama -> GGUF -> error stub."""
    name = Path(image_path).name
    print(f"[qwen] {name} ...", end=" ", flush=True)
    result = _via_ollama(image_path) or _via_gguf(image_path)
    if result:
        print("ok")
        return result
    print("FAILED")
    return {
        "selected":      False,
        "role":          "unavailable",
        "justification": (
            "Qwen 2.5-VL could not be reached -- "
            "run `ollama pull qwen2.5vl:3b` or place the GGUF files in models/."
        ),
    }


# =============================================================================
# Pipeline orchestration
# =============================================================================

def run_pipeline(
    prompt:    str = "black and white portrait of a person with high contrast lighting.",
    image_dir: str = "./dataset_images",
    top_k:     int = 5,
    output:    str = _OUTPUT_FILE,
) -> list[dict]:
    print()
    print("=" * 62)
    print(" CURATOR PIPELINE")
    print("=" * 62)
    print(f" Prompt    : {prompt}")
    print(f" Image dir : {image_dir}")
    print(f" Top-K     : {top_k}")
    print(f" Output    : {output}")
    print("=" * 62)
    print()

    # Phase 1 -- CLIP retrieval
    print("-- Phase 1: CLIP Literal Retrieval ----------------------")
    top_paths = clip_retrieve(prompt, image_dir, top_k=top_k)
    print()

    # Phase 2 -- Qwen evaluation
    print("-- Phase 2: Qwen 2.5-VL Curatorial Evaluation ----------")
    report: list[dict] = []
    for path in top_paths:
        evaluation = qwen_evaluate(path)
        entry = {
            "path":          path,
            "filename":      Path(path).name,
            "selected":      evaluation["selected"],
            "role":          evaluation["role"],
            "justification": evaluation["justification"],
        }
        report.append(entry)
        icon = "[YES]" if entry["selected"] else "[ NO]"
        print(f"  {icon}  [{entry['role']:<20}]  {Path(path).name}")
        print(f"         {entry['justification']}")
        print()

    # Save report
    out_path = Path(output)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    selected = sum(1 for e in report if e["selected"])
    print("=" * 62)
    print(f" Report saved  -> {out_path.resolve()}")
    print(f" Essay picks   : {selected} / {len(report)}")
    print("=" * 62)
    print()
    return report


# =============================================================================
if __name__ == "__main__":
    _prompt    = (sys.argv[1] if len(sys.argv) > 1
                  else "black and white portrait of a person with high contrast lighting.")
    _image_dir = sys.argv[2] if len(sys.argv) > 2 else "./dataset_images"
    _top_k     = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    run_pipeline(prompt=_prompt, image_dir=_image_dir, top_k=_top_k)
