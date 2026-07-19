"""
Critique Engine — Qwen2.5-VL-2B Visual Judge

Two public functions called by the server subprocess and the annotation queue:
  run_jury_critique(image_hash)    → {"critique", "think", "bbox_factors"}
  run_audit_annotation(image_hash) → {"score_factors", "think"}

Priority order:
  1. Qwen2.5-VL-2B GGUF (multimodal, real bounding boxes, 4 GB VRAM)
  2. Ollama qwen2.5vl:3b  (already installed, multimodal via API)
  3. Ollama deepseek-r1:8b (text-only, last resort)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

_ROOT       = Path(__file__).resolve().parent.parent
_MODEL_GGUF = _ROOT / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"
_MMPROJ     = _ROOT / "models" / "mmproj-qwen2.5-vl-2b-instruct-f16.gguf"

_llm: object = None   # cached Llama instance

# Grammar-constrained decoding for the contact-sheet swap verdict — same
# pattern as vlm_niche_detector.py's _GRAMMAR_SRC (grammar built once,
# passed per-call via grammar=). Only applies to the local GGUF path —
# Ollama's HTTP API has no GBNF grammar support, so that fallback still
# relies on _parse_swap_json's loose parsing.
_SWAP_GRAMMAR_SRC = r'''
root   ::= "{" ws '"action"' ws ":" ws action ws "," ws '"swap_slot"' ws ":" ws slot ws "," ws '"cited_aspect"' ws ":" ws aspect ws "," ws '"cited_value"' ws ":" ws value ws "," ws '"reason"' ws ":" ws string ws "}"
action ::= "\"accept\"" | "\"swap\""
slot   ::= "null" | [0-9] [0-9]?
aspect ::= "\"Composition\"" | "\"Lighting\"" | "\"Narrative\"" | "\"Human/Culture\"" | "\"Technical\"" | "\"none\""
value  ::= "null" | float
float  ::= "-"? [0-9]+ "." [0-9]+
string ::= "\"" ([^"\\] | "\\" .)* "\""
ws     ::= [ \t\r\n]*
'''
_swap_grammar: object = None   # cached LlamaGrammar instance

# Circuit breaker: some llama-cpp-python + GGUF combinations hit a native
# access-violation inside the grammar-constrained sampler (observed
# elsewhere in this codebase with a different GGUF on this build). It's
# caught as a Python exception, not a hard crash, but repeatedly retrying a
# known-broken path per call is wasteful — after the first failure, fall
# back to unconstrained decoding (+ the existing loose JSON parsing, which
# worked reliably before grammar was added) for the rest of the process.
_swap_grammar_broken = False


def _load_swap_grammar():
    global _swap_grammar
    if _swap_grammar is not None:
        return _swap_grammar
    try:
        from llama_cpp import LlamaGrammar
        _swap_grammar = LlamaGrammar.from_string(_SWAP_GRAMMAR_SRC)
    except Exception as e:
        print(f"[ce] swap grammar build failed ({e}) — falling back to unconstrained decoding")
        _swap_grammar = None
    return _swap_grammar


def _mark_swap_grammar_broken() -> None:
    global _swap_grammar_broken
    _swap_grammar_broken = True

# ── Ollama availability cache ─────────────────────────────────────────────────
_ollama_last_check: float = 0.0
_ollama_ok: bool = False
_OLLAMA_RECHECK_SECS = 60.0


def _check_ollama_available() -> bool:
    """Ping Ollama /api/version once per 60 s; returns False immediately if down."""
    global _ollama_last_check, _ollama_ok
    now = time.monotonic()
    if now - _ollama_last_check < _OLLAMA_RECHECK_SECS:
        return _ollama_ok
    try:
        import requests as _req
        _ollama_ok = _req.get("http://localhost:11434/api/version", timeout=2).ok
    except Exception:
        _ollama_ok = False
    _ollama_last_check = now
    return _ollama_ok


def get_ollama_ps() -> list[dict]:
    """
    Call /api/ps to get currently loaded models with their memory usage.
    Returns list of dicts: {name, size_vram, size_total, processor, until}.
    Empty list if Ollama is down or no models are loaded.
    """
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/ps", timeout=3)
        if not r.ok:
            return []
        models = r.json().get("models", [])
        out = []
        for m in models:
            out.append({
                "name":       m.get("name", ""),
                "size_vram":  m.get("size_vram", 0),
                "size_total": m.get("size", 0),
                "processor":  m.get("details", {}).get("quantization_level", ""),
                "until":      m.get("expires_at", ""),
            })
        return out
    except Exception:
        return []


# ── VRAM-safe downscaler ──────────────────────────────────────────────────────

def secure_image_for_vram(image_path: str, max_dimension: int = 1024) -> str:
    """
    Open image, downscale if either dimension exceeds max_dimension (aspect
    ratio preserved), encode as Base64 JPEG and return the string.
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > max_dimension or h > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model():
    global _llm
    if _llm is not None:
        return _llm
    if not _MODEL_GGUF.exists() or not _MMPROJ.exists():
        return None
    try:
        from llama_cpp import Llama
        # Prefer the dedicated Qwen2-VL handler; fall back to LLaVA15 for older builds
        try:
            from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _Handler
            print("[ce] Using Qwen2VLChatHandler")
        except ImportError:
            from llama_cpp.llama_chat_format import Llava15ChatHandler as _Handler
            print("[ce] Qwen2VLChatHandler not found — falling back to Llava15ChatHandler")

        chat_handler = _Handler(clip_model_path=str(_MMPROJ))
        _n_threads   = min(os.cpu_count() or 4, 8)
        print(f"[ce] Loading Qwen2.5-VL-2B  threads={_n_threads}  ctx=2048")
        _llm = Llama(
            model_path=str(_MODEL_GGUF),
            chat_handler=chat_handler,
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=_n_threads,
            verbose=False,
        )
        print("[ce] Qwen2.5-VL-2B ready.")
        return _llm
    except Exception as _e:
        print(f"[ce] GGUF load failed: {_e}")
        return None


def unload() -> None:
    """
    Release the Qwen2.5-VL-2B GGUF singleton. This module normally stays
    warm across independent per-image annotation requests (run_jury_critique/
    run_audit_annotation), which is correct for that use case — but Story
    Mode's batched contact-sheet revision loop (src/contact_sheet.py) needs
    to tear it down deterministically before the next GPU-relevant phase,
    so it doesn't stay resident longer than the loop that needed it.
    """
    global _llm
    _llm = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ── Coordinate normalisation ──────────────────────────────────────────────────

def _parse_bbox(text: str, img_w: int, img_h: int) -> Optional[dict]:
    """
    Extract the first bounding box from model output and return normalised
    {"x","y","w","h"} (all in [0,1]).  Handles:
      <box>(x1,y1,x2,y2)</box>
      <|box_start|>(x1,y1),(x2,y2)<|box_end|>
      [x1, y1, x2, y2]
    Qwen2.5-VL emits coords in a 0-1000 virtual space; actual pixel values
    are detected when they exceed 1000.
    """
    patterns = [
        r"<box>\s*\(?\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\)?",
        r"<\|box_start\|>\s*\((\d+),(\d+)\),\s*\((\d+),(\d+)\)",
        r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            # Values ≤ 1000 → 0-1000 virtual space; larger → raw pixels
            ref_x = 1000 if x2 <= 1000 else img_w
            ref_y = 1000 if y2 <= 1000 else img_h
            nx1 = max(0.0, min(1.0, x1 / ref_x))
            ny1 = max(0.0, min(1.0, y1 / ref_y))
            nx2 = max(0.0, min(1.0, x2 / ref_x))
            ny2 = max(0.0, min(1.0, y2 / ref_y))
            if nx2 > nx1 and ny2 > ny1:
                return {"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1}
    return None


def _region_from_bbox(bbox: Optional[dict]) -> str:
    """Map a normalised bbox centre to a named region string (frontend compat)."""
    if not bbox:
        return "center"
    cx = bbox["x"] + bbox["w"] / 2
    cy = bbox["y"] + bbox["h"] / 2
    row = "top" if cy < 0.33 else ("center" if cy < 0.67 else "bottom")
    col = "left" if cx < 0.4 else ("right" if cx > 0.6 else None)
    if row == "center" and col is None:
        return "center"
    if row == "center":
        return f"{col}-half"
    if col is None:
        return f"{row}-third"
    return f"{row}-{col}"


# ── Ollama helper ─────────────────────────────────────────────────────────────

# Ollama generation is slow on a 6 GB GPU — a vision model can take tens of
# seconds, and the first (cold) call must also load ~3 GB of weights into VRAM.
# A flat 5 s timeout aborted almost every real call. Use a short *connect*
# timeout so we still fail fast when Ollama is down, but a generous *read*
# timeout so legitimate slow generation completes. keep_alive keeps the model
# resident between annotations, and one retry covers the cold-start case where
# the first attempt is consumed by the model load.
_OLLAMA_CONNECT_TIMEOUT = 5      # s — is Ollama reachable?
_OLLAMA_READ_TIMEOUT    = 90     # s — allow slow VL generation + cold VRAM load
# Keep the ~3 GB VL model resident only briefly: long enough to stay warm while
# you browse photo-to-photo, but short enough that it frees the RAM/VRAM soon
# after you stop (was "5m", which held 3 GB idle for 5 minutes).
_OLLAMA_KEEP_ALIVE      = "30s"


def _ollama(prompt: str, model: str, max_tokens: int = 400) -> Optional[str]:
    import requests
    payload = {
        "model": model, "prompt": prompt, "stream": False,
        "keep_alive": _OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }
    for attempt in (1, 2):
        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=(_OLLAMA_CONNECT_TIMEOUT, _OLLAMA_READ_TIMEOUT),
            )
            if r.ok:
                return r.json().get("response", "").strip() or None
            print(f"[ce] Ollama/{model} HTTP {r.status_code}")
            return None
        except requests.exceptions.ReadTimeout:
            # Cold load likely consumed the budget; the model is resident now, so
            # a second attempt usually returns quickly. Give up after the retry.
            if attempt == 1:
                print(f"[ce] Ollama/{model} read-timeout (cold load?) — retrying once")
                continue
            print(f"[ce] Ollama/{model} timed out after retry ({_OLLAMA_READ_TIMEOUT}s)")
        except Exception as _e:
            print(f"[ce] Ollama/{model} failed: {_e}")
            break
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def run_jury_critique(image_hash: str) -> dict:
    """
    Generate a 3-paragraph visual jury critique.
    Returns {"critique": str, "think": str, "bbox_factors": list}.
    """
    import sys
    sys.path.insert(0, str(_ROOT / "src"))
    import lance_store as _ls

    record = _fetch_record(_ls, image_hash)
    if record is None:
        return {"error": f"Hash '{image_hash}' not found in LanceDB.",
                "critique": "", "think": "", "bbox_factors": []}

    image_path = record["path"]
    score      = float(record.get("score", 0.0))
    breakdown  = _parse_breakdown(record)
    archetype  = _dominant_style(record)
    filename   = Path(image_path).name
    profile    = (str(breakdown.get("semantic_profile", ""))
                  or record.get("reasoning_log", ""))[:80]

    prompt_text = (
        "You are a world-class street photo editor. "
        "Write exactly 3 short paragraphs: 1) Strengths  2) Weaknesses  3) Verdict. "
        "Be accurate: only list strengths and weaknesses that are actually visible "
        "in the image — if a paragraph has nothing real to say, say so in one line. "
        "The Verdict is a clear keep/cut call with the deciding reason. "
        f"Score: {score:.2f}. Archetype: {archetype}. "
        + (f"Profile: {profile}. " if profile else "")
        + "Be specific — reference exactly what you see. Under 180 words total. "
        "Sparingly use: <trigger type=\"blur\">text</trigger> for focus issues, "
        "<trigger type=\"heatmap\">text</trigger> for exposure, "
        "<trigger type=\"grid\">text</trigger> for composition."
    )

    llm = _load_model()

    # ── Qwen2.5-VL multimodal (GGUF) ─────────────────────────────────────────
    if llm is not None and Path(image_path).exists():
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as _im:
                img_w, img_h = _im.size

            b64    = secure_image_for_vram(image_path, max_dimension=1024)
            output = llm.create_chat_completion(  # type: ignore[union-attr]
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ]}],
                temperature=0.1,
                max_tokens=450,
            )
            raw      = (output["choices"][0]["message"]["content"] or "").strip()
            critique, think = _parse_think(raw)
            bboxes   = _extract_free_bboxes(raw, img_w, img_h)
            print(f"[ce] jury (qwen-vl gguf): {filename}  {len(critique)} chars  {len(bboxes)} bboxes")
            return {"critique": critique, "think": think, "bbox_factors": bboxes}
        except Exception as _e:
            print(f"[ce] Qwen-VL critique error ({_e}) — trying Ollama")

    # ── Ollama fallback — qwen2.5vl:3b first, then deepseek ──────────────────
    if _check_ollama_available():
        for model in ("qwen2.5vl:3b", "deepseek-r1:8b"):
            raw = _ollama(prompt_text, model=model, max_tokens=400)
            if raw:
                critique, think = _parse_think(raw)
                print(f"[ce] jury (ollama/{model}): {filename}  {len(critique)} chars")
                return {"critique": critique, "think": think, "bbox_factors": []}

    return {"error": "All critique backends unavailable. Install qwen2.5vl:3b via Ollama.",
            "critique": "", "think": "", "bbox_factors": []}


def run_contact_sheet_critique(
    sheet_path: str,
    slot_summaries: list[dict],
    style_prompt: str,
) -> dict:
    """
    View a rendered Story Mode contact sheet (a labeled grid of the current
    sequence — src/contact_sheet.py) and decide whether it should be
    revised. Returns {"action": "accept"|"swap", "swap_slot": int|None,
    "reason": str}.

    Reuses the same Qwen2.5-VL-2B GGUF singleton as run_jury_critique/
    run_audit_annotation (free if either already warmed it this session).
    The local GGUF path is grammar-constrained (_SWAP_GRAMMAR_SRC,
    LlamaGrammar, following vlm_niche_detector.py's pattern) — the Ollama
    fallback has no GBNF support, so it still relies on _parse_swap_json's
    loose brace-matching parse.

    Never raises — returns action="accept" on any failure so the caller's
    revision loop always terminates safely.
    """
    if not Path(sheet_path).exists():
        return {"action": "accept", "swap_slot": None, "reason": "no contact sheet"}

    prompt_text = (
        "You are a photo editor reviewing a curated street-photo sequence, shown "
        "as a numbered contact sheet (each cell labeled with its role and slot "
        f"number). Style brief: '{style_prompt[:150]}'. "
        f"Per-slot data: {json.dumps(slot_summaries, separators=(',', ':'))[:600]}. "
        "If every slot fits its role and the sequence flows well, respond ACCEPT. "
        "If exactly one slot clearly doesn't belong (wrong mood, weak composition, "
        "breaks pacing), respond SWAP with that slot number (0-indexed) and cite the "
        "specific aspect/value from the per-slot data driving your decision (or "
        "\"none\"/null if purely qualitative). "
        'Output ONLY JSON: {"action":"accept"|"swap","swap_slot":<int or null>,'
        '"cited_aspect":"Composition|Lighting|Narrative|Human/Culture|Technical|none",'
        '"cited_value":<float or null>,"reason":"<one sentence>"}'
    )

    llm = _load_model()
    if llm is not None:
        try:
            b64 = secure_image_for_vram(sheet_path, max_dimension=1280)
            base_kwargs: dict = dict(
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt_text},
                ]}],
                temperature=0.1,
                max_tokens=200,
            )
            output = None
            if not _swap_grammar_broken:
                grammar = _load_swap_grammar()
                if grammar is not None:
                    try:
                        output = llm.create_chat_completion(grammar=grammar, **base_kwargs)  # type: ignore[union-attr]
                    except Exception as _e_g:
                        print(f"[ce] grammar-constrained swap decoding failed ({_e_g}) — "
                              "disabling grammar for this process, retrying unconstrained")
                        _mark_swap_grammar_broken()
            if output is None:
                output = llm.create_chat_completion(**base_kwargs)  # type: ignore[union-attr]
            raw = (output["choices"][0]["message"]["content"] or "").strip()
            parsed = _parse_swap_json(raw)
            if parsed:
                print(f"[ce] contact-sheet critique (qwen-vl gguf): action={parsed['action']}")
                return parsed
        except Exception as _e:
            print(f"[ce] contact-sheet critique error ({_e}) — trying Ollama")

    if _check_ollama_available():
        raw = _ollama(prompt_text, model="qwen2.5vl:3b", max_tokens=200)
        if raw:
            parsed = _parse_swap_json(raw)
            if parsed:
                print(f"[ce] contact-sheet critique (ollama): action={parsed['action']}")
                return parsed

    return {"action": "accept", "swap_slot": None, "reason": "critique backend unavailable"}


def _parse_swap_json(raw: str) -> Optional[dict]:
    """Brace-balanced JSON extraction for the accept/swap verdict, tolerant
    of <think> preambles and markdown fences — same defensive style as
    _parse_factor_json below."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw).strip()
    raw = re.sub(r"\s*```\s*$", "", raw).strip()

    start = raw.find("{")
    if start < 0:
        return None
    depth = 0; end = -1; in_str = False; esc = False
    for ci, ch in enumerate(raw[start:], start):
        if esc:       esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str:    continue
        if   ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = ci + 1; break
    if end < 0:
        return None

    try:
        obj = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None

    action = str(obj.get("action", "accept")).lower()
    if action not in ("accept", "swap"):
        action = "accept"
    swap_slot = obj.get("swap_slot")
    try:
        swap_slot = int(swap_slot) if swap_slot is not None else None
    except (TypeError, ValueError):
        swap_slot = None
    if action == "swap" and swap_slot is None:
        action = "accept"

    _VALID_ASPECTS = {"Composition", "Lighting", "Narrative", "Human/Culture", "Technical"}
    cited_aspect = obj.get("cited_aspect")
    if cited_aspect not in _VALID_ASPECTS:
        cited_aspect = None
    cited_value = obj.get("cited_value")
    try:
        cited_value = float(cited_value) if cited_value is not None else None
    except (TypeError, ValueError):
        cited_value = None

    return {
        "action": action, "swap_slot": swap_slot,
        "cited_aspect": cited_aspect, "cited_value": cited_value,
        "reason": str(obj.get("reason", ""))[:200],
    }


def run_audit_annotation(image_hash: str) -> dict:
    """
    Generate 3 visual factor annotations with bounding boxes.
    Returns {"score_factors": list[dict], "think": str}.
    Each factor: {label, type, region, note, impact, bbox?}
    """
    import sys
    sys.path.insert(0, str(_ROOT / "src"))
    import lance_store as _ls

    record = _fetch_record(_ls, image_hash)
    if record is None:
        return {"score_factors": [], "error": f"Hash '{image_hash}' not found."}

    image_path    = record["path"]
    score         = float(record.get("score", 0.0))
    grade         = record.get("grade", "")
    breakdown     = _parse_breakdown(record)
    filename      = Path(image_path).name
    profile       = (str(breakdown.get("semantic_profile", ""))
                     or record.get("reasoning_log", ""))[:120]
    aspect_lines  = [
        f"{k}: {float(v):.3f}"
        for k, v in breakdown.items()
        if k != "semantic_profile" and isinstance(v, (int, float))
    ][:10]
    breakdown_str = ("  " + "\n  ".join(aspect_lines)) if aspect_lines else "  (none)"

    vl_prompt = (
        "Backend API. Output ONLY a JSON array — no prose, no markdown fences.\n"
        f"IMAGE: {filename} | SCORE: {score:.3f} ({grade})\n"
        f"ASPECTS:\n{breakdown_str}\n"
        f"PROFILE: {profile or 'N/A'}\n\n"
        "Identify EXACTLY 3 quality factors visible in the image. "
        "Draw a bounding box around each relevant area using "
        "<box>(x1,y1,x2,y2)</box> in 0-1000 coordinate space.\n"
        "TYPES: blur=sharpness/focus  heatmap=exposure/lighting  grid=composition\n"
        'FORMAT (return ONLY this):\n'
        '[{"label":"str","type":"blur|heatmap|grid","region":"str",'
        '"note":"str","impact":0.0,"bbox_raw":"<box>(x1,y1,x2,y2)</box>"}]\n'
        "impact: positive=strength, negative=weakness."
    )

    text_prompt = vl_prompt.replace(
        'Draw a bounding box around each relevant area using '
        '<box>(x1,y1,x2,y2)</box> in 0-1000 coordinate space.\n',
        'REGIONS: top-third center bottom-third full left-half right-half '
        'top-left top-right bottom-left bottom-right\n',
    ).replace('"bbox_raw":"<box>(x1,y1,x2,y2)</box>"', '"region":"str"')

    llm = _load_model()

    # ── Qwen2.5-VL multimodal (GGUF) ─────────────────────────────────────────
    if llm is not None and Path(image_path).exists():
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as _im:
                img_w, img_h = _im.size

            b64    = secure_image_for_vram(image_path, max_dimension=896)
            output = llm.create_chat_completion(  # type: ignore[union-attr]
                messages=[{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": vl_prompt},
                ]}],
                temperature=0.05,
                max_tokens=320,
            )
            raw     = (output["choices"][0]["message"]["content"] or "").strip()
            factors = _parse_factor_json(raw, img_w, img_h)
            if factors:
                print(f"[ce] annotation (qwen-vl gguf): {filename}  {len(factors)} factors")
                return {"score_factors": factors, "think": ""}
        except Exception as _e:
            print(f"[ce] Qwen-VL annotation error ({_e}) — trying Ollama")

    # ── Ollama fallback ───────────────────────────────────────────────────────
    if _check_ollama_available():
        for model in ("qwen2.5vl:3b", "deepseek-r1:8b"):
            raw = _ollama(text_prompt, model=model, max_tokens=300)
            if raw:
                factors = _parse_factor_json(raw, 1, 1)
                if factors:
                    print(f"[ce] annotation (ollama/{model}): {filename}  {len(factors)} factors")
                    return {"score_factors": factors, "think": ""}

    return {"score_factors": [], "error": "All annotation backends unavailable."}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_record(ls, image_hash: str) -> Optional[dict]:
    all_rows = ls.query_all(min_score=0.0)
    return next(
        (r for r in all_rows
         if Path(r["path"]).stem == image_hash
         or image_hash in Path(r["path"]).stem),
        None,
    )


def _parse_breakdown(record: dict) -> dict:
    bd = record.get("breakdown") or {}
    if isinstance(bd, str):
        try:
            return json.loads(bd)
        except Exception:
            return {}
    return bd


def _dominant_style(record: dict) -> str:
    bd = _parse_breakdown(record)
    aspects = {k: float(v) for k, v in bd.items() if isinstance(v, (int, float))}
    return max(aspects, key=aspects.get) if aspects else "unknown"  # type: ignore[arg-type]


def _parse_think(raw: str) -> tuple[str, str]:
    think = ""
    m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if m:
        think = m.group(1).strip()
    after = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
    if after:
        text = after.group(1).strip()
    else:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return (text or raw), think


def _extract_free_bboxes(text: str, img_w: int, img_h: int) -> list[dict]:
    """Collect all <box> annotations from free-text critique output."""
    out = []
    for m in re.finditer(
        r"<box>\s*\(?\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\)?", text
    ):
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        ref_x = 1000 if x2 <= 1000 else img_w
        ref_y = 1000 if y2 <= 1000 else img_h
        nx1 = max(0.0, min(1.0, x1 / ref_x))
        ny1 = max(0.0, min(1.0, y1 / ref_y))
        nx2 = max(0.0, min(1.0, x2 / ref_x))
        ny2 = max(0.0, min(1.0, y2 / ref_y))
        if nx2 > nx1 and ny2 > ny1:
            out.append({"x": nx1, "y": ny1, "w": nx2 - nx1, "h": ny2 - ny1})
    return out


def _parse_factor_json(raw: str, img_w: int, img_h: int) -> list[dict]:
    """Extract, validate, and return the JSON factor array from model output."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw).strip()
    raw = re.sub(r"\s*```\s*$",        "", raw).strip()

    start = raw.find("[")
    if start < 0:
        return []

    depth = 0; end = -1; in_str = False; esc = False
    for ci, ch in enumerate(raw[start:], start):
        if esc:       esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str:    continue
        if   ch == "[": depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0: end = ci + 1; break

    if end < 0:
        return []

    try:
        factors = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return []
    if not isinstance(factors, list):
        return []

    _VALID_TYPES   = {"blur", "heatmap", "grid"}
    _VALID_REGIONS = {
        "top-third", "center", "bottom-third", "full",
        "left-half", "right-half",
        "top-left", "top-right", "bottom-left", "bottom-right",
    }
    clean = []
    for f in factors:
        if not isinstance(f, dict):
            continue
        bbox     = _parse_bbox(f.get("bbox_raw", ""), img_w, img_h)
        ftype    = f.get("type",   "heatmap")
        fregion  = f.get("region", _region_from_bbox(bbox))
        entry: dict = {
            "label":  str(f.get("label", "Factor"))[:40],
            "type":   ftype   if ftype   in _VALID_TYPES   else "heatmap",
            "region": fregion if fregion in _VALID_REGIONS else "center",
            "note":   str(f.get("note",  ""))[:80],
            "impact": float(f.get("impact", 0.0)),
        }
        if bbox:
            entry["bbox"] = bbox
        clean.append(entry)

    return clean[:5]
