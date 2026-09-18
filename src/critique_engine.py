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
# Paths come from model_registry: these two literals named a Qwen2.5-VL-2B
# GGUF that does not exist on the Hub (the published build is 3B), so this
# loader could never succeed and every critique fell through to Ollama.
import model_registry as _mr
_MODEL_GGUF = _mr.vision_gguf_path()
_MMPROJ     = _mr.vision_mmproj_path()

_llm: object = None   # cached Llama instance

# Grammar-constrained decoding for the contact-sheet swap verdict — the GBNF
# source string is sent to the sidecar per-call (_SWAP_GRAMMAR_SRC in the
# payload). Only applies to the local GGUF path — Ollama's HTTP API has no
# GBNF grammar support, so that fallback still relies on _parse_swap_json's
# loose parsing.
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
_swap_grammar_broken = False


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


# ── Model loader → sidecar client (2026-09-16) ────────────────────────────────
# The 4 GB Qwen-VL GGUF no longer lives in this process. It runs in a
# disposable sidecar (vlm_sidecar.py) that EXITS when idle — process exit is
# the only complete RAM release on Windows (llama.cpp arenas, mmapped weights
# and the CRT heap survive every in-process unload; the old design measured
# 6 GB working set / 10 GB committed with the model "unloaded"). This module
# keeps ALL the cheap logic: prompt building, parsing, LanceDB context, the
# Ollama fallbacks. It only spawns/adopts/kills the sidecar and proxies infer.
import time as _time_ce
import socket as _socket_ce
import sys as _sys_ce
_LAST_USED = 0.0
_sidecar_port: int = 0

def _touch() -> None:
    global _LAST_USED
    _LAST_USED = _time_ce.monotonic()
    # Keep the residency registry's LRU stamp in sync so the server's idle
    # reaper never evicts a model that is actively being used.
    try:
        import model_residency as _res_t
        _res_t.touch("vision_vl_sidecar")
    except Exception:
        pass

def idle_seconds() -> float:
    """Seconds since the model was last actually used (∞ if never loaded)."""
    return (_time_ce.monotonic() - _LAST_USED) if _LAST_USED else 1e12

def _marker_path() -> Path:
    return _ROOT / "cache" / "vlm_sidecar.json"

def _read_marker() -> "dict | None":
    import sidecar_client as _sc
    return _sc.read_marker()

def _probe_health(port: int, timeout: float = 2.0) -> "dict | None":
    import sidecar_client as _sc
    return _sc.probe_health(port, timeout)

def _kill_sidecar() -> None:
    import sidecar_client as _sc
    _sc.kill()

def _ensure_sidecar() -> bool:
    """Adopt a healthy sidecar, or spawn one and wait for it to come up."""
    import sidecar_client as _sc
    ok = _sc.ensure()
    if ok:
        try:
            import model_residency as _res
            _res.register("vision_vl_sidecar", unload, 3.5)
        except Exception:
            pass
    return ok

def is_loaded() -> bool:
    """True when the sidecar's vision model is actually resident (cheap probe)."""
    m = _read_marker()
    if not m:
        return False
    h = _probe_health(int(m["port"]), timeout=1.5)
    return bool(h and (h.get("loaded") or {}).get("vision"))


def unload() -> None:
    """
    Tear down the vision model. This module normally stays warm across
    independent per-image annotation requests (run_jury_critique /
    run_audit_annotation), which is correct for that use case — but Story
    Mode's batched contact-sheet revision loop (src/contact_sheet.py) and the
    server's residency reaper need a deterministic teardown. The sidecar drops
    its vision slot and exits itself when nothing remains loaded — process
    exit (not in-place unload) is what makes the RAM actually come back.
    """
    import sidecar_client as _sc
    _sc.evict("vision")


def sweep_orphan_sidecar() -> bool:
    """Server-boot hygiene: kill a sidecar whose owning server is gone.

    Returns True when an orphan was terminated. A sidecar whose parent is
    alive (another running FirstCut instance) is left strictly alone.
    """
    import sidecar_client as _sc
    return _sc.sweep_orphan()


def _chat_complete(messages: list, *, temperature: float = 0.1,
                   max_tokens: int = 200,
                   grammar_src: "str | None" = None) -> "str | None":
    """One chat completion against the sidecar. Returns the assistant text or
    None. The shared client's single respawn-retry covers a sidecar that died
    between the health probe and the infer (idle exit racing a late call)."""
    _touch()
    if not _MODEL_GGUF.exists() or not _MMPROJ.exists():
        return None
    payload = {"kind": "vision", "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    gsrc = None if _swap_grammar_broken else grammar_src
    if gsrc:
        payload["grammar_src"] = gsrc
    j = None
    import sidecar_client as _sc
    for attempt in (1, 2):
        j = _sc.infer(payload)
        if j is not None:
            break
        _touch()
    if j is None:
        return None
    if j.get("ok"):
        return (j.get("text") or "").strip()
    if j.get("grammar_failed"):
        # Same semantics as the old in-process fallback: this build's grammar
        # sampler is broken — disable for good; the caller retries unconstrained.
        _mark_swap_grammar_broken()
        return None
    print(f"[ce] sidecar infer error: {j.get('error')}")
    return None


def _vision_complete(b64: str, prompt_text: str, *, max_tokens: int = 200,
                     temperature: float = 0.1) -> "str | None":
    """Vision chat completion, Ollama-first.

    Ollama runs the same Qwen2.5-VL-3B on CUDA: VRAM instead of the ~8 GB RAM
    the CPU-only sidecar llama build needs, and keep_alive self-unloads it.
    The sidecar stays as the fallback for machines without Ollama."""
    if _check_ollama_available():
        raw = _ollama_vision(b64, prompt_text, max_tokens=max_tokens,
                             temperature=temperature)
        if raw:
            return raw
    return _chat_complete(_vl_messages(b64, prompt_text),
                          temperature=temperature, max_tokens=max_tokens)


def _vl_messages(b64: str, prompt_text: str) -> list:
    """The one image+text chat payload every GGUF call site used to build."""
    return [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt_text},
    ]}]


def vision_available() -> bool:
    """True when the vision GGUF and its projector are both on disk.

    Cheap and side-effect free, so callers can gate a feature without paying the
    load. Both files are required — the projector alone cannot see, and the model
    alone cannot take an image.
    """
    return _MODEL_GGUF.exists() and _MMPROJ.exists()


def describe_image(image_path: str,
                   prompt_text: str,
                   *,
                   max_dimension: int = 1024,
                   max_tokens: int = 200,
                   temperature: float = 0.1) -> str:
    """Run the local vision model over one image. "" when it is unavailable.

    Exposed so other modules stop reimplementing this. fast_ingestion used to
    POST the same base64 payload to Ollama's /api/chat with the qwen2.5vl:3b tag
    — a tag no installer pulled — and returned "" on every machine that had not
    manually set Ollama up. Same model family, same prompt, now in-process.

    "" is a supported answer, not a failure: the caller's semantic profile is an
    enrichment, and the pipeline is defined without it.
    """
    if not vision_available() and not _check_ollama_available():
        return ""
    try:
        b64 = secure_image_for_vram(image_path, max_dimension=max_dimension)
        raw = _vision_complete(
            b64, prompt_text, temperature=temperature, max_tokens=max_tokens,
        )
        return (raw or "")
    except Exception as _e:
        print(f"[ce] describe_image failed ({_e})")
        return ""


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
# Preferred vision model: Ollama's Qwen2.5-VL-3B — the SAME model as the local
# GGUF, but on Ollama's CUDA runtime. The sidecar's llama build is CPU-only
# (llama_supports_gpu_offload() is False in this venv), so holding the local
# GGUF resident cost ~8 GB of COMMITTED RAM (weights + a 4.9 GB clip compute
# buffer the loader allocates from the mmproj's image config — not reducible
# from our side) and crushed the machine to <1 GB free during creative runs.
# Ollama puts those gigabytes in the idle 6 GB VRAM instead and self-unloads.
_OLLAMA_VL_MODEL        = "qwen2.5vl:3b"


def _ollama_vision(b64: str, prompt_text: str, *, max_tokens: int = 200,
                   temperature: float = 0.1) -> "str | None":
    """One vision chat completion against Ollama (GPU, RAM-free). The local
    sidecar stays as the fallback when Ollama isn't installed."""
    import requests as _rq
    payload = {
        "model": _OLLAMA_VL_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt_text, "images": [b64]}],
        "keep_alive": _OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    for attempt in (1, 2):
        try:
            r = _rq.post("http://localhost:11434/api/chat", json=payload,
                         timeout=(_OLLAMA_CONNECT_TIMEOUT, _OLLAMA_READ_TIMEOUT))
            if r.ok:
                msg = (r.json().get("message") or {}).get("content") or ""
                return msg.strip() or None
            print(f"[ce] Ollama vision HTTP {r.status_code}")
            return None
        except _rq.exceptions.ReadTimeout:
            # Cold VRAM load likely consumed the budget; the model is resident
            # now, so a second attempt usually returns quickly.
            if attempt == 1:
                print("[ce] Ollama vision read-timeout (cold load?) — retrying once")
                continue
            print("[ce] Ollama vision timed out after retry")
        except Exception as _e:
            print(f"[ce] Ollama vision failed: {_e}")
            return None
    return None


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

    # Rating-aware critique (2026-09-15): when the photographer has rated this
    # photo, the judge sees their own context and must reason from visible
    # evidence about whether THIS frame fits what they keep. It never changes
    # the verdict — it explains or challenges it from what is visible.
    _tctx = _taste_context(image_path, record)
    if _tctx:
        prompt_text += (
            "\n\n" + _tctx + "\n"
            "Write the critique FOR this photographer: point out what in THIS "
            "image aligns with what they keep, and what clashes. If the verdict "
            "is borderline, name the visible thing that would move it into the "
            "next band. Ground every claim in what is visible; do not repeat "
            "the context numbers back."
        )

    # ── Qwen2.5-VL multimodal (GGUF, in the sidecar process) ─────────────────
    if vision_available() and Path(image_path).exists():
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as _im:
                img_w, img_h = _im.size

            b64    = secure_image_for_vram(image_path, max_dimension=1024)
            raw = _vision_complete(
                b64, prompt_text, temperature=0.1, max_tokens=450,
            ) or ""
            critique, think = _parse_think(raw)
            bboxes   = _extract_free_bboxes(raw, img_w, img_h)
            print(f"[ce] jury (vision): {filename}  {len(critique)} chars  {len(bboxes)} bboxes")
            return {"critique": critique, "think": think, "bbox_factors": bboxes}
        except Exception as _e:
            print(f"[ce] Qwen-VL critique error ({_e})")

    # There used to be an Ollama fallback here (qwen2.5vl:3b, then deepseek-r1:8b)
    # whose failure message told the user to install a third-party service that
    # nothing in FirstCut's installer set up and no document mentioned. The
    # local GGUF above is the supported path; if it is absent, say so and name the
    # remedy we actually ship.
    return {"error": "The critique model is not installed. Run the model "
                     "downloader to fetch it — grading is unaffected.",
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

    if vision_available() or _check_ollama_available():
        try:
            b64 = secure_image_for_vram(sheet_path, max_dimension=1280)
            msgs = _vl_messages(b64, prompt_text)
            raw = None
            # Ollama first (GPU, RAM-free). Its swap output is parsed by
            # _parse_swap_json's loose brace-matching — the same path the old
            # Ollama fallback used, before the local GGUF existed.
            if _check_ollama_available():
                raw = _ollama_vision(b64, prompt_text, max_tokens=200)
            # Sidecar fallback, grammar-constrained when this build supports it.
            if raw is None and not _swap_grammar_broken and vision_available():
                raw = _chat_complete(
                    msgs, temperature=0.1, max_tokens=200,
                    grammar_src=_SWAP_GRAMMAR_SRC,
                )
            if raw is None:
                raw = _chat_complete(msgs, temperature=0.1, max_tokens=200) or ""
            parsed = _parse_swap_json(raw)
            if parsed:
                print(f"[ce] contact-sheet critique (vision): action={parsed['action']}")
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

    _a_ctx = ""
    try:
        _a_ctx = _taste_context(image_path, record)
    except Exception:
        _a_ctx = ""
    _a_line = (f"\nPHOTOGRAPHER CONTEXT: {_a_ctx.replace(chr(10), ' | ')}\n"
               "Weigh the 3 factors by what THIS photographer keeps and rejects.\n"
               if _a_ctx else "")

    vl_prompt = (
        "Backend API. Output ONLY a JSON array — no prose, no markdown fences.\n"
        f"IMAGE: {filename} | SCORE: {score:.3f} ({grade})\n"
        f"ASPECTS:\n{breakdown_str}\n"
        f"PROFILE: {profile or 'N/A'}\n"
        + _a_line +
        "\nIdentify EXACTLY 3 quality factors visible in the image. "
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

    # ── Qwen2.5-VL multimodal (GGUF, in the sidecar process) ─────────────────
    if vision_available() and Path(image_path).exists():
        try:
            from PIL import Image as _PIL
            with _PIL.open(image_path) as _im:
                img_w, img_h = _im.size

            b64    = secure_image_for_vram(image_path, max_dimension=896)
            raw = _vision_complete(
                b64, vl_prompt, temperature=0.05, max_tokens=320,
            ) or ""
            factors = _parse_factor_json(raw, img_w, img_h)
            if factors:
                print(f"[ce] annotation (vision): {filename}  {len(factors)} factors")
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

def _taste_context(image_path: str, record: dict) -> str:
    """Rating-aware context for the critique prompts (2026-09-15, Stages 2-3
    follow-on). Assembles what the photographer's own data says about THIS
    photo: their star rating, the calibrated verdict with its margin, the
    preference model's cull flag, a one-line taste profile from their rated
    anchors, and the closest rated relatives in their library.

    Returns '' whenever there is nothing personal to say (unrated photo, gate
    off, store unavailable) — callers then fall back to the legacy generic
    prompt. Everything is best-effort; a failure anywhere degrades to ''."""
    try:
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        import ratings_store as _rs
        import preference_model as _pm
        import numpy as _np

        path = str(image_path)
        stars = _rs.get(path)
        if stars <= 0:
            return ""                      # unrated → nothing personal to say

        lines: list[str] = []
        score = float(record.get("score", 0.0) or 0.0)
        verdict = str(record.get("grade", "") or "ungraded")

        # Calibrated verdict margin — how close this photo sits to a cut line.
        try:
            from rating_calibration import thresholds_from_records as _tfr
            _s_t, _m_t, _ = _tfr(_rs.load_records(),
                                 strong_default=0.60, mid_default=0.41)
            band = ("above the Keep line" if score >= _s_t
                    else "between the Keep and Maybe lines" if score >= _m_t
                    else "below the Maybe line")
            margin = min(abs(score - _s_t), abs(score - _m_t))
            lines.append(f"Verdict: {verdict} — {band}, {margin:.2f} from the "
                         f"nearest line ({'borderline' if margin < 0.04 else 'clear'})")
        except Exception:
            lines.append(f"Verdict: {verdict}")

        lines.append(f"Your rating: {stars} star" + ("s" if stars != 1 else ""))

        # Preference model verdict, if its gate is deployed.
        gate = _pm.load_gate() or {}
        if gate.get("active"):
            w = gate.get("weights")
            if w and isinstance(record.get("embedding"), (list, tuple)):
                try:
                    import numpy as _npx
                    from preference_model import build_anchor_banks, affinities
                    _recs = _rs.load_records()
                    _embs = _ls_embeds_for([r["path"] for r in _recs] + [path])
                    if path in _embs:
                        kb, rb, kp, rp = build_anchor_banks(_recs, _embs)
                        if len(kb) and len(rb):
                            ka, ra = affinities(_embs[path], kb, rb)
                            ps = float(_npx.dot(_npx.asarray(w),
                                                [score, ka, ra]))
                            lines.append(f"Preference model score: {ps:.2f} "
                                         f"(positive = looks like your keepers)")
                except Exception:
                    pass
            # Cull suggestion recorded by the last grade run, if any.
            try:
                import catalog_store as _cs
                for ph in _cs.load().get("photos", []):
                    if ph.get("path") == path:
                        if ph.get("pref_cull"):
                            lines.append("Preference model flagged this photo "
                                         "as looking like your typical rejects "
                                         "despite its verdict")
                        break
            except Exception:
                pass

        # Taste profile from the photographer's rated anchors.
        ts = _pm.taste_summary()
        if ts and (ts.get("keeper_leans") or ts.get("reject_leans")):
            pairs = (ts.get("keeper_leans") or []) + (ts.get("reject_leans") or [])
            pairs = sorted(pairs, key=lambda kv: -abs(kv[1]))[:4]
            lines.append("Your taste gaps (your keepers' average factor minus "
                         "your rejects', + means your keepers score higher): "
                         + ", ".join(f"{k} {g:+.2f}" for k, g in pairs))

        # Closest rated relatives in the library (embedding neighbours).
        emb = record.get("embedding")
        if isinstance(emb, (list, tuple)) and emb:
            try:
                import lance_store as _ls
                import numpy as _npx
                hits = _ls.vector_search(_npx.asarray(emb, dtype=_npx.float32),
                                         top_k=6)
                shown = 0
                for h in hits:
                    hp = h.get("path")
                    if not hp or hp == path:
                        continue
                    hs = _rs.get(hp)
                    if hs <= 0:
                        continue
                    lines.append(f"Closest photo in your library: "
                                 f"{Path(hp).name} — you rated it {hs} star"
                                 + ("s" if hs != 1 else ""))
                    shown += 1
                    if shown >= 2:
                        break
            except Exception:
                pass

        if not lines:
            return ""
        ctx = "PHOTOGRAPHER CONTEXT (their own ratings and history):\n- " \
              + "\n- ".join(lines)
        return ctx[:1600]                  # token cap for the 2-3B VL model
    except Exception as exc:
        print(f"[ce] taste context unavailable ({exc})")
        return ""


def _ls_embeds_for(paths: list) -> dict:
    """Embeddings for the given paths via the bulk scanner (IN-clause safe)."""
    import lance_store as _ls
    return _ls.query_embeddings_for_paths_bulk(paths)


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
