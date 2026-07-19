"""
Qwen2.5-VL-3B-Instruct Vision-Language Reasoning Engine

Speed target: 100 photos in ~2 minutes.

Architecture (hybrid):
    CLIP score  — computed instantly from SigLIP-2 embeddings already in hand.
    VLM reason  — Qwen2.5-VL-3B-Instruct looks at each image and writes the
                  critique; the score comes from CLIP so generation can be short
                  (~40 tokens vs 200), cutting per-photo time to ~0.5–1.5 s.

Typical throughput on 6 GB GPU (RTX 3060):
    ~0.8–2 s/photo  →  80–200 s for 100 photos

VRAM: ~2.2 GB INT4 — loads after SigLIP-2 unloads, frees before PersonalHead.
"""
from __future__ import annotations

import gc
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import os as _os_env

MODEL_ID        = "Qwen/Qwen2.5-VL-3B-Instruct"
# QWEN_VLM_DIR selects the vision grader weights (A/B switch between model
# generations — e.g. models/qwen3_vl for Qwen3-VL-4B). The model class is
# chosen at load time from the directory's config.json model_type.
MODEL_CACHE_DIR = Path(_os_env.environ.get("QWEN_VLM_DIR", "models/qwen_vlm"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Pre-quantised NF4 checkpoint. The grade worker is a fresh subprocess per run,
# so the in-process singleton never survives a cull — without this cache every
# run re-reads the FP16 shards and quantises them on the fly (the long
# stall at ~53%). Loading the saved 4-bit checkpoint takes seconds.
# Derived per model dir so two model generations never share a cache.
INT4_CACHE_DIR  = Path(str(MODEL_CACHE_DIR) + "_int4")


def _resolve_hf_model_dir(base: Path) -> Path:
    """Return the directory that actually holds the model weights.

    Handles both a flat layout (weights directly under `base`) and the HuggingFace
    cache layout (`base/models--*/snapshots/<hash>/`). The loader was pointed at the
    cache ROOT, so from_pretrained could not find the sharded weights and the whole
    grader silently fell back to SpecVLM CLIP — the cause of imprecise grades.
    """
    try:
        if any(base.glob("*.safetensors")) or (base / "config.json").exists():
            return base
        snaps = [p for p in base.glob("models--*/snapshots/*") if p.is_dir()]
        for s in sorted(snaps, key=lambda p: p.stat().st_mtime, reverse=True):
            if (s / "config.json").exists() and (
                any(s.glob("*.safetensors")) or (s / "model.safetensors.index.json").exists()
            ):
                return s
    except Exception:
        pass
    return base

# Short prompt — fewer output tokens = faster generation
# Accurate editor voice: states what actually decides the image, no tone words
# ("blunt"/"harsh" make small VLMs deflate everything to sound tough).
_REASON_PROMPT = """\
You are an experienced photo editor. Look at this photograph and write ONE \
specific sentence describing what actually decides this image: the strength \
that carries it or the flaw that limits it, naming the photographic principle \
involved (e.g. decisive moment, chiaroscuro, leading lines, bokeh isolation, \
Winogrand tilt, negative space). Be accurate and concrete — no generic praise, \
no exaggeration in either direction. Speak directly: "Your..." or "The..." \
(e.g. "The chiaroscuro carries the frame, but the focus plane landed behind \
the subject — zone focus at 2 m and shoot again"; \
or "You caught a decisive gesture but the horizon tilt reads as accidental \
rather than intentional — commit to level or an extreme Dutch angle.").\
"""

# Max image dimension for the MAIN scoring pass. Two-tier design (2026-06-11,
# user target: 1.5-2 s/photo): the main pass is fast triage at 336 px / large
# batch / short JSON; photos near a grade boundary get the expensive look —
# 672 px sampled passes in the borderline re-judge. 336's known weakness
# (night shots reading as mush) is exactly what the re-judge tier corrects.
_MAX_VLM_PX = 336
_MAX_NEW_TOKENS = 60   # one sentence is plenty (critique-only reason path)
# Scoring JSON: 2 short evidence phrases (strongest_element / main_flaw,
# written FIRST) + ~6 numeric fields ≈ 90-110 tokens. The critique SENTENCE
# was dropped from the scoring schema (2026-06-11, 1.5-2 s/photo target) —
# decode length is the dominant per-photo cost, and the UI critique is
# synthesised from the evidence phrases (deep critique stays on-demand).
# Truncation mid-object = unparseable = blank breakdown, so keep headroom.
_MAX_SCORE_TOKENS = 160


def _repair_json(s: str) -> str:
    """Best-effort repair of almost-JSON so json.loads can accept it.

    Handles the failure modes a token-truncated or sloppy VLM produces:
    trailing commas, single quotes, an unterminated final string, and missing
    closing brackets/braces. Returns the repaired string (may still be invalid;
    callers wrap in try/except)."""
    import re as _re
    if not s:
        return s
    t = s.strip()
    # Single-quoted keys/strings → double (only when no double quotes present,
    # to avoid mangling apostrophes inside a valid double-quoted string).
    if "'" in t and '"' not in t:
        t = t.replace("'", '"')
    # Drop trailing commas before } or ]
    t = _re.sub(r',\s*([}\]])', r'\1', t)
    # If a string was left open by truncation, close it.
    if t.count('"') % 2 == 1:
        t += '"'
    # Balance braces/brackets that truncation left open.
    t += "}" * max(0, t.count("{") - t.count("}"))
    t += "]" * max(0, t.count("[") - t.count("]"))
    return t


# ── Scoring prompt (full VLM grading — no CLIP dependency) ────────────────────

_SCORE_PROMPT_TEMPLATE = """\
You are a photography editor evaluating a photograph for a {mode} edit.

Rate this image honestly. Output ONLY valid JSON — no prose, no markdown fences:
{{"score":0,"composition":0,"lighting":0,"narrative":0,"human":0,"technical":0,"critique":"one sentence"}}

All values are integers 0–100. Apply this SAME scale to every field including each individual aspect:
  0:         not applicable — use 0 for human ONLY when absolutely no person exists in the frame
  below 35:  genuine failure in this dimension (severe blur; chaotic/blocked composition; blown or dead exposure; zero tension)
  35–50:     weak — present but not working (manageable softness; ordinary framing; flat light; incidental figures; nothing at stake)
  50–65:     competent — solid professional standard (clean; readable geometry; functional light; something is happening)
  65–80:     strong — portfolio-worthy in this dimension (precise; decisive hierarchy; clear mood; real gesture or tension caught)
  80+:       exceptional — this dimension defines the image

A photo worth keeping for a portfolio should score 60–75 on its best aspects.
Do NOT cluster scores between 40–55. Use the full range — reserve 35–50 for genuinely weak dimensions, 65+ for genuinely strong ones.

  score        overall culling verdict — holistic weighted judgment
  composition  framing, geometry, visual hierarchy, decisive use of space (65+ = every element earns its place)
  lighting     quality, direction, tonal depth, shadow control, mood (65+ = light has clear direction and mood)
  narrative    decisive moment ONLY — 65+ requires a specific unrepeatable action: peak gesture, expression, or juxtaposition frozen at exactly the right frame; a person simply walking or standing = 35–50; a static scene with no action = 35; 0 if no person visible
  human        human presence and gesture quality — 0 ONLY if NO person is visible anywhere in the frame
  technical    sharpness, exposure accuracy, noise, lens rendering (65+ = execution is invisible)
  critique     one sentence of direct feedback to the photographer: name the strongest photographic principle at work and the single most important thing to improve next. Be specific — name the technique or concept (e.g. "decisive moment", "chiaroscuro", "zone focusing") and give actionable guidance. Avoid generic phrases like "well-composed" or "good light".

Calibration: spread your scores across the full range. Roughly 30% of aspects should fall below 50, 40% between 50–65, and 30% above 65. If most scores land between 55–68, widen your distribution.
{rag_block}
JSON only:"""

# Modes where human presence is irrelevant — use environment-aware prompt instead.
_ENV_MODES: set[str] = {"architectural", "liminal", "fine art"}

# Prompt for architectural / liminal / fine art — replaces human/narrative with
# atmosphere + geometry so shots without subjects aren't penalised for missing people.
_SCORE_PROMPT_TEMPLATE_ENV = """\
You are a photography mentor evaluating a photograph for a {mode} edit.

Rate this image honestly. Output ONLY valid JSON — no prose, no markdown fences:
{{"score":0,"composition":0,"lighting":0,"atmosphere":0,"geometry":0,"technical":0,"critique":"one sentence"}}

All values are integers 0–100. Apply this SAME scale to every field including each individual aspect:
  below 35:  genuine failure in this dimension (severe blur; incoherent spatial structure; blown/dead exposure)
  35–50:     weak — present but not working (soft; flat light; no spatial tension; atmospheric drift)
  50–65:     competent — solid standard (clean; readable spatial structure; functional light and atmosphere)
  65–80:     strong — portfolio-worthy in this dimension (precise; compelling geometry or mood; clear visual intent)
  80+:       exceptional — this dimension defines the image

A strong architectural or fine-art image should score 65–80 on its dominant dimensions.
Do NOT cluster scores between 40–55. Use the full range.

  score        overall culling verdict — holistic weighted judgment
  composition  framing, spatial hierarchy, perspective, decisive use of space (65+ = spatial logic is airtight)
  lighting     quality, direction, tonal depth, shadow control, mood (65+ = light is directional with clear mood)
  atmosphere   sense of place, solitude, tension, environmental resonance, mood (65+ = atmosphere is unmistakable)
  geometry     structural lines, symmetry, rhythm, architectural clarity, spatial depth (65+ = geometry carries the image)
  technical    sharpness, exposure accuracy, noise, lens rendering (65+ = execution is invisible)
  critique     one sentence of direct feedback to the photographer: name the geometric or atmospheric principle at work and what would strengthen it next. Reference specific devices (leading lines, frame-within-frame, negative space, symmetry) and give actionable guidance.
{rag_block}
JSON only:"""

_MODE_LABELS: dict[str, str] = {
    "story":          "photo story",
    "competition":    "competition print",
    "classic street": "street photography",
    "street":         "street photography",
    "architectural":  "architectural photography",
    "fine art":       "fine art photography",
    "liminal":        "liminal / atmospheric photography",
}


# ── Ollama culling prompt templates ───────────────────────────────────────────
# Two variants: story (editorial emphasis) and competition (strict technical).
# Placeholders: {is_monochrome} {topiq_score} {yolo_detections} {rag_context}

STORY_SYSTEM_PROMPT_TEMPLATE = """\
You are a photography mentor and editorial judge reviewing a student's work for a photo story edit. \
Your feedback is educational — name photographic concepts, reference traditions, and give actionable guidance. \
Avoid color descriptors if the image is black-and-white.

=== TECHNICAL GROUNDING ===
- Black and White: {is_monochrome}
- Sharpness (TOPIQ 0-100, lower = blurry): {topiq_score}
- Subjects detected: {yolo_detections}

=== STYLE CONTEXT ===
{rag_context}

=== RESPONSE FORMAT ===
Return a SINGLE valid JSON object — no markdown, no prose outside the JSON. Every property required.

{{"culling_verdict":{{"global_score":0,"verdict_reasoning":"2 sentences: what photographic principle this image demonstrates and what the photographer should work on next — speak directly to the photographer"}},"aesthetic_critique":{{"narrative_arc":"3-4 sentences on decisive moment, gesture, and human condition — name the photographic tradition (Cartier-Bresson, Frank, Winogrand, Koudelka) this frame works in, what it achieves, and what's missing","geometry_composition":"2-3 sentences evaluating compositional intent — name specific devices used (leading lines, frame-within-frame, negative space, rule of thirds) and whether they succeed or need refinement"}},"spatial_localization_map":[{{"label":"anchor_subject|focal_point_miss|light_leak|blown_highlight|crushed_shadow|motion_blur|composition_anchor","bbox_2d":[x1,y1,x2,y2],"justification":"1-sentence educational note: why this region matters and what it teaches"}}]}}

bbox_2d: absolute pixel coordinates [x1,y1,x2,y2]. Output 2-3 entries. global_score: integer 0-100.

Begin your JSON processing now."""

COMPETITION_SYSTEM_PROMPT_TEMPLATE = """\
You are a photography competition judge and educator. Evaluate this image for a competition entry. \
Ground your verdict in objective technical criteria first, then aesthetic judgment. \
Teach the photographer exactly what is and isn't working at competition level. \
Avoid color descriptors if the image is black-and-white.

=== TECHNICAL GROUNDING ===
- Black and White: {is_monochrome}
- Sharpness (TOPIQ 0-100; below 40 = disqualifying blur): {topiq_score}
- Subjects detected: {yolo_detections}

=== COMPETITION RUBRIC ===
{rag_context}

=== RESPONSE FORMAT ===
Return a SINGLE valid JSON object — no markdown, no prose outside the JSON.

{{"culling_verdict":{{"global_score":0,"verdict_reasoning":"2 sentences: competition-level verdict explaining exactly what technical or artistic standard is or isn't met, with reference to what a jury would see"}},"aesthetic_critique":{{"narrative_arc":"3-4 sentences on decisive moment and emotional impact — name whether a moment has been caught, what tradition it belongs to, and what would make it a competition-worthy frame","geometry_composition":"2-3 sentences on compositional precision — name specific strengths and weaknesses using technical vocabulary (tonal depth, spatial hierarchy, tonal balance, geometric tension)"}},"spatial_localization_map":[{{"label":"anchor_subject|focal_point_miss|light_leak|blown_highlight|crushed_shadow|motion_blur|composition_anchor","bbox_2d":[x1,y1,x2,y2],"justification":"1-sentence: what this region reveals about technical or artistic success or failure"}}]}}

bbox_2d: absolute pixel coordinates. Output 2-3 entries. global_score: 0-100.
TOPIQ below 40 must pull global_score below 45 unless artistic blur intent is clearly evident.

Begin your JSON processing now."""


# ── Fast-scan prompt (numbers + bboxes only — no text generation) ─────────────
# Token budget: ~128 predicted tokens.  Runs 3–5× faster than the full prompt
# because the VLM skips all sentence generation.
# Output schema: {"global_score":N,"spatial_localization_map":[{...},...]}

FAST_SCAN_PROMPT_TEMPLATE = """\
Photography culling engine. Score this image and mark 2-3 spatial regions.

CPU metrics: bw={is_monochrome}  topiq={topiq_score}/100  persons={yolo_persons}
{rag_block}
Output ONLY this exact JSON structure — absolute pixel bbox coords, nothing else:
{{"global_score":0,"spatial_localization_map":[{{"label":"anchor_subject","bbox_2d":[x1,y1,x2,y2]}},{{"label":"focal_point_miss","bbox_2d":[x1,y1,x2,y2]}}]}}

label must be one of: anchor_subject  composition_anchor  focal_point_miss  blown_highlight  crushed_shadow  motion_blur
global_score is integer 0-100.  TOPIQ below 40 forces global_score below 45 unless artistic blur is evident.
JSON only — no markdown, no prose:"""

# ── Gemma 3 bbox-only spatial grounding ───────────────────────────────────────
# Gemma 3 acts as a pure visual grounding scanner — locating the anchor subject.
# ALL scoring is done in grade_pipeline_v2.py via deterministic Python math:
#   global_score = 0.40 * TOPIQ + 0.35 * SigLIP_RAG_sim + 0.25 * comp_fit
# where comp_fit = 1 - distance(bbox_center, nearest_thirds_node) / max_dist.
#
# Benefits: eliminates all LLM score hallucination, halves token output,
# gives Gemma a single focused task it executes reliably every time.
# num_predict=60 — bbox JSON is ~20 tokens.

GEMMA_SYSTEM_TEMPLATE = """\
You are a visual grounding system. Locate the primary subject in the image.
Active style context: {rag_block}
Output ONLY a raw JSON object. No markdown, no explanation."""

GEMMA_USER_PROMPT = """\
Locate the primary anchor subject (main person, animal, or focal point) in this photograph.

Output ONLY this JSON with normalized 0-1000 integer coordinates:
{{"spatial_localization_map":[{{"label":"anchor_subject","bbox_2d":[ymin,xmin,ymax,xmax]}}]}}

[ymin,xmin,ymax,xmax] = tightest bounding box, top-left to bottom-right, 0-1000 scale.
If no clear focal subject exists, use the most visually dominant region.
JSON only:"""


# ── Deep-text prompt (on-demand, text only — no scores or bboxes) ─────────────
# Called by /api/critique/details for a single selected photo.
# Token budget: ~600 predicted tokens.

GENERATE_DEEP_TEXT_PROMPT = """\
You are a photography mentor writing detailed feedback for a student. Look at this image. \
Do not output bounding boxes or scores. Write ONLY a valid JSON block with two fields. \
Speak directly to the photographer — name photographic concepts, traditions, and techniques. \
Give actionable guidance, not just description.
{{
  "narrative_arc": "<3-4 sentences: is a decisive moment present? Name the tradition this image works in (Cartier-Bresson, Frank, Winogrand, Koudelka, Arbus, Salgado). What human truth or gesture does it capture or miss? What would the photographer need to do differently to catch a stronger moment?>",
  "geometry_composition": "<2-3 sentences: name the compositional devices at work (leading lines, frame-within-frame, negative space, foreground-background layering, rule of thirds, symmetry). Are they working? What one compositional adjustment would most strengthen the frame?>"
}}
{rag_block}
JSON only — no markdown, no preamble:"""


# ── Ollama culling helper ──────────────────────────────────────────────────────

def _is_monochrome(path: str) -> bool:
    """Heuristic: True when image is grayscale or de-saturated (BW film scan)."""
    try:
        from PIL import Image as _PIL
        with _PIL.open(path) as img:
            if img.mode in ("L", "LA", "1"):
                return True
            arr = np.array(img.convert("RGB").resize((32, 32)), dtype=np.float32)
            sat = arr.max(axis=2) - arr.min(axis=2)
            return float(sat.mean()) < 12.0
    except Exception:
        return False


_FAST_SCAN_MODEL = "gemma3:4b"        # bulk scoring — Q4_K_M (3.3 GB, more accurate than Q2_K)
_DEEP_DIVE_MODEL = "qwen2.5vl:3b"  # on-demand critique — poetic prose

# Session flag: flips True after Ollama's GPU backend throws a CUDA kernel error
# ("device kernel image is invalid" — driver/Ollama build mismatch). Once set, all
# subsequent Ollama calls go straight to CPU (num_gpu=0) instead of crashing the
# llama-server with a 500 on every request.
_OLLAMA_GPU_BROKEN = False


def _get_pulled_models() -> set:
    """Return set of model names currently pulled in Ollama (empty on error)."""
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        if r.ok:
            return {m["name"] for m in r.json().get("models", [])}
    except Exception:
        pass
    return set()


def resolve_fast_scan_model() -> str:
    """Return best available fast-scan model: gemma3:4b → gemma3:4b-q2k → qwen fallback."""
    pulled = _get_pulled_models()
    if _FAST_SCAN_MODEL in pulled:          # "gemma3:4b" Q4_K_M preferred
        return _FAST_SCAN_MODEL
    if "gemma3:4b-q2k" in pulled:          # Q2_K acceptable fallback
        print(f"[vlm] {_FAST_SCAN_MODEL} not found — using gemma3:4b-q2k")
        return "gemma3:4b-q2k"
    print(f"[vlm] gemma3 not found in Ollama — falling back to {_DEEP_DIVE_MODEL} for fast scan")
    return _DEEP_DIVE_MODEL


def warmup_vlm_models() -> None:
    """
    Evict the deep-dive model if resident, then pre-load the fast-scan model only.
    Warming both simultaneously overflows 6 GB VRAM and pushes Qwen to CPU.
    """
    try:
        import requests as _req
        # Evict deep-dive model so it doesn't occupy VRAM alongside the fast-scan model.
        try:
            _req.post(
                "http://localhost:11434/api/generate",
                json={"model": _DEEP_DIVE_MODEL, "keep_alive": 0},
                timeout=5,
            )
            print(f"[vlm] evicted: {_DEEP_DIVE_MODEL}")
        except Exception:
            pass

        _pulled = _get_pulled_models()
        _fast   = resolve_fast_scan_model()
        _present = _fast in _pulled
        if not _present:
            print(f"[vlm] warmup skipped (not pulled): {_fast}")
            return
        try:
            _req.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": _fast, "stream": False,
                    "keep_alive": "5m",   # warm for the grade, not pinned forever (was -1 → never freed)
                    "options": {"num_predict": 1, "num_gpu": 999},
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=180,
            )
            print(f"[vlm] warmup ok: {_fast}")
        except Exception as _e:
            print(f"[vlm] warmup failed for {_fast}: {_e}")
    except ImportError:
        pass


def execute_vlm_culling_sync(
    image_path:  str,
    mode:        str,
    cpu_metrics: dict,
    rag_context: str,
    model:       str  = _FAST_SCAN_MODEL,
    timeout:     int  = 45,
    fast_scan:   bool = False,
) -> Optional[dict]:
    """
    Route images through the dual-VLM pipeline:

    fast_scan=True  → Gemma 3 (4B) bulk scoring — FAST_SCAN_PROMPT_TEMPLATE,
                      num_predict=128, keep_alive=0.  Target ~3 s/image.
    fast_scan=False → Qwen2.5-VL full critique templates, num_predict=512.

    Returns parsed JSON dict or None on failure.
    cpu_metrics keys: is_monochrome (bool), topiq_score (int 0-100),
                      yolo_detections (dict with "persons" key)
    """
    import base64, io, json as _json, re as _re, os as _os
    try:
        import requests as _req
    except ImportError:
        return None

    # Poison-pill guard: reject 0-byte or missing files before they reach PIL/GPU.
    if not _os.path.exists(image_path) or _os.path.getsize(image_path) == 0:
        print(f"[vlm_cull] Skipping empty/missing file: {Path(image_path).name}")
        return None

    _is_gemma = fast_scan and "gemma" in model.lower()
    _is_mono  = cpu_metrics.get("is_monochrome", False)  # bool; renamed to avoid shadowing module-level fn
    _topiq    = cpu_metrics.get("topiq_score",   50)
    _persons  = cpu_metrics.get("yolo_detections", {}).get("persons", 0)

    if _is_gemma:
        # Gemma 3: RAG injected via system role using the already-loaded rag_context parameter.
        _rag_phrases = [l.lstrip("- ") for l in (rag_context or "").splitlines() if l.strip()]
        _rag_block   = "\n".join(f"- {p}" for p in _rag_phrases[:5]) if _rag_phrases else "No style reference loaded."
        system_msg = GEMMA_SYSTEM_TEMPLATE.format(rag_block=_rag_block)
        messages    = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": GEMMA_USER_PROMPT, "images": []},  # images filled below
        ]
        num_predict = 60   # bbox JSON is ~20 tokens
    elif fast_scan:
        rag_block = (
            "Style reference:\n" + "\n".join(f"- {p}" for p in (rag_context or "").split("\n") if p.strip())
            if rag_context and rag_context != "No reference context provided."
            else ""
        )
        user_prompt = FAST_SCAN_PROMPT_TEMPLATE.format(
            is_monochrome = _is_mono,
            topiq_score   = _topiq,
            yolo_persons  = _persons,
            rag_block     = rag_block,
        )
        messages    = [{"role": "user", "content": user_prompt, "images": []}]
        num_predict = 128
    else:
        template = (
            STORY_SYSTEM_PROMPT_TEMPLATE
            if mode.lower() in ("story", "photo story")
            else COMPETITION_SYSTEM_PROMPT_TEMPLATE
        )
        user_prompt = template.format(
            is_monochrome   = _is_mono,
            topiq_score     = _topiq,
            yolo_detections = _json.dumps(cpu_metrics.get("yolo_detections", {})),
            rag_context     = rag_context or "No reference context provided.",
        )
        messages    = [{"role": "user", "content": user_prompt, "images": []}]
        num_predict = 512

    # Encode image — keep original dims for Gemma coord descaling
    try:
        from PIL import Image as _PIL
        with _PIL.open(image_path) as img:
            _orig_w, _orig_h = img.size
            if max(_orig_w, _orig_h) > 336:
                _scale = 336 / max(_orig_w, _orig_h)
                img = img.resize((int(_orig_w * _scale), int(_orig_h * _scale)), _PIL.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as _e:
        print(f"[vlm_cull] Image encode failed {Path(image_path).name}: {_e}")
        return None

    # Inject base64 into the last message that has an images list
    for _msg in reversed(messages):
        if "images" in _msg:
            _msg["images"] = [b64]
            break

    global _OLLAMA_GPU_BROKEN
    try:
        _temp  = 0.0 if _is_gemma else 0.3   # deterministic bbox localization
        _top_p = 0.1 if _is_gemma else 0.92  # restrict to highest-prob tokens

        def _cull_chat(num_gpu: int):
            return _req.post(
                "http://localhost:11434/api/chat",
                json={
                    "model":      model,
                    "stream":     False,
                    "keep_alive": "5m",   # warm across the batch, not pinned forever (was -1 → never freed)
                    "options":    {"temperature": _temp, "top_p": _top_p, "num_predict": num_predict, "num_gpu": num_gpu},
                    "messages":   messages,
                },
                timeout=timeout,
            )

        r = _cull_chat(0 if _OLLAMA_GPU_BROKEN else 999)
        if not r.ok:
            _body = (r.text or "")[:300]
            # Ollama GPU backend crash (CUDA kernel mismatch) → retry once on CPU,
            # then remember for the rest of the session so we don't keep crashing it.
            if r.status_code == 500 and not _OLLAMA_GPU_BROKEN and "CUDA" in _body:
                print("[vlm_cull] Ollama GPU backend failed (CUDA) — switching to CPU for this session")
                _OLLAMA_GPU_BROKEN = True
                r = _cull_chat(0)
            if not r.ok:
                print(f"[vlm_cull] Ollama returned {r.status_code}: {(r.text or '')[:200]}")
                return None
        raw = r.json().get("message", {}).get("content", "").strip()
    except _req.exceptions.Timeout:
        print(f"[vlm_cull] Timeout ({timeout}s) — Gemma locked on {Path(image_path).name}")
        return None
    except _req.exceptions.ConnectionError:
        print(f"[vlm_cull] Ollama disconnected — ConnectionError on {Path(image_path).name}")
        return None
    except Exception as _e:
        print(f"[vlm_cull] Ollama request failed: {_e}")
        return None

    # Strip markdown fences, extract JSON object
    raw = _re.sub(r"^```[a-zA-Z]*\s*", "", raw).strip()
    raw = _re.sub(r"\s*```\s*$",        "", raw).strip()
    m   = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if not m:
        print(f"[vlm_cull] No JSON in response for {Path(image_path).name}")
        return None
    try:
        _json_str = m.group()
        _json_str = _re.sub(r",\s*([\]}])", r"\1", _json_str)  # heal trailing commas
        data = _json.loads(_json_str)
        if fast_scan:
            gs = data.get("global_score", "?")
        else:
            gs = data.get("culling_verdict", {}).get("global_score", "?")
        print(f"[vlm_cull] {Path(image_path).name}: score={gs}"
              + ("  [gemma/fast]" if _is_gemma else "  [fast]" if fast_scan else ""))

        if _is_gemma:
            # Descale Gemma's [ymin,xmin,ymax,xmax] 0-1000 normalized bbox
            # to absolute pixel [x1,y1,x2,y2] for the frontend SVG overlay.
            _slm = data.get("spatial_localization_map", [])
            _descaled = []
            _bbox_center_norm = None
            for _b in _slm:
                _raw_bb = _b.get("bbox_2d", [])
                if len(_raw_bb) == 4:
                    _yn1, _xn1, _yn2, _xn2 = [max(0, min(1000, int(v))) for v in _raw_bb]
                    if _yn1 > _yn2: _yn1, _yn2 = _yn2, _yn1  # correct inverted y
                    if _xn1 > _xn2: _xn1, _xn2 = _xn2, _xn1  # correct inverted x
                    _descaled.append({
                        "label":   _b.get("label", "anchor_subject"),
                        "bbox_2d": [
                            int((_xn1 / 1000) * _orig_w),
                            int((_yn1 / 1000) * _orig_h),
                            int((_xn2 / 1000) * _orig_w),
                            int((_yn2 / 1000) * _orig_h),
                        ],
                    })
                    _bbox_center_norm = ((_xn1 + _xn2) / 2000, (_yn1 + _yn2) / 2000)
            data["spatial_localization_map"] = _descaled
            data["bbox_center_norm"] = _bbox_center_norm
            # Scoring done in grade_pipeline_v2.py — Gemma is bbox-only

        return data
    except Exception as _e:
        print(f"[vlm_cull] JSON parse error {Path(image_path).name}: {_e}")
        return None


def execute_vlm_text_deep_dive(
    image_path:  str,
    mode:        str,
    rag_context: Optional[str] = None,
    model:       str = _DEEP_DIVE_MODEL,
    timeout:     int = 180,
) -> Optional[dict]:
    """
    On-demand deep text critique for a single selected photo.

    Loads up to 5 RAG phrases from cache/rag_concepts.json (or uses the
    supplied rag_context string), then asks the VLM to write narrative_arc and
    geometry_composition prose — no scores, no bounding boxes.

    Returns {"narrative_arc": str, "geometry_composition": str} or None.
    Designed to be called by POST /api/critique/details.
    """
    import base64, io, json as _json, re as _re
    try:
        import requests as _req
    except ImportError:
        return None

    if rag_context is None:
        try:
            from pdf_rag import load_concepts as _lc
            phrases = _lc()
            rag_context = "\n".join(f"- {p}" for p in phrases[:5]) if phrases else ""
        except Exception:
            rag_context = ""

    rag_block = (
        f"=== STYLE REFERENCE ===\n{rag_context}"
        if rag_context.strip() else ""
    )

    prompt = GENERATE_DEEP_TEXT_PROMPT.format(rag_block=rag_block)

    try:
        from PIL import Image as _PIL
        with _PIL.open(image_path) as img:
            w, h = img.size
            if max(w, h) > 448:
                scale = 448 / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), _PIL.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as _e:
        print(f"[vlm_deep] Image encode failed {Path(image_path).name}: {_e}")
        return None

    global _OLLAMA_GPU_BROKEN

    def _chat(num_gpu: int):
        return _req.post(
            "http://localhost:11434/api/chat",
            json={
                "model":      model,
                "stream":     False,
                # Keep the VL model warm so it isn't unloaded between the deep
                # critique and the annotation path (which both use qwen2.5vl:3b) —
                # the old keep_alive:0 forced a cold reload every call, thrashing
                # the 6 GB GPU and blowing past the timeout. 30s frees it on idle.
                "keep_alive": "30s",
                "options":    {"temperature": 0.5, "top_p": 0.95, "num_predict": 350, "num_ctx": 2048, "num_gpu": num_gpu},
                "messages":   [{
                    "role":    "user",
                    "content": prompt,
                    "images":  [b64],
                }],
            },
            timeout=timeout,
        )

    # One retry on read-timeout: the first call may be consumed by the cold VRAM
    # load (~3 GB) under GPU contention; the second usually returns warm.
    raw = None
    for _attempt in (1, 2):
        try:
            # Skip the GPU attempt entirely once it has failed this session.
            r = _chat(0 if _OLLAMA_GPU_BROKEN else 999)
            if not r.ok:
                _body = (r.text or "")[:300]
                # Ollama's GPU llama-server crashes with a 500 + CUDA error when its
                # CUDA build doesn't match the installed driver/GPU. Retry on CPU once.
                if r.status_code == 500 and not _OLLAMA_GPU_BROKEN and "CUDA" in _body:
                    print(f"[vlm_deep] Ollama GPU backend failed (CUDA) — switching to CPU for this session")
                    _OLLAMA_GPU_BROKEN = True
                    r = _chat(0)
                if not r.ok:
                    print(f"[vlm_deep] Ollama returned {r.status_code}: {(r.text or '')[:200]}")
                    return None
            raw = r.json().get("message", {}).get("content", "").strip()
            break
        except _req.exceptions.ReadTimeout:
            if _attempt == 1:
                print(f"[vlm_deep] read-timeout (cold load / GPU contention) — retrying once")
                continue
            print(f"[vlm_deep] timed out after retry ({timeout}s)")
            return None
        except Exception as _e:
            print(f"[vlm_deep] Ollama request failed: {_e}")
            return None
    if raw is None:
        return None

    raw = _re.sub(r"^```[a-zA-Z]*\s*", "", raw).strip()
    raw = _re.sub(r"\s*```\s*$",        "", raw).strip()
    m   = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if not m:
        print(f"[vlm_deep] No JSON in response for {Path(image_path).name}")
        return None
    try:
        data = _json.loads(m.group())
        result = {
            "narrative_arc":       data.get("narrative_arc",       ""),
            "geometry_composition": data.get("geometry_composition", ""),
        }
        print(f"[vlm_deep] {Path(image_path).name}: "
              f"{len(result['narrative_arc'])} chars narrative, "
              f"{len(result['geometry_composition'])} chars geometry")
        return result
    except Exception as _e:
        print(f"[vlm_deep] JSON parse error {Path(image_path).name}: {_e}")
        return None


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class VLMResult:
    path:      str
    score:     float                          # comes from CLIP caller
    aspects:   Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    critique:  str = ""


@dataclass
class VLMScoredResult:
    """Full VLM-scored result — score and aspects from vision, not CLIP."""
    path:      str
    score:     float                          # 0–1 normalised
    breakdown: Dict[str, float]               # same keys as SpecVLMResult.breakdown
    critique:  str


# ── Reasoning formatter ────────────────────────────────────────────────────────

_ASPECT_DETAIL = {
    "Technical":   "sharpness, exposure, and noise",
    "Composition": "framing, geometry, and visual flow",
    "Lighting":    "directional light and tonal contrast",
    "Moment":      "decisive moment and story tension",
    "Human":       "emotional presence and candid energy",
}


_ASPECT_SHORT = {
    "Technical":   "Technical",
    "Composition": "Composition",
    "Lighting":    "Lighting",
    "Moment":      "Moment",
    "Human":       "Human",
}


def build_vlm_reasoning(
    score: float,
    aspects: Dict[str, float],
    critique: str,
) -> str:
    tier = "Strong" if score >= 0.60 else ("Mid" if score >= 0.41 else "Weak")
    pct  = int(round(score * 100))
    lines = [f"{tier}  {pct}%"]
    if critique:
        lines.append(critique)
    if aspects:
        lines.append("")
        for k, v in sorted(aspects.items(), key=lambda x: -x[1]):
            label = _ASPECT_SHORT.get(k, k)
            bar   = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"{label:<12} {bar}  {int(v*100)}%")
        top    = _ASPECT_SHORT.get(max(aspects, key=aspects.get), max(aspects, key=aspects.get))
        bottom = _ASPECT_SHORT.get(min(aspects, key=aspects.get), min(aspects, key=aspects.get))
        lines.append(f"\nBest: {top}   ·   Weakest: {bottom}")
    return "\n".join(lines)


# ── Grader class ───────────────────────────────────────────────────────────────

class QwenVLMGrader:
    """
    Loads Qwen2.5-VL-3B-Instruct as INT4 (~2.2 GB VRAM).

    grade_images() accepts pre-computed CLIP scores (from SigLIP-2) and uses
    the VLM only to generate per-photo reasoning text — keeping generation
    length short for maximum throughput.
    """

    _INT4_VRAM_GB = 2.2
    # Qwen2.5-VL-3B is ~3.75 B params; FP32 on CPU needs ~15 GB resident plus load
    # overhead. Require headroom before attempting it — otherwise the caller's
    # exception handler falls back to lightweight SpecVLM CLIP scoring (no OOM kill).
    _CPU_FP32_RAM_GB = 16.0

    def __init__(self, device: str = "auto", progress=None):
        _p = progress or (lambda f, d: None)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.device == "cuda":
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / 1e9
            if free_gb < self._INT4_VRAM_GB:
                print(
                    f"[qwen_vlm] Only {free_gb:.1f} GB free "
                    f"(need ~{self._INT4_VRAM_GB:.1f}) — falling back to CPU"
                )
                self.device = "cpu"

        _p(0.52, "Loading Vision Engine…")
        print(f"[qwen_vlm] Loading on {self.device} from {MODEL_CACHE_DIR}…")

        from transformers import AutoProcessor

        _model_dir = _resolve_hf_model_dir(MODEL_CACHE_DIR)
        if _model_dir != MODEL_CACHE_DIR:
            print(f"[qwen_vlm] Resolved HF-cache snapshot → {_model_dir}")

        # Pick the model class from the checkpoint's own config — lets
        # QWEN_VLM_DIR point at any Qwen VL generation without code changes.
        _model_type = ""
        try:
            import json as _json_cfg
            _model_type = _json_cfg.loads(
                (_model_dir / "config.json").read_text(encoding="utf-8")
            ).get("model_type", "")
        except Exception as _e_cfg:
            print(f"[qwen_vlm] config.json read failed ({_e_cfg}) — assuming qwen2_5_vl")

        self._model_type = _model_type   # drives per-model anchors + calibration

        if _model_type == "qwen3_vl":
            from transformers import Qwen3VLForConditionalGeneration as _VLCls
            print("[qwen_vlm] model_type=qwen3_vl → Qwen3VLForConditionalGeneration")
        elif _model_type in ("qwen2_5_vl", "qwen2_vl", ""):
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _VLCls
            except ImportError:
                try:
                    from transformers import Qwen2VLForConditionalGeneration as _VLCls
                except ImportError:
                    raise ImportError(
                        "transformers >= 4.45 with Qwen2-VL support required.\n"
                        "Run:  pip install --upgrade transformers"
                    )
        else:
            # Non-Qwen rivals (internvl, smolvlm, gemma3, …): the generic
            # image-text-to-text auto class resolves the concrete model from
            # the checkpoint config — same processor/chat-template contract.
            from transformers import AutoModelForImageTextToText as _VLCls
            print(f"[qwen_vlm] model_type={_model_type} → AutoModelForImageTextToText")
        base_kw = dict(
            pretrained_model_name_or_path=str(_model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )

        # Heartbeat — from_pretrained gives no progress callbacks, so without
        # this the bar freezes for the whole load and looks crashed.
        import threading as _thr
        _hb_stop = _thr.Event()
        _hb_t0   = time.time()

        def _heartbeat():
            while not _hb_stop.wait(5.0):
                _p(0.525, f"Loading Vision Engine… {int(time.time() - _hb_t0)}s elapsed")

        _thr.Thread(target=_heartbeat, daemon=True).start()
        try:
            if self.device == "cuda":
                # RAM preflight for the GPU INT4 load too. Loading the ~2.3 GB
                # INT4 cache (or re-quantising the FP16 model) materialises
                # weights in HOST RAM before the GPU transfer; on a RAM-starved
                # box — especially right after SigLIP consumed memory — this
                # OOM-kills the worker NATIVELY at the Qwen step (no traceback,
                # the cull just dies). Raise instead so grade_pipeline_v2's
                # `except _e_qwen` falls back to SpecVLM CLIP scoring and the
                # cull still COMPLETES. Tunable via QWEN_MIN_FREE_RAM_GB.
                try:
                    import os as _os_q, psutil as _ps_q
                    _free_gb = _ps_q.virtual_memory().available / 1e9
                    _need_q  = float(_os_q.environ.get("QWEN_MIN_FREE_RAM_GB", "3.0"))
                except Exception:
                    _free_gb = _need_q = None
                if _free_gb is not None and _free_gb < _need_q:
                    raise RuntimeError(
                        f"Insufficient RAM for Qwen load: {_free_gb:.1f} GB free, "
                        f"need ~{_need_q:.1f} GB — deferring to SpecVLM CLIP scoring."
                    )
                self._model = self._load_int4(_VLCls, base_kw, _p)
            else:
                # RAM preflight — the CPU path loads FP32 (~15 GB). Without headroom this
                # silently OOM-kills the grade worker. Bail out so the caller falls back
                # to SpecVLM CLIP scoring instead of crashing the whole run.
                try:
                    import psutil as _psutil
                    _free_gb = _psutil.virtual_memory().available / 1e9
                    if _free_gb < self._CPU_FP32_RAM_GB:
                        raise RuntimeError(
                            f"Insufficient RAM for Qwen CPU fallback: {_free_gb:.1f} GB free, "
                            f"need ~{self._CPU_FP32_RAM_GB:.0f} GB for FP32 load — "
                            f"deferring to SpecVLM CLIP scoring."
                        )
                except ImportError:
                    pass   # psutil absent — proceed and let the OS handle allocation
                _p(0.525, "Loading on CPU…")
                self._model = _VLCls.from_pretrained(
                    **base_kw, torch_dtype=torch.float32, device_map="cpu"
                )
        finally:
            _hb_stop.set()

        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            str(_model_dir), local_files_only=True, trust_remote_code=True
        )
        # Decoder-only generation requires LEFT padding. With the default right
        # padding, batched VL generation can emit empty/garbled completions (the
        # cause of "No parseable JSON … ''" → silent default scores). Force left
        # padding on both the processor and its inner tokenizer.
        try:
            self._processor.tokenizer.padding_side = "left"
            if hasattr(self._processor, "padding_side"):
                self._processor.padding_side = "left"
        except Exception as _e_pad:
            print(f"[qwen_vlm] could not set padding_side=left: {_e_pad}")

        # torch.compile is intentionally skipped: BitsAndBytes INT4 + torch.compile
        # causes a hard process crash on Windows at first inference (Triton/inductor
        # tries to JIT-compile CUDA kernels that BnB has already quantised, leading
        # to a fatal CUDA error that kills the entire Python process).
        # 0.53 keeps progress monotonic: grade_images_scored resumes at 0.53.
        _p(0.53, "Qwen2.5-VL ready…")
        print("[qwen_vlm] Model ready.")

    # ------------------------------------------------------------------
    def _load_int4(self, cls, base_kw: dict, progress=None):
        from transformers import BitsAndBytesConfig
        _p = progress or (lambda f, d: None)

        # ── Opt-in INT8 (set QWEN_QUANT=int8): closer to full-precision
        # judgment than NF4, ~3.8 GB VRAM, ~1.5× slower. A/B switch, not the
        # default — falls through to NF4 on insufficient VRAM or load failure.
        import os as _os
        if _os.environ.get("QWEN_QUANT", "").lower() == "int8":
            try:
                _free8 = torch.cuda.mem_get_info(0)[0] / 1e9
            except Exception:
                _free8 = 0.0
            if _free8 >= 4.2:
                try:
                    _p(0.525, "Loading INT8 weights (QWEN_QUANT=int8)…")
                    model = cls.from_pretrained(
                        **base_kw,
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                        device_map="auto",
                    )
                    print("[qwen_vlm] QWEN_QUANT=int8 — INT8 loaded (~3.8 GB VRAM)")
                    return model
                except Exception as _e8:
                    print(f"[qwen_vlm] INT8 load failed ({_e8}) — falling back to NF4")
            else:
                print(f"[qwen_vlm] QWEN_QUANT=int8 requested but only "
                      f"{_free8:.1f} GB free — falling back to NF4")

        # ── Pre-quantised cache hit: skip on-the-fly NF4 quantisation ────────
        if (INT4_CACHE_DIR / "config.json").exists() and any(INT4_CACHE_DIR.glob("*.safetensors")):
            try:
                _p(0.525, "Loading cached INT4 weights…")
                model = cls.from_pretrained(
                    pretrained_model_name_or_path=str(INT4_CACHE_DIR),
                    local_files_only=True,
                    trust_remote_code=True,
                    device_map="auto",
                )
                print("[qwen_vlm] INT4 cache hit — pre-quantised load (~2.2 GB VRAM)")
                return model
            except Exception as e_cache:
                # Corrupt/partial/incompatible cache — wipe it and re-quantise.
                print(f"[qwen_vlm] INT4 cache load failed ({e_cache}) — re-quantising")
                import shutil
                shutil.rmtree(INT4_CACHE_DIR, ignore_errors=True)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Python 3 deletes `except ... as var` bindings at block exit, so save
        # each error message before the except block ends.
        _e1_msg = _e2_msg = ""
        _p(0.525, "Quantising to INT4 (first run only — cached after)…")

        # 1st attempt: INT4 + flash_attn2 (best speed on Ampere+)
        try:
            model = cls.from_pretrained(
                **base_kw, quantization_config=bnb, device_map="auto",
                attn_implementation="flash_attention_2",
            )
            print("[qwen_vlm] INT4 + flash_attn2 loaded (~2.2 GB VRAM)")
            self._save_int4_cache(model)
            return model
        except Exception as e1:
            _e1_msg = str(e1)
            print(f"[qwen_vlm] flash_attn2 failed ({e1}) — retrying INT4 without it")

        # 2nd attempt: INT4 without flash_attn2 (Windows-safe)
        try:
            model = cls.from_pretrained(
                **base_kw, quantization_config=bnb, device_map="auto",
            )
            print("[qwen_vlm] INT4 loaded (~2.2 GB VRAM)")
            self._save_int4_cache(model)
            return model
        except Exception as e2:
            _e2_msg = str(e2)
            print(f"[qwen_vlm] INT4 failed ({e2}) — trying INT8 fallback")

        # 3rd attempt: INT8 (~3.5 GB VRAM, still fits in 6 GB)
        try:
            from transformers import BitsAndBytesConfig as _BnB
            bnb8 = _BnB(load_in_8bit=True)
            model = cls.from_pretrained(
                **base_kw, quantization_config=bnb8, device_map="auto",
            )
            print("[qwen_vlm] INT8 loaded (~3.5 GB VRAM)")
            return model
        except Exception as e3:
            raise RuntimeError(
                f"All GPU load attempts failed.\n  flash_attn2: {_e1_msg}\n  INT4: {_e2_msg}\n  INT8: {e3}"
            ) from e3

    # ------------------------------------------------------------------
    @staticmethod
    def _save_int4_cache(model) -> None:
        """Persist the quantised checkpoint so later worker processes skip
        quantisation. 4-bit serialisation needs bitsandbytes >= 0.41.3 —
        failure is non-fatal (a partial dir is wiped so the next run never
        loads a corrupt cache)."""
        import shutil
        try:
            INT4_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(INT4_CACHE_DIR), safe_serialization=True)
            print(f"[qwen_vlm] INT4 weights cached to {INT4_CACHE_DIR} — "
                  f"future runs skip quantisation")
        except Exception as e:
            print(f"[qwen_vlm] INT4 cache save skipped ({e})")
            shutil.rmtree(INT4_CACHE_DIR, ignore_errors=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _resize(path: str, max_px: int = _MAX_VLM_PX):
        """Load and resize image to max_px on the long edge.

        RAM lever: draft() lets libjpeg decode a JPEG at a reduced DCT scale
        (1/2, 1/4, 1/8) directly, so a 38 MP photo materialises at ~0.6 MP
        instead of ~116 MB before the LANCZOS downscale to max_px. No-op for
        PNG/already-small images; grading quality is unchanged (the frame ends
        at max_px either way). With BS+1 concurrent decodes this is the main
        peak-RAM driver during a cull.
        """
        from PIL import Image as _PIL
        img = _PIL.open(path)
        try:
            img.draft("RGB", (max_px, max_px))   # JPEG-only DCT downscale
        except Exception:
            pass
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)), _PIL.Resampling.LANCZOS
            )
        return img

    # ------------------------------------------------------------------
    def grade_images(
        self,
        paths: List[str],
        clip_scores: np.ndarray,          # (N,) float32 from SigLIP-2 CLIP
        clip_aspects: Optional[List[Dict[str, float]]] = None,
        progress=None,
    ) -> List[VLMResult]:
        """
        Generate per-photo reasoning.  Scores come from CLIP (already computed).

        Args:
            paths:        image file paths
            clip_scores:  calibrated CLIP scores (0-1) from SpecVLMPipeline
            clip_aspects: per-photo aspect dicts from SpecVLMPipeline (optional)
            progress:     (frac, msg) callback

        Returns:
            List[VLMResult] with score=clip_score and VLM critique in reasoning.
        """
        _p = progress or (lambda f, d: None)
        n  = len(paths)
        if clip_aspects is None:
            clip_aspects = [{}] * n

        results: List[VLMResult] = []
        t0 = time.time()

        # Pre-load next image in background while GPU runs on current one
        with ThreadPoolExecutor(max_workers=2) as pool:
            next_fut = pool.submit(self._resize, paths[0]) if n > 0 else None

            for i, path in enumerate(paths):
                # Retrieve pre-loaded image
                try:
                    img = next_fut.result() if next_fut else self._resize(path)
                except Exception as e:
                    print(f"[qwen_vlm] Preload failed {path}: {e}")
                    img = None

                # Kick off loading for next image
                next_fut = (
                    pool.submit(self._resize, paths[i + 1]) if i + 1 < n else None
                )

                # VLM critique
                if img is not None:
                    critique = self._run_one(img, path)
                    del img
                else:
                    critique = ""

                score   = float(clip_scores[i])
                aspects = clip_aspects[i]
                results.append(VLMResult(
                    path      = path,
                    score     = score,
                    aspects   = aspects,
                    reasoning = build_vlm_reasoning(score, aspects, critique),
                    critique  = critique,
                ))

                done    = i + 1
                elapsed = time.time() - t0
                if done < n:
                    eta_s   = int(elapsed / done * (n - done))
                    eta_str = f" — ~{eta_s // 60}m{eta_s % 60:02d}s left"
                else:
                    eta_str = ""
                frac = 0.56 + (done / n) * 0.29   # spans 0.56 → 0.85
                _p(frac, f"VLM reasoning: {done}/{n}{eta_str}")

        return results

    def _run_one(self, img, path: str) -> str:
        """Shared inference logic for a pre-loaded PIL image."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  _REASON_PROMPT},
            ],
        }]
        try:
            text   = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[img], return_tensors="pt", padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False
                )
            n_in = inputs["input_ids"].shape[1]
            result = self._processor.decode(
                out[0][n_in:], skip_special_tokens=True
            ).strip()
            del inputs, out  # free CUDA tensors before next image
            print(f"[qwen_vlm] {Path(path).name}: {result[:80]!r}")
            return result
        except Exception as e:
            print(f"[qwen_vlm] Inference failed {path}: {e}")
            return ""

    # Context sentences injected per image based on detected visual archetype.
    # Tells Qwen what register it's evaluating so it calibrates expectations
    # rather than applying a universal standard to every shot.
    _ARCH_CONTEXT: dict[str, str] = {
        "geometric_minimal":      "Shot register: architectural/geometric. Empty space and absence of subjects are intentional. Evaluate spatial lines, vanishing points, and minimalism.",
        "night_chiaroscuro":      "Shot register: low-key night or chiaroscuro. Deep shadows, intentional grain, and underexposure are stylistic tools — do NOT penalize them as technical failures.",
        "layered_portrait":       "Shot register: layered environmental portrait. Foreground blur and shallow depth of field are deliberate. Evaluate subject isolation and layered depth.",
        "raw_snapshot":           "Shot register: raw street documentary. Imperfect framing may be intentional. Evaluate whether a decisive moment or raw energy redeems the frame.",
        "maximalist_documentary": "Shot register: maximalist documentary. A dense, complex composition is expected. Evaluate storytelling richness and cultural density, not formal simplicity.",
    }

    # ------------------------------------------------------------------
    def grade_images_scored(
        self,
        paths:       List[str],
        mode:        str                  = "street",
        rag_phrases: Optional[List[str]]  = None,
        arch_hints:  Optional[Dict[str, str]] = None,
        progress                          = None,
        on_batch                          = None,
    ) -> List[VLMScoredResult]:
        """
        Full vision scoring — Qwen2.5-VL assigns absolute aspect scores (0-100)
        by looking at each image directly. No CLIP/SigLIP-2 dependency.

        arch_hints: optional {path: arch_label} dict from grade_pipeline_v2.
        When provided, each image gets a context line in its prompt so Qwen
        evaluates it on its own visual register rather than a universal standard.

        Returns VLMScoredResult list with breakdown keys matching SpecVLMResult:
        Composition, Lighting, Narrative, Human/Culture, Technical.
        """
        _p      = progress or (lambda f, d: None)
        n       = len(paths)
        phrases = rag_phrases or []

        # Framed as aspirational examples, not a checklist — a checklist reads
        # as pass/fail and drags scores down for every quality a photo never
        # aimed at (one strong quality matters more than five absent ones).
        rag_block  = (
            "\nQualities found in exemplary work (a strong photo shows one or "
            "two of these — absence of the others is not a flaw):\n"
            + "\n".join(f"  • {p}" for p in phrases[:8])
            + "\n"
        ) if phrases else ""
        from niche_registry import build_niche_prompt as _build_niche_prompt
        _base_prompt = _build_niche_prompt(
            mode, rag_block, model_family=getattr(self, "_model_type", "")
        )

        def _make_prompt(path: str) -> str:
            """Append arch context sentence before the final JSON-only line."""
            if not arch_hints:
                return _base_prompt
            ctx = self._ARCH_CONTEXT.get(arch_hints.get(path, ""), "")
            if not ctx:
                return _base_prompt
            if "JSON only:" in _base_prompt:
                return _base_prompt.replace("JSON only:", f"{ctx}\nJSON only:", 1)
            return _base_prompt + f"\n{ctx}"

        results: List[VLMScoredResult] = []
        t0 = time.time()

        # Derive batch size from available VRAM. At 448 px an image is ~530
        # vision tokens; with a 200-token completion the KV cache + activations
        # cost ~1.0 GB per slot. 0.5 GB is held back as reserve so WDDM never
        # spills to shared system memory (which silently runs 10× slower
        # instead of raising OOM). On a 6 GB card this yields BS=2-3.
        if self.device == "cuda":
            try:
                _free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
                # 0.65 GB/slot at 336 px (~300 vision tokens + 160 gen tokens).
                # Floor 1, not 2: under INT8 (QWEN_QUANT=int8) free VRAM is
                # genuinely tight and forcing BS 2 would spill to shared
                # memory (silent 10× slowdown). NF4 derives BS 4 on a free
                # 6 GB card, BS 2-3 with other models resident.
                #
                # Lever 1 (speed): NF4 decode is memory-bandwidth bound and the
                # per-step dequant tax only amortises across a LARGER batch. When
                # Qwen is the sole resident model (warm worker; SigLIP evicted
                # before grading) free VRAM is high, so a bigger batch is pure
                # throughput with no per-image quality change. Reserve/slot/ceil
                # are env-tunable so the sole-resident batch can be pushed and
                # benchmarked without risking the documented WDDM shared-memory
                # spill (which shows up as a 10× s/img regression, not an OOM).
                _reserve = float(_os_env.environ.get("QWEN_VRAM_RESERVE", "0.5"))
                _slot    = float(_os_env.environ.get("QWEN_VRAM_SLOT",    "0.65"))
                _bs_ceil = int(_os_env.environ.get("QWEN_BS_CEIL", "8"))
                _BS = max(1, min(_bs_ceil, int((_free_gb - _reserve) / _slot)))
            except Exception:
                _BS = 2
        else:
            _BS = 1
        print(f"[qwen_vlm] Batch size {_BS} for {n} images "
              f"({_free_gb:.1f} GB free)" if self.device == "cuda"
              else f"[qwen_vlm] Batch size {_BS} for {n} images")

        with ThreadPoolExecutor(max_workers=_BS + 1) as pool:
            # Pre-load the first batch before the loop starts
            _pre_futs = [
                pool.submit(self._resize, paths[j])
                for j in range(min(_BS, n))
            ]

            i = 0
            while i < n:
                batch_end  = min(i + _BS, n)
                cur_paths  = paths[i:batch_end]

                # Retrieve pre-loaded images for this batch
                cur_imgs: list = []
                for j, fut in enumerate(_pre_futs[:len(cur_paths)]):
                    try:
                        cur_imgs.append(fut.result())
                    except Exception as _e:
                        print(f"[qwen_vlm] Preload failed {cur_paths[j]}: {_e}")
                        cur_imgs.append(None)

                # Kick off pre-load for next batch immediately
                _pre_futs = [
                    pool.submit(self._resize, paths[j])
                    for j in range(batch_end, min(batch_end + _BS, n))
                ]

                # Batch-score valid images; fall back per-image on None
                valid_idx = [j for j, img in enumerate(cur_imgs) if img is not None]
                if valid_idx:
                    v_imgs    = [cur_imgs[j]   for j in valid_idx]
                    v_paths   = [cur_paths[j]  for j in valid_idx]
                    v_prompts = [_make_prompt(cur_paths[j]) for j in valid_idx]
                    scored    = self._score_batch(v_imgs, v_paths, v_prompts, mode)
                    for img in v_imgs:
                        del img
                else:
                    scored = []

                scored_iter = iter(scored)
                _batch_new: List[VLMScoredResult] = []
                for j, path in enumerate(cur_paths):
                    if cur_imgs[j] is not None:
                        score, breakdown, critique = next(scored_iter)
                    else:
                        score, breakdown, critique = 0.5, {}, ""
                    _r = VLMScoredResult(
                        path=path, score=score, breakdown=breakdown, critique=critique,
                    )
                    results.append(_r)
                    _batch_new.append(_r)

                # Resume checkpoint hook: persist each completed batch so a
                # cull interrupted mid-grade (e.g. window closed) resumes here
                # instead of re-grading from zero. Best-effort — never block.
                if on_batch is not None and _batch_new:
                    try:
                        on_batch(_batch_new)
                    except Exception as _ob_e:
                        print(f"[qwen_vlm] on_batch checkpoint failed: {_ob_e}")

                i       = batch_end
                done    = i
                elapsed = time.time() - t0
                eta_s   = int(elapsed / done * (n - done)) if done < n else 0
                # 0.53 → 0.65: starts where the loader ended so the bar never
                # moves backwards (it used to restart at 0.51 after 0.56).
                _p(
                    0.53 + (done / n) * 0.12,
                    f"Qwen grading: {done}/{n}"
                    + (f" — ~{eta_s // 60}m{eta_s % 60:02d}s left" if eta_s else ""),
                )

        # ── Borderline re-judging ─────────────────────────────────────────────
        # Grading mistakes that matter live near the cut lines (Weak<0.41,
        # Strong≥0.60), not at 0.85 or 0.25. Photos whose first-pass score
        # lands near a boundary get two extra passes at higher resolution
        # (672 px) with temperature sampling; axes are averaged across all
        # passes. Capped so a big cull pays a bounded premium.
        # Bands and pass count tuned for the 1.5-2 s/photo budget: one extra
        # 560 px sampled look (the first re-look carries most of the value;
        # a third judgment moved averages <0.03 in testing, and 672→560 px
        # cuts the re-look's vision tokens ~30% with no observed grade flips).
        # Lever 4 (speed): the re-judge is the discretionary half of grade cost
        # (up to _BL_MAX full second passes). Only photos that could actually
        # flip a bucket need it — i.e. within ~±0.03 of a cut line (0.41/0.60).
        # A photo at 0.45 or 0.56 is 0.04-0.05 clear of the nearest cut and a
        # second look essentially never moved it across, so tightening the bands
        # from ±0.06 to ±0.03 drops ~half the re-judges with no observed bucket
        # changes. Bands/cap env-tunable for re-benchmarking.
        _bl_w     = float(_os_env.environ.get("QWEN_REJUDGE_BAND", "0.03"))
        _BL_BANDS = ((0.41 - _bl_w, 0.41 + _bl_w), (0.60 - _bl_w, 0.60 + _bl_w))
        _BL_MAX   = int(_os_env.environ.get("QWEN_REJUDGE_MAX", "12"))
        _BL_PX    = 560

        def _is_borderline(s: float) -> bool:
            return any(lo <= s <= hi for lo, hi in _BL_BANDS)

        _bl = [i for i, r in enumerate(results) if r.breakdown and _is_borderline(r.score)]
        _bl.sort(key=lambda i: min(abs(results[i].score - 0.41),
                                   abs(results[i].score - 0.60)))
        _bl = _bl[:_BL_MAX]
        if _bl and self.device == "cuda":
            print(f"[qwen_vlm] Borderline re-judge: {len(_bl)} images "
                  f"(batched greedy @ {_BL_PX}px)")
            # Greedy + higher resolution = a genuinely independent second look
            # that is ALSO deterministic (the earlier sampled pass made grades
            # wobble between runs and forced serial generation; the resolution
            # change supplies the diversity sampling was providing). Batched
            # through _score_batch in chunks — ~2× faster than serial.
            _bl_chunk = 2   # 560 px ≈ 830 vision tokens/slot — 2 fits the budget
            for cb_start in range(0, len(_bl), _bl_chunk):
                chunk = _bl[cb_start : cb_start + _bl_chunk]
                imgs_hi, ok_idx = [], []
                for i in chunk:
                    try:
                        imgs_hi.append(self._resize(results[i].path, max_px=_BL_PX))
                        ok_idx.append(i)
                    except Exception as _e_hi:
                        print(f"[qwen_vlm] re-judge load failed "
                              f"{Path(results[i].path).name}: {_e_hi}")
                if not ok_idx:
                    continue
                scored_hi = self._score_batch(
                    imgs_hi,
                    [results[i].path for i in ok_idx],
                    [_make_prompt(results[i].path) for i in ok_idx],
                    mode,
                )
                for img in imgs_hi:
                    del img
                for i, (s2, bd2, _cr2) in zip(ok_idx, scored_hi):
                    if not bd2:
                        continue
                    r = results[i]
                    _keys  = set(r.breakdown) | set(bd2)
                    avg_bd = {k: float(np.mean([bd[k] for bd in (r.breakdown, bd2) if k in bd]))
                              for k in _keys}
                    avg_s  = float((r.score + s2) / 2.0)
                    crit   = r.critique or _cr2
                    results[i] = VLMScoredResult(
                        path=r.path, score=avg_s, breakdown=avg_bd, critique=crit,
                    )
                    print(f"[qwen_vlm]   {Path(r.path).name}: "
                          f"{r.score:.2f} → {avg_s:.2f} (2 looks)")
                _p(0.65, f"Refining borderline images: "
                         f"{min(cb_start + _bl_chunk, len(_bl))}/{len(_bl)}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    def _score_one(self, img, path: str, prompt: str, mode: str = "classic_street",
                   sample: bool = False) -> tuple[float, dict, str]:
        """Run one scoring inference. Returns (score_0_1, breakdown, critique).

        sample=True enables temperature sampling — used by the borderline
        re-judge pass, where averaging diverse samples beats one greedy run."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt},
            ],
        }]
        def _generate() -> str:
            text   = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=[img], return_tensors="pt", padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            gen_kw: dict = {"max_new_tokens": _MAX_SCORE_TOKENS}
            if sample:
                gen_kw.update(do_sample=True, temperature=0.7, top_p=0.9)
            else:
                gen_kw["do_sample"] = False
            out = None
            try:
                with torch.no_grad():
                    out = self._model.generate(**inputs, **gen_kw)
                n_in = inputs["input_ids"].shape[1]
                return self._processor.decode(
                    out[0][n_in:], skip_special_tokens=True
                ).strip()
            finally:
                del inputs, out  # free CUDA tensors before next image
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        try:
            raw = _generate()
            # Empty completion (usually a padding/decoder hiccup) → retry once
            # before _parse_score_json silently falls back to a default 0.5 score.
            if not raw:
                print(f"[qwen_vlm] empty completion {Path(path).name} — retrying once")
                raw = _generate()
            return self._parse_score_json(raw, path, mode)
        except Exception as _e:
            print(f"[qwen_vlm] Score inference failed {path}: {_e}")
            return 0.5, {}, ""

    # ------------------------------------------------------------------
    def _score_batch(
        self,
        imgs:    list,
        paths:   list,
        prompts: List[str],
        mode:    str = "classic_street",
    ) -> list:
        """Score a batch of images in a single Qwen forward pass (~1.5× throughput).
        Each image receives its own prompt (arch-context-aware).
        Falls back to serial _score_one() if batched inference fails.
        """
        if len(imgs) == 1:
            return [self._score_one(imgs[0], paths[0], prompts[0], mode)]

        messages_batch = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompts[j]},
            ]}]
            for j, img in enumerate(imgs)
        ]
        inputs = None
        out    = None
        try:
            texts = [
                self._processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                for msgs in messages_batch
            ]
            inputs = self._processor(
                text=texts, images=imgs, return_tensors="pt", padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.to("cuda") if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=_MAX_SCORE_TOKENS, do_sample=False)
            # n_in = padded input length; same for every item because of padding=True
            n_in = inputs["input_ids"].shape[1]
            # Lever 3 (speed): instrument real decode length. Per-row generated
            # tokens = non-pad tokens after n_in. If the max sits well under
            # _MAX_SCORE_TOKENS the cap is not binding and lowering it is a no-op
            # (generation already stops at EOS); if rows hit the cap, JSON is
            # being truncated (the historical blank-breakdown failure mode).
            if _os_env.environ.get("QWEN_TOKLOG"):
                try:
                    _pad = self._processor.tokenizer.pad_token_id
                    _gen = out[:, n_in:]
                    _lens = [int((row != _pad).sum()) for row in _gen]
                    print(f"[qwen_vlm] gen tokens (cap {_MAX_SCORE_TOKENS}): "
                          f"max={max(_lens)} mean={sum(_lens)/len(_lens):.0f} "
                          f"lens={_lens}")
                except Exception as _tle:
                    print(f"[qwen_vlm] toklog failed: {_tle}")
            results = []
            for bi, path in enumerate(paths):
                raw = self._processor.decode(
                    out[bi][n_in:], skip_special_tokens=True
                ).strip()
                if not raw:
                    # Empty batched completion — re-score this one serially (which
                    # retries) so it never silently defaults to 0.5.
                    print(f"[qwen_vlm] empty batched completion {Path(path).name} — re-scoring serially")
                    results.append(self._score_one(imgs[bi], path, prompts[bi], mode))
                else:
                    results.append(self._parse_score_json(raw, path, mode))
            return results
        except Exception as _be:
            print(f"[qwen_vlm] Batch-{len(imgs)} failed ({_be}) — serial fallback")
            # List comprehension defers VRAM release until after all iterations;
            # explicit loop + empty_cache() prevents accumulation across serial calls.
            _serial: list = []
            for _j, (_img, _p) in enumerate(zip(imgs, paths)):
                _serial.append(self._score_one(_img, _p, prompts[_j], mode))
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            return _serial
        finally:
            # Always release GPU tensors — critical to prevent VRAM leak on exception path.
            del out, inputs
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Per-model output calibration: fixed affine maps (absolute — identical for
    # every batch, never per-batch normalisation) that translate a model's
    # native score distribution onto the scale the downstream system was built
    # around (thresholds 0.41/0.60, fusion gates, anchor floors). Constants are
    # fitted offline on a calibration set graded by both models; missing entry
    # = identity. calibrated = clip(gain * x + offset).
    _OUTPUT_CALIBRATION: dict = {
        # "qwen3_vl": {"gain": ..., "offset": ...},   # fitted by _ab_fit_calibration
    }

    def _apply_output_calibration(self, score: float, breakdown: dict) -> tuple[float, dict]:
        cal = self._OUTPUT_CALIBRATION.get(getattr(self, "_model_type", ""))
        if not cal:
            return score, breakdown
        g, o = float(cal["gain"]), float(cal["offset"])
        _m = lambda v: float(np.clip(g * float(v) + o, 0.0, 1.0))
        return _m(score), {k: (_m(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                           for k, v in breakdown.items()}

    # ------------------------------------------------------------------
    def _parse_score_json(self, raw: str, path: str, mode: str = "classic_street") -> tuple[float, dict, str]:
        """Parse VLM JSON output → (score_0_1, breakdown_dict, critique).

        Resilient to the common ways Qwen drifts from "JSON only": markdown code
        fences, leading/trailing prose, nested objects (the old flat-only regex
        missed these), trailing commas, and single quotes. Last resort scrapes
        key:value pairs by regex so a malformed-but-readable answer still yields a
        breakdown instead of a blank MID-50 row.
        """
        import re as _re, json as _json
        from niche_registry import parse_niche_breakdown as _parse_niche

        data = self._extract_json_obj(raw)
        if data is None:
            data = self._scrape_kv(raw)

        if not data:
            print(f"[qwen_vlm] No parseable JSON {Path(path).name}: {raw[:120]!r}")
            return 0.5, {}, ""

        score, breakdown, critique = _parse_niche(data, mode)
        if not breakdown:
            print(f"[qwen_vlm] Empty breakdown {Path(path).name} (keys={list(data)[:8]}): {raw[:120]!r}")
        score, breakdown = self._apply_output_calibration(score, breakdown)
        print(f"[qwen_vlm] {Path(path).name}: score={score:.2f}  {critique[:60]!r}")
        return score, breakdown, critique

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_obj(raw: str):
        """Return the first JSON object in `raw` as a dict, or None.

        Strips markdown fences, then scans for a brace-balanced span (handles
        nested objects, which the old `\\{[^{}]*\\}` regex could not). Retries
        after light repair (trailing commas, single→double quotes)."""
        import re as _re, json as _json
        if not raw:
            return None
        s = raw.strip()
        # Strip ```json … ``` / ``` … ``` fences if present.
        fence = _re.search(r'```(?:json)?\s*(.*?)\s*```', s, _re.DOTALL | _re.IGNORECASE)
        if fence:
            s = fence.group(1).strip()

        start = s.find('{')
        if start == -1:
            return None
        # Brace-balanced scan, skipping braces inside strings.
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        candidate = s[start:end + 1] if end != -1 else s[start:]

        for attempt in (candidate, _repair_json(candidate)):
            try:
                obj = _json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _scrape_kv(raw: str):
        """Last-resort: scrape "key": value pairs from non-JSON text.

        Catches answers like  composition: 72, lighting: 65 ...  that never
        formed valid JSON. Returns a dict (possibly empty)."""
        import re as _re
        out: dict = {}
        # "key": 72   |   key: 72   |   'key': 72
        for m in _re.finditer(r'["\']?([A-Za-z][A-Za-z /_-]{1,28})["\']?\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)', raw):
            out[m.group(1).strip()] = float(m.group(2))
        # critique / one-liner sentence, if any quoted string with spaces remains
        cm = _re.search(r'["\']?(?:critique|comment|summary)["\']?\s*[:=]\s*["\']([^"\']{4,})["\']', raw, _re.IGNORECASE)
        if cm:
            out["critique"] = cm.group(1).strip()
        return out

    # ------------------------------------------------------------------
    def unload(self) -> None:
        # Do NOT call .cpu() — BitsAndBytes INT4 models don't support it and
        # it leaves VRAM partially occupied. Just drop the reference and let GC + empty_cache clean up.
        self._model     = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        print("[qwen_vlm] Unloaded.")
