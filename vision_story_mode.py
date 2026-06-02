#!/usr/bin/env python3
"""
vision_story_mode.py -- Vision-Verified Story Mode Pipeline

Replaces the text-only story mode with a four-stage in-memory vision pipeline:

  Stage 1  CLIP (openai/clip-vit-base-patch32)
           All images are encoded as 224x224 grayscale PIL objects. Pixel
           data is deleted immediately after embedding. No disk writes.

  Stage 2  Qwen 2.5-VL (temperature=0.0) via Ollama
           Each candidate is rendered as a 512x512 JPEG -> base64 string ->
           POST -> response parsed -> base64 deleted. No pixel data retained.

  Stage 3  DeepSeek-R1:8b (Ollama) -- narrative art direction
           Receives text metadata only (slot, filename, score). No images.

  Stage 4  TSP luminance reorder -> Judge's Verdict -> final_story_manifest.json

Output    final_story_manifest.json
          Fields: image_path, assigned_slot, approved_by_vision, curator_justification

Usage:
    python vision_story_mode.py ./dataset_images
    python vision_story_mode.py ./dataset_images "rain soaked streets at dusk"
    python vision_story_mode.py ./dataset_images "urban loneliness" 5
"""
from __future__ import annotations

import base64
import gc
import io
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from src.vram_manager import purge_vram as _purge_vram
except ImportError:
    def _purge_vram() -> None:  # type: ignore[misc]
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            if hasattr(_torch.cuda, "ipc_collect"):
                _torch.cuda.ipc_collect()

# =============================================================================
# Configuration
# =============================================================================

_IMAGE_EXTS          = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}
_OLLAMA_GEN_URL      = "http://localhost:11434/api/generate"
_CLIP_MODEL          = "openai/clip-vit-base-patch32"
_QWEN_GGUF           = Path(__file__).parent / "models" / "qwen2.5-vl-2b-instruct-q4_k_m.gguf"
_QWEN_MMPROJ         = Path(__file__).parent / "models" / "mmproj-qwen2.5-vl-2b-instruct-f16.gguf"
_QWEN_MODEL          = "qwen2.5vl:3b"   # kept for reference / fallback label
_DIRECTOR_MODEL      = "deepseek-r1:8b"
_OUTPUT_FILE         = "final_story_manifest.json"
_CLIP_BATCH          = 16
_CANDIDATES_PER_SLOT = 8   # top-k images to send to Qwen per slot

# Runtime thresholds — loaded from config.json if present, else defaults apply.
_CONFIG_PATH = Path(__file__).parent / "config.json"
_THRESHOLD_DEFAULTS: dict[str, float] = {
    "DOMINANT_AREA_PCT": 20.0,
    "H_CONTRAST_THRESH": 161.0,
    "OPENER_MAX_AREA":   30.0,
}


def _load_thresholds() -> dict[str, float]:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as _f:
            _raw = json.load(_f)
        return {
            **_THRESHOLD_DEFAULTS,
            **{k: float(v) for k, v in _raw.items() if isinstance(v, (int, float))},
        }
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(_THRESHOLD_DEFAULTS)


_CFG: dict[str, float] = _load_thresholds()

# Story slots in pacing order
_SLOT_NAMES = ["Opener", "Subject/Interaction", "Detail/Accent", "Closer/Resolution"]

# Per-slot CLIP retrieval queries
_SLOT_CLIP_QUERIES: dict[str, str] = {
    "Opener":
        "wide establishing shot empty street deep focus geographic urban landscape context",
    "Subject/Interaction":
        "people interacting mid-frame street candid human decisive moment social encounter",
    "Detail/Accent":
        "close-up detail texture macro shallow depth hands signage object visual metaphor",
    "Closer/Resolution":
        "figure walking away vanishing point receding perspective quiet resolution ending",
}

# Per-slot Qwen system prompts (verbatim structural tests per slot type)
_SLOT_SYSTEM_PROMPTS: dict[str, str] = {

    "Opener": """\
You are a documentary photography editor trained in archival photography theory \
(LIFE Magazine, Stephen Bull, LUMA Arles). \
Evaluate this image strictly as an OPENER/ESTABLISHING slot candidate.

=== CRITERIA -- ALL must pass to approve ===
1. LOW SUBJECT-TO-FRAME RATIO: Human subjects must occupy less than 30% of the frame.
2. DEEP DEPTH OF FIELD: Both foreground AND background must appear in focus, \
grounding the viewer in a physical environment.
3. CONTEXTUAL GROUNDING: The frame must establish geographic or situational context -- \
an identifiable location, time of day, social environment, or urban landscape.
4. HUMANIST DIGNITY: Camera must be at or near eye level with subjects. \
Any high-angle shot that reduces subjects to objects or strips their agency is grounds \
for immediate rejection.
5. INTENTIONAL GEOMETRY: The composition must contain at least one of: \
leading lines drawing the eye to a subject or vanishing point, \
framing shapes (archways, doorways, shadow frames), \
or clear rule-of-thirds placement with a dominant subject.

REJECT (assigned_slot = "Snapshot", approved_for_essay = false) if ANY criterion fails -- \
especially high-angle framing, absence of contextual information, or subjects \
dominating the majority of the frame.

=== SPATIAL GROUNDING (mandatory regardless of approval decision) ===
Detect and report the bounding boxes of two elements using coordinates normalized to 0-1000 \
(0 = top or left edge, 1000 = bottom or right edge). Format: [ymin, xmin, ymax, xmax].
  subject_bbox  : The primary subject -- the most important person, figure, or object.
  anchor_bbox   : The secondary structural element defining the composition -- \
a shadow, road surface, window frame, archway, horizon line, or wall plane.
  If no distinct anchor is visible, use the background plane [0, 0, 1000, 1000].

Output ONLY a single valid JSON object -- no markdown, no explanation, no text outside JSON:
{"approved_for_essay": <true|false>, "assigned_slot": "<Opener|Snapshot>", \
"curator_justification": "<exactly one sentence citing the specific criterion that drove this decision>", \
"subject_bbox": [ymin, xmin, ymax, xmax], "subject_label": "<label>", \
"anchor_bbox": [ymin, xmin, ymax, xmax], "anchor_label": "<label>"}""",

    "Subject/Interaction": """\
You are a documentary photography editor trained in archival photography theory. \
Evaluate this image strictly as a SUBJECT/INTERACTION slot candidate.

=== CRITERIA -- ALL must pass to approve ===
1. MID-RANGE FRAMING: Subjects must be framed waist-up or wider. \
Extreme close-ups, head-shots only, or telephoto-compressed full-body shots are disqualifying.
2. MULTI-SUBJECT OR SOCIAL TENSION: At least two subjects must be present, interacting, \
or in reciprocal spatial tension -- labor, urban friction, or social exchange between figures.
3. DECISIVE MOMENT: Spontaneous human action must lock into geometric alignment with \
background elements -- shadows, architectural lines, or vanishing points -- at the precise \
frame captured. A static pose with no geometric tension fails this test absolutely.
4. HUMANIST DIGNITY: Camera must be at or near eye level. \
Voyeuristic framings that strip subject agency must be rejected.
5. INTENTIONAL GEOMETRY: The compositional grid must be deliberate -- leading lines, \
rule-of-thirds subject placement, or framing elements that direct the eye.

REJECT (assigned_slot = "Snapshot") if subjects are absent, the moment is static and \
lacks geometric alignment with the environment, or the framing is exploitative.

=== SPATIAL GROUNDING (mandatory) ===
Report bounding boxes [ymin, xmin, ymax, xmax] normalized to 0-1000.
  subject_bbox  : The primary interacting figure or the figure closest to the camera.
  anchor_bbox   : The background geometric element (shadow, wall, road, architectural line) \
that the decisive moment aligns with.

Output ONLY a single valid JSON object:
{"approved_for_essay": <true|false>, "assigned_slot": "<Subject/Interaction|Snapshot>", \
"curator_justification": "<exactly one sentence>", \
"subject_bbox": [ymin, xmin, ymax, xmax], "subject_label": "<label>", \
"anchor_bbox": [ymin, xmin, ymax, xmax], "anchor_label": "<label>"}""",

    "Detail/Accent": """\
You are a documentary photography editor trained in archival photography theory. \
Evaluate this image strictly as a DETAIL/ACCENT slot candidate.

=== CRITERIA -- ALL must pass to approve ===
1. HIGH MAGNIFICATION: The frame must isolate a single dominant graphic element -- \
a hand, texture, signage, shadow pattern, or material surface. \
Wide-angle contextual shots cannot qualify for this slot.
2. SHALLOW DEPTH OF FIELD: The background must be visibly out of focus, \
creating clear visual separation and emphasis on the dominant element.
3. METAPHORIC FUNCTION: The isolated element must function as a visual pause -- \
a texture that implies a broader narrative or a detail that acts as punctuation \
within a documentary sequence.
4. HUMANIST DIGNITY: If human subjects appear, the camera must be at eye level. \
Voyeuristic framing of body parts that strips subject agency is disqualifying.
5. ONE DOMINANT GRAPHIC ELEMENT: The composition must be organized around exactly \
one graphic center. Multiple competing focal points disqualify this slot entirely.

REJECT (assigned_slot = "Snapshot") if the image has a wide contextual composition, \
if multiple graphic elements compete for dominance, \
or if the depth of field is deep and narrative rather than intimate and detail-focused.

=== SPATIAL GROUNDING (mandatory) ===
Report bounding boxes [ymin, xmin, ymax, xmax] normalized to 0-1000.
  subject_bbox  : The single dominant graphic element (hand, sign, texture, shadow pattern).
  anchor_bbox   : The out-of-focus background plane or the secondary texture surface \
providing the depth separation.

Output ONLY a single valid JSON object:
{"approved_for_essay": <true|false>, "assigned_slot": "<Detail/Accent|Snapshot>", \
"curator_justification": "<exactly one sentence>", \
"subject_bbox": [ymin, xmin, ymax, xmax], "subject_label": "<label>", \
"anchor_bbox": [ymin, xmin, ymax, xmax], "anchor_label": "<label>"}""",

    "Closer/Resolution": """\
You are a documentary photography editor trained in archival photography theory. \
Evaluate this image strictly as a CLOSER/RESOLUTION slot candidate.

=== CRITERIA -- ALL must pass to approve ===
1. COMPOSITIONAL FINALITY: The image must convey closure through at least one of: \
vanishing point perspective (architectural or natural lines converging at the horizon), \
a low-contrast atmospheric landscape with deep space and receding distance, \
or subjects explicitly moving away from the camera -- backs turned, receding figures, \
diminishing silhouettes.
2. NARRATIVE RESOLUTION: The emotional register must be resolution -- \
quiet, diminishing, settling -- not active, climactic, or confrontational.
3. HUMANIST DIGNITY: No exploitative angles that diminish or objectify subjects.
4. GEOMETRIC FINALITY: Converging lines, negative space expanding toward the horizon, \
or a figure-to-background ratio that emphasizes space and distance over subject presence.
5. NO PEAK ACTION: Active, front-facing, high-energy shots are disqualifying. \
This slot must not energetically compete with the Subject/Interaction slot.

REJECT (assigned_slot = "Snapshot") if the image is energetic or action-driven, \
if subjects face the camera with full frontal presence and high frame occupation, \
or if the composition lacks finality and closure and could serve as an opener instead.

=== SPATIAL GROUNDING (mandatory) ===
Report bounding boxes [ymin, xmin, ymax, xmax] normalized to 0-1000.
  subject_bbox  : The receding figure, silhouette, or the primary element moving away from camera. \
If no subject, use the dominant atmospheric zone (e.g., sky [0,0,400,1000]).
  anchor_bbox   : The vanishing point zone, horizon line, or the architectural surface \
providing the sense of depth and distance.

Output ONLY a single valid JSON object:
{"approved_for_essay": <true|false>, "assigned_slot": "<Closer/Resolution|Snapshot>", \
"curator_justification": "<exactly one sentence>", \
"subject_bbox": [ymin, xmin, ymax, xmax], "subject_label": "<label>", \
"anchor_bbox": [ymin, xmin, ymax, xmax], "anchor_label": "<label>"}""",
}

_EVAL_PROMPT = (
    "Evaluate this photograph for its assigned narrative slot. "
    "Apply all criteria strictly. "
    "Respond with a single JSON object only -- no other text."
)

# Apply OPENER_MAX_AREA from config to the Opener slot criterion text at runtime.
_SLOT_SYSTEM_PROMPTS["Opener"] = re.sub(
    r"less than \d+(?:\.\d+)?% of the frame",
    f"less than {int(_CFG['OPENER_MAX_AREA'])}% of the frame",
    _SLOT_SYSTEM_PROMPTS["Opener"],
)


# =============================================================================
# Data types
# =============================================================================

@dataclass
class SlotResult:
    slot_name:              str
    image_path:             str
    clip_score:             float
    approved_by_vision:     bool
    assigned_slot:          str    # the approved slot name, or "Snapshot" if rejected
    curator_justification:  str
    # Coordinate-grounded spatial data extracted by Qwen (bbox = [ymin,xmin,ymax,xmax] 0-1000)
    subject_bbox:   list = field(default_factory=lambda: [0, 0, 1000, 1000])
    subject_label:  str  = ""
    anchor_bbox:    list = field(default_factory=lambda: [0, 0, 1000, 1000])
    anchor_label:   str  = ""
    luminance:      float = 0.5   # mean frame luminance [0,1] computed during TSP


# =============================================================================
# Stage 1: CLIP in-memory embedding (all images + all slot queries)
# =============================================================================

def clip_stage(
    image_dir: str,
) -> tuple[list[str], list[list[float]], dict[str, list[float]]]:
    """
    Encode all images (224x224 L -> RGB) and all slot query strings with CLIP.
    The CLIP model is unloaded immediately after encoding.

    Returns:
        valid_paths : list of image paths that were successfully encoded
        img_embs    : parallel list of 512-d float image embeddings
        text_embs   : dict mapping slot_name -> 512-d float text embedding
    """
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    image_dir_path = Path(image_dir).resolve()
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir_path}")

    raw_paths = sorted([
        str(p) for p in image_dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    ])
    if not raw_paths:
        raise ValueError(f"No images found in {image_dir_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[clip] Loading {_CLIP_MODEL} on {device} ...")
    model     = CLIPModel.from_pretrained(_CLIP_MODEL, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL)
    model.eval()

    # -- Encode all images (224x224 grayscale then RGB for CLIP's 3-channel input)
    img_embs:    list[list[float]] = []
    valid_paths: list[str]         = []
    print(f"[clip] Embedding {len(raw_paths)} image(s) in batches of {_CLIP_BATCH} ...")

    with torch.no_grad():
        for i in range(0, len(raw_paths), _CLIP_BATCH):
            batch_paths = raw_paths[i : i + _CLIP_BATCH]
            pil_batch   = []
            ok_batch    = []
            for p in batch_paths:
                try:
                    with open(p, "rb") as fh:
                        stream = io.BytesIO(fh.read())
                    img = Image.open(stream).resize((224, 224)).convert("L").convert("RGB")
                    del stream
                    pil_batch.append(img)
                    ok_batch.append(p)
                except Exception as exc:
                    print(f"[clip]   skip {Path(p).name}: {exc}")
            if not pil_batch:
                continue
            inp  = processor(images=pil_batch, return_tensors="pt").to(device)
            del pil_batch
            feat = model.get_image_features(**inp).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            img_embs.extend(feat.cpu().tolist())
            valid_paths.extend(ok_batch[: feat.shape[0]])
            del feat, inp

    # -- Encode all slot queries at once
    slot_queries = [_SLOT_CLIP_QUERIES[s] for s in _SLOT_NAMES]
    with torch.no_grad():
        txt_inp   = processor(text=slot_queries, return_tensors="pt", padding=True).to(device)
        txt_feat  = model.get_text_features(**txt_inp).float()
        txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        txt_list  = txt_feat.cpu().tolist()
    text_embs = {_SLOT_NAMES[i]: txt_list[i] for i in range(len(_SLOT_NAMES))}

    del model, processor, txt_feat, txt_inp
    _purge_vram()

    print(f"[clip] Done: {len(valid_paths)} images + {len(_SLOT_NAMES)} slot queries encoded")
    return valid_paths, img_embs, text_embs


def _rank_for_slot(
    slot_name:    str,
    all_paths:    list[str],
    all_embs:     list[list[float]],
    text_embs:    dict[str, list[float]],
    already_used: set[str],
    top_k:        int = _CANDIDATES_PER_SLOT,
) -> list[tuple[float, str]]:
    """Rank unused images by cosine similarity to the slot's text embedding."""
    query_vec = np.array(text_embs[slot_name], dtype=np.float32)
    embs_arr  = np.array(all_embs, dtype=np.float32)
    sims      = embs_arr @ query_vec
    ranked = sorted(
        [
            (float(sims[i]), all_paths[i])
            for i in range(len(all_paths))
            if all_paths[i] not in already_used
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    return ranked[:top_k]


# =============================================================================
# Stage 2: Qwen 2.5-VL slot-specific gatekeeping
# =============================================================================

def _render_image_b64(image_path: str, size: int = 512) -> str:
    """
    Load image -> resize to size x size -> JPEG bytes -> base64 string.
    Returns empty string on failure. Caller must delete the return value after use.
    No raw pixels are retained after the function returns.
    """
    try:
        from PIL import Image
        with Image.open(image_path) as raw:
            img = raw.convert("RGB").resize((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        del img
        b64 = base64.b64encode(buf.getvalue()).decode()
        del buf
        return b64
    except Exception as exc:
        print(f"[qwen]   render failed {Path(image_path).name}: {exc}")
        return ""


def _sanitize_bbox(raw_val: object, fallback: list[int]) -> list[int]:
    """
    Coerce a raw value to a valid [ymin, xmin, ymax, xmax] list in 0-1000 scale.
    Returns fallback if the value is missing or malformed.
    """
    try:
        coords = [int(round(float(v))) for v in raw_val]  # type: ignore[union-attr]
        if len(coords) == 4 and all(0 <= c <= 1000 for c in coords):
            ymin, xmin, ymax, xmax = coords
            if ymin < ymax and xmin < xmax:   # valid non-degenerate box
                return coords
    except Exception:
        pass
    return fallback


def _parse_slot_json(raw: str, slot_name: str) -> Optional[dict]:
    """
    Strip <think> blocks, extract JSON, validate required fields.
    Extracts spatial bounding boxes when present; falls back gracefully.
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
    required = {"approved_for_essay", "assigned_slot", "curator_justification"}
    if not required.issubset(obj):
        return None
    assigned = str(obj.get("assigned_slot", "Snapshot")).strip()
    if assigned not in {slot_name, "Snapshot"}:
        assigned = "Snapshot"

    # Extract spatial grounding — use full-frame defaults if absent or invalid
    _full = [0, 0, 1000, 1000]
    subject_bbox = _sanitize_bbox(obj.get("subject_bbox"), _full)
    anchor_bbox  = _sanitize_bbox(obj.get("anchor_bbox"),  _full)

    return {
        "approved_for_essay":    bool(obj["approved_for_essay"]),
        "assigned_slot":         assigned,
        "curator_justification": str(obj.get("curator_justification", "")).strip(),
        "subject_bbox":          subject_bbox,
        "subject_label":         str(obj.get("subject_label", "subject")).strip()[:60],
        "anchor_bbox":           anchor_bbox,
        "anchor_label":          str(obj.get("anchor_label",  "background")).strip()[:60],
    }


def _qwen_evaluate(image_path: str, slot_name: str, llm: object) -> Optional[dict]:
    """
    Evaluate one image for one slot via the pre-loaded llama-cpp Qwen 2.5-VL instance.
    Base64 is deleted immediately after the call.
    Returns parsed dict or None.
    """
    b64 = _render_image_b64(image_path)
    if not b64:
        return None

    result = None
    try:
        messages = [
            {"role": "system", "content": _SLOT_SYSTEM_PROMPTS[slot_name]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": _EVAL_PROMPT},
                ],
            },
        ]
        resp = llm.create_chat_completion(  # type: ignore[union-attr]
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        raw    = resp["choices"][0]["message"]["content"].strip()
        result = _parse_slot_json(raw, slot_name)
    except Exception as exc:
        print(f"[qwen/{slot_name}] {type(exc).__name__}: {exc}")
    finally:
        del b64
        gc.collect()

    return result


def qwen_stage(
    all_paths:  list[str],
    all_embs:   list[list[float]],
    text_embs:  dict[str, list[float]],
) -> list[SlotResult]:
    """
    For each story slot:
      1. CLIP ranks top-k candidates (unused paths only).
      2. Qwen 2.5-VL (llama-cpp-python, temperature=0.0) evaluates each candidate.
      3. If none approved, use the highest CLIP-scored candidate as fallback.

    The Qwen GGUF model is loaded once, used for all slots, then unloaded with purge_vram().
    Returns one SlotResult per slot, in _SLOT_NAMES order.
    """
    from llama_cpp import Llama
    try:
        from llama_cpp.llama_chat_format import Qwen2VLChatHandler as _ChatHandler
    except ImportError:
        from llama_cpp.llama_chat_format import Llava15ChatHandler as _ChatHandler  # type: ignore[assignment]

    if not _QWEN_GGUF.exists():
        raise FileNotFoundError(
            f"Qwen GGUF not found: {_QWEN_GGUF}\n"
            "Download qwen2.5-vl-2b-instruct-q4_k_m.gguf and place it in models/"
        )
    if not _QWEN_MMPROJ.exists():
        raise FileNotFoundError(
            f"Qwen mmproj not found: {_QWEN_MMPROJ}\n"
            "Download mmproj-qwen2.5-vl-2b-instruct-f16.gguf and place it in models/"
        )

    import os as _os_cpu
    _n_threads = min(_os_cpu.cpu_count() or 4, 8)
    print(f"[qwen] Loading {_QWEN_GGUF.name} via llama-cpp-python  threads={_n_threads} ...")
    _chat_handler = _ChatHandler(clip_model_path=str(_QWEN_MMPROJ))
    _llm = Llama(
        model_path=str(_QWEN_GGUF),
        chat_handler=_chat_handler,
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=_n_threads,
        verbose=False,
    )
    print("[qwen] Model loaded.")

    results:      list[SlotResult] = []
    already_used: set[str]         = set()

    try:
      for slot_name in _SLOT_NAMES:
        print(f"\n[stage2] --- Slot: {slot_name} ---")
        candidates = _rank_for_slot(
            slot_name, all_paths, all_embs, text_embs, already_used,
            top_k=_CANDIDATES_PER_SLOT,
        )
        if not candidates:
            print(f"[stage2] WARNING: no candidates left for {slot_name}")
            continue

        print(f"[stage2] Evaluating top-{len(candidates)} CLIP matches ...")
        approved:    Optional[dict] = None
        best_result: Optional[dict] = None

        for clip_score, path in candidates:
            print(f"  [qwen] {Path(path).name} (clip={clip_score:+.4f}) ...", end=" ", flush=True)
            qr = _qwen_evaluate(path, slot_name, _llm)

            if qr is not None:
                icon = "APPROVED" if qr["approved_for_essay"] else "REJECTED"
                print(f"{icon} ({qr['assigned_slot']})")
                if qr["approved_for_essay"] and approved is None:
                    approved = {**qr, "_path": path, "_clip_score": clip_score}
                    break
                if best_result is None:
                    best_result = {**qr, "_path": path, "_clip_score": clip_score}
            else:
                print("UNAVAILABLE")
                if best_result is None:
                    best_result = {
                        "_path":             path,
                        "_clip_score":       clip_score,
                        "approved_for_essay": False,
                        "assigned_slot":     "Snapshot",
                        "curator_justification":
                            "Qwen 2.5-VL unavailable -- highest CLIP-scored candidate used as fallback.",
                    }

        chosen = approved or best_result or {
            "_path":             candidates[0][1],
            "_clip_score":       candidates[0][0],
            "approved_for_essay": False,
            "assigned_slot":     slot_name,
            "curator_justification":
                "No Qwen evaluation completed -- using top CLIP match as fallback.",
        }

        already_used.add(chosen["_path"])
        results.append(SlotResult(
            slot_name             = slot_name,
            image_path            = chosen["_path"],
            clip_score            = chosen["_clip_score"],
            approved_by_vision    = bool(chosen["approved_for_essay"]),
            assigned_slot         = slot_name if chosen["approved_for_essay"] else "Snapshot",
            curator_justification = chosen["curator_justification"],
            subject_bbox          = chosen.get("subject_bbox",  [0, 0, 1000, 1000]),
            subject_label         = chosen.get("subject_label", "subject"),
            anchor_bbox           = chosen.get("anchor_bbox",   [0, 0, 1000, 1000]),
            anchor_label          = chosen.get("anchor_label",  "background"),
        ))
        status = "APPROVED" if chosen["approved_for_essay"] else "FALLBACK"
        print(f"[stage2] {status}: {Path(chosen['_path']).name}")

    finally:
        del _llm, _chat_handler
        _purge_vram()
        print("[qwen] Model unloaded + VRAM purged.")

    return results


# =============================================================================
# Stage 3a: TSP luminance reorder
# =============================================================================

def _mean_luminance(path: str) -> float:
    """Mean luminance [0,1] via 64x64 grayscale PIL thumbnail. No pixel data retained."""
    try:
        from PIL import Image
        with Image.open(path) as raw:
            img = raw.convert("L")
        img.thumbnail((64, 64))
        lum = float(np.asarray(img, dtype=np.float32).mean() / 255.0)
        del img
        return lum
    except Exception:
        return 0.5


def tsp_luminance_reorder(slot_results: list[SlotResult]) -> list[SlotResult]:
    """
    TSP-style luminance smoothing over the selected slots.

    Constraints:
      - "Opener"            is fixed at position 0
      - "Closer/Resolution" is fixed at the last position
      - Middle slots are permuted to minimise max adjacent luminance delta

    Returns the reordered list.
    """
    n = len(slot_results)
    if n <= 2:
        return slot_results

    luminances = [_mean_luminance(r.image_path) for r in slot_results]
    for r, lum in zip(slot_results, luminances):
        r.luminance = lum

    opener_pos = next(
        (i for i, r in enumerate(slot_results) if r.slot_name == "Opener"), 0
    )
    closer_pos = next(
        (i for i, r in enumerate(slot_results) if r.slot_name == "Closer/Resolution"), n - 1
    )

    fixed_front = [opener_pos]
    fixed_back  = [closer_pos]
    free_slots  = [i for i in range(n) if i not in (opener_pos, closer_pos)]

    def _max_delta(order: list[int]) -> float:
        return max(
            abs(luminances[order[j]] - luminances[order[j + 1]])
            for j in range(len(order) - 1)
        ) if len(order) > 1 else 0.0

    best_free  = free_slots[:]
    best_delta = _max_delta(fixed_front + free_slots + fixed_back)

    if len(free_slots) <= 7:
        for perm in itertools.permutations(free_slots):
            delta = _max_delta(fixed_front + list(perm) + fixed_back)
            if delta < best_delta:
                best_delta = delta
                best_free  = list(perm)

    final_order = fixed_front + best_free + fixed_back
    print(f"[tsp] luminance reorder: best max delta-lum = {best_delta:.3f}")
    for j, idx in enumerate(final_order):
        print(f"  slot {j}: {slot_results[idx].slot_name} — lum={luminances[idx]:.3f}")

    return [slot_results[i] for i in final_order]


# =============================================================================
# Stage 3b: Spatial analysis + grounded Judge's Verdict + validation
# =============================================================================

def _derive_spatial_facts(r: SlotResult) -> dict:
    """
    Derive human-readable compositional facts from a SlotResult's bounding boxes.
    All coordinates are in 0-1000 scale, format [ymin, xmin, ymax, xmax].

    Returns a dict of facts to be injected into the R1 prompt as hard evidence.
    """
    s_ymin, s_xmin, s_ymax, s_xmax = r.subject_bbox
    a_ymin, a_xmin, a_ymax, a_xmax = r.anchor_bbox

    # Centers
    s_cx = (s_xmin + s_xmax) / 2.0
    s_cy = (s_ymin + s_ymax) / 2.0
    a_cx = (a_xmin + a_xmax) / 2.0

    # Horizontal position buckets (thirds)
    def _h_pos(cx: float) -> str:
        if cx < 333:   return "left-third"
        if cx < 667:   return "center"
        return "right-third"

    def _v_pos(cy: float) -> str:
        if cy < 333:   return "upper-third"
        if cy < 667:   return "middle"
        return "lower-third"

    # Horizontal gap: positive = subject xmin is to the right of anchor xmax
    h_gap = int(s_xmin - a_xmax)
    # Vertical gap: positive = subject ymin is below anchor ymax
    v_gap = int(s_ymin - a_ymax)

    # Subject area as percentage of frame
    s_area_pct = round((s_ymax - s_ymin) * (s_xmax - s_xmin) / 10000.0, 1)

    # Pre-computed compositional claim the narrative MUST use
    spatial_claim: str
    _h_thresh = _CFG["H_CONTRAST_THRESH"]
    _a_thresh = _CFG["DOMINANT_AREA_PCT"]

    if abs(h_gap) > _h_thresh:
        side      = "right" if h_gap > 0 else "left"
        opp_side  = "left"  if h_gap > 0 else "right"
        spatial_claim = (
            f"The {r.subject_label or 'subject'} ({s_xmin}-{s_xmax} on x-axis) "
            f"is displaced to the {side} of the frame, while the "
            f"{r.anchor_label or 'structural anchor'} occupies the {opp_side} zone "
            f"({a_xmin}-{a_xmax} on x-axis), creating horizontal axis contrast "
            f"across a gap of {abs(h_gap)} coordinate units."
        )
    elif abs(v_gap) > 200:
        pos       = "below" if v_gap > 0 else "above"
        spatial_claim = (
            f"The {r.subject_label or 'subject'} sits {pos} the "
            f"{r.anchor_label or 'structural anchor'} — a vertical separation "
            f"of {abs(v_gap)} coordinate units divides the frame."
        )
    elif s_area_pct > _a_thresh:
        spatial_claim = (
            f"The {r.subject_label or 'subject'} occupies {s_area_pct}% of the frame "
            f"({s_xmin},{s_ymin} to {s_xmax},{s_ymax}), dominating the composition."
        )
    else:
        spatial_claim = (
            f"The {r.subject_label or 'subject'} occupies {s_area_pct}% of the frame "
            f"at the {_h_pos(s_cx)}, with the "
            f"{r.anchor_label or 'structural anchor'} at the {_h_pos(a_cx)}."
        )

    return {
        "subject_h_pos":    _h_pos(s_cx),
        "subject_v_pos":    _v_pos(s_cy),
        "anchor_h_pos":     _h_pos(a_cx),
        "h_gap":            h_gap,
        "v_gap":            v_gap,
        "subject_area_pct": s_area_pct,
        "horizontal_contrast": abs(h_gap) > _h_thresh,
        "subject_dominant":    s_area_pct > _a_thresh,
        "spatial_claim":    spatial_claim,
    }


def generate_judges_verdict(
    slot_results: list[SlotResult],
    style_prompt: str,
) -> str:
    """
    Generate the Judge's Verdict via DeepSeek-R1:8b (Ollama).

    The prompt is coordinate-grounded: R1 receives a structured data packet
    containing Qwen's extracted bounding boxes, derived spatial facts, and
    per-image luminance values.  Generic phrases not backed by numerical
    evidence are explicitly forbidden.

    Returns verdict text, or empty string if the model is unavailable.
    """
    import urllib.request

    # Build grounded data packet
    lum_values = [r.luminance for r in slot_results]
    lum_range  = round(max(lum_values) - min(lum_values), 3) if lum_values else 0.0

    packet: list[dict] = []
    for i, r in enumerate(slot_results):
        facts = _derive_spatial_facts(r)
        packet.append({
            "sequence_position":  i + 1,
            "slot":               r.slot_name,
            "filename":           Path(r.image_path).name,
            "vision_approved":    r.approved_by_vision,
            "qwen_justification": r.curator_justification,
            "subject_label":      r.subject_label,
            "subject_bbox":       r.subject_bbox,
            "anchor_label":       r.anchor_label,
            "anchor_bbox":        r.anchor_bbox,
            "luminance":          round(r.luminance, 3),
            **facts,
        })

    # Identify forced spatial claims from grounded evidence
    forced_claims = [e["spatial_claim"] for e in packet if e.get("spatial_claim")]
    forced_block  = "\n".join(f"  - {c}" for c in forced_claims)

    # Determine which generic phrases are forbidden vs permitted
    anchor_labels_str = " ".join(e.get("anchor_label", "") for e in packet).lower()
    line_anchors = any(kw in anchor_labels_str for kw in
                       ("road", "rail", "pipe", "line", "path", "street"))
    drama_light  = lum_range > 0.30

    forbidden_block = (
        "  - 'powerful composition' (always forbidden -- generic)\n"
        + ("" if line_anchors else
           "  - 'leading lines' (no line-type anchor detected in bbox data)\n")
        + ("" if drama_light else
           "  - 'dynamic lighting' / 'dramatic lighting' "
           f"(luminance range {lum_range:.3f} is below the 0.30 threshold)\n")
        + "  - 'creates tension' (only permitted if you cite a specific h_gap value)\n"
        + "  - 'draws the eye' (only permitted if you cite specific coordinate evidence)"
    )

    prompt = (
        "=== COORDINATE-GROUNDED NARRATIVE PROTOCOL ===\n"
        "You are a documentary photography curator. "
        "You have been given precise numerical spatial data extracted by a vision model.\n\n"
        "STRICT RULES:\n"
        "1. Every compositional claim must be derivable from the numerical data below.\n"
        "2. FORBIDDEN phrases:\n"
        f"{forbidden_block}\n"
        "3. MANDATORY: Use each of the following pre-computed spatial claims verbatim "
        "or closely paraphrased -- they are factually grounded:\n"
        f"{forced_block}\n"
        f"4. If h_gap for any slot is > {int(_CFG['H_CONTRAST_THRESH'])} "
        f"or < -{int(_CFG['H_CONTRAST_THRESH'])}, "
        "you MUST describe horizontal axis contrast using explicit left/right language.\n"
        f"5. If subject_area_pct > {int(_CFG['DOMINANT_AREA_PCT'])} for any slot, "
        "you MUST describe subject presence or frame dominance.\n\n"
        f"Style brief: '{style_prompt[:150]}'\n\n"
        "Spatial data packet (Qwen-extracted bboxes, format [ymin,xmin,ymax,xmax] 0-1000):\n"
        f"{json.dumps(packet, indent=2)}\n\n"
        "Write 2-3 sentences. Use <think> to verify each claim against the data. "
        "After </think>, write only the final verdict. Keep verdict under 120 words."
    )

    try:
        payload = json.dumps({
            "model":   _DIRECTOR_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.1, "num_predict": 350},
        }).encode()
        req = urllib.request.Request(
            _OLLAMA_GEN_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "").strip()

        m = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if raw:
            print(f"[verdict] Generated ({len(raw)} chars)")
            return raw
    except Exception as exc:
        print(f"[verdict] {_DIRECTOR_MODEL} unavailable: {exc}")

    return ""


# =============================================================================
# Validation: verify narrative mirrors numerical spatial data
# =============================================================================

def validate_narrative(
    verdict:      str,
    slot_results: list[SlotResult],
) -> dict:
    """
    Programmatic test loop: check that the generated verdict text mirrors
    the numerical spatial data extracted by Qwen.

    Checks:
      1. No forbidden generic phrases used without coordinate evidence
      2. Horizontal contrast language present when |h_gap| > 200
      3. Dominance language present when subject_area_pct > 35
      4. At least one positional anchor term present
      5. Luminance-claim accuracy (drama words forbidden if range <= 0.30)

    Returns a dict:
      checks  : {check_name: True|False}
      passed  : int
      total   : int
      grade   : "PASS" | "WARN" | "FAIL"
      details : list[str] of human-readable findings
    """
    if not verdict:
        return {
            "checks":  {},
            "passed":  0,
            "total":   1,
            "grade":   "FAIL",
            "details": ["No verdict text generated."],
        }

    vl = verdict.lower()
    checks:  dict[str, bool] = {}
    details: list[str]       = []

    lum_values = [r.luminance for r in slot_results]
    lum_range  = max(lum_values) - min(lum_values) if lum_values else 0.0

    anchor_labels = " ".join(r.anchor_label for r in slot_results).lower()
    line_anchors  = any(kw in anchor_labels for kw in
                        ("road", "rail", "pipe", "line", "path", "street"))

    # ── Check 1: 'powerful composition' is never permitted ────────────────────
    chk = "powerful_composition" not in vl
    checks["no_powerful_composition"] = chk
    details.append(
        f"[{'PASS' if chk else 'FAIL'}] 'powerful composition': "
        + ("not present" if chk else "FOUND -- forbidden generic phrase")
    )

    # ── Check 2: 'leading lines' only when line-type anchor present ───────────
    has_ll = "leading lines" in vl
    if has_ll and not line_anchors:
        checks["leading_lines_grounded"] = False
        details.append("[FAIL] 'leading lines' used but no road/rail/line anchor in bbox data")
    else:
        checks["leading_lines_grounded"] = True
        if has_ll:
            details.append("[PASS] 'leading lines' permitted (line-type anchor detected)")
        else:
            details.append("[PASS] 'leading lines' not used")

    # ── Check 3: luminance claims require evidence ────────────────────────────
    drama_words = any(w in vl for w in ("dynamic lighting", "dramatic lighting",
                                        "chiaroscuro", "dramatic contrast"))
    if drama_words and not (lum_range > 0.30):
        checks["luminance_claim_grounded"] = False
        details.append(
            f"[FAIL] Lighting drama claimed but luminance range={lum_range:.3f} "
            f"is below threshold 0.30"
        )
    else:
        checks["luminance_claim_grounded"] = True
        msg = f"(lum_range={lum_range:.3f}, threshold=0.30)"
        details.append(
            f"[PASS] Luminance claim check {msg}: "
            + ("drama words present with sufficient range" if drama_words
               else "no unjustified lighting claims")
        )

    # ── Check 4: horizontal contrast language when h_gap evidence exists ──────
    _h_thresh = _CFG["H_CONTRAST_THRESH"]
    h_contrast_slots = [r for r in slot_results
                        if abs(_derive_spatial_facts(r)["h_gap"]) > _h_thresh]
    if h_contrast_slots:
        pos_terms = any(w in vl for w in
                        ("left", "right", "edge", "side", "horizontal", "axis",
                         "pushed", "displaced", "anchored", "far ", "offset"))
        checks["h_contrast_language"] = pos_terms
        slot_names = [r.slot_name for r in h_contrast_slots]
        details.append(
            f"[{'PASS' if pos_terms else 'FAIL'}] H-contrast check "
            f"(slots with |h_gap|>{int(_h_thresh)}: {slot_names}): "
            + ("positional language present" if pos_terms
               else "MISSING -- verdict must describe horizontal axis contrast")
        )
    else:
        checks["h_contrast_language"] = True
        details.append(
            f"[PASS] H-contrast check: no slots with |h_gap|>{int(_h_thresh)} -- not required"
        )

    # ── Check 5: dominance language when subject fills >35% of frame ─────────
    dominant_slots = [r for r in slot_results
                      if _derive_spatial_facts(r)["subject_dominant"]]
    if dominant_slots:
        dom_terms = any(w in vl for w in
                        ("dominat", "fill", "occupi", "presence", "overwhelm",
                         "foreground", "central figure", "large"))
        checks["dominance_language"] = dom_terms
        slot_names = [r.slot_name for r in dominant_slots]
        details.append(
            f"[{'PASS' if dom_terms else 'FAIL'}] Dominance check "
            f"(slots with area>35%: {slot_names}): "
            + ("dominance language present" if dom_terms
               else "MISSING -- verdict must acknowledge subject frame dominance")
        )
    else:
        checks["dominance_language"] = True
        details.append("[PASS] Dominance check: no slots with subject_area_pct>35% -- not required")

    # ── Check 6: at least one positional anchor term ──────────────────────────
    any_pos = any(w in vl for w in
                  ("left", "right", "center", "top", "bottom", "edge", "corner",
                   "frame", "foreground", "background", "horizon"))
    checks["positional_language_present"] = any_pos
    details.append(
        f"[{'PASS' if any_pos else 'FAIL'}] Positional language: "
        + ("at least one spatial term found" if any_pos
           else "MISSING -- verdict contains no spatial positioning language")
    )

    passed = sum(1 for v in checks.values() if v)
    total  = len(checks)
    ratio  = passed / total if total else 0.0
    grade  = "PASS" if ratio >= 1.0 else ("WARN" if ratio >= 0.67 else "FAIL")

    return {
        "checks":  checks,
        "passed":  passed,
        "total":   total,
        "grade":   grade,
        "details": details,
    }


# =============================================================================
# Main pipeline orchestrator
# =============================================================================

def run_vision_story_mode(
    image_dir:    str = "./dataset_images",
    style_prompt: str = "high contrast black and white street documentary sequence",
    output:       str = _OUTPUT_FILE,
) -> list[dict]:
    """
    Run all four stages and write final_story_manifest.json.

    Returns the manifest as a list of dicts.
    """
    print()
    print("=" * 68)
    print(" VISION STORY MODE -- In-Memory Vision-Verified Pipeline")
    print("=" * 68)
    print(f" Image dir    : {Path(image_dir).resolve()}")
    print(f" Style brief  : {style_prompt}")
    print(f" Output       : {output}")
    print(f" Slots        : {', '.join(_SLOT_NAMES)}")
    print("=" * 68)

    # ── Stage 1: CLIP ─────────────────────────────────────────────────────────
    print("\n>> Stage 1: CLIP In-Memory Embedding")
    print("-" * 50)
    valid_paths, img_embs, text_embs = clip_stage(image_dir)
    print(f"[stage1] {len(valid_paths)} images ready for slot matching")

    # ── Stage 2: Qwen per-slot gatekeeping ────────────────────────────────────
    print("\n>> Stage 2: Qwen 2.5-VL Slot-Specific Gatekeeping  (temperature=0.0)")
    print("-" * 50)
    slot_results = qwen_stage(valid_paths, img_embs, text_embs)

    if not slot_results:
        print("[error] No slot results -- aborting.")
        return []

    # ── Stage 3a: TSP luminance reorder ───────────────────────────────────────
    print("\n>> Stage 3a: TSP Luminance Reorder")
    print("-" * 50)
    slot_results = tsp_luminance_reorder(slot_results)

    # ── Stage 3b: DeepSeek-R1 Judge's Verdict (coordinate-grounded) ─────────
    print("\n>> Stage 3b: DeepSeek-R1 Judge's Verdict  (grounded protocol)")
    print("-" * 50)
    verdict = generate_judges_verdict(slot_results, style_prompt)
    if verdict:
        print(f"\n  \"{verdict}\"\n")
    else:
        verdict = (
            "Vision-verified documentary sequence assembled across four narrative "
            "slots: Opener, Subject/Interaction, Detail/Accent, and Closer/Resolution. "
            "Judge's Verdict unavailable -- install deepseek-r1:8b via Ollama."
        )

    # ── Stage 3c: Validation loop ─────────────────────────────────────────────
    print(">> Stage 3c: Narrative Grounding Validation")
    print("-" * 50)
    validation = validate_narrative(verdict, slot_results)
    for line in validation["details"]:
        print(f"  {line}")
    grade_line = (
        f"\n  Validation: {validation['passed']}/{validation['total']} checks passed "
        f"-- {validation['grade']}"
    )
    print(grade_line)

    # ── Stage 4a: Eye Feature Overlay Rendering ───────────────────────────────
    print("\n>> Stage 4a: Eye Feature Overlay Rendering")
    print("-" * 50)
    overlay_map: dict[str, str] = {}
    try:
        from canvas_renderer import render_story_overlays
        overlay_results = render_story_overlays(slot_results)
        for ov in overlay_results:
            if ov.get("overlay_url"):
                overlay_map[ov["image_id"]] = ov["overlay_url"]
            print(
                f"  [{ov['pixel_check']}] {ov['image_id']}"
                + (f" -> {ov['overlay_url']}" if ov.get("overlay_url") else "")
            )
    except Exception as exc:
        print(f"[canvas] overlay rendering failed: {exc}")

    # ── Stage 4: Manifest assembly ────────────────────────────────────────────
    manifest: list[dict] = []
    for r in slot_results:
        facts    = _derive_spatial_facts(r)
        image_id = Path(r.image_path).stem
        entry: dict = {
            "image_path":            r.image_path,
            "assigned_slot":         r.assigned_slot,
            "approved_by_vision":    r.approved_by_vision,
            "curator_justification": r.curator_justification,
            "subject_bbox":          r.subject_bbox,
            "subject_label":         r.subject_label,
            "anchor_bbox":           r.anchor_bbox,
            "anchor_label":          r.anchor_label,
            "luminance":             round(r.luminance, 3),
            "spatial_facts":         facts,
            "eye_overlay_url":       overlay_map.get(image_id),
        }
        manifest.append(entry)

    out_path = Path(output)
    _payload = json.dumps(
        {
            "style_prompt":    style_prompt,
            "judges_verdict":  verdict,
            "validation":      {
                "grade":   validation["grade"],
                "passed":  validation["passed"],
                "total":   validation["total"],
                "details": validation["details"],
            },
            "sequence": manifest,
        },
        indent=2,
        ensure_ascii=False,
    )
    _tmp_path = out_path.with_suffix(".tmp")
    _tmp_path.write_text(_payload, encoding="utf-8")
    import os as _os
    _os.replace(str(_tmp_path), str(out_path))

    # ── Summary ───────────────────────────────────────────────────────────────
    approved_n = sum(1 for r in slot_results if r.approved_by_vision)
    print()
    print("=" * 68)
    print(f" Manifest saved    -> {out_path.resolve()}")
    print(f" Slots filled      : {len(slot_results)} / {len(_SLOT_NAMES)}")
    print(f" Vision-approved   : {approved_n} / {len(slot_results)}")
    print(f" Grounding grade   : {validation['grade']}  "
          f"({validation['passed']}/{validation['total']} checks)")
    print("=" * 68)
    print()
    print(" Sequence:")
    for i, r in enumerate(slot_results):
        marker = "[V]" if r.approved_by_vision else "[-]"
        print(f"  {marker} [{i+1}] {r.assigned_slot:<22} {Path(r.image_path).name}")
        print(f"          {r.curator_justification[:90]}")
    print()
    if verdict:
        print(f" Judge's Verdict:\n  {verdict}")
        print()
    print("=" * 68)
    print()

    return manifest


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    _image_dir    = sys.argv[1] if len(sys.argv) > 1 else "./dataset_images"
    _style_prompt = (
        sys.argv[2] if len(sys.argv) > 2
        else "high contrast black and white street documentary sequence"
    )
    run_vision_story_mode(
        image_dir=_image_dir,
        style_prompt=_style_prompt,
    )
