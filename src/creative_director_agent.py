"""
Creative Director Agent — Phi-4-mini-reasoning GGUF (Literal Judge, JSON Schema Mode)

Multi-Stage Architecture
────────────────────────
Stage 1 · Director Brief (Phi-4 Mini, CPU GGUF, JSON Schema Mode)
    Input : User style prompt
    Output: Strict JSON — exactly 4 keys, no prose allowed:
              thematic_niche           str
              spatial_progression_sequence  list[str]  (5 elements, one per slot)
              color_profile_target     str
              structural_role_matrix   {slot → role_descriptor}

    Grammar enforcement: llama-cpp LlamaGrammar.from_json_schema() blocks any
    token that would violate the schema at the generation level. Hard RuntimeError
    is raised if the raw output cannot be parsed into all 4 required keys.

Stage 2 · Code Engine (Python + pymoo)
    The DirectorBrief is converted to boundary_params (per-slot thresholds,
    similarity limit, color emphasis) and passed directly into the pymoo
    ElementwiseProblem — no LLM involvement in candidate selection.
    Cluster deduplication is enforced via Python set operations.

Candidate Tokenization
──────────────────────
Before building the manifest, all image filenames, system directory paths, and
reasoning_log strings are stripped. Each candidate is mapped to a stable token
("IMG_01", "IMG_02", …) paired with only its numerical vision metrics:
    [Aesthetic_V2_5, MUSIQ_Sharpness, Depth_Class, Cluster_ID]

The token→path map is held in memory for result resolution; it is never sent
to the LLM context.

DeepSeek-R1 Verdict Safeguard
──────────────────────────────
The 8B Judge output is parsed with:
    re.search(r'</think>\\s*(.*)', response, re.DOTALL)
Everything before and including </think> is discarded. Only the clean
post-think string is stored in LanceDB / returned to the frontend.

Model paths
───────────
    models/phi4-mini-reasoning-q4.gguf   (Phi-4 Mini, CPU, Q4_K_M)
    models/deepseek-r1-8b-q5.gguf       (8B Judge, GPU, Q5_K_M)
"""
from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_GGUF_PATH       = Path("models/phi4-mini-reasoning-q4.gguf")
_JUDGE_GGUF_PATH = Path("models/deepseek-r1-8b-q5.gguf")

_EMPTY_KW    = {"empty", "liminal", "desert", "void", "abandoned", "desolate", "minimal", "sparse"}
_GEOMETRIC_KW = {"geometric", "architecture", "pattern", "grid", "lines", "symmetry", "abstract"}
_WARM_KW      = {"golden", "sunset", "sunrise", "warm", "orange", "amber"}
_DARK_KW      = {"night", "dark", "neon", "low-key", "shadow"}
_FLAT_KW      = {"overcast", "flat", "grey", "gray", "fog", "mist", "rain"}
_CONTRAST_KW  = {"harsh", "contrast", "dramatic", "midday", "silhouette"}


# ── Stage 1 Director Brief JSON Schema ───────────────────────────────────────
# Enforced at the token level via llama-cpp LlamaGrammar. No prose outside the
# 4-key boundary is ever generated or reaches the database writer.

_DIRECTOR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "thematic_niche": {
            "type": "string",
            "description": "One-phrase essence of the photographic theme"
        },
        "spatial_progression_sequence": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 5,
            "description": "Five spatial directives in slot order: Opener, Subject, Contrast, Detail, Closer"
        },
        "color_profile_target": {
            "type": "string",
            "description": "Concise tonal / color instruction, e.g. 'cool desaturated blues'"
        },
        "structural_role_matrix": {
            "type": "object",
            "properties": {
                "Opener":   {"type": "string"},
                "Subject":  {"type": "string"},
                "Contrast": {"type": "string"},
                "Detail":   {"type": "string"},
                "Closer":   {"type": "string"},
            },
            "required": ["Opener", "Subject", "Contrast", "Detail", "Closer"],
            "additionalProperties": False,
        },
    },
    "required": [
        "thematic_niche",
        "spatial_progression_sequence",
        "color_profile_target",
        "structural_role_matrix",
    ],
    "additionalProperties": False,
}

_STAGE1_PROMPT = (
    "You are a Literal Competition Judge for a street photography selection system.\n\n"
    "TASK: Translate the Style Brief into a strict structural constraint schema.\n"
    "Output ONLY a JSON object with these exact 4 keys:\n"
    "  thematic_niche            — one-phrase photographic theme\n"
    "  spatial_progression_sequence — array of EXACTLY 5 spatial directives for slots:\n"
    "                                 [Opener, Subject, Contrast, Detail, Closer]\n"
    "  color_profile_target      — concise tonal/color instruction\n"
    "  structural_role_matrix    — object with exactly 5 slot keys (Opener/Subject/Contrast/Detail/Closer),\n"
    "                              each mapped to a role descriptor string\n\n"
    "NO conversational prose. NO explanation. ONLY the JSON object.\n\n"
    'Style Brief: "{brief}"\n\nJSON:'
)

# Legacy rule-set prompt kept for internal YOLO gate only (not sent to sequence selection)
_RULE_SET_PROMPT = (
    "You are a Literal Competition Judge for a street photography selection system.\n\n"
    "LITERAL MODE: Treat the Style Brief as Boolean Constraints.\n"
    "- If the brief contains ANY of: empty, liminal, void, abandoned, desolate, minimal, "
    "no people, desert — set HARD_FILTER_PEOPLE to true.\n\n"
    "Output ONLY a JSON object with these exact fields:\n"
    "- HARD_FILTER_PEOPLE: true if brief implies empty/no-people scenes, else false\n"
    "- GEOMETRIC_PRIORITY: \"High\" | \"Medium\" | \"Low\"\n"
    "- LIGHTING_MOOD: one short phrase\n"
    "- BRIEF_KEYWORDS: array of 2-4 constraint words\n\n"
    'Style Brief: "{brief}"\n\nJSON:'
)

_VERDICT_PROMPT = (
    "You are a Competition Judge writing an official Verdict for a street photography portfolio.\n\n"
    'Style Brief: "{brief}"\n'
    "Photographic Theme: {thematic_niche}\n"
    "Target Tone/Color: {color_profile}\n\n"
    "Write a 2-3 sentence Judge's Verdict that addresses:\n"
    "1. How the sequence adheres to the brief's thematic and tonal constraints.\n"
    "2. The visual arc and progression across the 5 selected positions.\n"
    "3. Why this selection earns its place in the final portfolio.\n\n"
    "Selected Sequence (5 positions):\n{sequence}\n\nVerdict:"
)

_RATIONALE_PROMPT = (
    "You are the head photo editor. Review this finalized {n}-image sequence.\n"
    "Based on the scoring metrics and slot roles below, generate exactly one concise sentence\n"
    "explaining why each image was chosen for its specific slot.\n"
    "Return ONLY a JSON object mapping each image token to its rationale string. No prose outside the JSON.\n\n"
    "Sequence:\n{sequence}\n\nJSON:"
)

_llama: Optional[object] = None


# ── DirectorBrief dataclass ───────────────────────────────────────────────────

@dataclass
class DirectorBrief:
    thematic_niche:               str
    spatial_progression_sequence: list[str]
    color_profile_target:         str
    structural_role_matrix:       dict[str, str]

    # Derived fields — populated by extract_nsga3_boundary_params()
    boundary_params: dict = field(default_factory=dict)

    def __post_init__(self):
        # Validate 5-slot completeness
        required_slots = {"Opener", "Subject", "Contrast", "Detail", "Closer"}
        missing = required_slots - set(self.structural_role_matrix.keys())
        if missing:
            raise ValueError(f"DirectorBrief: structural_role_matrix missing slots: {missing}")
        if len(self.spatial_progression_sequence) != 5:
            raise ValueError(
                f"DirectorBrief: spatial_progression_sequence must have 5 elements, "
                f"got {len(self.spatial_progression_sequence)}"
            )


# ── Rationale schema builder ─────────────────────────────────────────────────

def _build_rationale_schema(tokens: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {t: {"type": "string"} for t in tokens},
        "required": tokens,
        "additionalProperties": False,
    }


# ── GGUF loader ───────────────────────────────────────────────────────────────

def _load_gguf() -> Optional[object]:
    global _llama
    if _llama is not None:
        return _llama
    if not _GGUF_PATH.exists():
        print(f"[agent] GGUF not found at {_GGUF_PATH} — keyword fallback active")
        return None
    try:
        from llama_cpp import Llama
        _llama = Llama(
            model_path=str(_GGUF_PATH),
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("[agent] Phi-4-mini-reasoning GGUF loaded (CPU, n_ctx=4096)")
        return _llama
    except ImportError:
        print("[agent] llama-cpp-python not installed — keyword fallback active")
    except Exception as e:
        print(f"[agent] GGUF load failed ({e}) — keyword fallback active")
    return None


def _build_grammar():
    """Return a LlamaGrammar for _DIRECTOR_SCHEMA, or None if llama_cpp is old."""
    try:
        from llama_cpp import LlamaGrammar
        g = LlamaGrammar.from_json_schema(json.dumps(_DIRECTOR_SCHEMA))
        return g
    except (ImportError, AttributeError, Exception):
        return None


# ── Keyword fallback helpers ──────────────────────────────────────────────────

def _keyword_rule_set(brief: str) -> dict:
    text = brief.lower()
    hard_filter  = any(kw in text for kw in _EMPTY_KW)
    geo_hits     = sum(1 for kw in _GEOMETRIC_KW if kw in text)
    geo_priority = "High" if geo_hits >= 2 else ("Medium" if geo_hits == 1 else "Low")
    if any(kw in text for kw in _WARM_KW):
        mood = "warm golden"
    elif any(kw in text for kw in _DARK_KW):
        mood = "low-key night"
    elif any(kw in text for kw in _FLAT_KW):
        mood = "flat overcast"
    elif any(kw in text for kw in _CONTRAST_KW):
        mood = "high contrast"
    else:
        mood = "natural ambient"
    kws = [kw for kw in (_EMPTY_KW | _GEOMETRIC_KW) if kw in text][:4]
    if not kws:
        kws = text.split()[:3]
    return {
        "HARD_FILTER_PEOPLE": hard_filter,
        "GEOMETRIC_PRIORITY": geo_priority,
        "LIGHTING_MOOD":      mood,
        "BRIEF_KEYWORDS":     kws,
    }


def _keyword_director_brief(brief: str) -> DirectorBrief:
    """Heuristic DirectorBrief when GGUF is absent."""
    rs    = _keyword_rule_set(brief)
    mood  = rs["LIGHTING_MOOD"]
    kws   = rs["BRIEF_KEYWORDS"]
    theme = " ".join(kws) if kws else "street photography"
    return DirectorBrief(
        thematic_niche               = theme,
        spatial_progression_sequence = [
            "Wide establishing shot",
            "Decisive moment focal point",
            "Dramatic tonal contrast",
            "Intimate texture detail",
            "Receding vanishing perspective",
        ],
        color_profile_target         = mood,
        structural_role_matrix       = {
            "Opener":   "Wide/Scale — cityscape or environment",
            "Subject":  "Focal Point — human element or visual anchor",
            "Contrast": "Luminance shift — dramatic tonal jump",
            "Detail":   "Macro/Texture — intimate close-up",
            "Closer":   "Vanishing/Finality — resolution and closure",
        },
    )


# ── Stage 1: Director Brief generation ───────────────────────────────────────

def generate_director_brief(brief: str) -> DirectorBrief:
    """
    Stage 1: Parse the user's style prompt and return a strict DirectorBrief.

    Phi-4 Mini uses JSON Schema grammar mode — any token that would produce
    prose outside the 4-key boundary is blocked at the generation level.
    RuntimeError is raised if the parsed output is missing any required key
    or fails DirectorBrief validation.

    Falls back to keyword heuristics when GGUF is absent.
    """
    if not (brief or "").strip():
        return _keyword_director_brief("")

    llm = _load_gguf()
    if llm is not None:
        grammar = _build_grammar()
        try:
            prompt = _STAGE1_PROMPT.format(brief=brief[:300])
            kwargs = dict(max_tokens=512, temperature=0.0, echo=False)
            if grammar is not None:
                kwargs["grammar"] = grammar
            out = llm(prompt, **kwargs)
            raw = out["choices"][0]["text"].strip()

            # Hard validate: strip any think-tag prefix, then parse JSON
            m = re.search(r'</think>\s*(.*)', raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start < 0 or end <= start:
                raise RuntimeError(
                    f"[agent] Stage 1: No JSON object found in Phi-4 output — "
                    f"prose outside schema boundary. Raw: {raw[:120]!r}"
                )

            parsed = json.loads(raw[start:end])

            # Hard-validate all 4 required keys
            missing = [
                k for k in ("thematic_niche", "spatial_progression_sequence",
                             "color_profile_target", "structural_role_matrix")
                if k not in parsed
            ]
            if missing:
                raise RuntimeError(
                    f"[agent] Stage 1: DirectorBrief schema violation — missing keys: {missing}. "
                    f"Terminating to prevent hallucinated sequence selection."
                )

            db = DirectorBrief(
                thematic_niche               = str(parsed["thematic_niche"]),
                spatial_progression_sequence = [str(x) for x in parsed["spatial_progression_sequence"]],
                color_profile_target         = str(parsed["color_profile_target"]),
                structural_role_matrix       = {
                    k: str(v) for k, v in parsed["structural_role_matrix"].items()
                },
            )
            print(f"[agent] DirectorBrief (Phi-4 JSON Schema): niche='{db.thematic_niche}' "
                  f"color='{db.color_profile_target}'")
            return db

        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            print(f"[agent] Stage 1 GGUF inference failed ({e}) — keyword fallback")

    db = _keyword_director_brief(brief)
    print(f"[agent] DirectorBrief (keywords): niche='{db.thematic_niche}'")
    return db


# ── Legacy rule_set (YOLO gate only) ─────────────────────────────────────────

def generate_rule_set(brief: str) -> dict:
    """
    Returns the HARD_FILTER_PEOPLE / GEOMETRIC_PRIORITY rule set used exclusively
    by the YOLO gate. NOT used for sequence selection — that is Stage 2 only.
    """
    if not (brief or "").strip():
        return {
            "HARD_FILTER_PEOPLE": False,
            "GEOMETRIC_PRIORITY": "Low",
            "LIGHTING_MOOD":      "natural ambient",
            "BRIEF_KEYWORDS":     [],
        }

    llm = _load_gguf()
    if llm is not None:
        try:
            prompt = _RULE_SET_PROMPT.format(brief=brief[:200])
            out    = llm(prompt, max_tokens=160, temperature=0.0, echo=False)
            raw    = out["choices"][0]["text"].strip()
            m = re.search(r'</think>\s*(.*)', raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                rs = {
                    "HARD_FILTER_PEOPLE": bool(parsed.get("HARD_FILTER_PEOPLE", False)),
                    "GEOMETRIC_PRIORITY": str(parsed.get("GEOMETRIC_PRIORITY", "Low")),
                    "LIGHTING_MOOD":      str(parsed.get("LIGHTING_MOOD", "natural ambient")),
                    "BRIEF_KEYWORDS":     list(parsed.get("BRIEF_KEYWORDS", [])),
                }
                print(f"[agent] Rule set (Phi-4 Literal Judge): {rs}")
                return rs
        except Exception as e:
            print(f"[agent] Rule set GGUF failed ({e}) — keyword fallback")

    rs = _keyword_rule_set(brief)
    print(f"[agent] Rule set (keywords): {rs}")
    return rs


# ── Candidate tokenization ────────────────────────────────────────────────────

def tokenize_candidates(
    candidates: list[dict],
) -> tuple[dict[str, str], list[dict]]:
    """
    Strip all image filenames, system paths, and reasoning_log strings from the
    candidate pool before any LLM interaction.

    Maps each candidate to a stable token key ("IMG_01", "IMG_02", …) paired
    only with its numerical vision metrics:
        Aesthetic_V2_5  — from breakdown["AestheticV25"] or score
        MUSIQ_Sharpness — from breakdown["Technical"]
        Depth_Class     — 0=unknown 1=foreground 2=midground 3=background
        Cluster_ID      — -1 if unique

    Returns:
        token_map     dict[token → original_path]   (never sent to LLM)
        tokenized     list of sanitized dicts for LLM context
    """
    token_map:  dict[str, str] = {}
    tokenized:  list[dict]     = []

    for i, c in enumerate(candidates):
        token = f"IMG_{i+1:02d}"
        token_map[token] = c.get("path", "")

        bd = c.get("breakdown", {}) or {}

        # Numerical metrics only — no strings, no paths, no free-form text
        # Explicit None checks: 0.0 is a valid score and must not fall through via 'or'
        _aes_raw = bd.get("Technical") if bd.get("Technical") is not None else bd.get("aesthetic")
        aes   = float(_aes_raw if _aes_raw is not None else c.get("score", 0.5))
        musiq = float(bd.get("Technical") if bd.get("Technical") is not None else c.get("score", 0.5))
        depth = int(c.get("depth_class", 0))
        cid   = int(c.get("cluster_id", -1) or -1)

        tokenized.append({
            "token":           token,
            "Aesthetic_V2_5":  round(aes,   3),
            "MUSIQ_Sharpness": round(musiq, 3),
            "Depth_Class":     depth,
            "Cluster_ID":      cid,
            # Internal fields kept for Stage 2 (not sent to LLM)
            "_score":          float(c.get("score", 0.5)),
            "_breakdown":      {k: float(v) for k, v in bd.items()
                                if isinstance(v, (int, float))},
            "_yolo_blocked":   bool(c.get("yolo_blocked", False)),
        })

    return token_map, tokenized


# ── Stage 2 boundary parameters ──────────────────────────────────────────────

def extract_nsga3_boundary_params(brief: DirectorBrief) -> dict:
    """
    Convert a DirectorBrief into numerical boundary parameters for the pymoo
    ElementwiseProblem. All logic here is Python array operations — no LLM.

    Returns:
        slot_thresholds  list[float]  — per-slot minimum role fitness (length 5)
        sim_limit        float        — maximum allowed pairwise cosine similarity
        color_emphasis   str          — "warm" | "cool" | "low_key" | "high_key" | "neutral"
        require_people   bool         — False = YOLO hard filter active
    """
    slot_names = ["Opener", "Subject", "Contrast", "Detail", "Closer"]

    # Per-slot threshold from structural_role_matrix descriptions
    # Slots with more specific role descriptors get higher thresholds
    slot_thresholds: list[float] = []
    for slot in slot_names:
        desc = brief.structural_role_matrix.get(slot, "").lower()
        if len(desc) > 40:
            thresh = 0.30   # detailed constraint → stricter fitness gate
        elif len(desc) > 20:
            thresh = 0.25
        else:
            thresh = 0.20   # minimal constraint → permissive
        slot_thresholds.append(thresh)

    # Color emphasis from color_profile_target
    cpt = brief.color_profile_target.lower()
    if any(w in cpt for w in ("warm", "golden", "amber", "orange")):
        color_emphasis = "warm"
    elif any(w in cpt for w in ("cool", "blue", "cold", "cyan")):
        color_emphasis = "cool"
    elif any(w in cpt for w in ("dark", "shadow", "low-key", "night", "noir")):
        color_emphasis = "low_key"
    elif any(w in cpt for w in ("bright", "high-key", "white", "blown")):
        color_emphasis = "high_key"
    else:
        color_emphasis = "neutral"

    # Tighter similarity limit for empty/abstract themes (near-duplicates more noticeable)
    niche = brief.thematic_niche.lower()
    sim_limit = 0.78 if any(w in niche for w in ("empty", "minimal", "abstract", "liminal")) else 0.82

    require_people = not any(
        w in niche for w in ("empty", "no people", "liminal", "void", "abandoned")
    )

    return {
        "slot_thresholds": slot_thresholds,
        "sim_limit":       sim_limit,
        "color_emphasis":  color_emphasis,
        "require_people":  require_people,
    }


# ── Stage 2 candidate selection (Python only — no LLM) ───────────────────────

def select_sequence_from_batch(
    candidates: list[dict],
    n_target: int,
    style_prompt: str,
    rule_set: dict,
) -> list[int]:
    """
    Legacy shim — Stage 2 candidate selection is now handled entirely in
    nsga3_sequencer.py via Python array operations.

    Returns [] to signal to the caller that NSGA-III should be used directly.
    The LLM is no longer involved in slot selection.
    """
    print("[agent] select_sequence_from_batch: Stage 2 delegated to NSGA-III Code Engine")
    return []


# ── DeepSeek-R1 Judge's Verdict ───────────────────────────────────────────────

def generate_judges_verdict_8b(
    selected_images: list[dict],
    style_prompt: str,
    roles: list[str],
    director_brief: Optional["DirectorBrief"] = None,
    scores: Optional[list[float]] = None,
) -> str:
    """
    Use DeepSeek-R1-Distill-Llama-8B Q5_K_M GGUF (GPU-offloaded) to write the
    official Judge's Verdict for the chosen story sequence.

    Output safeguard: re.search(r'</think>\\s*(.*)', response, re.DOTALL)
    strips everything before and including the closing think tag.  Only the
    clean downstream string is returned and stored in LanceDB.

    Returns empty string if GGUF absent or llama-cpp unavailable.
    """
    if not _JUDGE_GGUF_PATH.exists():
        print(f"[agent] 8B Judge GGUF not found at {_JUDGE_GGUF_PATH} — verdict skipped")
        return ""

    try:
        from llama_cpp import Llama

        seq_lines: list[str] = []
        for i, (img, role) in enumerate(zip(selected_images, roles)):
            # Aspects are stored as top-level keys, not nested under "breakdown"
            bd = img.get("breakdown", {}) or {}
            co = int(float(img.get("Composition",   bd.get("Composition",   0.5))) * 100)
            li = int(float(img.get("Lighting",      bd.get("Lighting",      0.5))) * 100)
            na = int(float(img.get("Narrative",     bd.get("Narrative",     0.5))) * 100)
            hc = int(float(img.get("Human/Culture", bd.get("Human/Culture", 0.5))) * 100)
            sc = int(float((scores[i] if scores and i < len(scores) else 0.5)) * 100)
            # Do NOT include filename or path in verdict prompt
            seq_lines.append(
                f"{i+1}. [{role.upper()}] Score:{sc}% Comp:{co}% Light:{li}% Narr:{na}% HC:{hc}%"
            )

        thematic_niche = (
            director_brief.thematic_niche if director_brief else "street photography"
        )
        color_profile = (
            director_brief.color_profile_target if director_brief else "natural ambient"
        )

        prompt = _VERDICT_PROMPT.format(
            brief          = style_prompt[:300],
            thematic_niche = thematic_niche,
            color_profile  = color_profile,
            sequence       = "\n".join(seq_lines),
        )

        judge_llm = Llama(
            model_path   = str(_JUDGE_GGUF_PATH),
            n_ctx        = 2048,
            n_gpu_layers = -1,
            n_threads    = 2,
            verbose      = False,
        )
        out  = judge_llm(prompt, max_tokens=400, temperature=0.4, echo=False)
        raw  = out["choices"][0]["text"].strip()

        # ── DeepSeek-R1 think-tag stripper ──────────────────────────────────
        # Strip everything before and including </think> — only the clean
        # post-reasoning narrative is stored or returned to the frontend.
        m = re.search(r'</think>\s*(.*)', raw, re.DOTALL)
        if m:
            text = m.group(1).strip()
            print(f"[agent] </think> stripped — extracted {len(text)} chars of verdict")
        elif "<think>" in raw:
            # Unclosed think block — discard entire response to prevent leakage
            print("[agent] DeepSeek-R1: unclosed <think> block — verdict discarded")
            text = ""
        else:
            text = raw

        # Take only the first paragraph (no multi-topic prose)
        verdict = text.split("\n\n")[0].strip()

        del judge_llm
        gc.collect()

        print(f"[agent] 8B Judge's Verdict: {len(verdict)} chars")
        return verdict

    except Exception as e:
        print(f"[agent] 8B Judge unavailable ({e})")
        return ""


# ── DeepSeek-R1 per-slot curation rationales ─────────────────────────────────

def generate_curation_rationales(
    sequence: list[dict],
    style_prompt: str = "",
) -> dict[str, str]:
    """
    Use DeepSeek-R1-Distill-8B to generate one-sentence per-slot rationales
    for the finalized NSGA-III sequence.

    Paths are tokenized (IMG_01…) before the LLM call — the model never sees
    filenames. Returns dict[path → rationale]. Falls back to {} on any error
    or if the GGUF is absent.
    """
    if not sequence or not _JUDGE_GGUF_PATH.exists():
        return {}

    token_map: dict[str, str] = {}   # IMG_0N → path
    tokens:    list[str]      = []
    seq_lines: list[str]      = []

    for i, item in enumerate(sequence):
        token = f"IMG_{i+1:02d}"
        path  = item.get("path", "")
        token_map[token] = path
        tokens.append(token)

        # Parse breakdown — may arrive as dict or JSON string
        bd_raw = item.get("breakdown", {}) or {}
        if isinstance(bd_raw, str):
            try:
                bd_raw = json.loads(bd_raw)
            except Exception:
                bd_raw = {}

        score  = int(float(item.get("score", item.get("slot_score", 0.5))) * 100)
        comp   = int(float(bd_raw.get("Composition",  0.5)) * 100)
        tech   = int(float(bd_raw.get("Technical",    0.5)) * 100)
        slot   = item.get("slot",    f"Slot {i+1}")
        route  = item.get("route_triggered", "")
        # First 60 chars of SpecVLM tags — no full path leakage
        tags   = (item.get("reasoning_log", "") or "")[:60].strip()

        line = f"{token} [{slot}] Score:{score}% Tech:{tech}% Comp:{comp}%"
        if route:
            line += f" Route:{route}"
        if tags:
            line += f" Tags:{tags!r}"
        seq_lines.append(line)

    schema = _build_rationale_schema(tokens)
    prompt = _RATIONALE_PROMPT.format(
        n        = len(sequence),
        sequence = "\n".join(seq_lines),
    )

    try:
        from llama_cpp import Llama

        grammar = None
        try:
            from llama_cpp import LlamaGrammar
            grammar = LlamaGrammar.from_json_schema(json.dumps(schema))
        except Exception:
            pass

        judge_llm = Llama(
            model_path   = str(_JUDGE_GGUF_PATH),
            n_ctx        = 2048,
            n_gpu_layers = -1,
            n_threads    = 2,
            verbose      = False,
        )
        kwargs: dict = dict(max_tokens=600, temperature=0.3, echo=False)
        if grammar is not None:
            kwargs["grammar"] = grammar

        out = judge_llm(prompt, **kwargs)
        raw = out["choices"][0]["text"].strip()

        del judge_llm
        gc.collect()

        # Strip DeepSeek think tokens — same pattern as generate_judges_verdict_8b
        m = re.search(r'</think>\s*(.*)', raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
        elif "<think>" in raw:
            print("[agent] rationales: unclosed <think> block — discarded")
            return {}

        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start < 0 or end <= start:
            print(f"[agent] rationales: no JSON object in output — raw: {raw[:80]!r}")
            return {}

        parsed: dict = json.loads(raw[start:end])

        result: dict[str, str] = {}
        for token, orig_path in token_map.items():
            rationale = str(parsed.get(token, "")).strip()
            if orig_path and rationale:
                result[orig_path] = rationale

        print(f"[agent] curation rationales: {len(result)}/{len(sequence)} generated")
        return result

    except Exception as e:
        print(f"[agent] curation rationales failed ({e})")
        return {}
