"""
Creative Director Agent — DeepSeek-R1-Distill-Qwen-1.5B GGUF Middleman

Runs strictly on CPU via llama-cpp-python.
Analyzes the Style Brief and emits a Rule Set JSON governing all curation filters.

Rule Set schema
───────────────
  HARD_FILTER_PEOPLE  bool    — True → person detection is a hard disqualifier
  GEOMETRIC_PRIORITY  str     — "Low" | "Medium" | "High"
  LIGHTING_MOOD       str     — one-phrase lighting descriptor
  BRIEF_KEYWORDS      list    — 2-4 key descriptive words from the brief

Model: models/deepseek-r1-1.5b-q4.gguf  (Q4_K_M, ~1 GB, CPU only)
Fallback: fast keyword-based heuristics when the GGUF is absent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_GGUF_PATH = Path("models/deepseek-r1-1.5b-q4.gguf")

_EMPTY_KW    = {"empty", "liminal", "desert", "void", "abandoned", "desolate", "minimal", "sparse"}
_GEOMETRIC_KW = {"geometric", "architecture", "pattern", "grid", "lines", "symmetry", "abstract"}
_WARM_KW      = {"golden", "sunset", "sunrise", "warm", "orange", "amber"}
_DARK_KW      = {"night", "dark", "neon", "low-key", "shadow"}
_FLAT_KW      = {"overcast", "flat", "grey", "gray", "fog", "mist", "rain"}
_CONTRAST_KW  = {"harsh", "contrast", "dramatic", "midday", "silhouette"}

_PROMPT = (
    "You are a Rule Set Generator for a street photography curation system.\n\n"
    "Given a Style Brief, output ONLY a JSON object with these exact fields:\n"
    "- HARD_FILTER_PEOPLE: true if the brief implies empty / no-people scenes, else false\n"
    "- GEOMETRIC_PRIORITY: \"High\" if brief implies geometry/architecture, "
    "\"Medium\" if mixed, \"Low\" otherwise\n"
    "- LIGHTING_MOOD: one short phrase describing implied lighting "
    "(e.g. \"flat overcast\", \"high contrast\", \"warm golden\")\n"
    "- BRIEF_KEYWORDS: JSON array of 2-4 key words extracted from the brief\n\n"
    'Style Brief: "{brief}"\n\n'
    "JSON:"
)

_llama: Optional[object] = None


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
            n_ctx=512,
            n_threads=4,
            n_gpu_layers=0,   # strictly CPU — VRAM reserved for 7B verifier
            verbose=False,
        )
        print("[agent] DeepSeek-R1-1.5B GGUF loaded (CPU)")
        return _llama
    except ImportError:
        print("[agent] llama-cpp-python not installed — keyword fallback active")
    except Exception as e:
        print(f"[agent] GGUF load failed ({e}) — keyword fallback active")
    return None


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


def generate_rule_set(brief: str) -> dict:
    """
    Analyze the style brief and return a Rule Set JSON.

    Uses DeepSeek-R1-1.5B GGUF (CPU) when available; falls back to keyword
    heuristics otherwise. Never raises — always returns a valid dict.
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
            prompt = _PROMPT.format(brief=brief[:200])
            out    = llm(prompt, max_tokens=160, temperature=0.0, echo=False)
            raw    = out["choices"][0]["text"].strip()
            start  = raw.find("{")
            end    = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                rule_set = {
                    "HARD_FILTER_PEOPLE": bool(parsed.get("HARD_FILTER_PEOPLE", False)),
                    "GEOMETRIC_PRIORITY": str(parsed.get("GEOMETRIC_PRIORITY", "Low")),
                    "LIGHTING_MOOD":      str(parsed.get("LIGHTING_MOOD", "natural ambient")),
                    "BRIEF_KEYWORDS":     list(parsed.get("BRIEF_KEYWORDS", [])),
                }
                print(f"[agent] Rule set (GGUF): {rule_set}")
                return rule_set
        except Exception as e:
            print(f"[agent] GGUF inference failed ({e}) — keyword fallback")

    rs = _keyword_rule_set(brief)
    print(f"[agent] Rule set (keywords): {rs}")
    return rs
