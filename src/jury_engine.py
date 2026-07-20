"""
Multi-persona jury — replaces the Phase-0 single-Ollama-call
generate_judges_verdict_8b stub.

Three personas (Purist, Storyteller, Formalist), one shared in-process GGUF
(models/deepseek-r1-8b-q5.gguf), called sequentially with different system
prompts/temperatures — not N separate model loads. Each verdict is
grammar-constrained (GBNF, following vlm_niche_detector.py's pattern) and
validated against the real per-image aspect data (src/signal_validator.py)
before it's allowed to count toward the jury's self-consistency check or
the final narrative.

GPU offload: this used to be hardcoded n_gpu_layers=0 (CPU-only), inherited
from pdf_rag.py's rationale for ITS OWN call site — "never compete with
SigLIP-2 VRAM" — but that rationale doesn't hold here. By the time the jury
runs (Step 5b, after the revision loop's own GPU model has already been
explicitly unloaded + purged), the GPU is idle; there is nothing left to
compete with. CPU-only was blindly copying a pattern designed for a
different point in the pipeline. _load_llm() now probes GPU offload with
an auto-backoff ladder (try full offload, step down on OOM, land on
CPU-only as the final fallback) — this can never be slower than the old
behavior, only faster, and never risks a hard crash from guessing a VRAM
budget wrong.

Self-consistency: if the validated personas' scores spread by more than
0.30 (same order of magnitude as vlm_niche_detector.py's existing 0.45-0.55
borderline band), one additional synthesis round runs — capped at exactly
one re-judge.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from signal_validator import Claim, validate_claims

_ROOT = Path(__file__).resolve().parent.parent
_GGUF = _ROOT / "models" / "deepseek-r1-8b-q5.gguf"

_VALID_ASPECTS = {"Composition", "Lighting", "Narrative", "Human/Culture", "Technical"}
_SPREAD_THRESHOLD = 0.30

# DeepSeek-R1 distills emit a chain-of-thought preamble before any real
# content, even for simple structured-output tasks — burns hundreds of
# tokens narrating reasoning nobody reads. Priming the completion with an
# empty <think></think> block (a well-known trick for this model family)
# tricks it into believing it already finished thinking, skipping straight
# to the answer. Cuts both latency and token count independent of the GPU
# offload win below.
_THINK_SKIP = "\n<think>\n\n</think>\n\n"

# Auto-backoff ladder for GPU offload: try full (-1), then step down. The
# 8B Q5 GGUF is ~5.7GB; a 6GB card has to share that with KV cache + CUDA
# context overhead, so full offload may not fit — back off rather than
# guess a fixed "safe" number that might not generalize to a different GPU.
_GPU_LAYER_LADDER = [-1, 28, 24, 20, 16, 12, 8, 4, 0]

_PERSONAS = [
    {"name": "Purist",      "temp": 0.15,
     "system": "You value technical craft and negative space above all. Penalize gimmicks."},
    {"name": "Storyteller", "temp": 0.45,
     "system": "You value narrative tension, decisive moment, and emotional pacing above technical polish."},
    {"name": "Formalist",   "temp": 0.30,
     "system": "You value geometry, composition, and visual rhythm across the sequence above individual shot merit."},
]

_JURY_GRAMMAR_SRC = r'''
root       ::= "{" ws '"verdict"' ws ":" ws string ws "," ws '"score"' ws ":" ws float ws "," ws '"cited_aspect"' ws ":" ws aspect ws "," ws '"cited_slot"' ws ":" ws slot ws "," ws '"cited_value"' ws ":" ws value ws "}"
aspect     ::= "\"Composition\"" | "\"Lighting\"" | "\"Narrative\"" | "\"Human/Culture\"" | "\"Technical\"" | "\"none\""
slot       ::= "null" | [0-9] [0-9]?
value      ::= "null" | float
float      ::= "-"? [0-9]+ "." [0-9]+
string     ::= "\"" ([^"\\] | "\\" .)* "\""
ws         ::= [ \t\r\n]*
'''

_llm: Optional[object] = None
_grammar: Optional[object] = None
_grammar_broken = False


def _load_llm():
    global _llm
    if _llm is not None:
        return _llm
    if not _GGUF.exists():
        return None

    from llama_cpp import Llama
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    ladder = _GPU_LAYER_LADDER if has_cuda else [0]
    last_err: Optional[Exception] = None
    for n_gpu in ladder:
        try:
            # flash_attn halves KV-cache memory (both VRAM and, on a CPU
            # fallback, system RAM) and speeds up attention on GPU — it
            # defaults to False in llama-cpp-python for no good reason here.
            # Only requested alongside actual GPU layers: some builds don't
            # support it on a pure-CPU context, and the n_gpu_layers=0 rung
            # is the last-resort fallback, so it must not be the one thing
            # that can fail with no further backoff available.
            _llm = Llama(
                model_path=str(_GGUF), n_ctx=1024, n_gpu_layers=n_gpu,
                flash_attn=(n_gpu != 0), verbose=False,
            )
            print(f"[jury] GGUF loaded (n_gpu_layers={n_gpu}, flash_attn={n_gpu != 0})")
            return _llm
        except Exception as e:
            last_err = e
            print(f"[jury] load failed at n_gpu_layers={n_gpu} ({e}) — backing off")
            continue

    print(f"[jury] GGUF load failed on every offload level: {last_err}")
    _llm = None
    return _llm


def _load_grammar():
    global _grammar
    if _grammar is not None:
        return _grammar
    try:
        from llama_cpp import LlamaGrammar
        _grammar = LlamaGrammar.from_string(_JURY_GRAMMAR_SRC)
    except Exception as e:
        print(f"[jury] grammar build failed ({e}) — falling back to unconstrained decoding")
        _grammar = None
    return _grammar


def unload() -> None:
    """Release the GGUF singleton. Now that _load_llm() may put it on GPU,
    this must actually free CUDA memory (empty_cache/ipc_collect), not just
    drop the Python reference — otherwise VRAM leaks across requests within
    the same server process, breaking the "one GPU model at a time"
    invariant for whatever runs next."""
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


def _extract_json_obj(raw: str) -> Optional[dict]:
    """Brace-balanced JSON extraction, tolerant of <think> preambles."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"</think>\s*(.*)", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
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
        import json
        obj = json.loads(raw[start:end])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_verdict(raw: str, persona_name: str) -> Optional[dict]:
    obj = _extract_json_obj(raw)
    if not obj or "verdict" not in obj:
        return None
    try:
        score = float(obj.get("score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))
    cited_aspect = obj.get("cited_aspect")
    if cited_aspect not in _VALID_ASPECTS:
        cited_aspect = None
    cited_slot = obj.get("cited_slot")
    try:
        cited_slot = int(cited_slot) if cited_slot is not None else None
    except (TypeError, ValueError):
        cited_slot = None
    cited_value = obj.get("cited_value")
    try:
        cited_value = float(cited_value) if cited_value is not None else None
    except (TypeError, ValueError):
        cited_value = None
    return {
        "persona":     persona_name,
        "verdict":     str(obj.get("verdict", ""))[:200],
        "score":       score,
        "cited_aspect": cited_aspect,
        "cited_slot":   cited_slot,
        "cited_value":  cited_value,
    }


def _build_slot_summary(selected_images: list[dict], roles: list[str], scores: list[float]) -> str:
    lines = []
    for i, (img, role, sc) in enumerate(zip(selected_images, roles, scores)):
        aspects = {k: v for k, v in img.items() if k != "filename" and isinstance(v, (int, float))}
        aspects_str = ",".join(f"{k}={v:.2f}" for k, v in list(aspects.items())[:3])
        fname = Path(img.get("filename", "")).name
        lines.append(f"{i}:{role}:{fname}:score={sc:.2f}:{aspects_str}")
    return " | ".join(lines)[:500]


def _run_persona(
    llm, persona: dict, slot_summary: str, style_prompt: str,
    niche: str, color: str,
) -> Optional[dict]:
    global _grammar_broken
    prompt = (
        f"{persona['system']}\n"
        f"ROLE: {persona['name']} juror delivering a verdict on a curated street-photo sequence.\n"
        f"Style brief: '{style_prompt[:150]}'. Theme: {niche}. Color: {color}.\n"
        f"Sequence (slot:role:file:score:aspects): {slot_summary}\n"
        "Give a one-sentence verdict, a 0.00-1.00 score, and cite the specific slot/aspect/value "
        "driving your score (or \"none\"/null if purely qualitative).\n"
        'Output ONLY JSON: {"verdict":"...","score":0.00,"cited_aspect":'
        '"Composition|Lighting|Narrative|Human/Culture|Technical|none","cited_slot":<int or null>,'
        '"cited_value":<float or null>}'
    ) + _THINK_SKIP
    # _THINK_SKIP primes the completion with an already-closed <think></think>
    # block, which DeepSeek-R1 distills treat as "reasoning already done" —
    # skips the chain-of-thought preamble entirely instead of just tolerating
    # it (the previous approach: no stop sequence + max_tokens=600 to let a
    # ~400-token reasoning ramble finish before the JSON appeared). With the
    # preamble skipped, the real answer is short — max_tokens drops accordingly.
    base_kwargs = dict(max_tokens=200, temperature=persona["temp"])

    if not _grammar_broken:
        grammar = _load_grammar()
        if grammar is not None:
            try:
                out = llm(prompt, grammar=grammar, **base_kwargs)
                return _parse_verdict(out["choices"][0]["text"], persona["name"])
            except Exception as _e:
                print(f"[jury] grammar-constrained decoding failed ({_e}) — "
                      "disabling grammar for this process, retrying unconstrained")
                _grammar_broken = True

    try:
        out = llm(prompt, **base_kwargs)
        return _parse_verdict(out["choices"][0]["text"], persona["name"])
    except Exception as e:
        print(f"[jury] {persona['name']} verdict failed: {e}")
        return None


def _run_synthesis(llm, verdicts: list[dict], style_prompt: str) -> Optional[dict]:
    """One re-judge round when personas disagree — Formalist's own prompt
    style, but summarizing the panel's spread rather than re-scoring blind."""
    summary = " | ".join(f"{v['persona']}:{v['score']:.2f}:{v['verdict'][:60]}" for v in verdicts)
    prompt = (
        "The jury disagreed on this sequence. Prior verdicts: " + summary[:500] + "\n"
        f"Style brief: '{style_prompt[:150]}'. "
        "Give a final synthesis: one-sentence verdict and a single 0.00-1.00 score "
        "representing the jury's consensus.\n"
        'Output ONLY JSON: {"verdict":"...","score":0.00,"cited_aspect":"none","cited_slot":null,"cited_value":null}'
    ) + _THINK_SKIP
    base_kwargs = dict(max_tokens=200, temperature=0.20)
    try:
        out = llm(prompt, **base_kwargs)
        return _parse_verdict(out["choices"][0]["text"], "Synthesis")
    except Exception as e:
        print(f"[jury] synthesis round failed: {e}")
        return None


def run_jury_panel(
    selected_images: list[dict],
    style_prompt: str,
    roles: list[str],
    director_brief,
    scores: list[float],
) -> tuple[list[dict], bool]:
    """
    Runs the 3-persona panel (+ one conditional synthesis round). Returns
    (validated_verdicts, rejudge_fired). Each verdict dict has
    {persona, verdict, score, cited_aspect, cited_slot, cited_value}.
    Verdicts whose cited evidence fails signal_validator.validate_claims()
    are excluded entirely — never counted toward the spread or the
    narrative. Never raises; returns ([], False) on total failure.
    """
    llm = _load_llm()
    if llm is None:
        return [], False

    try:
        niche = director_brief.thematic_niche if director_brief else "street photography"
        color = director_brief.color_profile_target if director_brief else "natural"
        slot_summary = _build_slot_summary(selected_images, roles, scores)
        aspects_by_slot = [
            {k: v for k, v in img.items() if k != "filename" and isinstance(v, (int, float))}
            for img in selected_images
        ]

        raw_verdicts = []
        for persona in _PERSONAS:
            v = _run_persona(llm, persona, slot_summary, style_prompt, niche, color)
            if v:
                raw_verdicts.append(v)

        validated: list[dict] = []
        for v in raw_verdicts:
            claim = Claim(text=v["verdict"], cited_aspect=v["cited_aspect"],
                          cited_value=v["cited_value"], cited_slot=v["cited_slot"])
            result = validate_claims([claim], aspects_by_slot)
            if result.passed:
                validated.append(v)
            else:
                print(f"[jury] {v['persona']} verdict rejected — {result.reason}")

        if not validated:
            return [], False

        rejudge_fired = False
        if len(validated) > 1:
            spread = max(v["score"] for v in validated) - min(v["score"] for v in validated)
            if spread > _SPREAD_THRESHOLD:
                rejudge_fired = True
                synth = _run_synthesis(llm, validated, style_prompt)
                if synth:
                    validated.append(synth)

        return validated, rejudge_fired
    finally:
        unload()


def generate_judges_verdict_8b(
    selected_images: list[dict],
    style_prompt: str,
    roles: list[str],
    director_brief,
    scores: list[float],
) -> Optional[str]:
    """
    Orchestrates the jury panel and formats the final human-readable
    narrative string. Same Optional[str] return contract as the Phase-0
    stub it replaces — never raises, returns None when the jury produces
    nothing usable.
    """
    try:
        verdicts, rejudge_fired = run_jury_panel(selected_images, style_prompt, roles, director_brief, scores)
        if not verdicts:
            return None
        panel = [v for v in verdicts if v["persona"] != "Synthesis"]
        lines = [f"{v['persona']}: {v['verdict']} (score {v['score']:.2f})" for v in panel]
        narrative = " ".join(lines)
        synth = next((v for v in verdicts if v["persona"] == "Synthesis"), None)
        if synth:
            narrative += f" Jury synthesis (panel disagreed): {synth['verdict']}"
        return narrative[:800] or None
    except Exception as e:
        print(f"[jury] verdict generation failed: {e}")
        return None
