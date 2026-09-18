"""
local_llm.py — the one text-LLM runtime, offline, via the disposable sidecar.

Why this module exists
----------------------
Four modules used to reach a local Ollama server over HTTP on port 11434:
creative_director (Story Mode selection and critique), fast_ingestion
(pixel_inspector), pdf_rag (RAG concept extraction) and critique_engine (as a
fallback). Nothing installed Ollama — not Setup.ps1, not requirements.txt — and
no user-facing document mentioned it, while CLAUDE.md rule 5 promised a fully
offline app. Worse, server.py refused every non-scan grade with a 503 when the
port did not answer, so a correct and complete install could not grade a photo.

Meanwhile jury_engine had already been running the same class of model locally
through llama_cpp for months, and critique_engine preferred a local GGUF and only
fell back to HTTP. The local path was the real one; Ollama was the leftover.

So: one loader, here. Every text caller shares it.

Sharing matters more than tidiness. Three callers each building their own Llama
would hold three copies of a 5.4 GB model. On the 16 GB laptop this app targets
that is not a slowdown, it is an out-of-memory kill — which is exactly why the
loader below is a singleton with a RAM preflight rather than a convenience
wrapper around a constructor.

Process model
-------------
This is imported by the SERVER process, which spawns grade_runner.py as a CUDA
subprocess. It must therefore never touch ``torch.cuda`` — not even
``is_available()``, which initialises a context in the parent and makes it fault
0xC0000005 with no traceback when the child exits. GPU presence comes from
tier_select's cached subprocess probe, the same way jury_engine asks.
"""
from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent

# Offload ladder: the vision/text GGUF loaders live in src/vlm_sidecar.py now
# (see _GPU_LAYER_LADDER there). This module is the CLIENT — it keeps the
# skip-reporting semantics and the public API, not the weights.
_load_attempted = False


def _setting(name: str, default):
    """run_profile is the declaration point for every knob, but this module must
    still work if it is imported stand-alone (a script, a test)."""
    try:
        import run_profile
        return run_profile.setting(name)
    except Exception:
        return default


def model_path() -> Path:
    override = str(_setting("FIRSTCUT_LOCAL_LLM_GGUF", "") or "")
    if override:
        return Path(override)
    import model_registry
    return model_registry.text_gguf_path()


def available() -> bool:
    """True when the weights are on disk. Does not load anything.

    Callers use this to decide whether to offer a feature, so it must stay cheap
    and must not have the side effect of pulling 5.4 GB into RAM.
    """
    return model_path().exists()


def _free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 999.0        # unknown → don't let the probe itself block a feature


_last_skip: Optional[str] = None


def last_skip_reason() -> Optional[str]:
    """Why the text model is not answering, or None when it is.

    A degraded Story run used to be a single print into a subprocess log, so a
    score-sorted sequence was indistinguishable from a curated one. Callers
    surface this string instead of the user guessing.
    """
    return _last_skip


def required_ram_gb() -> float:
    """Free RAM needed to load these weights, MEASURED against the weights.

    Peak RSS while loading at n_ctx=4096, on a machine with headroom:

        LFM2.5-VL-1.6B   file 0.68 GiB -> peak 1.25 GiB   (1.84x)
        Qwen3-4B         file 2.33 GiB -> peak 4.56 GiB   (1.96x)

    So roughly TWICE the file: weights, the KV cache at n_ctx=4096, and the
    allocator's working set during load.

    Two earlier versions were both wrong. A flat 6.0 was "the DeepSeek
    checkpoint plus headroom" frozen into a constant, refusing a 0.73 GB model
    as firmly as a 5.34 GiB one. Its replacement, `file x 1.15 + 0.5`, I
    invented and shipped as though measured -- the exact mistake commit 3f42c7f
    exists to document. It returned 3.17 GB for Qwen3-4B, so at 3.58 GB free the
    gate PASSED and the load then drove the machine to 0.00 GB available. A
    floor that admits the failure it exists to prevent is not a floor.

    Two data points is thin. It is two more than the last version had, and it
    errs high, which is the safe direction: refusing costs a fallback that is
    now reported, while admitting costs the user their machine.
    """
    override = float(_setting("FIRSTCUT_LOCAL_LLM_MIN_RAM_GB", 0.0) or 0.0)
    if override:
        return override
    try:
        return model_path().stat().st_size / 2 ** 30 * 2.0
    except Exception:
        return 6.0          # weights unreadable; keep the historical floor


# Models shipping a hybrid "thinking" mode that honour the /no_think switch.
# Measured on Qwen3-4B, same 6 trials: thinking ON was 6/6 at 31.9 s/answer,
# OFF was 6/6 at 1.7 s. Identical accuracy, 19x the time -- the think block is
# pure tax on "return one id", and it would put a single Art Director call over
# the budget for an entire Story run.
_THINKING_MODELS = ("qwen3",)


def _strip_thinking(text):
    """Remove a <think>...</think> wrapper from a completion.

    Even with /no_think, Qwen3 emits an EMPTY block: the first live call
    returned "<think>\n\n</think>\n\nREADY". Three callers each carry their own
    regex for this (written originally for DeepSeek-R1), which means every
    future caller has to know the quirk too. This module is the one that
    suppresses thinking, so it owes callers clean text.

    An UNCLOSED tag is left alone apart from the marker itself: a truncated
    generation still holds the only content there is, and discarding it would
    turn a partial answer into no answer.
    """
    if not text:
        return text
    import re as _re
    out = _re.sub("<think>.*?</think>", "", text, flags=_re.DOTALL)
    if "<think>" in out:                      # unclosed: drop the marker only
        out = out.replace("<think>", "")
    return out.strip()


def _suppress_thinking(system):
    """Append /no_think for models that reason by default.

    Never raises: probing the weight filename must not be the thing that stops
    a generation. Idempotent, so a caller that already asked for it is safe.
    """
    try:
        name = model_path().name.lower()
    except Exception:
        return system
    if not any(m in name for m in _THINKING_MODELS):
        return system
    if system and "/no_think" in system:
        return system
    return ((system + " ") if system else "") + "/no_think"


_OLLAMA_TEXT_TIMEOUT = (5, 120)   # (connect, read) — cold VRAM load budget


def _check_ollama_available() -> bool:
    """Ping Ollama /api/version once per 60 s; False immediately if down."""
    global _ollama_last_check, _ollama_ok
    try:
        now = time.monotonic()
    except Exception:
        return False
    if now - _ollama_last_check < 60.0:
        return _ollama_ok
    try:
        import requests as _rq
        _ollama_ok = _rq.get("http://localhost:11434/api/version",
                             timeout=2).ok
    except Exception:
        _ollama_ok = False
    _ollama_last_check = now
    return _ollama_ok


_ollama_last_check = 0.0
_ollama_ok = False


def _ollama_text(messages, *, max_tokens: int, temperature: float,
                 json_schema=None, model: str) -> Optional[str]:
    """One text chat completion against Ollama. None when it fails — callers
    fall back to the sidecar."""
    try:
        import requests as _rq
    except Exception:
        return None
    payload = {
        "model": model, "stream": False, "messages": messages,
        "keep_alive": "30s",   # self-unload: no resident RAM between requests
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_schema is not None:
        payload["format"] = json_schema   # Ollama structured output (JSON schema)
    for attempt in (1, 2):
        try:
            r = _rq.post("http://localhost:11434/api/chat", json=payload,
                         timeout=_OLLAMA_TEXT_TIMEOUT)
            if r.ok:
                msg = (r.json().get("message") or {}).get("content") or ""
                return msg.strip() or None
            print(f"[llm] Ollama/{model} HTTP {r.status_code}")
            return None
        except _rq.exceptions.ReadTimeout:
            if attempt == 1:
                print(f"[llm] Ollama/{model} read-timeout (cold load?) — retrying once")
                continue
        except Exception as e:
            print(f"[llm] Ollama/{model} failed: {e}")
            return None
    return None


def generate(prompt: str,
             *,
             system: Optional[str] = None,
             max_tokens: int = 400,
             temperature: float = 0.4,
             json_schema: Optional[dict] = None) -> Optional[str]:
    """Return the model's text, or None when no model is available.

    None is a first-class answer, not an error. Every caller here has a defined
    behaviour without an LLM — Story Mode ranks by score, pixel_inspector returns
    an empty note, pdf_rag falls back to its own extractor — and those fallbacks
    are why removing the Ollama gate is safe. Raising instead would convert a
    degraded feature into a broken grade.

    ``json_schema`` constrains decoding via GBNF — the schema travels to the
    sidecar, which builds the LlamaGrammar next to the model.

    The model runs in the disposable sidecar process (sidecar_client), NOT
    here: this server process used to hold a 1.5 GB llama.cpp resident whose
    RAM could never fully return after an in-place unload. Same public
    behaviour, same skip-reporting semantics, different address space.
    """
    global _last_skip, _load_attempted
    path = model_path()
    if not path.exists():
        _last_skip = f"no text model installed at {path.name}"
        print(f"[llm] {_last_skip} — text features disabled")
        return None
    if _load_attempted:
        return None                      # already failed; don't retry per call

    need = required_ram_gb()
    free = _free_ram_gb()
    if free < need:
        # Deliberately does NOT set _load_attempted. Free memory is transient:
        # a single refusal while Chrome was open must not latch the model off
        # for the life of the server (see the original in-process comment).
        _last_skip = (f"only {free:.1f} GB RAM free, needs ~{need:.1f} GB "
                      f"for {path.name}")
        print(f"[llm] {_last_skip} — skipping the text model rather than "
              f"pushing this machine into swap")
        return None

    try:
        import sidecar_client as _sc
    except Exception as e:
        _last_skip = f"sidecar client unavailable ({e})"
        return None

    system = _suppress_thinking(system)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]

    schema = json_schema
    # Ollama-first (2026-09-17): the text Art Director used to depend solely
    # on the sidecar, whose detached spawn is the flakiest link in the chain
    # (observed WinError 50 from a windowed server) — a spawn failure silently
    # degraded Story Mode to score-sort. Ollama is already installed with
    # capable text models, runs them on CUDA, and keep_alive self-unloads:
    # same accuracy, no RAM balloon, no spawn roulette. The sidecar stays as
    # the fallback for machines without Ollama.
    for _model in ("qwen3.5:4b", "llama3.2:latest"):
        if _check_ollama_available():
            raw = _ollama_text(messages, max_tokens=max_tokens,
                               temperature=temperature, json_schema=schema,
                               model=_model)
            if raw:
                _last_skip = None
                return _strip_thinking(raw)
    for attempt in (1, 2):
        payload = {"kind": "text", "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature,
                   "json_schema": schema}
        try:
            import model_residency as _res; _res.register('text_llm', unload, 1.5)
        except Exception:
            pass
        j = _sc.infer(payload)
        if j is None:
            _last_skip = "the model sidecar could not start"
            return None
        if j.get("ok"):
            _last_skip = None
            return _strip_thinking(j.get("text") or "")
        if j.get("grammar_failed") and attempt == 1:
            # This build cannot build the GBNF schema — retry unconstrained,
            # exactly like the old in-process grammar path did.
            print(f"[llm] grammar build failed ({j.get('error')}) — unconstrained decoding")
            schema = None
            continue
        # Weights missing in the sidecar, or a genuine load failure.
        _load_attempted = True
        _last_skip = f"the model failed to load ({j.get('error')})"
        print(f"[llm] {_last_skip}")
        return None
    return None


def unload() -> None:
    """Release the text model so a GPU-heavy stage can have the VRAM back.

    Targets the sidecar's text slot; when vision is still loaded the sidecar
    stays up for it, otherwise the process exits and ALL its RAM returns.
    Resets the attempt flag too: an unload is a deliberate act, and the next
    caller should get a fresh try rather than inherit an earlier failure.
    """
    global _load_attempted
    _load_attempted = False
    try:
        import sidecar_client as _sc
        _sc.evict("text")
    except Exception:
        pass
