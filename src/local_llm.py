"""
local_llm.py — the one text-LLM runtime, in-process, offline.

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
import threading
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent

# Offload ladder, best first. A build without enough VRAM fails at construction
# rather than at generation time, so backing off one rung at a time is how we
# find the level this machine can actually hold. 0 is pure CPU and must stay last
# — it is the rung with nothing after it, so nothing optional may ride on it.
_GPU_LAYER_LADDER = (-1, 20, 10, 0)

_llm = None
_load_attempted = False
_lock = threading.Lock()


def _setting(name: str, default):
    """run_profile is the declaration point for every knob, but this module must
    still work if it is imported stand-alone (a script, a test)."""
    try:
        import run_profile
        return run_profile.setting(name)
    except Exception:
        return default


def model_path() -> Path:
    override = str(_setting("FRAMEGRADE_LOCAL_LLM_GGUF", "") or "")
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


def _load():
    """Load the singleton, or return None. Never raises."""
    global _llm, _load_attempted
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        if _load_attempted:
            return None                      # already failed; don't retry per call
        _load_attempted = True

        path = model_path()
        if not path.exists():
            print(f"[llm] no text model at {path} — text features disabled")
            return None

        need = float(_setting("FRAMEGRADE_LOCAL_LLM_MIN_RAM_GB", 6.0))
        free = _free_ram_gb()
        if free < need:
            print(f"[llm] only {free:.1f} GB free, need ~{need:.1f} GB — "
                  f"skipping the text model rather than pushing this machine "
                  f"into swap")
            return None

        try:
            from llama_cpp import Llama
        except Exception as e:
            print(f"[llm] llama_cpp unavailable ({e}) — text features disabled")
            return None

        try:
            from tier_select import has_gpu
            has_cuda = has_gpu()
        except Exception:
            has_cuda = False

        ladder = _GPU_LAYER_LADDER if has_cuda else (0,)
        last_err: Optional[Exception] = None
        for n_gpu in ladder:
            try:
                _llm = Llama(
                    model_path=str(path),
                    n_ctx=4096,
                    n_gpu_layers=n_gpu,
                    # flash_attn halves the KV cache, but some builds reject it on
                    # a pure-CPU context. The n_gpu=0 rung is the last resort, so
                    # nothing optional may be the thing that fails there.
                    flash_attn=(n_gpu != 0),
                    n_threads=min(os.cpu_count() or 4, 8),
                    verbose=False,
                )
                print(f"[llm] {path.name} loaded (n_gpu_layers={n_gpu})")
                return _llm
            except Exception as e:
                last_err = e
                print(f"[llm] load failed at n_gpu_layers={n_gpu} ({e}) — backing off")
        print(f"[llm] load failed on every offload level: {last_err}")
        _llm = None
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

    ``json_schema`` constrains decoding via GBNF when llama_cpp supports it. The
    HTTP API this replaces had no grammar support at all, so callers that want
    structured output were parsing free text and hoping; they no longer have to.
    """
    llm = _load()
    if llm is None:
        return None

    grammar = None
    if json_schema is not None:
        try:
            import json as _json
            from llama_cpp import LlamaGrammar
            grammar = LlamaGrammar.from_json_schema(_json.dumps(json_schema))
        except Exception as e:
            print(f"[llm] grammar build failed ({e}) — unconstrained decoding")

    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    try:
        kwargs = dict(messages=messages, max_tokens=max_tokens,
                      temperature=temperature)
        if grammar is not None:
            kwargs["grammar"] = grammar
        out = llm.create_chat_completion(**kwargs)
        return (out["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        print(f"[llm] generation failed: {e}")
        return None


def unload() -> None:
    """Release the singleton so a GPU-heavy stage can have the VRAM back.

    Resets the attempt flag too: an unload is a deliberate act, and the next
    caller should get a fresh try rather than inherit an earlier failure.
    """
    global _llm, _load_attempted
    with _lock:
        _llm = None
        _load_attempted = False
    import gc
    gc.collect()
    try:
        import torch
        # is_initialized() FIRST. is_available() would itself create the CUDA
        # context this whole module is arranged to avoid in the server process.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
