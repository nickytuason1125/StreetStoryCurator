"""
model_registry.py — the one declaration of where optional model files live and
which Hugging Face repo provides each.

Why this module exists
----------------------
critique_engine asked for ``models/qwen2.5-vl-2b-instruct-q4_k_m.gguf``,
queue_manager gated the annotation daemon on that same literal, and server.py's
health endpoint listed it a third time. All three agreed with each other and all
three were wrong: **no Qwen2.5-VL-2B GGUF exists**. The published build is 3B
(``ggml-org/Qwen2.5-VL-3B-Instruct-GGUF``). So the vision path could never load,
every critique fell through to the Ollama fallback, and the fallback is what made
Ollama look mandatory.

A file path repeated in three modules with no way to obtain the file is how that
happens. One declaration, with the source next to the destination, so "where does
this come from" is answerable without a web search.

Kept dependency-free on purpose: imported by the server, by the downloader script
and by low-level modules, so it must not drag in torch, transformers or
huggingface_hub at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelFile:
    """One downloadable file: where it goes, and where it comes from."""
    key: str
    dest: Path
    repo: str
    filename: str
    size_gb: float
    purpose: str
    required: bool = False       # False = a feature degrades, grading is unaffected

    def present(self) -> bool:
        return self.dest.exists()


# ── Optional GGUF models ─────────────────────────────────────────────────────
# None of these is required to grade. Grading is SigLIP zero-shot plus TOPIQ; the
# files below power critique, annotations and Story Mode. Verified against the Hub.
GGUF_MODELS: tuple = (
    ModelFile(
        key="vision",
        dest=_ROOT / "models" / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        filename="Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        size_gb=2.2,
        purpose="jury critique, audit annotations, contact-sheet review",
    ),
    ModelFile(
        key="vision_mmproj",
        dest=_ROOT / "models" / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        repo="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
        filename="mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
        size_gb=1.4,
        purpose="the projector the vision model needs to see at all",
    ),
    ModelFile(
        key="text",
        dest=_ROOT / "models" / "deepseek-r1-8b-q5.gguf",
        repo="bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        filename="DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf",
        size_gb=5.7,
        purpose="Story Mode selection, Judge's Verdict, RAG concept extraction",
    ),
)

_BY_KEY = {m.key: m for m in GGUF_MODELS}


def gguf(key: str) -> ModelFile:
    return _BY_KEY[key]


def vision_gguf_path() -> Path:
    return _BY_KEY["vision"].dest


def vision_mmproj_path() -> Path:
    return _BY_KEY["vision_mmproj"].dest


def text_gguf_path() -> Path:
    return _BY_KEY["text"].dest


def missing_gguf() -> list:
    """Files that are declared but absent. Advisory — never a gate on grading."""
    return [m for m in GGUF_MODELS if not m.present()]


# ── Encoder tiers ────────────────────────────────────────────────────────────
# The lean fp16 checkpoints. scripts/build_lean_checkpoint.py owns the conversion
# and already holds this mapping; it is repeated here ONLY as data the downloader
# and the UI can read without importing a build script. If they ever disagree,
# build_lean_checkpoint.py is the authority — it is what actually writes the dir.
ENCODER_SOURCES = {
    "high": ("google/siglip2-giant-opt-patch16-384", 3.6, "Pro"),
    "mid":  ("google/siglip2-large-patch16-384",     1.7, "Balanced"),
    "low":  ("google/siglip2-base-patch16-384",      0.8, "Fast"),
}


def encoder_dir(tier: str) -> Optional[Path]:
    """Where a tier's lean checkpoint lives, per run_profile."""
    try:
        import run_profile
        return Path(run_profile.spec_for(tier).hf_dirname)
    except Exception:
        return None
