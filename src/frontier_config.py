"""
Frontier 2026 enforcement layer.

Activated via --force-frontier flag in main.py.
When active:
  - Legacy encoder fallbacks (SigLIP So400M, zero-emb) are blocked.
  - Legacy grader fallbacks (QAlign, NIMA, V1) are blocked.
  - LanceDB 1152-d schema triggers a forced drop + re-scan.
  - Pre-flight validates VRAM >= 5.0 GB and 2026 weight files present.
"""
from __future__ import annotations
import sys
import logging
from pathlib import Path

logger = logging.getLogger("frontier")

_FORCE_FRONTIER: bool = False


# ── Accessor / mutator ─────────────────────────────────────────────────────────

def is_force_frontier() -> bool:
    return _FORCE_FRONTIER


def set_force_frontier(val: bool) -> None:
    global _FORCE_FRONTIER
    _FORCE_FRONTIER = bool(val)
    if _FORCE_FRONTIER:
        logger.info("🔒 Frontier 2026 enforcement ACTIVE — all legacy fallbacks disabled")


# ── Pre-flight checks (called from main.py before server starts) ───────────────

def validate_vram_overhead(required_gb: float = 5.0) -> None:
    """
    Assert free VRAM >= required_gb when --force-frontier is active.
    sys.exit() with CRITICAL message if the check fails.
    """
    if not _FORCE_FRONTIER:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            sys.exit(
                "CRITICAL: --force-frontier requires CUDA. No GPU detected.\n"
                "Either install CUDA drivers or run without --force-frontier."
            )
        props       = torch.cuda.get_device_properties(0)
        total_gb    = props.total_memory / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        free_gb     = total_gb - reserved_gb
        if free_gb < required_gb:
            _log_vram_block(total_gb, reserved_gb, free_gb, required_gb)
            sys.exit(
                f"CRITICAL: Insufficient VRAM for Frontier Reasoning. "
                f"Need {required_gb:.1f} GB free, have {free_gb:.1f} GB. "
                "Close background GPU apps (browsers, games, other ML processes) and retry."
            )
        logger.info(
            f"✓ VRAM pre-flight: {free_gb:.1f} GB free / {total_gb:.1f} GB total "
            f"(threshold {required_gb:.1f} GB)"
        )
    except SystemExit:
        raise
    except Exception as exc:
        logger.warning(f"VRAM pre-flight check could not run: {exc}")


def _log_vram_block(total_gb: float, reserved_gb: float, free_gb: float, required_gb: float) -> None:
    try:
        import torch
        device_name = torch.cuda.get_device_name(0)
    except Exception:
        device_name = "unknown"
    logger.critical(
        "VRAM GATEKEEPER TRIGGERED\n"
        f"  Device  : {device_name}\n"
        f"  Total   : {total_gb:.2f} GB\n"
        f"  Reserved: {reserved_gb:.2f} GB\n"
        f"  Free    : {free_gb:.2f} GB  (need {required_gb:.1f} GB)\n"
        "  Common culprits: Chrome/Edge GPU acceleration, background ML processes, "
        "CUDA-enabled games or apps."
    )


def check_model_integrity() -> None:
    """
    Verify 2026 Frontier weight directories are populated.
    sys.exit() with download instructions if any are missing.
    """
    if not _FORCE_FRONTIER:
        return

    weight_exts = {".bin", ".safetensors", ".pt", ".pth"}
    siglip2_dir = Path("models/siglip2")
    verify_dir  = Path("models/deepseek/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B")

    def _has_weights(d: Path) -> bool:
        return d.exists() and any(
            f.suffix in weight_exts for f in d.rglob("*") if f.is_file()
        )

    # SigLIP-2 encoder is always required — hard abort if absent.
    if not _has_weights(siglip2_dir):
        sys.exit(
            "CRITICAL: SigLIP-2 weights missing (expected at models/siglip2/).\n"
            "Run: python -c \"from siglip2_encoder import _download_siglip2_if_needed; "
            "_download_siglip2_if_needed()\""
        )

    # DeepSeek 7B verifier is optional — warn if absent, draft-only mode will run.
    if not _has_weights(verify_dir):
        logger.warning(
            "Vision-R1-7B weights absent (models/deepseek/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/)."
            " 7B verification disabled — pipeline runs in draft-only mode."
        )
    else:
        logger.info("✓ Model integrity: SigLIP-2-Giant-NaFlex + Vision-R1-7B-NF4 weights found")
