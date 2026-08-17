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

def _probe_vram_subprocess() -> "tuple[float, float] | None":
    """(total_gb, reserved_gb), or None when there is no usable GPU.

    Runs in a THROWAWAY subprocess. This function is called from main.py, which
    is an ancestor of the grade runner and therefore of the isolated encode and
    IQA subprocesses. Touching torch.cuda here initialises a CUDA context in that
    ancestor, and when a GPU child later exits the ancestor faults with
    0xC0000005 and no traceback — the crash class tier_select.has_gpu() and
    run_profile._gpu_present() both go out of their way to avoid. Reintroducing
    it here would defeat both of them.
    """
    import subprocess
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch\n"
             "if torch.cuda.is_available() and torch.cuda.device_count() > 0:\n"
             "    p = torch.cuda.get_device_properties(0)\n"
             "    print(p.total_memory, torch.cuda.memory_reserved())\n"
             "else:\n"
             "    print('none')\n"],
            capture_output=True, text=True, timeout=180)
        out = r.stdout.strip().splitlines()[-1].strip()
        if out == "none":
            return None
        total, reserved = out.split()
        return float(total) / 1e9, float(reserved) / 1e9
    except Exception:
        return None


def validate_vram_overhead(required_gb: float = 5.0) -> None:
    """
    Assert free VRAM >= required_gb when --force-frontier is active.
    sys.exit() with CRITICAL message if the check fails.

    A machine with no GPU is a SUPPORTED configuration, not a failure. This used
    to sys.exit("--force-frontier requires CUDA"), which meant the documented CLI
    entry point refused to start on precisely the low-RAM CPU laptops the tier
    ladder exists to serve. tier_select caps such machines at the Fast encoder
    and grades them on the CPU; there is nothing here for this check to protect.
    """
    if not _FORCE_FRONTIER:
        return
    try:
        probe = _probe_vram_subprocess()
        if probe is None:
            logger.info(
                "VRAM pre-flight skipped — no GPU detected. Running on the CPU; "
                "the encoder tier will be selected to fit."
            )
            return
        total_gb, reserved_gb = probe
        free_gb = total_gb - reserved_gb
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
    # nvidia-smi, not torch.cuda — see _probe_vram_subprocess. server.py's polled
    # telemetry endpoint reads the device name the same way for the same reason.
    device_name = "unknown"
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            device_name = r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
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

    root = Path(__file__).resolve().parent.parent

    # ANY installed encoder tier satisfies this. The old check demanded
    # models/siglip2 specifically — the 7 GB open_clip fp32 fallback — so a lean
    # install carrying only models/siglip2_B_hf_fp16 (753 MB), which is exactly
    # what a CPU laptop should receive, was rejected as "weights missing".
    try:
        import tier_select
        import run_profile as _rp
        installed = [t for t in _rp.TIERS if tier_select.available(t)]
    except Exception as exc:                     # pragma: no cover — import guard
        logger.warning(f"Model integrity check could not run: {exc}")
        return

    if not installed:
        sys.exit(
            "CRITICAL: no vision encoder weights are installed.\n"
            "Run:  python -m scripts.fetch_models\n"
            "It selects the right encoder for this machine and downloads only that."
        )

    labels = ", ".join(tier_select.label(t) for t in installed)
    logger.info(f"✓ Model integrity: encoder tiers installed — {labels}")

    # The GGUF jury/critique model is optional; without it those panels fall back.
    if not (root / "models" / "deepseek-r1-8b-q5.gguf").exists():
        logger.warning(
            "Jury GGUF absent (models/deepseek-r1-8b-q5.gguf). "
            "Jury critique disabled — grading is unaffected."
        )
