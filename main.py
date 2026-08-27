"""
FrameGrade — CLI entrypoint.

Usage
-----
    python main.py                          # normal
    python main.py --port 8080
    python main.py --force-frontier         # demand the Pro stack

--force-frontier  [OPTIONAL]
    Activates the Frontier 2026 enforcement layer before the server starts:
      1. Model integrity check — aborts if no encoder tier is installed.
      2. VRAM pre-flight    — aborts if a GPU is present with < 5.0 GB free.
                              Skipped entirely on a CPU-only machine.
      3. Legacy fallbacks   — permanently disabled for the life of the process.

    It used to be REQUIRED, and it exited on "No GPU detected" — so the
    documented entry point refused to start on the low-RAM CPU laptops the
    encoder tier ladder exists to serve. Enforcement is a deliberate choice for
    a machine that can meet it, not the price of admission.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FrameGrade — Frontier 2026 Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--force-frontier",
        action="store_true",
        default=os.environ.get("FORCE_FRONTIER", "").strip() in ("1", "true", "yes"),
        help=(
            "Enforce Frontier 2026 stack — no legacy fallbacks. "
            "Requires SigLIP-2 + DeepSeek-7B weights and ≥5 GB free VRAM. "
            "(Also activated by env: FORCE_FRONTIER=1)"
        ),
    )
    p.add_argument("--host",   default="127.0.0.1", help="Bind host  (default: 127.0.0.1)")
    p.add_argument("--port",   type=int, default=8000, help="Bind port  (default: 8000)")
    p.add_argument("--reload", action="store_true", default=False, help="Hot-reload (dev only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve working dir so relative paths (models/, cache/, frontend/dist/) work.
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    # Apply flag before any model imports — frontier_config is a module-level singleton.
    from frontier_config import (
        set_force_frontier,
        validate_vram_overhead,
        check_model_integrity,
    )
    set_force_frontier(args.force_frontier)

    # Say it before the cull, not during it. This costs a psutil read and no
    # model imports, and it never blocks: browsing, rating and exporting all
    # work with no memory to spare, so refusing to start would take those away
    # to prevent a failure that already reports itself where it happens.
    try:
        from system_check import read_and_assess
        _sys_state = read_and_assess()
        (logger.warning if _sys_state.level in ("tight", "insufficient")
         else logger.info)(_sys_state.message)
    except Exception as _e:                       # never let the check stop launch
        logger.info(f"System check skipped: {_e}")

    if args.force_frontier:
        logger.info("--force-frontier active — running pre-flight checks…")
        check_model_integrity()                   # aborts if no encoder is installed
        validate_vram_overhead(required_gb=5.0)   # skipped when there is no GPU
        logger.info("Pre-flight passed")
    else:
        logger.info(
            "Starting in normal mode. The encoder tier is selected at grade time "
            "to fit this machine; pass --force-frontier to demand the Pro stack."
        )

    import uvicorn
    logger.info(
        f"Starting FrameGrade  host={args.host}  port={args.port}  "
        f"force_frontier={args.force_frontier}"
    )
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
