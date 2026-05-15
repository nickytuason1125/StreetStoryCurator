"""
Street Story Curator — CLI entrypoint.

Usage
-----
    python main.py --force-frontier         # required — Frontier 2026 strict mode
    python main.py --force-frontier --port 8080
    python main.py --host 0.0.0.0 --force-frontier

--force-frontier  [REQUIRED]
    Activates the Frontier 2026 enforcement layer before the server starts:
      1. Model integrity check — aborts if SigLIP-2 or Vision-R1-7B weights are absent.
      2. VRAM pre-flight    — aborts if free VRAM < 5.0 GB.
      3. Legacy fallbacks   — permanently disabled for the life of the process.

When launched through the Tauri desktop app, server.py is invoked directly
(no argparse), so set env var FORCE_FRONTIER=1 for packaged builds.
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
        description="Street Story Curator — Frontier 2026 Edition",
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


_FRONTIER_REQUIRED_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║            FRONTIER 2026 — ENFORCEMENT REQUIRED                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Street Story Curator requires --force-frontier to launch.       ║
║  Legacy graders have been permanently removed.                   ║
║                                                                  ║
║  Run:   python main.py --force-frontier                          ║
║     or  FORCE_FRONTIER=1 python main.py                          ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main() -> None:
    args = parse_args()

    if not args.force_frontier:
        print(_FRONTIER_REQUIRED_BANNER, file=sys.stderr)
        sys.exit(1)

    # Resolve working dir so relative paths (models/, cache/, frontend/dist/) work.
    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    # Apply flag before any model imports — frontier_config is a module-level singleton.
    from frontier_config import (
        set_force_frontier,
        validate_vram_overhead,
        check_model_integrity,
    )
    set_force_frontier(True)

    logger.info("--force-frontier active — running pre-flight checks…")
    check_model_integrity()              # aborts if 2026 weights are missing
    validate_vram_overhead(required_gb=5.0)   # aborts if VRAM is insufficient
    logger.info("Pre-flight passed — all Frontier 2026 requirements met")

    import uvicorn
    logger.info(
        f"Starting Street Story Curator  host={args.host}  port={args.port}  "
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
