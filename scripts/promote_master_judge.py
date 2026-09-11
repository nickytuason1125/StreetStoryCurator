"""Promote a WON MasterJudge challenger into the shipped master algorithm.

The final step of the two-phase master-algo contract:

    BUILD  ratings train challengers (scripts/master_backtest.py --fit)
           → a challenger that beats the incumbent on held-out photos is
             marked promoted=true in cache/master_judge.json
    SHIP   THIS script bakes that promoted record into
           data/master_judge_defaults.json — shipped with the app, graded
           by default for every install, no flags, no ratings
    RUN    users just grade; they never rate anything

REFUSES to ship an unpromoted, stale, or incomplete record: a challenger
that lost its exam can never become the master. Use --show to inspect the
current cache and shipped records without writing anything.

Usage:
    venv\\Scripts\\python.exe scripts\\promote_master_judge.py --show
    venv\\Scripts\\python.exe scripts\\promote_master_judge.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 — UTF-8 so the report can never die in
# a print after the promotion already succeeded.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# scripts/ has no __init__.py — it works as a namespace package only when
# the repo root is on sys.path. Running this file directly (its documented
# usage) puts scripts/ itself on sys.path[0], not the repo root, so
# `import scripts.master_backtest` inside check_aadb_gate() would otherwise
# crash instead of gating. pytest already puts the repo root on sys.path,
# so this is a no-op duplicate under test.
sys.path.insert(0, str(_ROOT))

import master_judge as mj  # noqa: E402


def check_aadb_gate(min_rho: float = 0.149) -> tuple:
    """(passes, reason). min_rho defaults to the chance-level rank agreement
    already measured for this project's baseline grader
    (project_grading_measurement_bounds: +0.149) — a head that can't beat
    chance on its own labeled AADB data is noise, not signal, regardless of
    what it does to the live baseline's rho_holdout."""
    import scripts.master_backtest as mb
    metrics = mb.read_aadb_metrics()
    if metrics is None:
        return False, "no trained AADB head found — run aadb_setup.py first"
    rho = metrics.get("rho_holdout")
    if rho is None or rho != rho:  # None or NaN
        return False, f"AADB held-out rho is undefined ({rho})"
    if rho < min_rho:
        return False, f"AADB held-out rho {rho} is below the chance-level floor {min_rho}"
    return True, f"AADB held-out rho {rho} clears the chance-level floor {min_rho}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", action="store_true",
                    help="inspect cache + shipped records without writing")
    args = ap.parse_args()

    def _dump(label: str, path: Path) -> None:
        if not path.exists():
            print(f"{label}: (absent) {path}")
            return
        d = json.loads(path.read_text(encoding="utf-8"))
        print(f"{label}: {path}")
        print(f"    promoted      : {d.get('promoted')}")
        print(f"    n             : {d.get('n')}")
        print(f"    rho holdout   : {d.get('rho_holdout')}")
        print(f"    rho baseline  : {d.get('rho_baseline')}")
        print(f"    generated     : {d.get('generated')}")
        if d.get("shipped_from_cache_at"):
            print(f"    shipped at    : {d.get('shipped_from_cache_at')}")

    _dump("CACHE  ", mj._WEIGHTS_PATH)
    _dump("SHIPPED", mj._SHIPPED_PATH)
    if args.show:
        return 0

    if "AADB" in mj.FEATURES:  # only gate on this once the feature exists
        aadb_ok, aadb_reason = check_aadb_gate()
        print(f"[promote] AADB gate: {aadb_reason}")
        if not aadb_ok:
            print("[promote] REFUSING promotion — AADB gate not satisfied.")
            return 1

    print("\n[promote] attempting promotion…")
    res = mj.promote_to_shipped()
    print(json.dumps(res, indent=2))
    if not res.get("shipped"):
        print("\n[promote] NOT shipped. Only a challenger that WON its "
              "held-out exam can become the master algorithm.")
        return 1
    print("\n[promote] SHIPPED. This master judge is now part of the "
          "algorithm: every install grades with it by default, no flags, "
          "no ratings. Commit data/master_judge_defaults.json and release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
