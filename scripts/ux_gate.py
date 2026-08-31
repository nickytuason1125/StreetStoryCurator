"""ux_gate.py — UX regression gate (Part 4 of the UX suite).

Runs the API benchmark (Part 1) and the real-browser benchmark (Part 2),
then checks their results against committed thresholds. Any breach =
non-zero exit, so a release build can gate on it.

  usage:
    venv\\Scripts\\python.exe scripts\\ux_gate.py [--skip-browser]

Thresholds live in THRESHOLDS below (committed — changes are reviewable).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# Warm-path thresholds. Generous enough to pass on a loaded dev machine,
# tight enough to catch real regressions (today's numbers are 5-50x under).
THRESHOLDS = {
    "api": {
        "health_warm.p50": 100,
        "catalog_slim.p50": 500,       # cached slim serve (was 2038)
        "photo_detail_hot.p50": 50,    # cached parse (was 938)
        "thumb_warm_avg.p50": 25,
        "star_roundtrip.p50": 1500,
    },
    "browser": {
        "domContentLoaded": 3000,
        "gridPopulated": 8000,         # resume -> 20+ thumbs (was n/a)
        "interactionMs": 15000,        # loupe (full-res load; known gap)
        "heapAfterScrollMB": 600,
    },
    "browser_floor": {
        "scroll.scrollFps": 45,
    },
}

FAILS: list[str] = []


def check(name: str, value, limit) -> None:
    if value is None:
        FAILS.append(f"{name}: no measurement")
        print(f"   [FAIL] {name}: no measurement")
        return
    ok = value <= limit
    if not ok:
        FAILS.append(f"{name}: {value} > {limit}")
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {value} (limit {limit})")


def check_floor(name: str, value, floor) -> None:
    if value is None:
        FAILS.append(f"{name}: no measurement")
        print(f"   [FAIL] {name}: no measurement")
        return
    ok = value >= floor
    if not ok:
        FAILS.append(f"{name}: {value} < {floor}")
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {value} (floor {floor})")


def main() -> int:
    # 1. API bench (grade excluded — the gate is about UX latency)
    print("== API benchmark ==")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_api.py"),
                    "--skip-grade"], cwd=str(ROOT), check=False,
                   capture_output=True, text=True)

    # 2. browser bench
    if "--skip-browser" not in sys.argv:
        print("== browser benchmark ==")
        subprocess.run(["node", str(ROOT / "frontend" / "scripts" / "ux_bench.mjs")],
                       cwd=str(ROOT / "frontend"), check=False,
                       capture_output=True, text=True)

    # 3. thresholds
    print("== thresholds ==")
    api = {}
    browser = {}
    try:
        api = json.loads((REPORTS / "bench_api.json").read_text(encoding="utf-8"))["sections"]
    except Exception as e:
        FAILS.append(f"bench_api.json unreadable: {e}")
    try:
        browser = json.loads((REPORTS / "ux_bench.json").read_text(encoding="utf-8"))["checks"]
    except Exception as e:
        FAILS.append(f"ux_bench.json unreadable: {e}")

    for path, limit in THRESHOLDS["api"].items():
        sec, _, field = path.partition(".")
        check(path, (api.get(sec) or {}).get(field), limit)
    for path, limit in THRESHOLDS["browser"].items():
        check(path, browser.get(path), limit)
    for path, floor in THRESHOLDS["browser_floor"].items():
        sec, _, field = path.partition(".")
        check_floor(path, (browser.get(sec) or {}).get(field), floor)

    # 4. report
    gate = {
        "fails": FAILS,
        "thresholds": THRESHOLDS,
        "api": {k: v for k, v in api.items()},
        "browser": browser,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ux_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    md = ["# UX Gate", ""]
    if FAILS:
        md += ["**FAILED**"] + [f"- {f}" for f in FAILS]
    else:
        md += ["**PASS** — all UX thresholds met."]
    (REPORTS / "ux_gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nUX GATE: " + ("FAIL — " + "; ".join(FAILS) if FAILS else "PASS"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())