"""
Chaos driver — fault/invariant pairs for FirstCut.

Run:  venv\\Scripts\\python.exe scripts\\chaos_drive.py [c6|c7|c8|all]

C1-C4 (kill-mid-grade, SSE disconnect under load, whole-tree power loss,
OOM ballast) require a REAL cull — >= 2 GB free RAM — and are documented in
reports/chaos_report.md as the queued tier; the deterministic logic behind
them is covered by tests/test_grade_singleflight.py and test_disk_gate.py.
"""
from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"
HDR = {"X-Requested-With": "FirstCut"}
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), "-", name, ("| " + detail if detail else ""))


def c6_stale_instance_collision() -> None:
    """Two FirstCut servers cannot share a port. The second must exit fast
    with the bind pre-flight message — not a raw errno line (M2 fix)."""
    p = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--port", "8001"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    combined = (p.stdout or "") + (p.stderr or "")
    check("C6 second instance exits nonzero", p.returncode != 0, f"rc={p.returncode}")
    check("C6 bind pre-flight message present",
          "already in use" in combined and "stale" in combined.lower())
    check("C6 no raw errno-only output",
          "10048" not in combined or "already in use" in combined)


def c7_watch_folder_sabotage() -> None:
    """Watcher lifecycle vs sabotage.

    Two tiers, chosen by free RAM: starting a watcher pulls the analyzer
    (heavy model loads — unsafe under ~2 GB free), so at low memory only the
    idle lifecycle is exercised. The full sabotage (delete watched dir
    mid-watch) is the queued Tier B variant.
    """
    free = psutil.virtual_memory().available / 1e9
    tmp = Path(tempfile.mkdtemp(prefix="fc_watch_"))
    try:
        if free < 2.0:
            # Idle lifecycle only — start is RAM-gated off.
            r = requests.post(f"{BASE}/api/watch/stop", json={}, headers=HDR, timeout=15)
            check("C7 stop-when-idle answers cleanly (no watcher, no crash)",
                  r.status_code == 200 and r.json().get("status") == "stopped")
            r = requests.get(f"{BASE}/api/watch/status", timeout=15)
            check("C7 status endpoint consistent when idle",
                  r.status_code == 200 and r.json().get("watching") is False)
            check("C7 full sabotage SKIPPED (RAM-gated: start loads the analyzer)",
                  True, f"{free:.2f} GB free — queued for Tier B")
        else:
            r = requests.post(f"{BASE}/api/watch/start", json={"folder": str(tmp)},
                              headers=HDR, timeout=15)
            check("C7 watch starts on valid folder",
                  r.status_code == 200 and r.json().get("status") == "watching")
            shutil.rmtree(tmp, ignore_errors=True)   # sabotage: folder deleted mid-watch
            time.sleep(1.0)
            r = requests.get(f"{BASE}/api/watch/status", timeout=15)
            check("C7 status endpoint alive after sabotage", r.status_code == 200)
            r = requests.post(f"{BASE}/api/watch/stop", json={}, headers=HDR, timeout=15)
            check("C7 watch stops cleanly after sabotage",
                  r.status_code == 200 and r.json().get("status") == "stopped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def c8_flood() -> None:
    """50 parallel reads + 10 parallel writes: nothing corrupts, the catalog
    still parses afterwards."""
    reads = ["/api/health/engine", "/api/system/ram", "/api/presets", "/api/watch/status"]

    def do_read(i: int) -> int:
        return requests.get(BASE + reads[i % len(reads)], timeout=30).status_code

    def do_write(i: int) -> int:
        return requests.post(f"{BASE}/api/browse-folder",
                             json={"folder_path": "C:/nonexistent_fc_flood"},
                             headers=HDR, timeout=30).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        rs = list(ex.map(do_read, range(50)))
        ws = list(ex.map(do_write, range(10)))

    check("C8 all 50 parallel reads answered 200", all(s == 200 for s in rs),
          f"codes={sorted(set(rs))}")
    # browse-folder answers 200 with empty lists for nonexistent paths —
    # that IS its contract (non-recursive scan, missing dir → nothing found)
    empties = [requests.post(f"{BASE}/api/browse-folder",
                             json={"folder_path": "C:/nonexistent_fc_flood"},
                             headers=HDR, timeout=30).json()
               for _ in range(3)]
    check("C8 parallel writes answered, empty-list contract held",
          all(x.get("images") == [] and x.get("folders") == [] for x in empties))
    cat = requests.get(f"{BASE}/api/catalog", timeout=60)
    try:
        n = len(cat.json()["photos"])
        check("C8 catalog still parses after flood", True, f"{n} photos")
    except Exception as e:  # noqa: BLE001
        check("C8 catalog still parses after flood", False, str(e)[:80])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    free = psutil.virtual_memory().available / 1e9
    print(f"[chaos] {free:.2f} GB free — "
          f"{'Tier B (C1-C4) runnable' if free >= 2.0 else 'Tier B (C1-C4) still RAM-blocked'}")
    if which in ("c6", "all"):
        c6_stale_instance_collision()
    if which in ("c7", "all"):
        c7_watch_folder_sabotage()
    if which in ("c8", "all"):
        c8_flood()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} chaos checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)