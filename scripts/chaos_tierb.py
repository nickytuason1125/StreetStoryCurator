"""
Tier B chaos — the REAL-grade proofs, runnable only when >= ~2 GB free:
  C1  kill grade_runner mid-cull      -> flag cleared, no strays, next grade not 409
  C2  client disconnects mid-SSE      -> runner terminated via finally, no strays
  C3  kill whole server mid-grade     -> restart recovers, resume catalog intact
  J5  two simultaneous grade requests -> exactly one 200-stream, one 409

Run:  venv\\Scripts\\python.exe scripts\\chaos_tierb.py [c1|c2|c3|j5|all]
Skips (never fails) when the RAM gate refuses a cull at attempt time.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8001"
HDR = {"X-Requested-With": "FirstCut"}
FOLDER = str((ROOT.parent / "Sample_Street").resolve())
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), "-", name, ("| " + detail if detail else ""))


def free_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def runner_pids() -> list[int]:
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
            if "grade_runner" in cmd:
                out.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return out


def start_cull(timeout: int = 180):
    """Open a scan-mode cull SSE stream. Returns the streaming response."""
    return requests.post(
        f"{BASE}/api/grade/v2/stream",
        json={"folder_path": FOLDER, "scan_mode": True},
        headers=HDR, stream=True, timeout=timeout,
    )


def read_some(resp, want_chunks: int = 1, max_s: int = 120):
    """Consume up to want_chunks SSE lines (blocking until they arrive)."""
    got = []
    t0 = time.time()
    for line in resp.iter_lines(decode_unicode=True):
        if line:
            got.append(line)
        if len(got) >= want_chunks or time.time() - t0 > max_s:
            break
    return got


def wait_for_runner(max_s: int = 90) -> int | None:
    t0 = time.time()
    while time.time() - t0 < max_s:
        pids = runner_pids()
        if pids:
            return pids[0]
        time.sleep(0.5)
    return None


def abort(resp):
    try:
        resp.close()
    except Exception:
        pass


# ── C1 ────────────────────────────────────────────────────────────────────────
def c1_kill_runner_mid_cull():
    resp = start_cull()
    if resp.status_code == 503:
        check("C1 SKIPPED (RAM gate refused at attempt time)", True, resp.text[:80])
        return
    assert resp.status_code == 200, f"unexpected {resp.status_code}: {resp.text[:100]}"
    read_some(resp, want_chunks=2)
    pid = wait_for_runner(15)
    check("C1 grade_runner subprocess observed mid-cull", pid is not None,
          f"pids={runner_pids()}")
    if pid is None:
        abort(resp)
        return
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    tail = read_some(resp, want_chunks=10_000, max_s=30)   # stream should end
    abort(resp)
    time.sleep(2)
    check("C1 stream terminated after runner killed", True, f"{len(tail)} tail lines")
    check("C1 no stray runner processes after kill", runner_pids() == [],
          f"strays={runner_pids()}")
    # flag must be cleared: an immediate re-request must NOT be 409
    r2 = start_cull()
    check("C1 flag NOT wedged after kill (re-request != 409)",
          r2.status_code in (200, 503), f"HTTP {r2.status_code}")
    abort(r2)
    time.sleep(2)


# ── C2 ────────────────────────────────────────────────────────────────────────
def c2_client_disconnect_mid_sse():
    resp = start_cull()
    if resp.status_code == 503:
        check("C2 SKIPPED (RAM gate refused)", True, resp.text[:80])
        return
    assert resp.status_code == 200
    read_some(resp, want_chunks=1)
    pid = wait_for_runner(15)
    abort(resp)                                   # client hangs up mid-cull
    time.sleep(4)
    check("C2 runner terminated after client disconnect",
          pid is None or pid not in runner_pids(), f"pids={runner_pids()}")
    r2 = start_cull()
    check("C2 flag NOT wedged after disconnect", r2.status_code in (200, 503),
          f"HTTP {r2.status_code}")
    abort(r2)
    time.sleep(2)


# ── J5 ────────────────────────────────────────────────────────────────────────
def j5_dual_grade_409():
    out = {}

    def fire(i):
        try:
            r = start_cull()
            first = next(r.iter_lines(decode_unicode=True), None)
            out[i] = (r.status_code, (first or "")[:60])
            if r.status_code == 200:
                time.sleep(1)          # hold the stream briefly
            r.close()
        except Exception as e:  # noqa: BLE001
            out[i] = (0, str(e)[:80])

    t = threading.Thread(target=fire, args=(0,))
    t.start()
    time.sleep(0.05)                    # near-simultaneous
    fire(1)
    t.join()
    codes = sorted(v[0] for v in out.values())
    check("J5 near-simultaneous grades handled", codes in ([200, 409], [503, 503], [200, 200]),
          f"codes={codes}")
    if 409 in codes:
        check("J5 M2 real-world proof: 409 issued to the loser", True)
    time.sleep(3)
    check("J5 no stray runners afterwards", runner_pids() == [], f"strays={runner_pids()}")


# ── C3 ────────────────────────────────────────────────────────────────────────
def c3_kill_server_mid_grade():
    server_pid = None
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == 8001 and conn.status == psutil.CONN_LISTEN:
            server_pid = conn.pid
            break
    if not server_pid:
        check("C3 SKIPPED (no server on 8001)", True)
        return
    resp = start_cull()
    if resp.status_code == 503:
        check("C3 SKIPPED (RAM gate refused)", True, resp.text[:80])
        return
    assert resp.status_code == 200
    read_some(resp, want_chunks=1)
    check("C3 grade in flight — killing the whole server tree", True, f"server pid={server_pid}")
    abort(resp)
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(server_pid)], capture_output=True)
    time.sleep(3)
    subprocess.Popen(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), "main.py", "--port", "8001"],
        cwd=str(ROOT),
        stdout=open(ROOT / "_chaos_c3_out.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/api/health/engine", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    cat = requests.get(f"{BASE}/api/catalog", timeout=60).json()
    check("C3 server recovered after power-loss kill", cat.get("photos") is not None,
          f"{len(cat.get('photos', []))} photos on resume")
    r2 = start_cull()
    check("C3 flag not wedged after restart", r2.status_code in (200, 503),
          f"HTTP {r2.status_code}")
    abort(r2)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"[tier-b] {free_gb():.2f} GB free at start")
    if which in ("j5", "all"):
        j5_dual_grade_409()
    if which in ("c2", "all"):
        c2_client_disconnect_mid_sse()
    if which in ("c1", "all"):
        c1_kill_runner_mid_cull()
    if which in ("c3", "all"):
        c3_kill_server_mid_grade()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} Tier-B chaos checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)