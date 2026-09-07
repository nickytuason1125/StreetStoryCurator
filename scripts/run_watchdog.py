r"""
run_watchdog.py — audit trail for a live grading run.

Samples every 10 s and appends to reports/run_audit_<ts>.log:
  * free RAM, process census (launcher / backend / grade_runner / encode_worker)
  * server liveness (port 8000)
  * new interesting crash.log lines ([v2], encode_worker, REFUSED, errors, done)
  * catalog.json mtime/size changes (saves / checkpoint writes)
  * encoder_source marker (torch<->ONNX migration commit point)

Runs up to 3 hours, then exits. Pure observation — touches nothing.

Run:  venv\\Scripts\\pythonw.exe scripts\\run_watchdog.py
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRASH_LOG = ROOT / "crash.log"
CATALOG = ROOT / "cache" / "catalog.json"
SRC_MARKER = ROOT / "cache" / "encoder_source.txt"
MAX_SECONDS = 3 * 60 * 60
POLL = 10

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
AUDIT = ROOT / "reports" / f"run_audit_{ts}.log"

WATCH_PATTERNS = ("[v2]", "encode_worker", "REFUSED", "Refused", "error", "Error",
                  "WinError", "MemoryError", "Tight RAM", "advisory", "Encoder:",
                  "done", "checkpoint", "recovered", "FATAL", "ONNX")

_audit = AUDIT.open("a", encoding="utf-8", buffering=1)


def audit(line: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    _audit.write(f"[{stamp}] {line}\n")


def census() -> str:
    import psutil
    roles = {"launcher": 0, "backend": 0, "grade_runner": 0, "grade_worker": 0,
             "encode_worker": 0, "other_py": 0}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            n = (p.info["name"] or "").lower()
            if "python" not in n:
                continue
            cl = " ".join(p.info["cmdline"] or [])
            if "local_launcher" in cl:
                roles["launcher"] += 1
            if "server_impl" in cl or "--server-only" in cl:
                roles["backend"] += 1
            if "grade_runner" in cl:
                roles["grade_runner"] += 1
            if "grade_worker" in cl:
                roles["grade_worker"] += 1
            if "encode_worker" in cl:
                roles["encode_worker"] += 1
            if not any(k in cl for k in ("local_launcher", "server_impl", "grade_runner",
                                         "grade_worker", "encode_worker")):
                roles["other_py"] += 1
        except Exception:
            pass
    return " ".join(f"{k}={v}" for k, v in roles.items())


def free_ram() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1 << 30)
    except Exception:
        return -1.0


def server_up() -> bool:
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:8000/api/config",
                                     headers={"X-Requested-With": "FirstCut"})
        return urllib.request.urlopen(req, timeout=3).status == 200
    except Exception:
        return False


def main() -> int:
    audit(f"WATCHDOG START — audit file: {AUDIT}")
    crash_pos = CRASH_LOG.stat().st_size if CRASH_LOG.exists() else 0
    cat_key = None
    if CATALOG.exists():
        st = CATALOG.stat()
        cat_key = (st.st_mtime, st.st_size)
    marker_prev = SRC_MARKER.read_text(encoding="utf-8").strip() if SRC_MARKER.exists() else "(none)"
    audit(f"initial: encoder_source={marker_prev} catalog_key={cat_key}")

    server_seen_up = False
    grade_seen = False
    t_end = time.time() + MAX_SECONDS
    while time.time() < t_end:
        try:
            ram = free_ram()
            server = server_up()
            if server:
                server_seen_up = True
            line = f"ram={ram:.2f}GB server={'up' if server else 'down'} {census()}"

            # crash.log deltas
            if CRASH_LOG.exists():
                size = CRASH_LOG.stat().st_size
                if size > crash_pos:
                    with CRASH_LOG.open("r", encoding="utf-8", errors="replace") as f:
                        f.seek(crash_pos)
                        new = f.read(size - crash_pos)
                    crash_pos = size
                    hits = [ln.strip() for ln in new.splitlines()
                            if ln.strip() and any(w in ln for w in WATCH_PATTERNS)
                            and not ln.strip().startswith("[uvicorn]")]
                    for h in hits[-12:]:
                        audit("LOG " + h[:180])
                        if "done" in h and '"done"' in h:
                            audit(">>> GRADE STREAM REPORTED DONE <<<")
                        if "REFUSED" in h or "Refused" in h:
                            audit(">>> GRADE REFUSED <<<")
                        if "WinError" in h or "MemoryError" in h or "FATAL" in h:
                            audit(">>> CRASH SIGNATURE <<<")
                    if any("grade_worker" in h or "encode_worker" in h for h in hits):
                        grade_seen = True

            # catalog changes (saves / checkpoint writes)
            if CATALOG.exists():
                st = CATALOG.stat()
                key = (st.st_mtime, st.st_size)
                if key != cat_key:
                    audit(f"CATALOG CHANGED: mtime/size {cat_key} -> {key}")
                    cat_key = key

            # encoder-source commit point (torch<->ONNX migration marker)
            if SRC_MARKER.exists():
                marker = SRC_MARKER.read_text(encoding="utf-8").strip()
                if marker != marker_prev:
                    audit(f"ENCODER SOURCE COMMITTED: {marker_prev} -> {marker}")
                    marker_prev = marker

            audit(line)
        except Exception as e:
            audit(f"watchdog poll error: {type(e).__name__}: {e}")
        time.sleep(POLL)

    audit("WATCHDOG END (max duration reached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())