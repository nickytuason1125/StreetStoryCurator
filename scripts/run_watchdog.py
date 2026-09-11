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
# FIRSTCUT_WATCHDOG_HOURS overrides the 3 h default — a long cull session
# should not lose its audit trail halfway through ("max duration reached").
try:
    _watch_hours = float(os.environ.get("FIRSTCUT_WATCHDOG_HOURS", "3") or 3)
except ValueError:
    _watch_hours = 3.0
MAX_SECONDS = max(1, int(_watch_hours * 60 * 60))
try:
    POLL = max(1, int(os.environ.get("FIRSTCUT_WATCHDOG_POLL_S", "10") or 10))
except ValueError:
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


def census() -> dict:
    import psutil
    roles = {"launcher": 0, "backend": 0, "grade_runner": 0, "grade_worker": 0,
             "encode_worker": 0, "other_py": 0}
    for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            n = (p.info["name"] or "").lower()
            if "python" not in n:
                continue
            # venv redirector shims (2026-09-07 audit fix): venv\Scripts\python(w).exe
            # is a tiny launcher that spawns the BASE interpreter as a child process
            # and waits. Every venv-spawned process therefore appears TWICE in the
            # process list — shim (exe under venv\Scripts) + real worker (exe = the
            # base Python, running with the venv's site-packages) — which inflated
            # every role 2x in past audits ("launcher=6" really meant 3 launchers,
            # "backend=4" meant 2). Count only the real worker; the shim carries no
            # role of its own.
            exe = (p.info["exe"] or "").replace("/", "\\").lower()
            if "\\venv\\scripts\\" in exe:
                continue
            cl = " ".join(p.info["cmdline"] or [])
            # Mutually exclusive roles (2026-09-07 audit fix): the detached
            # backend runs local_launcher.py --server-only, so it matches BOTH
            # "local_launcher" and "--server-only" — independent ifs counted it
            # in both roles, inflating launcher by the number of backends.
            # One process, one role; most specific pattern wins.
            if "--server-only" in cl or "server_impl" in cl:
                roles["backend"] += 1
            elif "grade_runner" in cl:
                roles["grade_runner"] += 1
            elif "grade_worker" in cl:
                roles["grade_worker"] += 1
            elif "encode_worker" in cl:
                roles["encode_worker"] += 1
            elif "local_launcher" in cl:
                roles["launcher"] += 1
            else:
                roles["other_py"] += 1
        except Exception:
            pass
    return roles


def census_line(roles: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in roles.items())


def free_ram() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1 << 30)
    except Exception:
        return -1.0


def cull_progress():
    """(percent, desc, age_s) from the newest grade progress file, or None.

    The 2026-09-07 trail showed grade_runner=1 for an hour with no idea how
    far the cull had actually gotten — the SSE progress file is the source of
    truth, so sample it here. Best effort: any problem reads as 'no info'.

    The age travels with the value so the caller can refuse to report a
    PHANTOM cull: progress files live in %TEMP% and outlive their run, and a
    dead run's file had the watchdog reporting "cull=46%" hours after the
    grade died (2026-09-08 morning) — a different lie, same effect as the
    stall blindness.
    """
    import glob
    import json as _json
    try:
        _tmp = os.environ.get("TEMP", os.environ.get("TMP", ""))
        files = glob.glob(os.path.join(_tmp, "*.gradereq.json.progress.jsonl"))
        if not files:
            return None
        newest = max(files, key=os.path.getmtime)
        age = time.time() - os.path.getmtime(newest)
        last = ""
        with open(newest, "r", encoding="utf-8", errors="replace") as f:
            for last in f:
                pass
        d = _json.loads(last)
        return round(float(d.get("progress", 0)) * 100), str(d.get("desc", ""))[:48], age
    except Exception:
        return None


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
    # RAM pathline v2, Phase 2 — progress-shape detection. The watchdog used to
    # count processes only, so a crawling or silently-restarting cull looked
    # healthy for hours: on 2026-09-08 progress went 0→46%, reset to 7% (the
    # GPU attempt failed and the ladder fell back to a full CPU re-encode) and
    # then crawled at ~1%/hour — the census said grade_runner=1 the whole time.
    _prog_val = None          # last seen progress %
    _last_advance_t = None    # time of the last value change
    _stale_warned = False     # phantom-cull guard: warn once per run, not per poll
    # End-of-run summary counters (2026-09-07 plan Phase 4).
    polls = up_polls = crash_sig = catalog_saves = 0
    ram_min = ram_max = None
    t_end = time.time() + MAX_SECONDS
    while time.time() < t_end:
        try:
            ram = free_ram()
            server = server_up()
            polls += 1
            if server:
                server_seen_up = True
                up_polls += 1
            if ram >= 0:
                ram_min = ram if ram_min is None else min(ram_min, ram)
                ram_max = ram if ram_max is None else max(ram_max, ram)
            _roles = census()
            line = f"ram={ram:.2f}GB server={'up' if server else 'down'} {census_line(_roles)}"
            _runner_alive = _roles.get("grade_runner", 0) > 0
            _prog = cull_progress()
            if _prog:
                _pct, _desc, _age = _prog
                # Phantom-cull guard: only report a cull if a runner is alive
                # OR the progress file was written recently.
                if _age > 600 and not _runner_alive:
                    if not _stale_warned:
                        audit(f">>> STALE cull progress file ignored ({int(_age // 60)} min old, "
                              "no grade_runner alive) — refusing to report a phantom cull <<<")
                        _stale_warned = True
                else:
                    line += f" cull={_pct}% [{_desc}]"
                    _now_m = time.monotonic()
                    if _prog_val is not None:
                        if _pct < _prog_val - 2:
                            audit(f">>> CULL PROGRESS RESET: {_prog_val}% -> {_pct}% "
                                  "(an encode attempt failed and the ladder re-ran from "
                                  "scratch — GPU->CPU fallback re-encodes everything) <<<")
                        if _pct != _prog_val:
                            _last_advance_t = _now_m
                        elif _last_advance_t is not None:
                            _stall_s = _now_m - _last_advance_t
                            if _stall_s >= 900:
                                line += f" [STALL {int(_stall_s // 60)}m — progress unchanged]"
                    else:
                        _last_advance_t = _now_m
                    _prog_val = _pct

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
                            crash_sig += 1
                            audit(">>> CRASH SIGNATURE <<<")
                    if any("grade_worker" in h or "encode_worker" in h for h in hits):
                        grade_seen = True

            # catalog changes (saves / checkpoint writes)
            if CATALOG.exists():
                st = CATALOG.stat()
                key = (st.st_mtime, st.st_size)
                if key != cat_key:
                    catalog_saves += 1
                    audit(f"CATALOG CHANGED: mtime/size {cat_key} -> {key}")
                    cat_key = key

            # encoder-source commit point (torch<->ONNX migration marker)
            if SRC_MARKER.exists():
                marker = SRC_MARKER.read_text(encoding="utf-8").strip()
                if marker != marker_prev:
                    audit(f"ENCODER SOURCE COMMITTED: {marker_prev} -> {marker}")
                    marker_prev = marker

            audit(line)
            # Live status snapshot — one file, one glance, no trail-diving.
            try:
                (ROOT / "reports" / "watchdog_latest.txt").write_text(
                    f"{AUDIT.name}\n[{datetime.now().strftime('%H:%M:%S')}] {line}\n",
                    encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            audit(f"watchdog poll error: {type(e).__name__}: {e}")
        time.sleep(POLL)

    # End-of-run verdict (2026-09-07 plan Phase 4): the trail used to end with
    # a bare "max duration reached", leaving the reader to reconstruct what
    # happened by hand. Write the numbers next to the events.
    # End of watch: if a grade is STILL running, say so loudly. The 2026-09-08
    # trail expired at 09:06 while the cull was crawling at the wall, and the
    # SUMMARY read like a clean end — the reader had to notice the absence.
    try:
        if census().get("grade_runner", 0) > 0:
            audit(">>> WATCHDOG EXPIRED WITH A GRADE STILL RUNNING — "
                  "relaunch the watchdog to keep the audit trail, and check on the cull <<<")
    except Exception:
        pass
    audit("WATCHDOG END — SUMMARY")
    audit(f"  polls={polls}  server_up={up_polls}/{polls}")
    _lo = f"{ram_min:.2f}" if ram_min is not None else "?"
    _hi = f"{ram_max:.2f}" if ram_max is not None else "?"
    audit(f"  RAM free min/max: {_lo} / {_hi} GB")
    audit(f"  crash signatures: {crash_sig}   catalog saves: {catalog_saves}")
    audit(f"  grade activity seen: {grade_seen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())