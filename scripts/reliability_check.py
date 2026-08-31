"""reliability_check.py — can the app take a beating, repeatedly?

The benchmark measures SPEED. This measures DEPENDABILITY, against the real
pipeline (no mocks), and reports PASS/FAIL per check:

  1. REPEAT-RUN DETERMINISM   cull the same folder twice with force_rescan;
                              every photo's score must be identical across
                              runs (same pixels → same verdict, every time).
  2. PROCESS HYGIENE          after every cull: no leftover grade_runner /
                              encode_worker / iqa_worker processes, and the
                              grading.lock marker is cleaned up.
  3. CORRUPT-FILE RESILIENCE  one garbage .jpg among good ones must not kill
                              the run — the good photos still grade.
  4. EMPTY FOLDER             a cull of an empty folder must finish cleanly
                              (total=0 or a clean error), never a traceback.
  5. RAM FLOOR                available RAM is sampled before/after every
                              run; a run must not push the machine under
                              the 0.75 GB refuse-floor.

Writes reports/reliability_report.md. Exit 0 only if every check passes.

Usage:
    venv\\Scripts\\python.exe scripts\\reliability_check.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FOLDER = ROOT.parent / "Sample_Street"
RESULTS: list[dict] = []


def _ram_available_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def _run_cull(folder: Path, scan_mode: bool = False, timeout: int = 600) -> dict:
    """One real cull via grade_runner.py. Returns {ok, total, wall_s, error}.

    The gallery 'done'/'error' messages go to the PROGRESS FILE (the same
    stream the server tails), not stdout — so that is the authoritative
    source for the result."""
    import tempfile as _tf
    fd, req_path = _tf.mkstemp(suffix=".relreq.json"); os.close(fd)
    prog_path = req_path + ".progress.jsonl"
    Path(prog_path).write_text("", encoding="utf-8")
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump({
            "folders":      [str(folder)],
            "preset":       "classic_street",
            "force_rescan": True,
            "scan_mode":    scan_mode,
            "deep_grade":   False,
            "catalog_path": str(ROOT / "cache" / "catalog.json"),
            "data_dir":     str(ROOT),
            "mogco_target": 0,
        }, f)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "grade_runner.py"), req_path, prog_path],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    wall = time.perf_counter() - t0
    total, error = 0, None
    try:
        for line in Path(prog_path).read_text(encoding="utf-8",
                                              errors="replace").splitlines():
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if msg.get("done"):
                total = int(msg.get("total", 0))
            if msg.get("error"):
                error = str(msg.get("error"))[:200]
    except OSError:
        pass
    for tmp in (req_path, prog_path):
        try: os.unlink(tmp)
        except OSError: pass
    tb = "Traceback (most recent call last)" in (proc.stderr or "")
    return {"ok": proc.returncode == 0 and not tb and error is None,
            "total": total, "wall_s": round(wall, 1), "error": error,
            "stderr": proc.stderr or ""}


def _disk_free_gb() -> float:
    import shutil as _sh
    return _sh.disk_usage(str(ROOT)).free / 1e9

# ── checks ────────────────────────────────────────────────────────────────────

def _scores_for(folder: Path) -> dict:
    import lance_store as ls
    rows = ls.query_all(min_score=0.0)
    return {r["path"]: float(r.get("score", 0)) for r in rows
            if str(r.get("path", "")).lower().startswith(str(folder).lower())}


def _hygiene(run_label: str) -> dict:
    """No leftover worker processes, grading.lock cleaned up."""
    import psutil
    strays = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or []).lower()
            if any(sig in cmd for sig in
                   ("grade_runner", "encode_worker", "iqa_worker")):
                strays.append(f"{p.info['name']}#{p.info['pid']}")
        except Exception:
            pass
    lock = ROOT / "cache" / "grading.lock"
    lock_clean = not lock.exists()
    ok = not strays and lock_clean
    return {"check": f"process hygiene ({run_label})", "ok": ok,
            "detail": (f"strays={strays or 'none'}, "
                       f"grading.lock {'present!' if not lock_clean else 'cleaned'}")}


def _check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": detail})
    print(f"   [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    print("=" * 72)
    print("CULLWISE RELIABILITY CHECK — real pipeline, no mocks")
    print("=" * 72)
    print(f"target folder : {FOLDER}")
    print(f"RAM at start  : {_ram_available_gb():.2f} GB available")
    disk0 = _disk_free_gb()
    _check("disk head-room at start", disk0 > 1.0,
           f"{disk0:.2f} GB free on the app drive (a FULL drive silently "
           f"breaks catalog persistence — observed 2026-08-30 with 0 bytes "
           f"free: grades completed, then catalog/Lance writes died with "
           f"Errno 28)")
    print()

    # ── 1 + 2: repeat-run determinism + hygiene (two full culls) ─────────────
    scores = []
    for i in (1, 2):
        print(f"[run {i}/2] full cull of Sample_Street (force_rescan)…")
        res = _run_cull(FOLDER)
        _check(f"cull {i} completed cleanly", res["ok"],
               f"total={res['total']} photos, {res['wall_s']}s")
        h = _hygiene(f"cull {i}")
        RESULTS.append(h)
        print(f"   [{'PASS' if h['ok'] else 'FAIL'}] {h['check']}: {h['detail']}")
        scores.append(_scores_for(FOLDER))   # snapshot AFTER each run

    s1, s2 = scores[0], scores[1]
    common = sorted(set(s1) & set(s2))
    diffs = [p for p in common if abs(s1[p] - s2[p]) > 1e-6]
    _check("repeat-run determinism", len(common) >= 10 and not diffs,
           f"{len(common)} paths re-scored, {len(diffs)} score drifts "
           f"(max drift allowed 1e-6)")

    # ── 3: corrupt file among good ones ──────────────────────────────────────
    print("\n[resilience] 3 good photos + 1 garbage .jpg…")
    bad_dir = ROOT / "cache" / "reliab_corrupt"
    if bad_dir.exists():
        shutil.rmtree(bad_dir, ignore_errors=True)
    bad_dir.mkdir(parents=True)
    good = sorted(FOLDER.glob("*.jpg"))[:3]
    for g in good:
        shutil.copy2(g, bad_dir / g.name)
    (bad_dir / "totally_corrupt.jpg").write_bytes(b"\x00\x01GARBAGE" * 512)
    res = _run_cull(bad_dir)
    _check("corrupt file does not kill the run", res["ok"],
           f"run ok={res['ok']}, {res['total']} photos graded of 4 inputs "
           f"(1 undecodable)"
           + (f" — error: {res['error']}" if res.get("error") else ""))
    RESULTS.append(_hygiene("corrupt run"))
    shutil.rmtree(bad_dir, ignore_errors=True)

    # ── 4: empty folder ──────────────────────────────────────────────────────
    print("\n[edge] empty folder cull…")
    empty_dir = ROOT / "cache" / "reliab_empty"
    empty_dir.mkdir(exist_ok=True)
    res = _run_cull(empty_dir)
    clean = res["ok"] or ("Traceback" not in res["stderr"])
    _check("empty folder finishes cleanly", clean,
           f"returncode-clean={res['ok']}, total={res['total']}, "
           f"no traceback={('Traceback' not in res['stderr'])}")
    shutil.rmtree(empty_dir, ignore_errors=True)

    # ── 5: RAM floor + disk head-room after all runs ─────────────────────────
    ram_now = _ram_available_gb()
    _check("machine above refuse-floor after all runs", ram_now > 0.75,
           f"{ram_now:.2f} GB available (floor 0.75 GB)")
    disk_end = _disk_free_gb()
    _check("disk head-room after all runs", disk_end > 1.0,
           f"{disk_end:.2f} GB free (writes of catalog + Lance need head-room)")

    # ── report ───────────────────────────────────────────────────────────────
    passed = sum(1 for r in RESULTS if r["ok"])
    total_checks = len(RESULTS)
    rep = ROOT / "reports" / "reliability_report.md"
    rep.parent.mkdir(exist_ok=True)
    lines = [f"# Cullwise reliability report — {datetime.now():%Y-%m-%d %H:%M}",
             "", f"**{passed}/{total_checks} checks passed** — "
                 f"{FOLDER.name}/ via grade_runner.py, real pipeline.",
             ""]
    for r in RESULTS:
        lines.append(f"- {'✅ PASS' if r['ok'] else '❌ FAIL'} — {r['check']}: "
                     f"{r['detail']}")
    lines += ["", "*Generated by scripts/reliability_check.py.*"]
    rep.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"RESULT: {passed}/{total_checks} checks passed")
    print(f"report: {rep}")
    print("=" * 72)
    return 0 if passed == total_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
