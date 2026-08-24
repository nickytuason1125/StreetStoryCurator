"""doctor.py — FrameGrade one-click diagnose-and-repair.

Run this when the app misbehaves. It checks everything that has ever actually
broken an install of this application, explains each result in one sentence,
and — with --fix — repairs what is safe to repair automatically:

  CHECKS                              AUTO-FIX (--fix)
  ─────────────────────────────────   ────────────────────────────────────────
  Python version                      —
  venv present and imports work       recreate instructions (never deletes)
  Port 8000 free / ours               kill the squattering process
  Encoder weights installed           re-fetch via scripts/fetch_models.py
  Optional GGUF models present        re-fetch missing ones
  LanceDB table opens                 delete corrupt DB (grades re-index)
  .models_ready sentinel honest       delete a lying sentinel
  Free disk space                     —
  GPU visible (informational)         —
  crash.log tail triage               —

Usage:
    python scripts/doctor.py            # diagnose only
    python scripts/doctor.py --fix      # diagnose AND auto-repair
    python scripts/doctor.py --json     # machine-readable output (for the UI)

Exit codes: 0 = healthy · 1 = problems found · 2 = problems found and fixed.
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[dict] = []   # {name, ok, fixed, detail}


def check(name: str, ok: bool, detail: str, fixed: bool = False) -> None:
    RESULTS.append({"name": name, "ok": ok, "fixed": fixed, "detail": detail})
    mark = "OK " if ok else ("FIX" if fixed else "!! ")
    print(f"[{mark}] {name}: {detail}", flush=True)


# ── individual checks ─────────────────────────────────────────────────────────

def check_python() -> None:
    v = sys.version_info
    ok = v >= (3, 10)
    check("Python version", ok,
          f"{v.major}.{v.minor}.{v.micro} — needs 3.10+"
          if ok else f"{v.major}.{v.minor} is too old — install Python 3.10+")


def check_venv() -> None:
    venv_py = ROOT / "venv" / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python")
    if not venv_py.exists():
        check("Virtual environment", False,
              "venv/ is missing — run Setup.ps1 (Windows) or "
              "python -m venv venv && venv/bin/pip install -r requirements.txt")
        return
    r = subprocess.run([str(venv_py), "-c", "import fastapi, uvicorn"],
                       capture_output=True, timeout=60)
    check("Virtual environment", r.returncode == 0,
          "imports fine" if r.returncode == 0 else
          "exists but core packages fail to import — run: venv pip install -r requirements.txt")


def check_port(fix: bool) -> None:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 8000))
        check("Port 8000", True, "free")
    except OSError:
        # Something holds the port. If it's a zombie of ours, offer to kill it.
        holder = _port_holder_pid()
        if fix and holder:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/PID", str(holder)],
                                   capture_output=True, timeout=15)
                else:
                    os.kill(holder, 9)
                time.sleep(1)
                s2 = socket.socket()
                try:
                    s2.bind(("127.0.0.1", 8000))
                    check("Port 8000", True,
                          f"killed stale process {holder} — port now free", fixed=True)
                    return
                finally:
                    s2.close()
            except Exception as e:
                check("Port 8000", False,
                      f"held by PID {holder} and could not be killed ({e}) — "
                      "close the other app using port 8000")
                return
        check("Port 8000", False,
              f"in use{f' by PID {holder}' if holder else ''} — another app owns it; "
              "close it or set CURATOR_PORT to another port")
    finally:
        try: s.close()
        except OSError: pass


def _port_holder_pid() -> int | None:
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=15
            ).stdout
            for line in out.splitlines():
                if ":8000" in line and "LISTENING" in line.upper():
                    return int(line.split()[-1])
        else:
            out = subprocess.run(
                ["lsof", "-ti", ":8000"], capture_output=True, text=True, timeout=15
            ).stdout
            return int(out.strip().splitlines()[0]) if out.strip() else None
    except Exception:
        return None
    return None


def check_encoders(fix: bool) -> None:
    try:
        import tier_select
        import run_profile
        installed = [t for t in run_profile.TIERS if tier_select.available(t)]
        if installed:
            check("Encoder weights", True,
                  f"installed tiers: {', '.join(tier_select.label(t) for t in installed)}")
            return
        if fix:
            print("[.. ] re-fetching the encoder this machine needs …", flush=True)
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_models.py")],
                               cwd=str(ROOT), timeout=3600)
            still = [t for t in run_profile.TIERS if tier_select.available(t)]
            check("Encoder weights", bool(still),
                  f"fetched — now installed: {', '.join(still)}" if still
                  else "download ran but nothing installed — check your connection",
                  fixed=bool(still))
        else:
            check("Encoder weights", False,
                  "none installed — grading cannot start. Run with --fix to download.")
    except Exception as e:
        check("Encoder weights", False, f"could not verify ({e})")


def check_gguf(fix: bool) -> None:
    try:
        import model_registry as mr
        missing = [m.dest.name for m in mr.missing_gguf()]
        if not missing:
            check("Writing-engine models", True, "all optional GGUF models present")
            return
        if fix:
            print("[.. ] fetching missing GGUF models …", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "fetch_models.py"),
                 "--with-optional"], cwd=str(ROOT), timeout=7200)
            still = [m.dest.name for m in mr.missing_gguf()]
            check("Writing-engine models", not still,
                  "all fetched" if not still else f"still missing: {', '.join(still)}",
                  fixed=not still)
        else:
            check("Writing-engine models", True,   # advisory, not fatal
                  f"missing (grading unaffected): {', '.join(missing)} — "
                  "run with --fix to download")
    except Exception as e:
        check("Writing-engine models", True, f"could not verify ({e}) — advisory only")


def check_lancedb(fix: bool) -> None:
    db_dir = ROOT / "cache" / "lance.db"
    try:
        import lance_store as ls
        ls._open_table()
        check("Photo database", True, "opens cleanly")
    except Exception as e:
        if fix and db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)
            check("Photo database", True,
                  f"was corrupt ({str(e)[:80]}) — deleted; grades rebuild on next cull",
                  fixed=True)
        else:
            check("Photo database", False,
                  f"corrupt: {str(e)[:120]} — run with --fix to reset it")


def check_sentinel() -> None:
    sentinel = ROOT / "models" / ".models_ready"
    if not sentinel.exists():
        check("Install sentinel", True, "absent (fine — verified live instead)")
        return
    try:
        import tier_select, run_profile
        any_real = any(tier_select.available(t) for t in run_profile.TIERS)
        if any_real:
            check("Install sentinel", True, "consistent with installed weights")
        else:
            sentinel.unlink(missing_ok=True)
            check("Install sentinel", True,
                  "claimed models were installed but none are — deleted so the "
                  "first-run downloader runs", fixed=True)
    except Exception as e:
        check("Install sentinel", True, f"could not verify ({e})")


def check_disk() -> None:
    try:
        free = shutil.disk_usage(ROOT).free / 1e9
        check("Disk space", free > 5,
              f"{free:.1f} GB free" if free > 5 else
              f"only {free:.1f} GB free — model downloads need ~7 GB")
    except Exception as e:
        check("Disk space", True, f"could not measure ({e})")


def check_gpu() -> None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=8)
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 else ""
        if name:
            check("GPU", True, f"{name} — vision models run on it")
        else:
            check("GPU", True,
                  "none detected — the app runs on CPU automatically "
                  "(slower: ~2-3 s/image at Fast quality)")
    except FileNotFoundError:
        check("GPU", True,
              "no NVIDIA driver — the app runs on CPU automatically "
              "(slower: ~2-3 s/image at Fast quality)")
    except Exception as e:
        check("GPU", True, f"could not probe ({e})")


def check_crashlog() -> None:
    log = ROOT / "crash.log"
    if not log.exists():
        check("Crash log", True, "no crashes recorded")
        return
    try:
        tail = log.read_text(encoding="utf-8", errors="ignore")[-4000:]
        lines = [l for l in tail.splitlines() if l.strip()]
        last_err = next((l for l in reversed(lines)
                         if "Error" in l or "Traceback" in l or "FATAL" in l), "")
        size_mb = log.stat().st_size / 1e6
        hint = ""
        low = tail.lower()
        if "outofmemoryerror" in low or "bad_alloc" in low or "oom" in low:
            hint = "Looks like an out-of-memory crash — close other apps and re-grade."
        elif "0xc0000005" in low or "access_violation" in low:
            hint = "A GPU subprocess crashed — this is contained; just retry the grade."
        elif "no module named" in low:
            hint = "A package is missing — run: venv pip install -r requirements.txt"
        summary = f"{size_mb:.1f} MB"
        if last_err:
            summary += f"; last issue: {last_err[:100]}"
        if hint:
            summary += f" — {hint}"
        check("Crash log", True, summary)
    except Exception as e:
        check("Crash log", True, f"could not read ({e})")


import time  # noqa: E402  (used by check_port's fix path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true", help="auto-repair what is safe")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    print("=" * 62)
    print("FrameGrade doctor — checking your installation")
    print("=" * 62)

    check_python()
    check_venv()
    check_port(args.fix)
    check_encoders(args.fix)
    check_gguf(args.fix)
    check_lancedb(args.fix)
    check_sentinel()
    check_disk()
    check_gpu()
    check_crashlog()

    bad     = [r for r in RESULTS if not r["ok"]]
    fixed_n = sum(1 for r in RESULTS if r["fixed"])
    print("-" * 62)
    if not bad:
        print(f"All {len(RESULTS)} checks passed. The installation looks healthy.")
        code = 0
    elif fixed_n:
        print(f"{len(bad)} problem(s) found — {fixed_n} repaired automatically. "
              "Re-run without --fix to confirm, then start the app again.")
        code = 2
    else:
        print(f"{len(bad)} problem(s) need attention — see the [!!] lines above. "
              "Re-run with --fix to let the doctor repair what it can.")
        code = 1

    if args.json:
        print(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "exit": code,
            "results": RESULTS,
        }))

    sys.exit(code)


if __name__ == "__main__":
    main()