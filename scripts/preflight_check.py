"""
preflight_check.py — one command before trusting a cull.

Born from the 2026-09-07 incident: every failure that night (wiped catalog,
stale grade lock, port war, silent tier downgrade, encoder CUDA OOM) was
discoverable in advance but nothing checked for it. This script does.

Checks:
  1. catalog.json — exists, parses, photo count (+ backup rotations present)
  2. grading.lock  — absent / live / stale (a provably stale one is swept)
  3. backend       — /api/whoami identity on port 8000 (if running)
  4. RAM           — available vs admission thresholds (full 3.8 / scan 2.0)
  5. VRAM          — free on device 0 (encoder headroom)
  6. tier pins     — SIGLIP_TIER / FIRSTCUT_LITE visible in this environment
  7. single window launcher — shim-aware process census

Exit code 0 = no FAIL lines; 1 = at least one FAIL. WARN never fails the run.
Run:  venv\\Scripts\\python.exe scripts\\preflight_check.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

FULL_NEED_GB = 3.8
SCAN_NEED_GB = 2.0

checks = []   # (status, name, detail)


def _check(status, name, detail=""):
    checks.append((status, name, detail))


def main() -> int:
    # 1 ── catalog -----------------------------------------------------------
    cat = ROOT / "cache" / "catalog.json"
    try:
        if cat.exists():
            d = json.loads(cat.read_text(encoding="utf-8"))
            n = len(d.get("photos", []))
            _check("PASS", "catalog", f"valid, {n} photos, {cat.stat().st_size / 1e6:.1f} MB")
        else:
            _check("FAIL", "catalog", "cache/catalog.json MISSING — restore from a .bak.N before grading")
    except Exception as exc:
        _check("FAIL", "catalog", f"UNREADABLE ({exc}) — restore from a .bak.N before grading")
    baks = sorted(ROOT.glob("cache/catalog.json.bak.*"))
    if baks:
        _check("PASS", "catalog backups", f"{len(baks)} rotation(s) on disk")
    else:
        _check("WARN", "catalog backups", "none yet — created by the first grade checkpoint")

    # 2 ── grade lock --------------------------------------------------------
    try:
        from grade_lock import grade_in_progress, sweep_stale
        if grade_in_progress():
            _check("WARN", "grade lock", "a grade is LIVE — a new cull will be refused (409) until it finishes")
        elif sweep_stale():
            _check("PASS", "grade lock", "stale lock from a dead grade was swept")
        else:
            _check("PASS", "grade lock", "clear")
    except Exception as exc:
        _check("WARN", "grade lock", f"could not evaluate ({exc})")

    # 3 ── backend identity --------------------------------------------------
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/whoami",
                                     headers={"X-Requested-With": "FirstCut"})
        with urllib.request.urlopen(req, timeout=3) as r:
            who = json.loads(r.read().decode()).get("app")
        if who == "FirstCut":
            _check("PASS", "backend", "serving on :8000 (identity=FirstCut)")
        else:
            _check("FAIL", "backend", f"port 8000 answered but identity={who!r} — wrong program owns the port")
    except Exception:
        _check("WARN", "backend", "not running — launch_hidden.vbs will spawn it")

    # 4 ── RAM ---------------------------------------------------------------
    try:
        import psutil
        avail = psutil.virtual_memory().available / 1e9
        if avail >= FULL_NEED_GB:
            _check("PASS", "RAM", f"{avail:.2f} GB available — full quality admissible (needs {FULL_NEED_GB})")
        elif avail >= SCAN_NEED_GB:
            _check("WARN", "RAM", f"{avail:.2f} GB available — cull auto-downgrades to Scan (full needs {FULL_NEED_GB})")
        else:
            _check("FAIL", "RAM", f"only {avail:.2f} GB available — even Scan needs {SCAN_NEED_GB}; close apps")
    except Exception as exc:
        _check("WARN", "RAM", f"could not measure ({exc})")

    # 5 ── VRAM --------------------------------------------------------------
    try:
        import encode_worker as _ew
        vram = _ew._free_vram_gb()
        if vram is None:
            _check("WARN", "VRAM", "could not measure (nvidia-smi missing?) — encoder picks a provider unaided")
        elif vram >= 1.5:
            _check("PASS", "VRAM", f"{vram:.2f} GB free — CUDA encoder session admissible")
        else:
            _check("WARN", "VRAM", f"only {vram:.2f} GB free — encoder skips CUDA (CPU fallback, slower)")
    except Exception as exc:
        _check("WARN", "VRAM", f"could not measure ({exc})")

    # 6 ── tier pins ---------------------------------------------------------
    tier = os.environ.get("SIGLIP_TIER", "")
    lite = os.environ.get("FIRSTCUT_LITE", "")
    if tier == "low" and lite == "1":
        _check("PASS", "tier pins", "SIGLIP_TIER=low, FIRSTCUT_LITE=1 (safe config for 16 GB)")
    else:
        _check("WARN", "tier pins",
               f"SIGLIP_TIER={tier or '(unset)'} FIRSTCUT_LITE={lite or '(unset)'} — "
               "launch via launch_hidden.vbs to pin Lite on a 16 GB machine")

    # 7 ── single window launcher (shim-aware) -------------------------------
    try:
        import psutil
        launchers = set()
        me = os.getpid()
        for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
            try:
                if p.info["pid"] == me or "python" not in (p.info["name"] or "").lower():
                    continue
                exe = (p.info["exe"] or "").replace("/", "\\").lower()
                if "\\venv\\scripts\\" in exe:
                    continue  # venv redirector shim — the base child carries the role
                cl = " ".join(p.info["cmdline"] or [])
                if "local_launcher" in cl and "--server-only" not in cl:
                    launchers.add(p.info["pid"])
            except Exception:
                continue
        if len(launchers) <= 1:
            _check("PASS", "launcher", f"{len(launchers)} window launcher running")
        else:
            _check("FAIL", "launcher", f"{len(launchers)} window launchers {sorted(launchers)} — "
                                       "kill extras or the port war starts again")
    except Exception as exc:
        _check("WARN", "launcher", f"could not enumerate ({exc})")

    # ── report --------------------------------------------------------------
    print("\nFirstCut preflight")
    print("=" * 64)
    for status, name, detail in checks:
        print(f"  [{status}] {name:<16} {detail}")
    print("=" * 64)
    fails = sum(1 for s, _, _ in checks if s == "FAIL")
    warns = sum(1 for s, _, _ in checks if s == "WARN")
    print(f"{fails} FAIL, {warns} WARN, {len(checks) - fails - warns} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
