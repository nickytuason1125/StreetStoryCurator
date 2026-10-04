"""perf_guard — keep grades identical and culls fast (2026-10-04).

Two ledgers, because they fail differently:

  CORRECTNESS  golden set graded in isolation; every grade, score and aspect
               must match the recorded reference EXACTLY. The pipeline is
               deterministic (fixed ONNX batch per device, aligned chunks), so
               any difference is a real change, never noise.
  PERFORMANCE  (a) work counters — decodes per stage, detector images, model
               loads, worker starts (src/work_counters.py). Load-independent;
               a counter that goes UP fails.
               (b) wall time — median of repeats, judged only when the machine
               had comparable free RAM; otherwise reported INCONCLUSIVE instead
               of a false pass/fail (the same cull measured 156 s and 515 s on
               one day purely from other apps' RAM use).

Golden runs freeze the taste loop (empty ratings file, master judge off): it is
meant to move grades as you rate, and this guards the grading machinery.

    venv\\Scripts\\python.exe scripts\\perf_guard.py record --set quick
    venv\\Scripts\\python.exe scripts\\perf_guard.py check  --set quick
    venv\\Scripts\\python.exe scripts\\perf_guard.py check  --set full --repeat 3

Exit code 0 = pass (or timing inconclusive), 1 = failure, 3 = golden photos unavailable.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "golden"
PY = ROOT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)

# Comparable-conditions rule for wall time.
MIN_FREE_START_GB = 3.5
MIN_FREE_DURING_GB = 0.5          # the cull itself uses ~3 GB; only catch external pressure
SLOWER_TOLERANCE = 0.15            # fail if median wall > baseline * 1.15
# Counters that legitimately depend on free RAM (encode chunk plan → session
# loads / worker spawns). Compared only when timing is comparable.
# Scheduling-dependent: RAM-planned encode chunks, phase A on/off, and one
# YuNet per decode thread that happened to get work.
RAM_SENSITIVE = ("model_load.onnx_vision", "worker.encode", "worker.detect", "model_load.yunet")
# Depends on cache state, not code: the text-probe encode only runs when the
# probe cache (cache/probe_embs.*) is missing or its prompts changed. Reported,
# never failed.
CACHE_SENSITIVE = ("model_load.onnx_text",)


# ── photo sets ────────────────────────────────────────────────────────────────

def load_sets() -> dict:
    with open(GOLDEN / "sets.json", encoding="utf-8") as f:
        return json.load(f)


def expand(set_def: dict) -> dict:
    """{folder: [paths]} from a set definition of folder + glob + range."""
    import glob
    out: dict = {}
    for part in set_def["parts"]:
        folder = os.path.normpath(part["folder"])
        files = sorted(os.path.normpath(f) for f in glob.glob(os.path.join(folder, part["glob"])))
        sel = files[part.get("start", 0): part.get("start", 0) + part["count"]]
        if len(sel) < part["count"]:
            print(f"[perf_guard] {part['folder']}: only {len(sel)} of {part['count']} "
                  f"photos found — is the drive connected? (exit 3 = set unavailable)")
            raise SystemExit(3)
        out.setdefault(folder, []).extend(sel)
    return out


# ── environment / versions ────────────────────────────────────────────────────

_VERSIONS_SCRIPT = r'''
import json, hashlib, os, subprocess, importlib.metadata as md
from pathlib import Path
pk = {}
for n in ("torch", "onnxruntime-gpu", "onnxruntime", "numpy", "pillow", "rawpy",
          "opencv-python", "opencv-python-headless", "transformers", "pyiqa", "timm",
          "scipy", "scikit-learn"):
    try: pk[n] = md.version(n)
    except Exception: pass
try:
    import torch
    pk["torch.cuda"] = torch.version.cuda
    pk["cudnn"] = torch.backends.cudnn.version()
except Exception: pass
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=20).stdout.strip()
    pk["gpu"] = gpu
except Exception: pass
root = Path(r"%ROOT%")
def fp(p):
    h = hashlib.sha256(); size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(1 << 20)); f.seek(max(0, size - (1 << 20))); h.update(f.read(1 << 20))
    return f"{size}:{h.hexdigest()[:16]}"
models = {}
for pat in ("models/onnx/*.onnx", "models/dfine_nano/*.safetensors", "models/*.onnx"):
    for p in sorted(root.glob(pat)):
        try: models[str(p.relative_to(root))] = fp(p)
        except Exception: pass
hub = Path.home() / ".cache" / "torch" / "hub" / "pyiqa"
for p in sorted(hub.glob("*.pth")):
    models["pyiqa/" + p.name] = fp(p)
for p in ("cache/probe_embs.hash",):
    q = root / p
    if q.exists(): models[p] = q.read_text().strip()
print(json.dumps({"packages": pk, "models": models}))
'''


def versions() -> dict:
    r = subprocess.run([str(PY), "-c", _VERSIONS_SCRIPT.replace("%ROOT%", str(ROOT))],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": (r.stderr or r.stdout)[-500:]}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=str(ROOT)).stdout.strip()
    except Exception:
        return ""


# ── one isolated grade ────────────────────────────────────────────────────────

class _Sampler:
    def __init__(self):
        import psutil
        self.ps = psutil
        self.min_free = self.ps.virtual_memory().available / 2**30
        self.cpu = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(1.0):
            self.min_free = min(self.min_free, self.ps.virtual_memory().available / 2**30)
            self.cpu.append(self.ps.cpu_percent(None))

    def __enter__(self):
        self.ps.cpu_percent(None)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=3)


def run_once(explicit: dict, inject: dict | None = None) -> dict:
    import psutil
    work = Path(tempfile.mkdtemp(prefix="perfguard_"))
    (work / "cache").mkdir()
    ratings = work / "ratings_empty.json"
    ratings.write_text("{}", encoding="utf-8")
    counters = work / "counters"
    req, prog = work / "req.json", work / "prog.jsonl"
    prog.write_text("")
    req.write_text(json.dumps({
        "folders": list(explicit), "preset": "classic_street", "force_rescan": True,
        "scan_mode": False, "deep_grade": False,
        "catalog_path": str(work / "catalog.json"), "data_dir": str(work),
        "mogco_target": 0, "explicit_paths": explicit,
    }))
    env = dict(os.environ)
    for k in list(env):                       # nothing from the shell may steer it
        if k.startswith(("FIRSTCUT_", "SIGLIP_")):
            del env[k]
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "FIRSTCUT_LANCE_DIR": str(work / "lance.db"),
        "FIRSTCUT_RATINGS_PATH": str(ratings),
        "FIRSTCUT_MASTER_JUDGE_OFF": "1",
        "FIRSTCUT_WORK_COUNTERS": str(counters),
        "SIGLIP_TIER": "high",
    })
    env.update(inject or {})          # self-test only: prove the guard can fail
    free_start = psutil.virtual_memory().available / 2**30
    log = open(work / "run.log", "w", encoding="utf-8")
    t0 = time.perf_counter()
    with _Sampler() as smp:
        rc = subprocess.run([str(PY), str(ROOT / "grade_runner.py"), str(req), str(prog)],
                            cwd=str(ROOT), env=env, stdout=log, stderr=log).returncode
    wall = time.perf_counter() - t0
    log.close()
    sys.path.insert(0, str(ROOT / "src"))
    import work_counters
    cat = {}
    if (work / "catalog.json").exists():
        cat = json.loads((work / "catalog.json").read_text(encoding="utf-8"))
    grades = {}
    for p in cat.get("photos", []):
        bd = {k: v for k, v in (p.get("breakdown") or {}).items()
              if isinstance(v, (int, float, str, bool)) and not str(k).startswith("_grade_sig")}
        grades[p["path"]] = {"grade": p.get("grade"), "score": p.get("score"), "breakdown": bd}
    n = sum(len(v) for v in explicit.values())
    return {"rc": rc, "wall_s": round(wall, 1), "photos": n, "graded": len(grades),
            "free_start_gb": round(free_start, 2), "free_min_gb": round(smp.min_free, 2),
            "cpu_mean": round(statistics.mean(smp.cpu), 1) if smp.cpu else None,
            "counters": work_counters.collect(str(counters)), "grades": grades,
            "work": str(work)}


def comparable(t: dict) -> bool:
    return t["free_start_gb"] >= MIN_FREE_START_GB and t["free_min_gb"] >= MIN_FREE_DURING_GB


# ── record / check ────────────────────────────────────────────────────────────

def cmd_record(args) -> int:
    explicit = expand(load_sets()[args.set])
    runs = [run_once(explicit) for _ in range(args.repeat)]
    bad = [r for r in runs if r["rc"] != 0 or r["graded"] != r["photos"]]
    if bad:
        print(f"[perf_guard] record aborted: a run failed (rc={bad[0]['rc']}, "
              f"graded {bad[0]['graded']}/{bad[0]['photos']}) — see {bad[0]['work']}\\run.log")
        return 1
    ref = runs[0]
    for r in runs[1:]:
        if r["grades"] != ref["grades"]:
            print("[perf_guard] record aborted: repeated runs disagree — the pipeline is "
                  "not deterministic on this machine; fix that before recording.")
            return 1
    good = [r for r in runs if comparable(r)]
    if not good:
        print(f"[perf_guard] WARNING: no run had comparable conditions (≥{MIN_FREE_START_GB} GB "
              f"free at start, ≥{MIN_FREE_DURING_GB} GB throughout) — timing baseline left empty.")
    base = {
        "set": args.set, "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(), "versions": versions(),
        "grades": ref["grades"], "counters": ref["counters"],
        "timing": {"median_wall_s": statistics.median(r["wall_s"] for r in good) if good else None,
                   "runs": [{k: r[k] for k in ("wall_s", "free_start_gb", "free_min_gb", "cpu_mean")}
                            for r in runs]},
    }
    GOLDEN.mkdir(exist_ok=True)
    out = GOLDEN / f"{args.set}.baseline.json"
    out.write_text(json.dumps(base, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[perf_guard] recorded {out.name}: {len(ref['grades'])} photos, "
          f"median {base['timing']['median_wall_s']} s, commit {base['commit']}")
    return 0


def _diff_grades(ref: dict, got: dict) -> list:
    out = []
    for p in sorted(set(ref) | set(got)):
        a, b = ref.get(p), got.get(p)
        if a != b:
            if a is None or b is None:
                out.append(f"{Path(p).name}: {'missing now' if b is None else 'new photo'}")
            elif a["grade"] != b["grade"] or a["score"] != b["score"]:
                out.append(f"{Path(p).name}: {a['grade']} {a['score']} -> {b['grade']} {b['score']}")
            else:
                keys = [k for k in set(a["breakdown"]) | set(b["breakdown"])
                        if a["breakdown"].get(k) != b["breakdown"].get(k)]
                out.append(f"{Path(p).name}: breakdown {', '.join(sorted(keys)[:4])}")
    return out


def cmd_check(args) -> int:
    path = GOLDEN / f"{args.set}.baseline.json"
    if not path.exists():
        print(f"[perf_guard] no baseline {path.name} — run `record --set {args.set}` first")
        return 1
    base = json.loads(path.read_text(encoding="utf-8"))
    explicit = expand(load_sets()[args.set])
    inject = dict(kv.split("=", 1) for kv in (args.inject or []))
    if inject:
        print(f"[perf_guard] SELF-TEST: injecting {inject} — this run SHOULD fail")
    runs = [run_once(explicit, inject) for _ in range(args.repeat)]
    failed = False

    # 1. correctness — exact
    for i, r in enumerate(runs):
        if r["rc"] != 0 or r["graded"] != r["photos"]:
            print(f"FAIL  run {i+1}: grade run failed (rc={r['rc']}, graded {r['graded']}/{r['photos']})"
                  f" — {r['work']}\\run.log")
            failed = True
            continue
        d = _diff_grades(base["grades"], r["grades"])
        if d:
            failed = True
            print(f"FAIL  grades: {len(d)}/{len(base['grades'])} photos differ from the golden reference")
            for line in d[:15]:
                print(f"        {line}")
        else:
            print(f"PASS  grades: {len(r['grades'])}/{len(base['grades'])} identical (run {i+1})")

    # 2. work counters — must not grow
    ref_c, got_c = base["counters"], runs[0]["counters"]
    timing_ok = [r for r in runs if comparable(r)]
    grew, shrank = [], []
    for k in sorted(set(ref_c) | set(got_c)):
        a, b = ref_c.get(k, 0), got_c.get(k, 0)
        if k in RAM_SENSITIVE and not (timing_ok and base["timing"]["median_wall_s"]):
            continue
        if k in CACHE_SENSITIVE:
            if ref_c.get(k, 0) != got_c.get(k, 0):
                print(f"NOTE  {k}: {ref_c.get(k, 0)} -> {got_c.get(k, 0)} (probe cache was "
                      f"rebuilt this run — cache state, not a code change)")
            continue
        if b > a:
            grew.append(f"{k}: {a} -> {b}")
        elif b < a:
            shrank.append(f"{k}: {a} -> {b}")
    if grew:
        failed = True
        print(f"FAIL  work counters grew (more work per cull): " + "; ".join(grew))
    else:
        print("PASS  work counters: no stage does more work than the baseline")
    if shrank:
        print("NOTE  less work than the baseline (improvement — re-record to lock it in): "
              + "; ".join(shrank))

    # 3. wall time — only under comparable conditions
    bt = base["timing"]["median_wall_s"]
    rebuilt = [k for k in CACHE_SENSITIVE if got_c.get(k, 0) > ref_c.get(k, 0)]
    if bt and rebuilt:
        print(f"INCONCLUSIVE  timing: this run rebuilt a cache ({', '.join(rebuilt)}) — "
              f"one-time work the baseline did not do. Re-run for a speed verdict.")
    elif not bt:
        print("SKIP  timing: the baseline has no comparable-conditions timing")
    elif not timing_ok:
        r = runs[0]
        print(f"INCONCLUSIVE  timing: machine too loaded to judge ({r['free_start_gb']} GB free at "
              f"start, {r['free_min_gb']} GB at worst; needs ≥{MIN_FREE_START_GB}/≥{MIN_FREE_DURING_GB}). "
              f"Close apps and re-run for a speed verdict.")
    else:
        med = statistics.median(r["wall_s"] for r in timing_ok)
        ratio = med / bt
        if ratio > 1 + SLOWER_TOLERANCE:
            failed = True
            print(f"FAIL  timing: median {med:.1f} s vs baseline {bt:.1f} s ({ratio:.2f}x slower)")
        else:
            print(f"PASS  timing: median {med:.1f} s vs baseline {bt:.1f} s ({ratio:.2f}x)")

    # 4. versions — flagged
    v_now = versions()
    vdiff = [f"{sec}.{k}: {base['versions'].get(sec, {}).get(k)} -> {v_now.get(sec, {}).get(k)}"
             for sec in ("packages", "models")
             for k in sorted(set(base["versions"].get(sec, {})) | set(v_now.get(sec, {})))
             if base["versions"].get(sec, {}).get(k) != v_now.get(sec, {}).get(k)]
    if vdiff:
        print("WARN  versions changed since the baseline (GPU maths is only reproducible on the "
              "same libraries/models): " + "; ".join(vdiff[:8]))
    print("RESULT", "FAIL" if failed else "PASS")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "check"):
        s = sub.add_parser(name)
        s.add_argument("--set", default="quick")
        s.add_argument("--repeat", type=int, default=1)
        if name == "check":
            s.add_argument("--inject", action="append", metavar="KEY=VAL",
                           help="self-test: set an env var for the graded run to prove "
                                "the guard catches a regression")
    args = ap.parse_args()
    return cmd_record(args) if args.cmd == "record" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
