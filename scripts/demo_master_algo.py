"""MASTER ALGO DEMO — the two-phase contract, live, on real pictures.

Demonstrates the full journey with zero console gymnastics:

  A. THE EXAM RESULT  — the shipped master judge (data/
     master_judge_defaults.json, graded BY DEFAULT, no flags, no ratings)
     re-scores every rated photo in the store and is compared against the
     incumbent grader AND the photographer's stars: who is closer, per photo
     and overall.
  B. FRESH PICTURES   — grades Sample_Street (12 untouched street frames)
     through the REAL pipeline (SigLIP-2 → CLIP → IQA → MasterJudge →
     grades) exactly as the server does, and prints the verdicts. No star
     rating exists for these; the master algo graded them standalone.

An HTML report with thumbnails is written to output/master_algo_demo.html.

Usage:
    venv\\Scripts\\python.exe scripts\\demo_master_algo.py [--no-fresh] [--limit 12]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import master_judge as mj  # noqa: E402


# ── Part A: the exam result, on rated pictures ────────────────────────────────

def _band(score: float) -> str:
    return "Strong" if score >= 0.60 else ("Weak" if score < 0.41 else "Mid")


def part_a(limit: int) -> list:
    import numpy as np
    judge, w = mj.active()
    if not judge:
        print("No active master judge — nothing to demo.")
        return []
    print("=" * 72)
    print("PART A — THE SHIPPED MASTER JUDGE vs THE INCUMBENT, ON RATED PICS")
    print("=" * 72)
    print(f"source        : {judge['_source']}")
    print(f"blend weight  : {w:.2f}")
    print(f"exam result   : rho {judge['rho_holdout']} (master) vs "
          f"{judge['rho_baseline']} (incumbent) on {judge['n_holdout']} "
          f"held-out photos the judge never trained on")

    rows = mj.collect_rows()
    rows = [r for r in rows
            if r.get("breakdown") and np.isfinite(mj.feature_vector(r["breakdown"])).all()]
    master = mj.predict_many([r["breakdown"] for r in rows],
                             [r["score"] for r in rows], weights=judge)
    stars = np.array([r["stars"] for r in rows], dtype=np.float64) / 5.0
    machine = np.array([r["score"] for r in rows], dtype=np.float64)

    err_m = np.abs(master - stars)
    err_i = np.abs(machine - stars)
    wins = int((err_m < err_i - 1e-9).sum())
    losses = int((err_m > err_i + 1e-9).sum())

    print(f"\nfull baseline (n={len(rows)}; NOTE: includes photos the judge "
          f"trained on — the honest generalization number is the holdout "
          f"above):")
    print(f"   incumbent rho vs stars : {mj.spearman(machine, stars):+.4f}")
    print(f"   master    rho vs stars : {mj.spearman(master, stars):+.4f}")
    print(f"   closer to the photographer's verdict: master {wins} — "
          f"incumbent {losses} — tie {len(rows) - wins - losses}")

    by_star = {}
    for r in rows:
        by_star.setdefault(r["stars"], []).append(r)
    print("\n   mean |error| per star bucket (master vs incumbent):")
    for s in sorted(by_star):
        idx = [i for i, r in enumerate(rows) if r["stars"] == s]
        em = float(np.mean(err_m[idx]))
        ei = float(np.mean(err_i[idx]))
        better = "master" if em < ei else "incumbent"
        print(f"     {s}★ n={len(idx):<4} master {em:.3f}  incumbent {ei:.3f}  → {better}")

    # demo selection: the biggest moves, spread across star buckets
    delta = master - machine
    picked = []
    for s in sorted(by_star):
        idx = [i for i, r in enumerate(rows) if r["stars"] == s]
        idx.sort(key=lambda i: -abs(delta[i]))
        picked.extend(idx[:max(1, limit // 6)])
    picked = picked[:limit]

    print(f"\n   {len(picked)} demo pictures (biggest score moves per star bucket):")
    print(f"   {'stars':<5} {'incumbent':>9} {'master':>7} {'move':>7}  verdict change")
    for i in picked:
        r = rows[i]
        print(f"   {r['stars']:<5} {machine[i]:>9.3f} {master[i]:>7.3f} "
              f"{delta[i]:>+7.3f}  {_band(machine[i]):>6} → {_band(master[i])}")

    _write_html(rows, picked, machine, master, stars, judge, w)
    return [(rows[i], float(machine[i]), float(master[i])) for i in picked]

def _write_html(rows, picked, machine, master, stars, judge, w) -> None:
    out = _ROOT / "output" / "master_algo_demo.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for i in picked:
        r = rows[i]
        p = Path(r["path"])
        uri = "file:///" + p.resolve().as_posix()
        moved = "moved" if _band(machine[i]) != _band(master[i]) else ""
        cards.append(f"""
        <div class="card {moved}">
          <img src="{uri}" onerror="this.style.opacity=0.15">
          <div class="info">
            <div class="name">{p.name}</div>
            <div class="row">photographer <b>{r['stars']}★</b></div>
            <div class="row">incumbent <b>{machine[i]:.3f}</b> ({_band(machine[i])})</div>
            <div class="row">master&nbsp;&nbsp;&nbsp; <b>{master[i]:.3f}</b> ({_band(master[i])})</div>
            <div class="row delta">move <b>{master[i] - machine[i]:+.3f}</b></div>
          </div>
        </div>""")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Master Algo Demo</title><style>
body{{background:#111;color:#eee;font-family:Segoe UI,sans-serif;margin:24px}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}}
.card{{background:#1c1c1c;border-radius:10px;overflow:hidden}}
.card.moved{{outline:2px solid #e6b800}}
.card img{{width:100%;height:180px;object-fit:cover;display:block}}
.info{{padding:8px 10px;font-size:13px}} .name{{font-weight:600;margin-bottom:4px}}
.delta{{color:#e6b800}} .metric{{background:#1c1c1c;border-radius:8px;padding:10px 14px;margin:8px 0}}
</style></head><body>
<h1>Master Algo — shipped judge (source: {judge['_source']}, blend weight {w:.2f})</h1>
<div class="metric">Held-out exam: <b>ρ {judge['rho_holdout']}</b> vs incumbent <b>{judge['rho_baseline']}</b>
on {judge['n_holdout']} never-trained photos · trained on {judge['n']} master-rated frames · λ {judge.get('lambda')}</div>
<div class="grid">{''.join(cards)}</div>
</body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"\n   HTML report with thumbnails → {out}")

# ── Part B: fresh pictures, graded by the master algo, zero ratings ──────────

def part_b(timeout_min: int = 20) -> int:
    import numpy as np
    print("\n" + "=" * 72)
    print("PART B — FRESH PICTURES: Sample_Street graded by the real pipeline")
    print("=" * 72)
    folder = _ROOT.parent / "Sample_Street"
    fresh = sorted(folder.glob("*.jpg"))
    if not fresh:
        print(f"no *.jpg in {folder} — skipping part B")
        return 1
    print(f"folder       : {folder}  ({len(fresh)} photos, never rated)")

    import lance_store as ls
    before = {r["path"] for r in ls.query_all(min_score=0.0)}

    req = {
        "folders":      [str(folder)],
        "preset":       "Classic Street",
        "force_rescan": True,
        "scan_mode":    False,
        "deep_grade":   False,
        "catalog_path": str(_ROOT / "data" / "cache" / "catalog.json"),
        "data_dir":     str(_ROOT),
        "mogco_target": 0,
    }
    import tempfile, os
    fd, req_path = tempfile.mkstemp(suffix=".gradereq.json"); os.close(fd)
    prog_path = req_path + ".progress.jsonl"
    Path(req_path).write_text(json.dumps(req), encoding="utf-8")
    Path(prog_path).write_text("", encoding="utf-8")

    print("running the REAL grade subprocess (SigLIP-2 → CLIP → IQA → "
          "MasterJudge → grades)…")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "grade_runner.py"), req_path, prog_path],
        cwd=str(_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=timeout_min * 60, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if proc.returncode != 0:
        print(f"grade runner exited {proc.returncode} — tail of its log:")
        print("\n".join((proc.stdout or "").splitlines()[-25:]))
        print("\n".join((proc.stderr or "").splitlines()[-25:]))
        return 1
    print(f"grade finished in {time.time() - t0:.0f}s")

    after = {r["path"]: r for r in ls.query_all(min_score=0.0)}
    fresh_rows = [r for p, r in after.items()
                  if p not in before and Path(p).is_relative_to(folder)]
    if not fresh_rows:
        fresh_rows = [r for p, r in after.items()
                      if Path(p).is_relative_to(folder)]
    fresh_rows.sort(key=lambda r: -float(r.get("score") or 0))
    print(f"\ngraded {len(fresh_rows)} fresh pictures — the SHIPPED master "
          f"judge was active in this run (no ratings exist for any of them):")
    print(f"   {'score':>6}  verdict")
    for r in fresh_rows:
        print(f"   {float(r.get('score') or 0):>6.3f}  {r.get('grade', '')}"
              f"   {Path(r['path']).name}")
    blend_lines = [ln for ln in (proc.stdout or "").splitlines()
                   if "MasterJudge blend" in ln or "PersonalHead weights" in ln]
    if blend_lines:
        print("\npipeline log check:")
        for ln in blend_lines[-3:]:
            print(f"   {ln}")
    else:
        print("\npipeline log check: NOTE: no MasterJudge blend line in the "
              "runner output — inspect manually")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-fresh", action="store_true",
                    help="skip grading Sample_Street (part B)")
    ap.add_argument("--limit", type=int, default=12,
                    help="demo pictures per report")
    args = ap.parse_args()

    part_a(args.limit)
    if not args.no_fresh:
        return part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
