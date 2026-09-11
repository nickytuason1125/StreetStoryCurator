"""Master ground-truth backtest — measures the grader against YOUR stars.

The photographer's own ratings (LX3 keepers + TPE entries, tagged tpe_master
in cache/user_ratings.json) are the master file: the only human-accurate
quality labels this system has. This script joins every rating against the
machine's stored judgement and answers, with numbers:

1. COVERAGE          how many ratings can be measured at all (live store row
                     vs rating-time score snapshot vs nothing).
2. RANK AGREEMENT    Spearman ρ between machine score and stars — overall and
                     per grader path (clip / deep). This is the legitimacy
                     metric: everything else in the pipeline is decoration
                     if this is low.
3. ASPECT SIGNAL     ρ of EACH aspect in the stored breakdown against stars —
                     which of the five/six hand-weighted dimensions actually
                     carry the photographer's signal, and which are dead
                     weight the pipeline is being steered by.
4. BAND MONOTONICITY mean stars per Strong/Mid/Weak band — the absolute
                     thresholds only mean something if the bands order.

With --fit, also retrains the MasterJudge head (src/master_judge.py) on this
same baseline and reports its champion/challenger verdict.

Usage:  venv\\Scripts\\python.exe scripts/master_backtest.py [--fit]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows consoles default to cp1252 — UTF-8 so the report can never die in a
# print after the measurement already succeeded.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# models/, not cache/: cache/ is ephemeral/derived and gets cleared
# routinely; the durable head lives in models/aadb_head.npz, and this file
# must live next to it or a cache wipe makes promote_master_judge's AADB
# gate falsely report "no trained head found" (see aadb_setup.py).
_AADB_METRICS_PATH = _ROOT / "models" / "aadb_head_metrics.json"


def read_aadb_metrics() -> "dict | None":
    """The AADB-only held-out rho written by aadb_setup.py, or None if the
    head has never been trained."""
    import json
    if not _AADB_METRICS_PATH.exists():
        return None
    try:
        return json.loads(_AADB_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true",
                    help="also refit the MasterJudge head after reporting")
    args = ap.parse_args()

    import numpy as np

    import master_judge as mj

    rows = mj.collect_rows()
    if not rows:
        print("No measurable master ratings — rate photos (with a stored "
              "machine score) first.")
        return 1

    W = 64
    n_snap = sum(1 for r in rows if r["from_snapshot"])
    print("=" * W)
    print("FIRSTCUT MASTER BACKTEST (ground truth = your stars)")
    print("=" * W)
    print(f"1. COVERAGE")
    print(f"   measurable ratings:   {len(rows)}")
    print(f"   from live store row:  {len(rows) - n_snap}")
    print(f"   from rating snapshot: {n_snap}")
    print()

    stars = np.array([r["stars"] for r in rows], dtype=np.float64)
    machine = np.array([r["score"] for r in rows], dtype=np.float64)

    print(f"2. RANK AGREEMENT (machine score vs your stars)")
    rho = mj.spearman(machine, stars)
    if np.isnan(rho):
        print("   Spearman rho = undefined (constant score or too few rows)")
    else:
        verdict = ("excellent" if rho > .5 else "good" if rho > .35 else
                   "weak" if rho > .15 else "poor")
        print(f"   Spearman rho = {rho:+.3f}   (n={len(rows)})   {verdict}")
    graders = {}
    for r in rows:
        g = (r["breakdown"] or {}).get("_grader") or "unknown"
        graders.setdefault(g, []).append(r)
    for g, rs in sorted(graders.items()):
        if len(rs) >= 3:
            rg = mj.spearman([r["score"] for r in rs], [r["stars"] for r in rs])
            tag = "nan" if np.isnan(rg) else f"{rg:+.3f}"
            print(f"     grader={g:<8} rho={tag}  (n={len(rs)})")
    print()

    aadb = read_aadb_metrics()
    if aadb is not None:
        print(f"\nAADB (external human judgment):")
        print(f"  held-out rho = {aadb['rho_holdout']} "
              f"(n_train={aadb['n_train']}, n_holdout={aadb['n_holdout']})")
    else:
        print("\nAADB: no trained head found (run aadb_setup.py to add this check)")

    print(f"3. ASPECT SIGNAL (each stored aspect vs your stars)")
    fmat = np.stack([mj.feature_vector(r["breakdown"] or {}) for r in rows])
    for j, name in enumerate(mj.FEATURES):
        col = fmat[:, j]
        ok = np.isfinite(col)
        ra = mj.spearman(col[ok], stars[ok]) if ok.sum() >= 3 else float("nan")
        tag = "nan   " if np.isnan(ra) else f"{ra:+.3f}"
        bar = "#" * max(0, int(abs(ra) * 40)) if not np.isnan(ra) else ""
        print(f"   {name:<14} rho={tag}  (n={int(ok.sum()):<4}) {bar}")
    print()

    print(f"4. BAND MONOTONICITY (mean stars per grade band)")
    bands = {"Strong": [], "Mid": [], "Weak": []}
    for r in rows:
        grade = str(r.get("grade") or "")
        for b in bands:
            if grade.startswith(b):
                bands[b].append(r["stars"])
                break
    means = []
    for b in ("Strong", "Mid", "Weak"):
        v = bands[b]
        m = float(np.mean(v)) if v else float("nan")
        means.append(m)
        tag = "n/a  " if np.isnan(m) else f"{m:.2f}"
        print(f"   {b:<7} n={len(v):<4} mean stars = {tag}")
    finite = [m for m in means if not np.isnan(m)]
    if len(finite) == 3:
        print(f"   ordering Strong >= Mid >= Weak : "
              f"{'PASS' if finite[0] >= finite[1] >= finite[2] else 'FAIL'}")
    else:
        print("   ordering: UNDETERMINED (grade not recorded in the stored "
              "breakdown for some bands)")
    print("=" * W)

    if args.fit:
        print("\n[master_backtest] fitting the MasterJudge head "
              "(champion/challenger)…")
        stats = mj.fit()
        import json
        print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
