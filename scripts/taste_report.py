r"""
Taste agreement report — how well the machine and your taste head know you.

This is the "trust dial" made measurable. Every star rating snapshots the
machine score and the PersonalHead score AT THE MOMENT OF RATING
(ratings_store.get_score_snapshot), so agreement can be measured across
re-grades that wipe or reshuffle live LanceDB rows.

Metrics
-------
1. Machine agreement    — bucket(machine score) vs bucket(your stars)
                          (4-5★ Strong / 3★ Mid / 1-2★ Weak; score >=0.60 /
                           0.41-0.60 / <0.41), with the confusion matrix.
2. Taste discrimination — AUC over all (high=4-5★, low=1-2★) pairs: the
                          fraction where personal_score(high) > personal_
                          score(low). 0.50 = coin flip, 1.00 = perfect
                          separation.
3. Monotonicity         — mean personal score per star level; your head
                          should climb as your stars do.

Read-only. Writes reports/taste_report.md.

Run:  venv\Scripts\python.exe scripts/taste_report.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ratings_store                               # noqa: E402


def bucket_score(s: float) -> str:
    if s >= 0.60:
        return "Strong"
    if s >= 0.41:
        return "Mid"
    return "Weak"


def bucket_stars(stars: int) -> str:
    if stars >= 4:
        return "Strong"
    if stars == 3:
        return "Mid"
    return "Weak"


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    ratings = ratings_store.load()                  # {path: stars}
    rows = []                                      # (stars, score, personal)
    legacy = 0
    for path, stars in ratings.items():
        snap = ratings_store.get_score_snapshot(path)
        if not snap or snap.get("score") is None:
            legacy += 1
            continue
        rows.append((int(stars), float(snap["score"]),
                     float(snap["personal_score"])
                     if snap.get("personal_score") is not None else None))

    by_star: dict[int, list[tuple[float, float | None]]] = {}
    for stars, sc, ps in rows:
        by_star.setdefault(stars, []).append((sc, ps))

    # 1. Machine agreement + confusion
    conf = {t: {p: 0 for p in ("Strong", "Mid", "Weak")}
            for t in ("Strong", "Mid", "Weak")}
    for stars, sc, _ps in rows:
        conf[bucket_stars(stars)][bucket_score(sc)] += 1
    n = len(rows)
    agree = sum(conf[t][t] for t in conf)

    # 2. Taste AUC over (hi, lo) pairs
    hi = [ps for stars, _sc, ps in rows if stars >= 4 and ps is not None]
    lo = [ps for stars, _sc, ps in rows if stars <= 2 and ps is not None]
    win = tie = total = 0
    for h in hi:
        for l in lo:
            total += 1
            if h > l:
                win += 1
            elif h == l:
                tie += 1
    auc = (win + 0.5 * tie) / total if total else None

    lines = [
        "# Taste agreement report",
        f"_{time.strftime('%Y-%m-%d %H:%M')} — {n} rated photos with score snapshots"
        f" ({legacy} legacy ratings without snapshots excluded)_",
        "",
        "## 1. Machine grader vs your stars",
        f"- Bucket agreement: **{agree}/{n} = {100.0 * agree / n:.1f}%**",
        "",
        "| your stars → | machine: Strong | machine: Mid | machine: Weak |",
        "|---|---|---|---|",
    ]
    for t in ("Strong", "Mid", "Weak"):
        lines.append(f"| {t} | {conf[t]['Strong']} | {conf[t]['Mid']} | {conf[t]['Weak']} |")

    lines += ["", "## 2. Taste head discrimination (AUC)"]
    if auc is None:
        lines.append("- Not computable — needs personal-score snapshots on "
                     "both a 4-5★ and a 1-2★ photo.")
    else:
        verdict = ("perfect" if auc >= 0.95 else
                   "strong" if auc >= 0.80 else
                   "useful" if auc >= 0.65 else "weak — rate more photos")
        lines.append(f"- **AUC = {auc:.3f}** over {len(hi)}x{len(lo)} = {total} "
                     f"high/low pairs ({verdict})")
        lines.append("- (0.50 = coin flip, 1.00 = your head ranks every "
                     "photo you rated highly above every photo you rated low)")

    lines += ["", "## 3. Monotonicity — mean scores per star level", "",
              "| stars | n | machine score | taste score |", "|---|---|---|---|"]
    for stars in sorted(by_star):
        scs = [sc for sc, _ in by_star[stars]]
        pss = [ps for _, ps in by_star[stars] if ps is not None]
        lines.append(f"| {stars}★ | {len(scs)} | {mean(scs):.3f} | "
                     f"{mean(pss):.3f} |" if pss else
                     f"| {stars}★ | {len(scs)} | {mean(scs):.3f} | — |")

    lines += ["", "## Reading it",
               "- Machine agreement is the grader's baseline trust level.",
               "- Taste AUC is the PersonalHead's — the number the 0.20→0.70",
               "  confidence-adaptive blend earns its authority with.",
               "- Monotonicity should climb with stars. If it doesn't, the",
               "  head is learning noise, not you."]

    report = "\n".join(lines)
    out = ROOT / "reports" / "taste_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())