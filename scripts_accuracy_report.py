"""Grade-accuracy audit — measures how legitimate the machine's grades are.

Answers three questions with numbers, from durable stores only (never the
ephemeral catalog):

1. RANK AGREEMENT  Spearman correlation between the machine's score and the
   photographer's stars, over every photo he bothered to rate. This is THE
   legitimacy metric: a grader that disagrees with a photographer's ranking
   is decoration regardless of how sophisticated its pipeline is.
2. BAND MONOTONICITY  Mean stars per grade band (Strong/Mid/Weak). Bands must
   be ordered; a violation means the thresholds are miscalibrated.
3. PERSONAL SHIFT  Same correlation using personal_score (after PersonalHead),
   so you can see whether taste-learning moved grades TOWARD your judgements.

Usage:  python scripts_accuracy_report.py   (from street-story-curator/)
Exit code 0 always — this is a report, not a gate.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent   # this script lives in street-story-curator/
sys.path.insert(0, str(ROOT / "src"))


def _rankdata(xs):
    """Average ranks (ties share the mean), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


# ── Reporting an absence ─────────────────────────────────────────────────────
# A correlation of NaN is not a bad score, it is the absence of a measurement,
# and the two must never render the same way. Formatting `float("nan")` through
# `{:+.3f}` produces "+nan", which reads as a number; deriving a verdict from it
# is worse still, because every comparison against NaN is False and the ladder
# below silently bottoms out at the harshest label. These helpers keep "we could
# not measure this" a distinct outcome from "we measured this and it was poor".

def _no_variance_reason(xs, ys, n, x_label):
    """Why is this correlation undefined? Returns a sentence, or None."""
    if n < 3:
        return f"only {n} rated photo(s) - need at least 3 to rank."
    if len(set(xs)) == 1:
        return (f"{x_label} is constant ({xs[0]:.3f}) across all {n} rated "
                f"photos - there is no ranking to correlate.")
    if len(set(ys)) == 1:
        return (f"every one of the {n} rated photos has the same star rating "
                f"- there is no ranking to correlate against.")
    return f"the correlation is undefined for these {n} photos."


def rho_verdict(rho):
    """Interpretation ladder for section 1. NaN is not the bottom rung."""
    if rho is None or math.isnan(rho):
        return None
    return ("excellent" if rho > .5 else "good" if rho > .35 else
            "weak" if rho > .15 else "poor")


def band_ordering(means):
    """True / False / None, where None means a band was empty.

    `means` carries NaN for a band with no photos in it. Comparing NaN always
    yields False, so the old expression reported an outright FAIL for a catalog
    that had merely never produced a Weak photo — a calibration alarm raised by
    missing data. Undetermined is its own answer.
    """
    if any(math.isnan(m) for m in means):
        return None
    return means[0] >= means[1] >= means[2]


def personal_shift_lines(ppairs, rho):
    """Section 3, as a list of lines, so it can be tested without file I/O.

    `rho` is section 1's correlation, or None when section 1 had no data; the
    shift is a difference between two correlations and cannot be stated when
    either one is missing.
    """
    ps, pst = (list(t) for t in zip(*ppairs)) if ppairs else ([], [])
    n = len(ppairs)
    rho_p = spearman(ps, pst)
    lines = ["", "3. PERSONAL SHIFT (after PersonalHead taste learning)"]

    if math.isnan(rho_p):
        lines.append(f"   Spearman rho = undefined   (n={n})")
        lines.append(f"   {_no_variance_reason(ps, pst, n, 'personal_score')}")
        return lines

    lines.append(f"   Spearman rho = {rho_p:+.3f}   (n={n})")
    if rho is not None and not math.isnan(rho):
        d = rho_p - rho
        arrow = "toward" if d > 0 else "away from"
        lines.append(f"   taste learning moved agreement {arrow} your "
                     f"judgements by {abs(d):+.3f}")
    return lines


def main():
    ratings = json.loads(
        (ROOT / "cache" / "user_ratings.json").read_text(encoding="utf-8"))
    ratings = ratings.get("ratings", ratings)
    if not ratings:
        print("No user ratings found — rate some photos first.")
        return

    import lance_store as ls
    rows = {}
    try:
        rows = {r["path"]: r for r in ls.query_all(min_score=0.0)}
    except Exception as _e:
        print(f"[report] lance store unavailable ({_e}) — falling back to catalog")
    if not rows:
        # The durable store can be wiped by an encoder-tier switch or re-grade;
        # the catalog keeps the CURRENT grades and uses identical path keys.
        cat = ROOT / "data" / "cache" / "catalog.json"
        for cand in (cat, ROOT / "cache" / "catalog.json"):
            if cand.exists():
                data = json.loads(cand.read_text(encoding="utf-8"))
                for p in data.get("photos", []):
                    if isinstance(p.get("score"), (int, float)):
                        rows[p["path"]] = {"score": p["score"],
                                           "personal_score": p.get("personal_score"),
                                           "grade": p.get("grade", "")}
                print(f"[report] joined against catalog: {len(rows)} scored photos")
                break

    pairs = []      # (score, stars)
    ppairs = []     # (personal_score, stars)
    missing = 0
    for path, stars_raw in ratings.items():
        r = rows.get(path)
        try:
            stars = int(stars_raw)
        except Exception:
            continue
        if r is None:
            missing += 1
            continue
        s = r.get("score")
        p = r.get("personal_score")
        if isinstance(s, (int, float)):
            pairs.append((float(s), stars))
        if isinstance(p, (int, float)) and p != 0:
            ppairs.append((float(p), stars))

    W = 62
    print("=" * W)
    print("FRAMEGRADE GRADE-ACCURACY AUDIT")
    print("=" * W)
    print(f"rated photos:          {len(ratings)}")
    print(f"  with machine score:  {len(pairs)}")
    print(f"  store miss:          {missing}  (rated before last re-grade)")
    print()

    rho = None
    if pairs:
        scores, stars = zip(*pairs)
        rho = spearman(list(scores), list(stars))
        print("-" * W)
        print(f"1. RANK AGREEMENT (machine score vs your stars)")
        verdict = rho_verdict(rho)
        if verdict is None:
            print(f"   Spearman rho = undefined   (n={len(pairs)})")
            print("   " + _no_variance_reason(list(scores), list(stars),
                                              len(pairs), "the machine score"))
        else:
            print(f"   Spearman rho = {rho:+.3f}   (n={len(pairs)})")
            print(f"   interpretation: {verdict}")
        print()
        print(f"2. BAND MONOTONICITY (mean stars per band)")
        bands = {"Strong": [], "Mid": [], "Weak": []}
        for path, stars_raw in ratings.items():
            g = rows.get(path, {}).get("grade", "")
            try:
                st = int(stars_raw)
            except Exception:
                continue
            for b in bands:
                if g.startswith(b):
                    bands[b].append(st)
        means = []
        for b in ("Strong", "Mid", "Weak"):
            v = bands[b]
            m = sum(v) / len(v) if v else float("nan")
            means.append(m)
            print(f"   {b:<7} n={len(v):<4} mean stars = {m:.2f}")
        ok = band_ordering(means)
        label = ("UNDETERMINED (a band has no rated photos)" if ok is None
                 else "PASS" if ok else "FAIL")
        print(f"   ordering Strong >= Mid >= Weak : {label}")

    if ppairs:
        for line in personal_shift_lines(ppairs, rho):
            print(line)
    print("=" * W)


if __name__ == "__main__":
    main()
