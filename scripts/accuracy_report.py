"""How well do grades match YOUR stars? One table, every change.

Usage: venv\\Scripts\\python.exe scripts/accuracy_report.py [--strong 0.58 --mid 0.39]
"""
import argparse
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import accuracy as acc  # noqa: E402
import master_judge as mj  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--strong", type=float, default=0.58)
ap.add_argument("--mid", type=float, default=0.39)
args = ap.parse_args()

rows = mj.collect_rows()
if not rows:
    sys.exit("no rated photos with a machine score — rate some photos first")
true = [acc.stars_to_class(r["stars"]) for r in rows]
pred = [acc.score_to_class(r["score"], args.strong, args.mid) for r in rows]
print(f"rated photos measured : {len(rows)}")
print(f"3-class agreement     : {acc.agreement(pred, true):.3f}")
print(f"spearman (score~stars): {mj.spearman([r['score'] for r in rows], [r['stars'] for r in rows]):.3f}")
cm = acc.confusion(pred, true)
print("confusion  (rows = YOUR grade, cols = machine)   Weak  Mid  Strong")
for name, row in zip(("Weak", "Mid", "Strong"), cm):
    print(f"  {name:6}{'':38}{row[0]:5}{row[1]:5}{row[2]:7}")
by = collections.defaultdict(list)
for r, t, p in zip(rows, true, pred):
    by[acc.shoot_of(r["path"])].append(t == p)
print("per shoot:")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):4}  {sum(v) / len(v):.2f}  {k}")
