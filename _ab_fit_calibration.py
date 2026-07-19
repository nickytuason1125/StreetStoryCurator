"""Step 1 gate + Step 3 fit: ranking analysis of the anchored 2B vs baseline,
then affine calibration constants (match mean/std on the shared photo set).

Usage: python _ab_fit_calibration.py _ab_baseline_qwen25_3b.json _ab_qwen3_2b_anchored.json
"""
import json, sys
import numpy as np
from scipy import stats as _st

base = json.load(open(sys.argv[1], encoding="utf-8"))["results"]
cand = json.load(open(sys.argv[2], encoding="utf-8"))["results"]
common = sorted(set(base) & set(cand))
sb = np.array([base[k]["score"] for k in common])
sc = np.array([cand[k]["score"] for k in common])

print(f"n={len(common)}")
print(f"baseline: mean={sb.mean():.3f} std={sb.std():.3f} range=[{sb.min():.2f},{sb.max():.2f}]")
print(f"candidate: mean={sc.mean():.3f} std={sc.std():.3f} range=[{sc.min():.2f},{sc.max():.2f}]")

# ── Step 1: ranking gate ─────────────────────────────────────────────────────
rho, _ = _st.spearmanr(sc, sb)
pear   = float(np.corrcoef(sc, sb)[0, 1])
arr = np.array(common)
top_b, top_c = set(arr[np.argsort(-sb)[:8]]), set(arr[np.argsort(-sc)[:8]])
bot_b, bot_c = set(arr[np.argsort(sb)[:8]]),  set(arr[np.argsort(sc)[:8]])
print(f"\nRANKING GATE: spearman={rho:.3f}  pearson={pear:.3f}  "
      f"top8={len(top_b & top_c)}/8  bottom8={len(bot_b & bot_c)}/8")
verdict = "PASS" if rho >= 0.60 else ("MARGINAL" if rho >= 0.45 else "FAIL")
print(f"VERDICT: {verdict} (pass >= 0.60, fail < 0.45)")

# ── Step 3: affine fit (candidate -> baseline scale) ─────────────────────────
gain   = float(sb.std() / max(sc.std(), 1e-6))
offset = float(sb.mean() - gain * sc.mean())
mapped = np.clip(gain * sc + offset, 0, 1)
print(f"\nAFFINE FIT: gain={gain:.4f}  offset={offset:.4f}")
print(f"mapped: mean={mapped.mean():.3f} std={mapped.std():.3f} "
      f"range=[{mapped.min():.2f},{mapped.max():.2f}]  mean|diff vs base|={np.mean(np.abs(mapped-sb)):.3f}")

# bucket agreement after mapping (absolute thresholds 0.41/0.60)
def bucket(x): return 2 if x >= 0.60 else (1 if x >= 0.41 else 0)
agree = sum(1 for a, b in zip(mapped, sb) if bucket(a) == bucket(b))
print(f"bucket agreement after mapping: {agree}/{len(common)}")

# biggest residual disagreements (for the user's eye)
res = sorted(common, key=lambda k: -abs((gain * cand[k]["score"] + offset) - base[k]["score"]))[:5]
print("\nlargest residual disagreements (photo: mapped_2B vs baseline):")
for k in res:
    print(f"  {k}: {np.clip(gain*cand[k]['score']+offset,0,1):.2f} vs {base[k]['score']:.2f}")
