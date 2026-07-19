"""Train the PersonalHead on the user's star ratings as a taste baseline, then
measure whether the blended grade (0.80*model + 0.20*head) aligns better with
the user's stars. One-off calibration tool."""
import sys, json, os, random
sys.path.insert(0, 'src')
import numpy as np
from scipy.stats import spearmanr

import lance_store as ls
import personal_head as ph

STAR_GRADE = lambda s: "Strong ✅" if s >= 4 else ("Mid ⚠️" if s == 3 else "Weak ❌")

# ── Join catalog stars with LanceDB embeddings ───────────────────────────────
cat = json.load(open('cache/catalog.json', encoding='utf-8'))
items = cat if isinstance(cat, list) else (cat.get('photos') or cat.get('gallery') or [])
stars_by_path = {x['path']: int(x['stars']) for x in items
                 if isinstance(x, dict) and int(x.get('stars', 0) or 0) > 0}
rows = ls.query_all(min_score=0.0)
emb_by_path = {r['path']: np.asarray(r['embedding'], dtype=np.float32) for r in rows}
score_by_path = {r['path']: float(r.get('score', 0.5)) for r in rows}

data = [(p, stars_by_path[p], emb_by_path[p], score_by_path[p])
        for p in stars_by_path if p in emb_by_path]
print(f"joined {len(data)} rated photos with embeddings + scores")

# ── Baseline alignment BEFORE training ───────────────────────────────────────
def blended_align(tag):
    embs = np.stack([d[2] for d in data])
    head = ph.score(embs)                      # [0,1]
    model = np.array([d[3] for d in data])
    blend = 0.80 * model + 0.20 * head
    starv = np.array([d[1] for d in data])
    ug = np.array([2 if s >= 4 else (1 if s == 3 else 0) for s in starv])
    def bucket(v): return 2 if v >= 0.60 else (1 if v >= 0.41 else 0)
    mg = np.array([bucket(v) for v in blend])
    rho = spearmanr(starv, blend)[0]
    agree = (ug == mg).mean()
    strong_as_weak = int(np.sum((ug == 2) & (mg == 0)))
    print(f"  [{tag}] Spearman(stars,blend)={rho:.3f}  bucket_agree={agree:.0%}  "
          f"user-Strong->Weak={strong_as_weak}  head_range=[{head.min():.2f},{head.max():.2f}]")
    return rho, agree

print("\nBEFORE training:")
blended_align("before")

# ── Build balanced ranking pairs across grade tiers ──────────────────────────
by_grade = {"Strong ✅": [], "Mid ⚠️": [], "Weak ❌": []}
for p, s, e, sc in data:
    by_grade[STAR_GRADE(s)].append(e)
print(f"\ntier counts: Strong={len(by_grade['Strong ✅'])} "
      f"Mid={len(by_grade['Mid ⚠️'])} Weak={len(by_grade['Weak ❌'])}")

random.seed(0)
pairs = []
def mk(hi_grade, lo_grade, n):
    hi, lo = by_grade[hi_grade], by_grade[lo_grade]
    if not hi or not lo: return
    for _ in range(n):
        pairs.append({"emb1": random.choice(hi), "grade1": hi_grade,
                      "emb2": random.choice(lo), "grade2": lo_grade})
mk("Strong ✅", "Weak ❌", 120)
mk("Strong ✅", "Mid ⚠️", 120)
mk("Mid ⚠️", "Weak ❌", 120)
random.shuffle(pairs)
print(f"training on {len(pairs)} balanced ranking pairs...")

# 3 epochs over the pairs for a stable fit
for ep in range(3):
    loss = ph.update_batch(pairs)
    print(f"  epoch {ep+1}: mean loss {loss:.4f}")

print("\nAFTER training:")
blended_align("after")
print("\nPersonalHead saved to models/personal_head.pt")
