import json, sys
import numpy as np

files = sys.argv[1:]
data = {}
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    name = f.replace("_ab_", "").replace(".json", "")
    data[name] = d

names = list(data)
print(f"{'model':<22} {'s/img':>6} {'load_s':>7} {'mean':>6} {'std':>6} {'min':>5} {'max':>5} {'blank':>6}")
for n, d in data.items():
    sc = np.array([v["score"] for v in d["results"].values()])
    blank = sum(1 for v in d["results"].values() if not v["breakdown"])
    print(f"{n:<22} {d['s_per_img']:>6} {d['t_load']:>7} {sc.mean():>6.3f} {sc.std():>6.3f} "
          f"{sc.min():>5.2f} {sc.max():>5.2f} {blank:>6}")

if len(names) >= 2:
    a, b = names[0], names[1]
    ra, rb = data[a]["results"], data[b]["results"]
    common = sorted(set(ra) & set(rb))
    sa = np.array([ra[k]["score"] for k in common])
    sb = np.array([rb[k]["score"] for k in common])
    print(f"\npearson r = {np.corrcoef(sa, sb)[0,1]:.3f}   mean|diff| = {np.mean(np.abs(sa-sb)):.3f}")
    # rank agreement on top-8 / bottom-8
    top_a = set(np.array(common)[np.argsort(-sa)[:8]])
    top_b = set(np.array(common)[np.argsort(-sb)[:8]])
    bot_a = set(np.array(common)[np.argsort(sa)[:8]])
    bot_b = set(np.array(common)[np.argsort(sb)[:8]])
    print(f"top-8 overlap: {len(top_a & top_b)}/8   bottom-8 overlap: {len(bot_a & bot_b)}/8")
    print(f"\n{'photo':<16} {a[:14]:>14} {b[:14]:>14}  largest disagreements")
    diffs = sorted(common, key=lambda k: -abs(ra[k]["score"] - rb[k]["score"]))[:5]
    for k in diffs:
        print(f"{k:<16} {ra[k]['score']:>14.2f} {rb[k]['score']:>14.2f}")
    print(f"\nsample critiques from {b}:")
    for k in common[:3]:
        print(f"  {k}: {rb[k]['critique'][:110]}")
