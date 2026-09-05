"""Analyze people-search discrimination: who matches whom, at what distances."""
import json
import os
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def search(path: str, idx: int = 0) -> dict:
    q = urllib.parse.quote(path)
    req = urllib.request.Request(
        f"http://127.0.0.1:8001/api/people-search?path={q}&idx={idx}",
        headers={"X-Requested-With": "FirstCut"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())


q = r"C:\Users\Nicky Tuason\Desktop\LAS\LAS2-37.jpg"
d = search(q)
ms = d["matches"]
print(f"total matches: {len(ms)}")
by_group = {}
for m in ms:
    p = m["path"]
    if "\\LAS\\" in p:
        g = "LAS"
    elif "scratchpad" in p:
        g = "4STAR(temp)"
    else:
        g = "other"
    by_group.setdefault(g, []).append(m["distance"])
for g, dists in by_group.items():
    print(f"  {g}: {len(dists)} matches, distances {min(dists):.3f}-{max(dists):.3f}")

print("\nLAS matches in detail:")
for m in ms:
    if "\\LAS\\" in m["path"]:
        print(f"  {m['distance']:.4f}  {os.path.basename(m['path'])}")

print("\nfar end of the match list (least similar kept):")
for m in ms[-5:]:
    print(f"  {m['distance']:.4f}  {os.path.basename(m['path'])}")

print("\ndistance histogram (bin = 0.05):")
import collections
hist = collections.Counter(int(m["distance"] / 0.05) for m in ms)
for b in sorted(hist):
    bar = "#" * hist[b]
    print(f"  {b * 0.05:.2f}-{b * 0.05 + 0.05:.2f}: {hist[b]:3d} {bar}")
