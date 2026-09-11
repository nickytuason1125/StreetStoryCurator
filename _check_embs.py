import json, os, sys
sys.path.insert(0, r"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\src")
import numpy as np
import lance_store as ls
from pipeline_stages import cluster_similar, mark_duplicate_groups

d = json.load(open(r"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\cache\catalog.json", encoding="utf-8"))
paths = [p["path"] for p in d["photos"]]
print("paths:", len(paths))
m = ls.query_embeddings_by_paths(paths)
print("embeddings returned:", len(m))
embs = np.stack([np.asarray(m.get(p), dtype=np.float32) if p in m else np.zeros((1536,), dtype=np.float32) for p in paths])
cid = cluster_similar(embs)
scores = np.asarray([p["score"] for p in d["photos"]], dtype=np.float64)
flags = mark_duplicate_groups(cid, scores, paths)
n_cl = sum(1 for c in cid if c >= 0)
b = sum(1 for f in flags if "Best of" in f)
dup = sum(1 for f in flags if "Duplicate" in f)
print(f"clustered: {n_cl}  best-of: {b}  dup-losers: {dup}")