"""Re-point ratings whose file moved. A rating is relinked only when its
(parent-folder name, file name) pair matches EXACTLY ONE file under the
search roots — camera file names repeat across cards, so anything ambiguous
is reported, never guessed. Dry run unless --apply.

Usage: venv\\Scripts\\python.exe scripts/relink_ratings.py --search "D:\\Photos" "C:\\Users\\Nicky Tuason\\Desktop" [--apply]
"""
import argparse
import collections
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ratings_store as rs  # noqa: E402
from photo_identity import fingerprint  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--search", nargs="+", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

raw = rs._read_raw()
orphans = {p: v for p, v in raw.items() if not os.path.exists(p)}
index = collections.defaultdict(list)
for root in a.search:
    for dp, _, files in os.walk(root):
        for f in files:
            index[(Path(dp).name.lower(), f.lower())].append(os.path.join(dp, f))

linked, ambiguous, missing = {}, 0, 0
for p in orphans:
    key = (Path(p).parent.name.lower(), Path(p).name.lower())
    hits = index.get(key, [])
    if len(hits) == 1 and hits[0] not in raw:
        linked[p] = hits[0]
    elif len(hits) > 1:
        ambiguous += 1
    else:
        missing += 1
print(f"orphaned ratings: {len(orphans)}  relinkable: {len(linked)}  "
      f"ambiguous: {ambiguous}  not found: {missing}")
by_dir = collections.Counter(str(Path(n).parent) for n in linked.values())
for d, c in by_dir.most_common(10):
    print(f"  {c:4}  -> {d}")
if a.apply and linked:
    with rs._lock:
        cur = rs._read_raw()
        for old, new in linked.items():
            e = cur.pop(old)
            e = e if isinstance(e, dict) else {"stars": int(e)}
            e["fp"] = fingerprint(new) or e.get("fp")
            e["size"] = os.path.getsize(new)
            e["relinked_from"] = old
            cur[new] = e
        rs._atomic_write(rs._PATH, cur)
        rs._atomic_write(rs._BACKUP, cur)
    print(f"relinked {len(linked)} — now run scripts/backfill_rating_features.py")
