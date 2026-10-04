"""Re-grade COPIES of every rated photo still on disk and attach the fresh
features + score to its rating. Copies keep the user's library untouched; the
LanceDB/catalog rows the throwaway grade writes are purged afterwards.

Usage: venv\\Scripts\\python.exe scripts/backfill_rating_features.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UNIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UNIT / "src"))
sys.path.insert(0, str(UNIT / ".claude" / "skills" / "run-framegrade"))
import ratings_store as rs  # noqa: E402
import driver  # noqa: E402  (_lance_purge: removes rows the grade wrote into the real store)

stars = {p: s for p, s in rs.load().items() if os.path.exists(p)}
if not stars:
    sys.exit("no rated photos found on disk")
root = Path(tempfile.mkdtemp(prefix="fg_backfill_"))
copy_to_orig, folders, parent_ids = {}, set(), {}
for p in stars:
    src = Path(p)
    d = root / f"f{parent_ids.setdefault(str(src.parent), len(parent_ids))}"
    d.mkdir(exist_ok=True)
    dst = d / src.name
    if dst.exists():
        continue
    shutil.copy2(src, dst)
    copy_to_orig[str(dst)] = p
    folders.add(str(d))
print(f"copied {len(copy_to_orig)} rated photos into {len(folders)} folders", flush=True)
req, prog = root / "req.json", root / "p.jsonl"
prog.write_text("", encoding="utf-8")
req.write_text(json.dumps({"folders": sorted(folders), "preset": "Classic Street",
    "force_rescan": True, "scan_mode": False, "deep_grade": False,
    "catalog_path": str(root / "c.json"), "data_dir": str(root),
    "mogco_target": 0, "sample_limit": 0}), encoding="utf-8")
try:
    with open(root / "run.log", "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run([str(UNIT / "venv" / "Scripts" / "python.exe"), "grade_runner.py",
                        str(req), str(prog)], cwd=str(UNIT), stdout=lf, stderr=subprocess.STDOUT,
                       timeout=7200, env=dict(os.environ, PYTHONIOENCODING="utf-8", SIGLIP_TIER="high"))
    import lance_store  # noqa: E402
    t = lance_store._connect_or_create().to_pandas()
    t = t[t["path"].str.startswith(str(root))]
    n = 0
    for _, r in t.iterrows():
        orig = copy_to_orig.get(r["path"])
        bd = r["breakdown"] if isinstance(r["breakdown"], dict) else json.loads(r["breakdown"] or "{}")
        if orig and rs.attach_features(orig, bd, score=float(r["score"])):
            n += 1
    print(f"attached features to {n}/{len(stars)} rated photos")
finally:
    driver._lance_purge(str(root))
    shutil.rmtree(root, ignore_errors=True)
