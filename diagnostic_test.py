"""
Diagnostic test: run grade_pipeline_v2 on target images in isolation.
Prints dominant_idx, raw blend, post-gate score, narr-clutter fires, etc.
"""
import sys, os, shutil, tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

print(f"CWD: {os.getcwd()}")
print(f"cache/archetype_embs.npy exists: {(ROOT / 'cache' / 'archetype_embs.npy').exists()}")
print(f"cache/archetype_embs.hash exists: {(ROOT / 'cache' / 'archetype_embs.hash').exists()}")

TARGETS = [
    Path(r"C:\Users\Nicky Tuason\Desktop\tpe2026_2\TPE26-10.jpg"),   # Coin Laundry — should be Strong/Mid
    Path(r"C:\Users\Nicky Tuason\Desktop\tpe2026_2\TPE26-102.jpg"),  # Market Table — should be Mid (blurry, clutter)
]

for target in TARGETS:
    if not target.exists():
        print(f"\nERROR: {target.name} not found — skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC TEST — {target.name}")
    print(f"{'='*60}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        dst = Path(tmpdir) / target.name
        shutil.copy2(str(target), str(dst))

        from grade_pipeline_v2 import run_v2

        result = run_v2(
            folder_path=tmpdir,
            preset="Classic Street",
            force_rescan=True,
            progress=lambda f, d: print(f"  [{f:5.1%}] {d}"),
            mogco_target=0,
        )

    gallery = result.get("gallery", [])
    print(f"\n--- FINAL RESULT: {target.name} ---")
    if gallery:
        photo = gallery[0]
        print(f"  grade       : {photo.get('grade')}")
        print(f"  final score : {photo.get('score')}")
        bd = photo.get('breakdown', {})
        for k, v in bd.items():
            print(f"  {k:<16}: {v}")
    else:
        print("  No gallery results")
        print(f"  Keys: {list(result.keys())}")
