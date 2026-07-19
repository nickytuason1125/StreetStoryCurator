import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from grade_pipeline_v2 import run_v2

def prog(frac, desc=""):
    print(f"[{frac*100:5.1f}%] {desc}", flush=True)

t0 = time.time()
out = run_v2(
    r"C:\Users\Nicky Tuason\Desktop\test_batch_25",
    preset="classic_street",
    force_rescan=True,
    scan_mode=False,
    progress=prog,
)
dt = time.time() - t0
g = out.get("data", out.get("gallery", []))
print(f"\nTOTAL {dt:.0f}s for {len(g)} photos = {dt/max(len(g),1):.1f}s/photo")
strong = sum(1 for r in g if "Strong" in str(r.get("grade","")))
mid    = sum(1 for r in g if "Mid"    in str(r.get("grade","")))
weak   = sum(1 for r in g if "Weak"   in str(r.get("grade","")))
print(f"Strong={strong}  Mid={mid}  Weak={weak}")
for r in sorted(g, key=lambda r: -float(r.get("score") or 0)):
    bd = r.get("breakdown") or {}
    tech = bd.get("Technical")
    print(f"  {float(r.get('score') or 0):.2f}  {str(r.get('grade',''))[:8]:<8} "
          f"tech={tech if tech is not None else '-'}  {os.path.basename(r.get('path',''))}")
