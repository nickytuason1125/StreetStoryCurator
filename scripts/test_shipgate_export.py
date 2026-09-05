"""Ship-gate: one live XMP export through the real server, 3 grades,
original-integrity check, Lightroom-parseable XML, correct color labels."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import ratings_store  # noqa: E402
API = "http://127.0.0.1:8001"
PHOTOS = [
    Path(r"C:\Users\Nicky Tuason\Desktop\LAS\LAS2-1.jpg"),
    Path(r"C:\Users\Nicky Tuason\Desktop\LAS\LAS2-11.jpg"),
    Path(r"C:\Users\Nicky Tuason\Desktop\LAS\LAS2-13.jpg"),
]
DEST = ROOT / "output" / "shipgate_xmp"
GRADES = ["Strong ✅", "Mid ⚠️", "Weak ❌"]
LABELS = {"Strong": "Green", "Mid": "Yellow", "Weak": "Red"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


fails = []
def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'} — {name}" + (f" | {detail}" if detail else ""))
    if not ok:
        fails.append(name)


before = {p: md5(p) for p in PHOTOS}
DEST.mkdir(parents=True, exist_ok=True)
for old in DEST.glob("*.xmp"):
    old.unlink()

payload = [{"path": str(p), "grade": g, "score": 0.86, "personal_score": 0.79,
            "critique": "Ship-gate check", "breakdown": {"Lighting": 0.8}}
           for p, g in zip(PHOTOS, GRADES)]
req = urllib.request.Request(
    f"{API}/api/export/metadata",
    data=json.dumps({"photos": payload, "dest": str(DEST)}).encode(),
    headers={"Content-Type": "application/json", "X-Requested-With": "FirstCut"})
with urllib.request.urlopen(req, timeout=120) as r:
    resp = json.loads(r.read().decode())
check("API accepted export", r.status == 200, str(resp)[:80])

NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
user_stars = ratings_store.load()
for p, g in zip(PHOTOS, GRADES):
    xmp = DEST / (p.stem + ".xmp")
    check(f"sidecar written: {p.stem}.xmp", xmp.exists())
    if not xmp.exists():
        continue
    tree = ET.parse(xmp)
    desc = tree.getroot().find(f".//{{{NS_RDF}}}Description")
    label = desc.get("{http://ns.adobe.com/photoshop/1.0/}Label")
    rating = desc.get("{http://ns.adobe.com/xap/1.0/}Rating")
    # Contract: Rating = the USER's stars, present iff the user rated it.
    # Never score-derived, never fabricated (session contract 2026-09).
    stars = user_stars.get(str(p))
    want = str(int(stars)) if stars else None
    check(f"  {g.split()[0]}: Label={label} Rating={rating} (user stars: {stars})",
          label == LABELS[g.split()[0]] and rating == want)
    check(f"  {g.split()[0]}: original byte-identical", md5(p) == before[p])

print("\nSHIP-GATE EXPORT:", "ALL PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
