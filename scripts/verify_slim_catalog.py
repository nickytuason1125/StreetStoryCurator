"""One-shot verification of the slim-catalog contract (slim GET, detail GET,
merge-on-save). Run: venv\\Scripts\\python.exe scripts\\verify_slim_catalog.py"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from routers import misc  # noqa: E402  (mounts lazily via server_impl)


def main() -> int:
    # 1. slim catalog: no breakdown keys
    resp = asyncio.run(misc.get_catalog(full=False))
    photos = json.loads(resp.body).get("photos", [])
    slim_bad = sum(1 for p in photos if "breakdown" in p)
    print(f"slim rows: {len(photos)} | rows still carrying breakdown: {slim_bad}")

    # 2. full catalog still serves breakdowns
    respf = asyncio.run(misc.get_catalog(full=True))
    photos_f = json.loads(respf.body).get("photos", [])
    full_ok = sum(1 for p in photos_f if "breakdown" in p)
    print(f"full rows: {len(photos_f)} | rows carrying breakdown: {full_ok}")

    if not photos:
        print("catalog empty — nothing further to verify")
        return 0

    # 3. photo-detail returns the breakdown for one path
    p0 = photos[0]["path"]
    respd = asyncio.run(misc.get_photo_detail(path=p0))
    det = json.loads(respd.body).get("photo") or {}
    print(f"photo-detail for selected path: breakdown present = {'breakdown' in det}")

    # 4. save with SLIM rows (exactly what a frontend save sends) — the stored
    #    breakdowns must survive the merge.
    asyncio.run(misc.save_catalog({"photos": photos, "folders": []}))
    stored = json.loads((ROOT / "data" / "cache" / "catalog.json")
                        .read_text(encoding="utf-8")).get("photos", [])
    survived = sum(1 for p in stored if "breakdown" in p)
    print(f"after slim save: {survived}/{len(stored)} stored rows keep breakdown")
    return 0 if slim_bad == 0 and full_ok > 0 and survived > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
