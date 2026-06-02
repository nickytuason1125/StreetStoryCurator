import json, sys

SSE_FILE = r"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\regrade_small.txt"

with open(SSE_FILE, encoding="utf-8") as f:
    content = f.read()

data = None
for line in content.splitlines():
    stripped = line.strip()
    if stripped.startswith("data:") and '"done": true' in stripped:
        try:
            data = json.loads(stripped[5:].strip())
            break
        except Exception as e:
            print("Parse error:", e)

if data is None:
    print("done payload not found")
    sys.exit(1)

targets = {"TPE26-10.jpg", "TPE26-102.jpg"}
for photo in data.get("data", []):
    if photo["filename"] in targets:
        print(f"\n=== {photo['filename']} ===")
        print(f"  Grade    : {photo['grade']}")
        print(f"  Score    : {photo['score']}")
        bd = photo.get("breakdown", {})
        for k, v in bd.items():
            print(f"  {k:<16}: {v}")

print(f"\n=== BATCH SUMMARY ===")
for k in ("total", "strong", "mid", "weak"):
    if k in data:
        print(f"  {k}: {data[k]}")
