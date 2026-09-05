"""Live e2e test: /api/people-search with a real face from the People index."""
import json
import sys
import urllib.parse
import urllib.request

query_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Nicky Tuason\Desktop\LAS\LAS2-37.jpg"
face_idx = sys.argv[2] if len(sys.argv) > 2 else "0"

q = urllib.parse.quote(query_path)
req = urllib.request.Request(
    f"http://127.0.0.1:8001/api/people-search?path={q}&idx={face_idx}",
    headers={"X-Requested-With": "FirstCut"},
)
d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
print(f"query: {query_path} face #{face_idx}")
print(f"indexed: {d['indexed']} | matches: {len(d['matches'])}")
for m in d["matches"][:12]:
    print(f"  {m['distance']:.4f}  {m['path']}")
