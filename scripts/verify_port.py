"""Verify port 8000 ownership + backend identity, writing to verify_port.txt."""
import json
import urllib.request

out = []
try:
    # who owns 8000
    import psutil
    owner = None
    for c in psutil.net_connections(kind="inet"):
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == 8000 and c.pid:
            try:
                pr = psutil.Process(c.pid)
                owner = f"pid {c.pid} = {pr.name()}: {' '.join(pr.cmdline() or [])[:110]}"
            except Exception:
                owner = f"pid {c.pid}"
    out.append(f"port 8000 owner: {owner}")

    req = urllib.request.Request("http://127.0.0.1:8000/api/config",
                                 headers={"X-Requested-With": "FirstCut"})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
    out.append(f"config keys: {sorted(d.keys())[:6]}")

    req = urllib.request.Request("http://127.0.0.1:8000/api/catalog",
                                 headers={"X-Requested-With": "FirstCut"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    if raw[:2] == b"\x1f\x8b":
        import gzip
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    out.append(f"catalog photos: {len(d.get('photos', []))}")
    out.append("VERDICT: real backend serving" if len(d.get("photos", [])) > 60000
               else "VERDICT: suspicious responder")
except Exception as e:
    out.append(f"FAILED: {type(e).__name__}: {e}")

Path = __import__("pathlib").Path
Path("verify_port.txt").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
