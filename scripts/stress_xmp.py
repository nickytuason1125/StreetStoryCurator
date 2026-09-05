r"""
Stress test for the XMP sidecar export (/api/export/metadata).

Five dimensions, because "it worked on two files" is not evidence:
  1. VOLUME       — one request, ~800 real photos from the rated sets.
  2. CONCURRENCY  — 8 parallel requests while a probe thread measures
                    /api/config latency, to expose event-loop blocking.
  3. ADVERSARIAL  — unicode/quotes/newlines, 20 KB critique, malformed
                    grades, nonexistent paths, path-escape attempts.
  4. INTEGRITY    — MD5 of sampled ORIGINALS before/after everything.
  5. ENRICHMENT   — every xmp:Rating must equal the user's actual stars
                    in the durable ratings store (0 -> attribute absent).

Run:  venv\Scripts\python.exe scripts\stress_xmp.py
Requires the server on 127.0.0.1:8001 (uses the documented
X-Requested-With: FirstCut header for non-browser access).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

API = "http://127.0.0.1:8001"
DEST = ROOT / "output" / "xmp_stress"
FOLDERS = [
    Path(r"C:\Users\Nicky Tuason\Desktop\tpe2026_2"),
    Path(r"C:\Users\Nicky Tuason\Desktop\LX3 2024\DCIM\110_PANA"),
    Path(r"C:\Users\Nicky Tuason\Desktop\LX3 2024\DCIM\111_PANA"),
]
EXTS = {".jpg", ".jpeg", ".rw2", ".raw", ".dng", ".png", ".tif", ".tiff"}
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
GRADES = ["Strong ✅", "Mid ⚠️", "Weak ❌"]
LABELS = {"Strong": "Green", "Mid": "Yellow", "Weak": "Red"}

import ratings_store                      # noqa: E402  (enrichment oracle)

_pass, _fail = [], []


def check(name, ok, detail=""):
    (_pass if ok else _fail).append(name)
    print(f"  {'PASS' if ok else 'FAIL'} — {name}" + (f" | {detail}" if detail else ""))


def post(photos, dest=None):
    body = {"photos": photos}
    if dest:
        body["dest"] = dest
    req = urllib.request.Request(
        f"{API}/api/export/metadata",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "X-Requested-With": "FirstCut"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, {"detail": e.read().decode()[:200]}


def probe_until(t_stop):
    lat = []
    while time.time() < t_stop:
        t0 = time.time()
        try:
            urllib.request.urlopen(f"{API}/api/config", timeout=30).read(64)
        except Exception:
            pass
        lat.append(time.time() - t0)
        time.sleep(0.05)
    return lat


def probe_until_event(done):
    import threading
    lat = []
    while not done.is_set():
        t0 = time.time()
        try:
            urllib.request.urlopen(f"{API}/api/config", timeout=30).read(64)
        except Exception:
            pass
        lat.append(time.time() - t0)
        time.sleep(0.05)
    return lat


def pcts(lat):
    ms = sorted(x * 1000 for x in lat)
    return (ms[len(ms) // 2], ms[min(len(ms) - 1, int(len(ms) * 0.99))], ms[-1])


def parse_desc(xmp_path: Path) -> dict:
    raw = xmp_path.read_bytes()
    root = ET.fromstring(raw.decode("utf-8-sig"))
    d = root.find(f".//{{{NS_RDF}}}Description")
    return {k.split("}")[1]: v for k, v in d.attrib.items()
            if "}" in k and NS_RDF not in k and "adobe:ns:meta" not in k}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def main() -> int:
    photos = [p for f in FOLDERS for p in sorted(f.iterdir())
              if p.suffix.lower() in EXTS]
    print(f"corpus: {len(photos)} photos from 3 folders")
    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("*.xmp"):
        old.unlink()
    stars = ratings_store.load()
    idx = {p: i for i, p in enumerate(photos)}

    # -- 4. Integrity baseline (sampled originals) --------------------------
    sample = photos[::37]
    before = {p: md5(p) for p in sample}
    print(f"integrity baseline: md5 of {len(sample)} originals")

    # -- 1. VOLUME: one request, everything ----------------------------------
    print(f"\n[1] VOLUME — single request, {len(photos)} photos")
    payload = [{"path": str(p), "grade": GRADES[i % 3],
                "score": 0.30 + (i % 70) / 100,
                "personal_score": 0.40 + (i % 60) / 100,
                "critique": f'Frame {i} — "layered" framing\nsecond line ✅',
                "breakdown": {"Lighting": 0.8, "Narrative": 0.65}}
               for p, i in idx.items()]
    t0 = time.time()
    import threading
    _done = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fp = pool.submit(probe_until_event, _done)
        code, resp = post(payload, dest=str(DEST))
        dt = time.time() - t0
        _done.set()
        vol_lat = fp.result()
    v50, v99, vmax = pcts(vol_lat)
    print(f"  volume probe: n={len(vol_lat)} p50={v50:.0f}ms p99={v99:.0f}ms max={vmax:.0f}ms")
    check("event loop stays live during an 800-photo export (max < 500ms)",
          vmax < 500, f"max {vmax:.0f}ms")
    exported = resp.get("exported", 0)
    errors = [r for r in resp.get("results", []) if "error" in r]
    check(f"status 200 (got {code})", code == 200)
    check(f"exported == sent ({exported}/{len(photos)})", exported == len(photos))
    check(f"zero per-photo errors (got {len(errors)})", not errors,
          errors[0]["error"][:80] if errors else "")
    check(f"throughput >= 50 photos/s ({dt:.2f}s total, {len(photos)/dt:.0f}/s)",
          len(photos) / dt >= 50)
    n_files = len(list(DEST.glob("*.xmp")))
    check(f"file count matches ({n_files})", n_files == len(photos))
    bad = []
    for p in DEST.glob("*.xmp"):
        try:
            parse_desc(p)
        except Exception as e:
            bad.append(f"{p.name}: {e}")
    check(f"all {n_files} sidecars parse as valid XMP", not bad,
          bad[0] if bad else "")
# -- 5. ENRICHMENT: rating attr == user stars, exactly -------------------
    print("\n[5] ENRICHMENT — xmp:Rating must equal the user's real stars")
    stems, mismatch, fabricated, lbl_bad = {}, [], [], []
    for orig in photos:
        stems.setdefault(orig.stem, orig)
    for p in DEST.glob("*.xmp"):
        orig = stems.get(p.stem)
        if orig is None:
            continue
        attrs = parse_desc(p)
        got, want = attrs.get("Rating"), stars.get(str(orig), 0)
        if want > 0 and got != str(want):
            mismatch.append(f"{p.stem}: xmp {got} vs store {want}")
        if want == 0 and got is not None:
            fabricated.append(f"{p.stem}: rating {got} fabricated")
        if attrs.get("Label") != LABELS[GRADES[idx[orig] % 3].split()[0]]:
            lbl_bad.append(p.stem)
    n_rated = len([p for p in photos if str(p) in stars])
    check(f"all {n_rated} rated photos match the store exactly", not mismatch,
          mismatch[0] if mismatch else "")
    check("no rating fabricated for unrated photos", not fabricated,
          fabricated[0] if fabricated else "")
    check("every Label matches its grade bucket", not lbl_bad,
          lbl_bad[0] if lbl_bad else "")

    # -- 2. CONCURRENCY: 8x100 while probing latency -------------------------
    print("\n[2] CONCURRENCY — 8 parallel requests of 100 photos + latency probe")
    with ThreadPoolExecutor(max_workers=9) as pool:
        f = pool.submit(probe_until, time.time() + 15)
        time.sleep(0.1)
        t0 = time.time()
        futures = [pool.submit(post,
                    [{"path": str(p), "grade": GRADES[(i + w) % 3], "score": 0.5}
                     for i, p in enumerate(photos[w * 100:(w + 1) * 100])],
                    dest=str(DEST)) for w in range(8)]
        results = [b.result() for b in futures]
        dt_c = time.time() - t0
        lat = f.result()
    ok_codes = all(c == 200 and r.get("exported") == 100 for c, r in results)
    check(f"all 8 concurrent batches exported 100/100 ({dt_c:.1f}s)", ok_codes)
    lat_ms = sorted(x * 1000 for x in lat)
    p50 = lat_ms[len(lat_ms) // 2]
    p99 = lat_ms[min(len(lat_ms) - 1, int(len(lat_ms) * 0.99))]
    print(f"  probe: n={len(lat_ms)} p50={p50:.0f}ms p99={p99:.0f}ms max={lat_ms[-1]:.0f}ms")
    check("server stayed responsive during export (p99 < 2000ms)", p99 < 2000,
          f"p99 {p99:.0f}ms")
# -- 3. ADVERSARIAL payloads ---------------------------------------------
    print("\n[3] ADVERSARIAL — hostile payloads must fail per-photo, not per-batch")
    real = str(photos[0])
    adv = [
        ("20KB critique", {"path": real, "grade": "Strong ✅", "score": 0.9,
                           "critique": "x" * 20000}),
        ("lowercase grade", {"path": real, "grade": "strong", "score": 0.9}),
        ("empty grade", {"path": real, "grade": "", "score": 0.9}),
        ("no grade key", {"path": real, "score": 0.9}),
        ("unicode critique", {"path": real, "grade": "Mid ⚠️",
                              "critique": '引号 "quotes" <tag> \n emoji ✅ 日本語'}),
        ("breakdown with quotes", {"path": real, "grade": "Mid ⚠️",
                                   "breakdown": {'a"b': 'c<d>'}}),
        ("nonexistent path", {"path": r"C:\does\not\exist.jpg", "grade": "Mid ⚠️"}),
        ("path escape attempt", {"path": "..\\..\\windows\\system32\\config.sys",
                                 "grade": "Mid ⚠️"}),
        ("null-ish values", {"path": real, "grade": None, "score": None}),
    ]
    ok_adv = True
    for name, meta in adv:
        code, resp = post([meta], dest=str(DEST))
        one = resp.get("results", [{}])[0]
        if not (code == 200 and ("sidecar" in one or "error" in one)):
            ok_adv = False
            print(f"  FAIL — {name}: HTTP {code} {json.dumps(resp)[:120]}")
    check("all 9 hostile payloads handled without crashing the endpoint", ok_adv)
    code, resp = post([
        {"path": real, "grade": "Strong ✅", "score": 0.9},
        {"path": r"C:\nope.jpg", "grade": "Mid ⚠️"}], dest=str(DEST))
    r0, r1 = resp.get("results", [{}, {}])
    check("mixed batch: good exports, bad errors, exported==1",
          code == 200 and "sidecar" in r0 and "error" in r1
          and resp.get("exported") == 1)
    post([{"path": real, "grade": "strong", "score": 0.9}], dest=str(DEST))
    check("lowercase grade writes no Label (no silent colour assignment)",
          "Label" not in parse_desc(DEST / (Path(real).stem + ".xmp")))
    post([{"path": real, "grade": "Weak ❌", "stars": 99}], dest=str(DEST))
    check("stars=99 clamps to 5",
          parse_desc(DEST / (Path(real).stem + ".xmp")).get("Rating") == "5")

    # -- 4b. Integrity after everything --------------------------------------
    print("\n[4] INTEGRITY — originals after all stress")
    changed = [str(p) for p in sample if md5(p) != before[p]]
    check(f"all {len(sample)} sampled originals byte-identical", not changed,
          changed[0] if changed else "")

    print(f"\n{'=' * 60}\nSTRESS RESULT: {len(_pass)} passed, {len(_fail)} failed")
    for f in _fail:
        print(f"  FAILED: {f}")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())