"""bench_api.py — API-layer latency benchmark (Part 1 of the UX suite).

Benchmarks the endpoints the frontend actually calls, against a RUNNING
backend (installed app or `python server.py`). No server is spawned.

  usage:
    venv\\Scripts\\python.exe scripts\\bench_api.py [--base http://127.0.0.1:8000]
        [--iters 12] [--thumbs 20] [--details 10] [--skip-grade]

Measures cold/warm latency percentiles per endpoint, payload sizes, and a
grade_runner pass. Writes reports/bench_api.md + reports/bench_api.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT.parent / "Sample_Street"


def pct(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    vs = sorted(values)
    n = len(vs)

    def p(q: float) -> float:
        return round(vs[min(n - 1, int(q * n))], 1)

    return {"n": n, "p50": p(0.50), "p90": p(0.90), "p99": p(0.99),
            "max": round(vs[-1], 1), "mean": round(statistics.mean(vs), 1)}


def timed(fn, iters: int, warmup: int = 1) -> tuple[list[float], list[int]]:
    lat, sizes = [], []
    for i in range(iters + warmup):
        t0 = time.perf_counter()
        size = fn()
        if i >= warmup:
            lat.append((time.perf_counter() - t0) * 1000)
            sizes.append(int(size or 0))
    return lat, sizes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--thumbs", type=int, default=20)
    ap.add_argument("--details", type=int, default=10)
    ap.add_argument("--skip-grade", action="store_true")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    import requests
    s = requests.Session()
    out: dict = {"base": base, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "sections": {}}

    def record(section: str, data: dict) -> None:
        out["sections"][section] = data
        print(f"  {section}: {json.dumps(data, default=str)[:150]}")

    try:
        assert s.get(f"{base}/api/health", timeout=5).status_code == 200
    except Exception as e:
        print(f"backend not reachable at {base}: {e}")
        print("start the app (or python server.py) first.")
        return 1

    lat, _ = timed(lambda: len(s.get(f"{base}/api/health", timeout=10).content), args.iters)
    record("health_warm", pct(lat))
    lat, _ = timed(lambda: len(s.get(f"{base}/api/health/engine", timeout=30).content), 4)
    record("engine_health", pct(lat))

    lat, sizes = timed(lambda: len(s.get(f"{base}/api/catalog", timeout=120).content),
                       min(args.iters, 10))
    slim = pct(lat)
    slim["bytes"] = max(sizes) if sizes else 0
    record("catalog_slim", slim)

    lat, sizes = timed(lambda: len(s.get(f"{base}/api/catalog",
                                         params={"full": 1}, timeout=180).content), 2, warmup=0)
    full = pct(lat)
    full["bytes"] = max(sizes) if sizes else 0
    record("catalog_full", full)

    cat = s.get(f"{base}/api/catalog", timeout=120).json()
    photos = cat.get("photos", [])
    print(f"  catalog rows: {len(photos)}")
    return main2(args, base, s, out, photos)


def main2(args, base, s, out, photos) -> int:
    def record(section: str, data: dict) -> None:
        out["sections"][section] = data
        print(f"  {section}: {json.dumps(data, default=str)[:150]}")

    if photos:
        lat, _ = timed(lambda: len(s.get(f"{base}/api/photo-detail",
                                         params={"path": photos[0]["path"]}, timeout=30).content),
                       args.details)
        record("photo_detail_hot", pct(lat))

        thumbs = [p["path"] for p in photos[:args.thumbs]]
        cold_lat = []
        for p in thumbs:
            t0 = time.perf_counter()
            s.get(f"{base}/api/thumb", params={"path": p}, timeout=60)
            cold_lat.append((time.perf_counter() - t0) * 1000)
        record("thumb_first_touch", pct(cold_lat))

        lat, sizes = timed(lambda: sum(len(s.get(f"{base}/api/thumb",
                                                 params={"path": p}, timeout=30).content)
                                       for p in thumbs), 5, warmup=1)
        warm = pct([l / max(1, len(thumbs)) for l in lat])
        warm["bytes_per_thumb"] = (max(sizes) // max(1, len(thumbs))) if sizes else 0
        record("thumb_warm_avg", warm)

        target = photos[0]["path"]
        prior = next((p.get("stars") for p in photos if p["path"] == target), None)
        lat, _ = timed(lambda: s.post(f"{base}/api/personal/star",
                                      json={"path": target, "stars": 5}, timeout=30).status_code,
                       3, warmup=0)
        record("star_roundtrip", pct(lat))
        s.post(f"{base}/api/personal/star",
               json={"path": target, "stars": prior if prior is not None else 0}, timeout=30)

    lat, _ = timed(lambda: len(s.get(f"{base}/api/places", timeout=30).content), 4)
    record("places", pct(lat))

    if not args.skip_grade and FOLDER.exists():
        import os
        import tempfile as tf
        fd, req_path = tf.mkstemp(suffix=".bench.json")
        os.close(fd)
        prog = req_path + ".progress.jsonl"
        Path(prog).write_text("", encoding="utf-8")
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump({"folders": [str(FOLDER)], "preset": "classic_street",
                       "force_rescan": False, "scan_mode": True,
                       "deep_grade": False,
                       "catalog_path": str(ROOT / "cache" / "catalog.json"),
                       "data_dir": str(ROOT), "mogco_target": 0}, f)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(ROOT / "grade_runner.py"),
                               req_path, prog], cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=600)
        total_s = time.perf_counter() - t0
        ticks = 0
        for line in Path(prog).read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                if json.loads(line):
                    ticks += 1
            except Exception:
                continue
        record("grade_pass_scan", {"ok": proc.returncode == 0 and ticks > 0,
                                   "progress_lines": ticks,
                                   "total_s": round(total_s, 1)})
        for tmp in (req_path, prog):
            try:
                os.unlink(tmp)
            except OSError:
                pass

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "bench_api.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# API Latency Benchmark", "",
          f"Base: `{base}` · Generated: {out['generated']}", "",
          "| Endpoint | n | p50 (ms) | p90 | p99 | max | bytes |",
          "|---|---|---|---|---|---|---|"]
    for sec, d in out["sections"].items():
        if "p50" in d:
            md.append(f"| {sec} | {d.get('n')} | {d.get('p50')} | {d.get('p90')} "
                      f"| {d.get('p99')} | {d.get('max')} | {d.get('bytes', d.get('bytes_per_thumb', '—'))} |")
        else:
            md.append(f"| {sec} | — | — | — | — | — | {json.dumps(d)[:80]} |")
    (out_dir / "bench_api.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nreport: reports/bench_api.md")
    return 0

# PART2_ANCHOR

if __name__ == "__main__":
    raise SystemExit(main())