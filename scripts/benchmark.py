"""benchmark.py — repeatable performance & accuracy harness for FrameGrade.

Measures, on this machine, right now:
  1. CULL      — full grade of dataset_images/ through the real grade_runner
                 subprocess: wall time, seconds/image, peak RSS of the whole
                 process tree (runner + encode_worker + iqa_worker), and the
                 per-stage timeline from the progress stream.
  2. SEQUENCE  — MOGCO beam sequencing over the graded library: wall time and
                 determinism (same inputs → identical output, run twice).
  3. STORY     — (optional, --full-story) the full Creative Direction pipeline
                 in story AND competition modes, including role coverage of the
                 produced sequence.

Usage:
    python scripts/benchmark.py                  # cull + sequence
    python scripts/benchmark.py --scan           # cull in fast scan mode
    python scripts/benchmark.py --full-story     # also run story/competition
    python scripts/benchmark.py --skip-cull      # sequence only (needs a prior grade)

Output: reports/benchmark_report.md (+ the raw numbers on stdout).

Everything runs against the REAL pipeline — no mocks — so the numbers are the
numbers a user would get.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATASET = ROOT / "dataset_images"
REPORTS = ROOT / "reports"


# ── RSS sampling ──────────────────────────────────────────────────────────────

class TreeRSSSampler:
    """Samples the total RSS of a process tree every `interval` seconds until
    stopped. Peak is read after the tree exits. psutil only — no other deps."""

    def __init__(self, pid: int, interval: float = 0.5):
        self._pid = pid
        self._interval = interval
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import psutil
            parent = psutil.Process(self._pid)
            while not self._stop.is_set():
                total = 0
                try:
                    procs = [parent, *parent.children(recursive=True)]
                    for p in procs:
                        try:
                            total += p.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                self._peak = max(self._peak, total)
                self._stop.wait(self._interval)
        except Exception:
            pass   # sampling must never break the benchmark

    def stop(self) -> float:
        """Stop sampling; return peak RSS in GB."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak / 1e9


def _run_cull(scan_mode: bool, deep_grade: bool) -> dict:
    """Run one grade through grade_runner.py; return timing/RAM/result stats."""
    import tempfile as _tf
    fd, req_path = _tf.mkstemp(suffix=".benchreq.json"); os.close(fd)
    prog_path = req_path + ".progress.jsonl"
    open(prog_path, "w", encoding="utf-8").close()
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump({
            "folders":      [str(DATASET)],
            "preset":       "classic_street",
            "force_rescan": True,
            "scan_mode":    scan_mode,
            "deep_grade":   deep_grade,
            "catalog_path": str(ROOT / "cache" / "catalog.json"),
            "data_dir":     str(ROOT),
            "mogco_target": 0,
        }, f)

    flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "grade_runner.py"), req_path, prog_path],
        cwd=str(ROOT), creationflags=flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sampler = TreeRSSSampler(proc.pid)
    sampler.start()

    stages: list[tuple[float, str]] = []
    result: dict = {}
    deadline = time.time() + 3600
    while time.time() < deadline:
        if proc.poll() is not None and not _prog_has_new(prog_path, len(stages)):
            break
        for line in _read_lines(prog_path, len(stages)):
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if "progress" in msg:
                stages.append((msg["progress"], msg.get("desc", "")))
            if msg.get("done"):
                result = msg
            if msg.get("error"):
                result = {"error": msg.get("error")}
        if proc.poll() is not None and result:
            break
        time.sleep(0.25)

    elapsed = time.perf_counter() - t0
    peak_gb = sampler.stop()
    proc.wait(timeout=30)
    for tmp in (req_path, prog_path):
        try: os.unlink(tmp)
        except OSError: pass

    n = int(result.get("total", 0))
    return {
        "mode":        "scan" if scan_mode else ("deep" if deep_grade else "full"),
        "ok":          "error" not in result and n > 0,
        "error":       result.get("error"),
        "photos":      n,
        "wall_s":      round(elapsed, 1),
        "s_per_image": round(elapsed / n, 2) if n else None,
        "peak_rss_gb": round(peak_gb, 2),
        "stages":      stages[-6:],   # last few ticks for the timeline
    }


_prog_cache: dict = {"n": 0}
def _read_lines(path: str, from_idx: int) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return lines[from_idx:]
    except OSError:
        return []

def _prog_has_new(path: str, current: int) -> bool:
    try:
        return len(open(path, "r", encoding="utf-8").read().splitlines()) > current
    except OSError:
        return False


# ── Sequence benchmarks ───────────────────────────────────────────────────────

def _graded_rows() -> list[dict]:
    import lance_store as ls
    rows = ls.query_all(min_score=0.0)
    return [r for r in rows if float(r.get("score", 0)) > 0]


def _run_sequence(rows: list[dict]) -> dict:
    """MOGCO beam sequencing over the graded library, twice, for determinism."""
    from mogco_sequencer import run_mogco_sequence
    out = {}
    times, sigs = [], []
    for i in range(2):
        t0 = time.perf_counter()
        beam = run_mogco_sequence(vibe_vec=None, target=5, min_score=0.45, beam_width=4)
        times.append(round(time.perf_counter() - t0, 2))
        sigs.append(json.dumps(beam.get("paths", [])))
    out["deterministic"] = sigs[0] == sigs[1]
    out["time_s"] = times[0]
    out["slots"] = len(beam.get("paths", []))
    out["global_score"] = beam.get("global_score")
    return out


def _run_story_full(rows: list[dict]) -> dict:
    """Full Creative Direction pipeline, story + competition modes."""
    import numpy as np
    from creative_director import run_creative_direction
    rows_sorted = sorted(rows, key=lambda r: float(r.get("score", 0)), reverse=True)[:60]
    paths      = [r["path"] for r in rows_sorted]
    embeddings = [np.array(r["embedding"], dtype=np.float32) for r in rows_sorted]
    scores     = [float(r.get("score", 0.5)) for r in rows_sorted]
    outdir     = ROOT / "cache" / "bench_story_out"
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for mode in ("story", "competition"):
        t0 = time.perf_counter()
        try:
            r = run_creative_direction(
                strong_paths=paths, embeddings=embeddings, scores=scores,
                anchor_path="", output_dir=str(outdir),
                style_prompt="moody urban evening, available light",
                n_target=5, avoid_paths=[], progress=lambda *a, **k: None,
                mode=mode,
            )
            outputs = [o for o in r.get("outputs", []) if o.get("success")]
            roles = [o.get("params", {}).get("role", "?") for o in outputs]
            res[mode] = {
                "ok": True,
                "time_s": round(time.perf_counter() - t0, 1),
                "frames": len(outputs),
                "roles_filled": len(set(roles)),
                "role_list": sorted(set(roles)),
                "fallback": bool(r.get("director_fallback")),
            }
        except Exception as e:
            res[mode] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    return res


# ── Report ────────────────────────────────────────────────────────────────────

def _write_report(cull: dict | None, seq: dict | None, story: dict | None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / "benchmark_report.md"
    L: list[str] = []
    L.append(f"# FrameGrade benchmark — {datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"Machine: {os.cpu_count()} logical cores · "
             f"{_ram_total():.1f} GB RAM · dataset: {DATASET.name}/")
    L.append("")
    if cull:
        L.append("## Cull")
        L.append(f"- Mode: **{cull['mode']}** · ok: {cull['ok']}"
                 + (f" · error: {cull['error']}" if cull.get("error") else ""))
        L.append(f"- Photos graded: **{cull['photos']}**")
        L.append(f"- Wall time: **{cull['wall_s']} s** "
                 f"({cull['s_per_image']} s/image)")
        L.append(f"- Peak process-tree RSS: **{cull['peak_rss_gb']} GB**")
        L.append("- Last pipeline stages seen:")
        for frac, desc in cull["stages"]:
            L.append(f"  - {int(frac*100)}% — {desc[:90]}")
        L.append("")
    if seq:
        L.append("## Sequence (MOGCO beam)")
        L.append(f"- Time: **{seq['time_s']} s** for {seq['slots']} slots")
        L.append(f"- Deterministic across two runs: **{seq['deterministic']}**")
        L.append(f"- Global score: {seq['global_score']}")
        L.append("")
    if story:
        L.append("## Story / Competition (full pipeline)")
        for mode, r in story.items():
            if r.get("ok"):
                L.append(f"- **{mode}**: {r['time_s']} s · {r['frames']} frames · "
                         f"{r['roles_filled']} distinct roles {r['role_list']}"
                         + (" · FALLBACK (no art direction)" if r["fallback"] else ""))
            else:
                L.append(f"- **{mode}**: FAILED — {r.get('error')}")
        L.append("")
    L.append("---")
    L.append("*Generated by scripts/benchmark.py against the real pipeline.*")
    p.write_text(chr(10).join(L), encoding="utf-8")
    return p


def _ram_total() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true", help="cull in fast scan mode")
    ap.add_argument("--deep", action="store_true", help="cull with Deep Grade (Qwen)")
    ap.add_argument("--skip-cull", action="store_true", help="reuse existing grades")
    ap.add_argument("--full-story", action="store_true",
                    help="also run the full story/competition pipeline (loads LLMs)")
    args = ap.parse_args()

    print(f"[bench] dataset: {DATASET} ({len(list(DATASET.glob('*')))} files)")

    cull = None
    if not args.skip_cull:
        print("[bench] stage 1/3: cull …")
        cull = _run_cull(scan_mode=args.scan, deep_grade=args.deep)
        print(f"[bench]   → {cull['photos']} photos, {cull['wall_s']} s, "
              f"peak {cull['peak_rss_gb']} GB, ok={cull['ok']}")
    else:
        print("[bench] stage 1/3: cull SKIPPED")

    print("[bench] stage 2/3: sequence …")
    rows = _graded_rows()
    print(f"[bench]   → {len(rows)} graded rows in LanceDB")
    seq = _run_sequence(rows) if len(rows) >= 5 else {"error": "need ≥5 graded rows"}
    print(f"[bench]   → {seq}")

    story = None
    if args.full_story:
        print("[bench] stage 3/3: story + competition (loads local LLMs) …")
        story = _run_story_full(rows)
        print(f"[bench]   → {story}")
    else:
        print("[bench] stage 3/3: story SKIPPED (pass --full-story to include)")

    report = _write_report(cull, seq, story)
    print(f"[bench] report written: {report}")


if __name__ == "__main__":
    main()