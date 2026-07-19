"""
grade_runner.py — run ONE grade as a clean, disposable subprocess.

Why this exists
---------------
The persistent multiprocessing grade worker (grade_worker.py, spawned via
multiprocessing) proved fragile in the windowed app: it repeatedly died with a
C-level 0xC0000005 at GPU-process boundaries (a GPU child exiting faulted the
multiprocessing-spawn parent), while an ordinary `python` process running the
exact same pipeline completed 100% of the time. So the grade now runs here, in a
plain `subprocess.Popen` child of the server — NOT a multiprocessing spawn child.
It inherits the direct-run stability by construction, and if it ever does crash
it is a separate OS process: the server and window are unaffected, and the
durable catalog checkpoint + Resume recover the work.

Contract
--------
    python grade_runner.py <request_json> <progress_jsonl>

request_json : {folders, preset, force_rescan, scan_mode, deep_grade,
                mogco_target, sample_limit, detect_only, catalog_path, data_dir}
progress_jsonl: this process APPENDS one JSON object per line:
                {"progress": float, "desc": str}         progress ticks
                {"done": true, "total": int, ...}        final gallery result
                {"error": str, "traceback": str}         failure

It reuses grade_worker.grade_worker_main VERBATIM — the only difference from the
old multiprocessing path is that the "queue" is a line-appended file instead of a
multiprocessing.Queue, so the catalog write, grading.lock lifecycle, story-skip,
and error handling are all identical and battle-tested.
"""
import os, sys, json, traceback as _tb
from pathlib import Path

# Suppress cmd-window flashes (same as grade_worker.py / server.py).
try:
    import suppress_console  # noqa: F401
except Exception:
    pass


class _FileQueue:
    """Queue-compatible shim: .put(msg) appends one JSON line to the progress
    file and flushes, so the server can tail it for SSE. Matches the subset of
    multiprocessing.Queue that grade_worker_main uses (just .put)."""

    def __init__(self, path: str) -> None:
        self._path = path

    def put(self, msg: dict) -> None:
        try:
            line = json.dumps(
                msg,
                default=lambda o: o.item() if hasattr(o, "item") else str(o),
            )
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass   # never let progress reporting crash the grade


def main() -> None:
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    try:
        # utf-8, NOT the Windows cp1252 default: run_v2 prints '→' etc.; a cp1252
        # stdout raises UnicodeEncodeError mid-grade and aborts the run.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

    req_path, prog_path = sys.argv[1], sys.argv[2]
    q = _FileQueue(prog_path)
    try:
        req = json.loads(Path(req_path).read_text(encoding="utf-8"))
        from grade_worker import grade_worker_main
        grade_worker_main(
            q,
            req["folders"],
            req["preset"],
            req["force_rescan"],
            req["scan_mode"],
            req["catalog_path"],
            req["data_dir"],
            mogco_target=req.get("mogco_target", 0),
            sample_limit=req.get("sample_limit", 0),
            detect_only=req.get("detect_only", False),
            deep_grade=req.get("deep_grade", False),
        )
    except BaseException as exc:
        # Surface any top-level failure to the server as a normal error line so the
        # SSE stream reports it instead of the client hanging.
        try:
            q.put({"error": str(exc) or type(exc).__name__,
                   "traceback": _tb.format_exc()})
        except Exception:
            pass
    # Clean process exit: no torch CUDA atexit games needed here because this
    # process does no direct GPU work — SigLIP and IQA run in their OWN
    # subprocesses (encode_worker / iqa_worker) and have already exited.


if __name__ == "__main__":
    main()
