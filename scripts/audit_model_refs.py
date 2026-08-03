"""
Which files under models/ does FrameGrade actually load?

A static grep cannot answer this: paths get built at runtime, so a weight can be
loaded without its name appearing in any source file. Guessing wrong means
quarantining something the app needs, which is why this uses two passes and
treats only the INTERSECTION of "unreferenced" and "never opened" as dead.

  static   scan source text for each file's name and its parent dir's name
  dynamic  record every file under models/ actually opened during a real run

sys.addaudithook alone is not enough - ONNX Runtime and torch open weights from
C++, bypassing the Python audit hook - so the dynamic pass polls open handles
across the whole process tree instead. Model loads take seconds, so a 200 ms
poll catches them reliably.

A git-tracked file is never a candidate, whatever the scans say. models/ is
mostly gitignored weights but a handful of small configs are tracked, and a
loader may find one by convention without naming it in source.

Usage:
    python scripts/audit_model_refs.py --report
    python scripts/audit_model_refs.py --trace "venv/Scripts/python.exe grade_runner.py req.json out.jsonl" --report
"""
from __future__ import annotations

import argparse
import subprocess
import threading
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".json", ".txt", ".bat", ".sh", ".ps1", ".spec", ".md"}

# Never walk these. models/ is 23 GB and venv/ is huge; rglob'ing either to look
# for source text costs minutes and finds nothing useful. deprecated/ is excluded
# ON PURPOSE - a reference from retired code must NOT keep a weight alive, or
# nothing would ever be reclaimable.
SKIP_DIRS = {"venv", ".venv", "models", ".git", "__pycache__", "node_modules",
             "deprecated", "cache", "output", "dataset_images", "logs",
             "frontend", "_quarantine"}


def _iter_model_files(models_dir: Path):
    for p in models_dir.rglob("*"):
        if p.is_file() and "_quarantine" not in p.parts:
            yield p


def static_refs(models_dir: Path, search_roots: list[Path]) -> set[Path]:
    """Model files whose name (or parent dir name) appears in source text.

    Matches on the parent directory too: a checkpoint dir is referenced once by
    directory name, never shard by shard, so matching only filenames would call
    every shard of a live checkpoint dead.
    """
    models_dir = Path(models_dir).resolve()
    blobs: list[str] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            files = [root]
        else:
            files = [f for f in root.rglob("*")
                     if f.is_file() and f.suffix.lower() in SOURCE_SUFFIXES
                     and not SKIP_DIRS.intersection(f.parts)]
        for f in files:
            try:
                blobs.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    text = "\n".join(blobs)

    refs: set[Path] = set()
    for p in _iter_model_files(models_dir):
        names = {p.name}
        if p.parent != models_dir:
            names.add(p.parent.name)
        if any(n in text for n in names):
            refs.add(p)
    return refs


def git_tracked(models_dir: Path) -> set[Path]:
    """Files under models_dir that git tracks. Never quarantine these."""
    models_dir = Path(models_dir).resolve()
    try:
        out = subprocess.run(["git", "ls-files", "-z", str(models_dir)],
                             cwd=models_dir.parent, capture_output=True,
                             text=True, timeout=60)
        if out.returncode != 0:
            return set()
        return {(models_dir.parent / rel).resolve()
                for rel in out.stdout.split("\0") if rel.strip()}
    except Exception:
        return set()


def trace_opens(cmd: list[str], models_dir: Path, poll_s: float = 0.2) -> set[Path]:
    """Files under models_dir opened by `cmd` or any of its children."""
    import psutil

    models_dir = Path(models_dir).resolve()
    seen: set[Path] = set()
    stop = threading.Event()

    def _under(path_str: str):
        try:
            p = Path(path_str).resolve()
        except Exception:
            return None
        return p if models_dir in p.parents else None

    def _poll(proc):
        while not stop.is_set():
            try:
                targets = [proc] + proc.children(recursive=True)
            except psutil.Error:
                targets = []
            for t in targets:
                try:
                    for f in t.open_files():
                        hit = _under(f.path)
                        if hit:
                            seen.add(hit)
                except psutil.Error:
                    continue
            stop.wait(poll_s)

    popen = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    watcher = None
    try:
        watcher = threading.Thread(target=_poll, args=(psutil.Process(popen.pid),),
                                   daemon=True)
        watcher.start()
        popen.wait()
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=5)
    return seen


def audit(models_dir: Path, search_roots: list[Path] | None = None,
          traced: set[Path] | None = None,
          tracked: set[Path] | None = None) -> dict:
    """Intersect the passes. A candidate is untouched by ALL of them."""
    models_dir = Path(models_dir).resolve()
    if search_roots is None:
        root = models_dir.parent
        search_roots = [root / "src", root / "scripts", root]
    if tracked is None:
        tracked = git_tracked(models_dir)

    referenced = (static_refs(models_dir, search_roots)
                  | set(traced or set())
                  | set(tracked))
    candidates = {p for p in _iter_model_files(models_dir) if p not in referenced}
    return {
        "referenced": referenced,
        "candidates": candidates,
        "bytes_reclaimable": sum(p.stat().st_size for p in candidates),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--trace", default="", help="command to run while tracing")
    ap.add_argument("--traced-out", default="", help="append traced paths here")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models).resolve()
    traced: set[Path] = set()
    if args.trace:
        print(f"[audit] tracing: {args.trace}")
        traced = trace_opens(args.trace.split(), models_dir)
        print(f"[audit] {len(traced)} model files opened at runtime")
        if not traced:
            print("[audit] WARNING: nothing was traced. Either the command did "
                  "no model I/O, or the poller is broken - do NOT quarantine "
                  "on this result.")
        if args.traced_out:
            with open(args.traced_out, "a", encoding="utf-8") as fh:
                for p in sorted(traced):
                    fh.write(str(p) + "\n")

    result = audit(models_dir, traced=traced)
    if args.report:
        print(f"\n{'CANDIDATE (unreferenced, never opened, untracked)':<58} SIZE")
        by_size = sorted(result["candidates"], key=lambda p: -p.stat().st_size)
        for p in by_size[:40]:
            print(f"  {str(p.relative_to(models_dir)):<56} "
                  f"{p.stat().st_size / 1e9:>6.2f} GB")
        print(f"\n  reclaimable: {result['bytes_reclaimable'] / 1e9:.2f} GB "
              f"across {len(result['candidates'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
