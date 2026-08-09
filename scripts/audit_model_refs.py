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

# Deliberately NO .md. Prose that mentions a filename does not open it, and
# including markdown made the audit self-defeating: a spec listing a weight as a
# removal candidate thereby marked it "referenced" and kept it alive. Code,
# config and launchers can genuinely name a weight; documentation cannot load one.
SOURCE_SUFFIXES = {".py", ".json", ".txt", ".bat", ".sh", ".ps1", ".spec"}

# Never walk these. models/ is 23 GB and venv/ is huge; rglob'ing either to look
# for source text costs minutes and finds nothing useful. deprecated/ is excluded
# ON PURPOSE - a reference from retired code must NOT keep a weight alive, or
# nothing would ever be reclaimable.
SKIP_DIRS = {"venv", ".venv", "models", ".git", "__pycache__", "node_modules",
             "deprecated", "cache", "output", "dataset_images", "logs",
             "frontend", "_quarantine", "docs"}


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
        # Every ancestor name up to models_dir, not just the immediate parent.
        # HuggingFace caches nest weights as
        #   siglip2/models--timm--X/snapshots/<hash>/open_clip_pytorch_model.bin
        # so the immediate parent is a content hash that appears in no source
        # file. Matching only that parent called a live fallback checkpoint dead.
        names = {p.name}
        for anc in p.parents:
            if anc == models_dir:
                break
            names.add(anc.name)
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


def can_trace_children() -> bool:
    """Can this process observe ANOTHER process's open files?

    On Windows the answer is usually NO without elevation: psutil.open_files()
    returns entries for the calling process and an empty list for a child.
    Measured on this machine - self: 3 entries, child: 0.

    That makes the dynamic pass unreliable exactly where it matters, so callers
    must check this before treating an empty trace as evidence of anything.
    """
    import subprocess
    import sys
    import time

    import psutil
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time,sys; f=open(sys.executable,'rb'); time.sleep(1.2)"])
    try:
        time.sleep(0.5)
        try:
            return len(psutil.Process(proc.pid).open_files()) > 0
        except Exception:
            return False
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait(timeout=5)


def trace_opens(cmd: list[str], models_dir: Path, poll_s: float = 0.2) -> set[Path]:
    """Files under models_dir opened by `cmd` or any of its children.

    UNRELIABLE ON WINDOWS. See can_trace_children(): without elevation
    psutil cannot enumerate a child's handles, so this returns an empty set
    regardless of what the command actually loaded. An empty result is NOT
    evidence that nothing was opened. The static pass carries the audit on
    such platforms.
    """
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

    errors: list = []

    def _poll(proc):
        # Catch Exception, not just psutil.Error. On Windows open_files() can
        # raise OSError/MemoryError under memory pressure, and a bare
        # `except psutil.Error` let that kill the polling thread outright — the
        # trace then returned an empty set, which reads as "this binary opened
        # nothing" and would mark every weight dead. A tracer that can fail
        # silently is worse than no tracer.
        while not stop.is_set():
            try:
                targets = [proc] + proc.children(recursive=True)
            except Exception as err:
                errors.append(repr(err))
                targets = []
            for t in targets:
                try:
                    for f in t.open_files():
                        hit = _under(f.path)
                        if hit:
                            seen.add(hit)
                except Exception as err:
                    errors.append(repr(err))
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
    if errors and not seen:
        # Distinguish "observed nothing" from "could not observe".
        uniq = sorted(set(errors))[:3]
        print(f"[audit] WARNING: the poller hit {len(errors)} errors and saw "
              f"nothing. This is NOT evidence the command opened no models. "
              f"Examples: {uniq}")
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
    ap = argparse.ArgumentParser(
        epilog="Pass the traced command after a bare '--', so paths containing "
               "spaces survive: audit_model_refs.py --report -- python run.py 'a b.json'")
    ap.add_argument("--models", default="models")
    ap.add_argument("--traced-out", default="", help="append traced paths here")
    ap.add_argument("--report", action="store_true")
    # REMAINDER, not a quoted string: splitting a command on whitespace breaks
    # on "C:\Program Files\...", and under Git Bash the shell rewrites Windows
    # paths inside a quoted argument before argparse ever sees them.
    ap.add_argument("trace_cmd", nargs=argparse.REMAINDER,
                    help="command to run while tracing, after '--'")
    args = ap.parse_args()

    cmd = [a for a in args.trace_cmd if a != "--"]
    models_dir = Path(args.models).resolve()
    traced: set[Path] = set()
    if cmd:
        print(f"[audit] tracing: {cmd}")
        traced = trace_opens(cmd, models_dir)
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
