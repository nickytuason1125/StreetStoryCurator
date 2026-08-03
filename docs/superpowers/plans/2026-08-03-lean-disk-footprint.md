# Lean Disk Footprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut FrameGrade's on-disk footprint from ~24 GB to ~10–17 GB and stop the LanceDB store growing without bound on every re-cull.

**Architecture:** Three independent changes. (1) The existing `compact_after_write()` gains version reaping via `Table.optimize(cleanup_older_than=…)`, with the retention window declared in `run_profile.SETTINGS`. (2) A dead, syntactically-invalid module is deleted. (3) A two-pass audit — static reference scan plus a runtime open-file trace — identifies unreferenced model weights, which are quarantined rather than deleted.

**Tech Stack:** Python 3.12, lancedb 0.30.2, pyarrow, pytest, psutil.

## Global Constraints

- **Never call `torch.cuda.*` in a parent process** — not even `is_available()`. It initialises CUDA and reintroduces the 0xC0000005 parent-fault. Probe in a subprocess.
- **Tier state and settings are declared in exactly one place:** `src/run_profile.py`. Never re-derive a tier value or hardcode a setting default in another module.
- **All env vars must be declared in `run_profile.SETTINGS`.** `setting()` raises on undeclared names.
- **Never print or dump a container inside an `except` block** — handled errors previously OOM'd this way.
- **Cull-path failures must degrade, not raise.** Housekeeping must never fail a cull whose grades are already committed.
- **Verification bar for any pipeline change:** 169 tests pass, AND a real Pro-tier LX3 cull yields `Strong=62 Mid=324 Weak=128` with **zero** per-photo score/grade drift.
- **Tier pinning for verification runs:** `SIGLIP_TIER=high SIGLIP_MIN_FREE_RAM_GB=1.2 SIGLIP_HARD_MIN_RAM_GB=1.0`. Below 3 GB free the ladder silently drops to Balanced, which changes every score.
- **Leave `cache/encoder_source.txt` at its live value** during verification so embeddings are reused from LanceDB.
- Python interpreter is `venv/Scripts/python.exe`. Working directory is `street-story-curator/`.

---

### Task 1: LanceDB version retention

**Files:**
- Modify: `src/run_profile.py` (add one entry to `SETTINGS`, near `FRAMEGRADE_LANCE_CHUNK` at line 111)
- Modify: `src/lance_store.py:555-569` (`compact_after_write`)
- Test: `tests/test_lance_retention.py` (create)

**Interfaces:**
- Consumes: `run_profile.setting(name) -> Any`; `lance_store._open_table() -> lancedb.table.LanceTable`; `lance_store._lock` (a `threading.Lock`).
- Produces: `lance_store.compact_after_write() -> None` — unchanged signature, now also reaps versions. Nothing downstream needs to change.

**Background the implementer needs:**

`cache/lance.db/photos.lance` is 859 MB holding ~10 MB of vectors: 350 data fragments and 409 versions for 1,745 rows. `compact_files()` merges fragments but does **not** delete old versions, so every re-cull leaves its history behind.

In lancedb 0.30.2 the table API is:

```
optimize(*, cleanup_older_than: Optional[timedelta] = None,
         delete_unverified: bool = False, retrain: bool = False)
```

`optimize()` does compaction **and** version reaping in one call, replacing `compact_files()`. LanceDB always retains the current version regardless of the window, so a zero-length window is safe.

`delete_unverified=False` is required, not optional: all three tier tables (`photos.lance`, `photos_mid.lance`, `photos_low.lance`) live in one database, and a reader on another tier may hold references to fragments this call cannot verify as orphaned.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lance_retention.py`:

```python
"""
LanceDB version retention.

photos.lance reached 859 MB holding ~10 MB of vectors — 409 versions left behind
because compact_files() merges fragments but never deletes old versions. These
tests pin the fix: history is reaped, current data is untouched, and a cleanup
failure can never fail a cull whose grades are already committed.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_lance_retention.py -v
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _make_table(tmp_path, n_versions: int = 5):
    """A real LanceDB table with several versions of history."""
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    db = lancedb.connect(str(tmp_path / "t.db"))
    schema = pa.schema([
        pa.field("path", pa.string()),
        pa.field("embedding", pa.list_(pa.float32(), 4)),
    ])
    tbl = db.create_table("photos", schema=schema)
    for v in range(n_versions):
        tbl.merge_insert("path").when_matched_update_all().when_not_matched_insert_all().execute(
            pa.table({"path": [f"img{i}.jpg" for i in range(3)],
                      "embedding": [[float(v), 0.0, 0.0, 0.0] for _ in range(3)]})
        )
    return tbl


def test_cleanup_reaps_history_but_keeps_rows(tmp_path):
    tbl = _make_table(tmp_path, n_versions=5)
    before = len(tbl.list_versions())
    rows_before = tbl.count_rows()
    assert before > 1, "fixture must produce history to reap"

    tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=False)

    assert len(tbl.list_versions()) < before, "old versions must be reaped"
    assert tbl.count_rows() == rows_before, "cleanup must not lose rows"


def test_current_version_always_survives(tmp_path):
    """A zero-length retention window must not leave an unreadable table."""
    tbl = _make_table(tmp_path, n_versions=3)
    tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=False)

    assert len(tbl.list_versions()) >= 1
    assert tbl.count_rows() == 3
    assert tbl.search([0.0, 0.0, 0.0, 0.0]).limit(1).to_list(), "table must stay queryable"


def test_retention_setting_is_declared():
    """Undeclared settings raise, so this also pins the spelling."""
    import run_profile
    days = run_profile.setting("FRAMEGRADE_LANCE_RETENTION_DAYS")
    assert isinstance(days, int)
    assert days == 7, "default retention window is 7 days"


def test_cleanup_failure_never_raises(monkeypatch):
    """Housekeeping runs AFTER grades are committed. It must degrade, not raise."""
    import lance_store

    class Boom:
        def optimize(self, **kw):
            raise RuntimeError("simulated lance failure")

    monkeypatch.setattr(lance_store, "_open_table", lambda: Boom())
    lance_store.compact_after_write()      # must return normally
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv\Scripts\python.exe -m pytest tests/test_lance_retention.py -v`

Expected: `test_retention_setting_is_declared` FAILS with `KeyError: undeclared setting 'FRAMEGRADE_LANCE_RETENTION_DAYS'`. The two `optimize` tests should already pass (they exercise lancedb directly, proving the API behaves as assumed). `test_cleanup_failure_never_raises` passes against the current implementation and guards the rewrite.

- [ ] **Step 3: Declare the setting**

In `src/run_profile.py`, immediately after the `FRAMEGRADE_LANCE_CHUNK` line (line 111):

```python
    "FRAMEGRADE_LANCE_RETENTION_DAYS": Setting(
        int, 7, "keep LanceDB versions this many days; history was unbounded"),
```

- [ ] **Step 4: Rewrite `compact_after_write`**

Replace `src/lance_store.py:555-569` entirely:

```python
def compact_after_write() -> None:
    """
    Compact fragments AND reap old versions after a bulk write.

    LanceDB appends each upsert as a new fragment and keeps every prior version.
    compact_files() merged the fragments but left the history, so photos.lance
    reached 859 MB holding ~10 MB of vectors across 409 versions — growth
    unbounded in the number of culls, not the number of photos.

    optimize() does both in one call. Two deliberate choices:

      * delete_unverified=False — all three tier tables share one database, and
        a reader on another tier may hold references to fragments this call
        cannot prove are orphaned.
      * A retention WINDOW rather than "keep only current", so a cull that
        writes bad grades can still be rolled back.

    Safe to skip: this runs after grades are durably committed, so a failure
    here must never fail the cull.
    """
    from datetime import timedelta
    try:
        import run_profile as _rp
        days = max(0, int(_rp.setting("FRAMEGRADE_LANCE_RETENTION_DAYS")))
    except Exception:
        days = 7
    try:
        tbl = _open_table()
        with _lock:
            tbl.optimize(cleanup_older_than=timedelta(days=days),
                         delete_unverified=False)
        print(f"[lance] Compaction + version cleanup done (retention {days}d)")
    except Exception as e:
        print(f"[lance] Compaction skipped ({e})")
```

- [ ] **Step 5: Run the new tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_lance_retention.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the full suite**

Run: `venv\Scripts\python.exe -m pytest tests -q`
Expected: `173 passed` (169 existing + 4 new). Takes ~11 minutes.

- [ ] **Step 7: Reclaim the existing 859 MB**

The new code only bounds *future* growth; the 409 accumulated versions need one manual pass. Back up first — this deletes data.

```bash
cp -r cache/lance.db "$SCRATCH/lance.db.bak"
du -sh cache/lance.db
venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import lance_store as ls
from datetime import timedelta
tbl = ls._open_table()
print('versions before:', len(tbl.list_versions()), 'rows:', tbl.count_rows())
tbl.optimize(cleanup_older_than=timedelta(0), delete_unverified=False)
print('versions after :', len(tbl.list_versions()), 'rows:', tbl.count_rows())
"
du -sh cache/lance.db
```

Expected: row count identical before and after; `photos.lance` drops from ~859 MB to tens of MB.

**Note:** `_open_table()` opens only the table for the *current* tier. Repeat with `SIGLIP_TIER=mid` and `SIGLIP_TIER=low` to reclaim `photos_mid.lance` (13 MB) and `photos_low.lance` (24 MB). Those are small; the high tier is the whole win.

- [ ] **Step 8: Verify the store is intact**

```bash
venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import lance_store as ls
print('rows:', ls.count())
"
venv/Scripts/python.exe -c "
import json,pathlib
c=json.loads(pathlib.Path('cache/catalog.json').read_text(encoding='utf-8'))
print('catalog photos:', len(c) if isinstance(c,list) else len(c.get('photos',c)))
"
```

Expected: rows unchanged from Step 7; catalog still reports 1745 photos.

- [ ] **Step 9: Commit**

```bash
git add src/run_profile.py src/lance_store.py tests/test_lance_retention.py
git commit -m "fix: reap LanceDB versions after write

compact_files() merged fragments but never deleted old versions, so
photos.lance grew to 859 MB holding ~10 MB of vectors across 409
versions - unbounded in the number of culls. optimize() does both.

delete_unverified=False because all three tier tables share one
database and a reader on another tier may hold references."
```

---

### Task 2: Delete the dead `lance_migration` module

**Files:**
- Delete: `src/lance_migration.py`
- Test: `tests/test_lance_retention.py` (append one test)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. This module has no consumers — that is the point.

**Background the implementer needs:**

The file is dead three times over and should be verified as such before deletion, not taken on faith:

1. **Line 1 is the literal text `but ar`**, before the module docstring. `ast.parse` fails with `SyntaxError: invalid syntax`. The module cannot be imported.
2. **Nothing imports it.** Ripgrep across every `.py` finds no reference.
3. **`cache/lancedb_v2/` does not exist**, which proves it never ran — the module calls `DB_DIR.mkdir(parents=True, exist_ok=True)` at import time.

It is worse than dead. It defines a *second* LanceDB (`cache/lancedb_v2`, table `photos_v2`) whose schema disagrees with the live store — it has a `confidence` column the live schema lacks, and stores `breakdown` as a JSON string where the live store uses a struct. Anyone who "fixed" line 1 without reading on would create a parallel vector store beside the real one, at a CWD-relative path, as an import side effect. It also migrates from SQLite and FAISS, neither of which the project uses.

- [ ] **Step 1: Verify all three claims before deleting**

```bash
venv/Scripts/python.exe -c "import ast; ast.parse(open('src/lance_migration.py',encoding='utf-8').read())"
```
Expected: `SyntaxError: invalid syntax` pointing at line 1.

```bash
grep -rn --include=*.py lance_migration src/ tests/ scripts/ *.py
```
Expected: matches only inside `src/lance_migration.py` itself, if any.

```bash
ls -d cache/lancedb_v2
```
Expected: `No such file or directory`.

If any expectation does not hold, STOP — the module is not dead and this task needs rethinking.

- [ ] **Step 2: Write the regression test**

Append to `tests/test_lance_retention.py`:

```python
def test_no_second_lancedb_is_ever_created():
    """cache/lancedb_v2 was a parallel store with a conflicting schema.

    The module that created it (src/lance_migration.py) has been deleted. This
    guards against it coming back: a second store would silently split grades
    across two databases.
    """
    assert not (_ROOT / "src" / "lance_migration.py").exists(), \
        "lance_migration.py is dead code that creates a conflicting second store"
    assert not (_ROOT / "cache" / "lancedb_v2").exists(), \
        "a second LanceDB appeared - something is importing a migration module"
```

- [ ] **Step 3: Run it to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_lance_retention.py::test_no_second_lancedb_is_ever_created -v`
Expected: FAIL — `lance_migration.py is dead code…`, because the file still exists.

- [ ] **Step 4: Delete the file**

```bash
git rm src/lance_migration.py
```

- [ ] **Step 5: Run the test and the full suite**

Run: `venv\Scripts\python.exe -m pytest tests/test_lance_retention.py -v`
Expected: 5 passed.

Run: `venv\Scripts\python.exe -m pytest tests -q`
Expected: `174 passed`. No collection errors — confirms nothing imported the module.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore: delete dead lance_migration module

Line 1 was the literal text 'but ar' - the file raised SyntaxError and
could never be imported. Nothing referenced it and its cache/lancedb_v2
directory never existed, confirming it never ran.

It defined a second LanceDB with a schema conflicting with the live
store, created at import time on a CWD-relative path. Fixing line 1
without reading further would have split grades across two databases."
```

---

### Task 3: Build the model-weight audit

**Files:**
- Create: `scripts/audit_model_refs.py`
- Test: `tests/test_model_audit.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `static_refs(models_dir: Path, search_roots: list[Path]) -> set[Path]` — model files named in source text.
  - `trace_opens(cmd: list[str], models_dir: Path, poll_s: float = 0.2) -> set[Path]` — model files actually opened while `cmd` runs.
  - `audit(models_dir: Path) -> dict` — `{"referenced": set[Path], "candidates": set[Path], "bytes_reclaimable": int}`.

**Background the implementer needs:**

A static grep is not sufficient, and the project's own guidance names model weights as the case where caution beats cleverness. Paths get built at runtime, so a file can be loaded without its name appearing in any source file.

Two passes, and a file must be untouched by **both** to become a candidate:

- **Static** — scan source text for each model file's name and its parent directory's name.
- **Dynamic** — record every file under `models/` actually opened while a real cull runs.

For the dynamic pass, `sys.addaudithook` catches Python-level `open()` but misses native loads: ONNX Runtime and torch open weights from C++, bypassing the Python audit hook. Poll `psutil.Process.open_files()` across the whole process tree instead. Model loads take seconds, so a 200 ms poll reliably catches them. Use both — the audit hook is nearly free and catches short-lived Python reads the poller might miss between samples.

**This task builds and tests the tool only.** It moves nothing. Task 4 uses it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_audit.py`:

```python
"""
The model-weight audit must not produce false positives.

A false positive here means quarantining a weight the app actually loads, so
these tests pin the two properties that matter: a file opened at runtime is
never a candidate, and a file named in source is never a candidate.

Run:  venv\\Scripts\\python.exe -m pytest tests/test_model_audit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "src"))

import audit_model_refs as amr          # noqa: E402


def test_static_scan_finds_a_named_weight(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    (models / "used.onnx").write_bytes(b"x" * 16)
    (models / "unused.onnx").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "loader.py").write_text("PATH = 'models/used.onnx'\n", encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert (models / "used.onnx") in refs
    assert (models / "unused.onnx") not in refs


def test_static_scan_matches_by_parent_dir(tmp_path):
    """Checkpoint dirs are referenced by directory name, not by each shard."""
    models = tmp_path / "models"
    ckpt = models / "siglip2_hf_fp16"; ckpt.mkdir(parents=True)
    (ckpt / "model-00001.safetensors").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "enc.py").write_text("DIR = 'models/siglip2_hf_fp16'\n", encoding="utf-8")

    refs = amr.static_refs(models, [src])
    assert (ckpt / "model-00001.safetensors") in refs


def test_trace_catches_a_runtime_open(tmp_path):
    """The whole point: a file opened at runtime but named nowhere in source."""
    models = tmp_path / "models"; models.mkdir()
    target = models / "runtime_only.bin"
    target.write_bytes(b"x" * (4 * 1024 * 1024))

    reader = tmp_path / "reader.py"
    reader.write_text(
        "import time,sys\n"
        "f=open(sys.argv[1],'rb')\n"
        "f.read()\n"
        "time.sleep(1.5)\n"
        "f.close()\n",
        encoding="utf-8")

    opened = amr.trace_opens([sys.executable, str(reader), str(target)],
                             models, poll_s=0.1)
    assert target in opened


def test_candidates_exclude_everything_referenced(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    (models / "keep.onnx").write_bytes(b"x" * 16)
    (models / "drop.onnx").write_bytes(b"x" * 16)
    src = tmp_path / "src"; src.mkdir()
    (src / "a.py").write_text("'models/keep.onnx'\n", encoding="utf-8")

    result = amr.audit(models, search_roots=[src], traced=set())
    assert (models / "drop.onnx") in result["candidates"]
    assert (models / "keep.onnx") not in result["candidates"]
    assert result["bytes_reclaimable"] == 16
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_model_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'audit_model_refs'`.

- [ ] **Step 3: Implement the audit tool**

Create `scripts/audit_model_refs.py`:

```python
"""
Which files under models/ does FrameGrade actually load?

A static grep cannot answer this: paths get built at runtime, so a weight can
be loaded without its name appearing in any source file. Guessing wrong means
quarantining something the app needs, which is why this uses two passes and
treats only the intersection of "unreferenced" and "never opened" as dead.

  static   scan source text for each file's name and its parent dir's name
  dynamic  record every file under models/ actually opened during a real run

sys.addaudithook alone is not enough - ONNX Runtime and torch open weights from
C++, bypassing the Python audit hook - so the dynamic pass polls open handles
across the whole process tree as well.

Usage:
    python scripts/audit_model_refs.py --report
    python scripts/audit_model_refs.py --trace "venv/Scripts/python.exe grade_runner.py req.json out.jsonl"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".json", ".txt", ".bat", ".sh", ".ps1", ".spec"}

# Never walk these. `models/` is 23 GB and `venv/` is huge; rglob'ing either to
# look for source text costs minutes and finds nothing useful. `deprecated/` is
# excluded on purpose - a reference from retired code must NOT keep a weight
# alive, or nothing would ever be reclaimable.
SKIP_DIRS = {"venv", ".venv", "models", ".git", "__pycache__", "node_modules",
             "deprecated", "cache", "output", "dataset_images"}


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
    blobs: list[str] = []
    for root in search_roots:
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


def trace_opens(cmd: list[str], models_dir: Path, poll_s: float = 0.2) -> set[Path]:
    """Files under models_dir opened by `cmd` or any of its children."""
    import psutil

    models_dir = models_dir.resolve()
    seen: set[Path] = set()
    stop = threading.Event()

    def _under(path_str: str):
        try:
            p = Path(path_str).resolve()
        except Exception:
            return None
        return p if models_dir in p.parents else None

    def _poll(proc: psutil.Process):
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
    try:
        watcher = threading.Thread(target=_poll, args=(psutil.Process(popen.pid),),
                                   daemon=True)
        watcher.start()
        popen.wait()
    finally:
        stop.set()
        watcher.join(timeout=5)
    return seen


def audit(models_dir: Path, search_roots: list[Path] | None = None,
          traced: set[Path] | None = None) -> dict:
    """Intersect the two passes. Candidates are untouched by BOTH."""
    models_dir = Path(models_dir).resolve()
    if search_roots is None:
        root = models_dir.parent
        search_roots = [root / "src", root / "scripts", root]
    referenced = static_refs(models_dir, search_roots) | set(traced or set())
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
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    models_dir = Path(args.models).resolve()
    traced: set[Path] = set()
    if args.trace:
        print(f"[audit] tracing: {args.trace}")
        traced = trace_opens(args.trace.split(), models_dir)
        print(f"[audit] {len(traced)} model files opened at runtime")

    result = audit(models_dir, traced=traced)
    if args.report:
        print(f"\n{'CANDIDATE (unreferenced, never opened)':<58} SIZE")
        by_size = sorted(result["candidates"], key=lambda p: -p.stat().st_size)
        for p in by_size[:40]:
            print(f"  {str(p.relative_to(models_dir)):<56} "
                  f"{p.stat().st_size/1e9:>6.2f} GB")
        print(f"\n  reclaimable: {result['bytes_reclaimable']/1e9:.2f} GB "
              f"across {len(result['candidates'])} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_model_audit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_model_refs.py tests/test_model_audit.py
git commit -m "feat: add two-pass model weight audit

Static grep alone cannot prove a weight unused - paths are built at
runtime. Adds a dynamic pass that polls open handles across the process
tree, because ONNX Runtime and torch open weights from C++ and bypass
sys.addaudithook. Only the intersection of both passes is a candidate."
```

---

### Task 4: Trace, quarantine, verify

**Files:**
- Create: `models/_quarantine/` (populated by this task)
- Modify: none

**Interfaces:**
- Consumes: `audit_model_refs.audit()` and `.trace_opens()` from Task 3.
- Produces: a populated `models/_quarantine/` and a written manifest. No code changes.

**Background the implementer needs:**

The audit can only prove a weight unused *for the paths it exercises*. A mode never triggered will not appear in the trace. That is precisely why this quarantines rather than deletes — and why the trace must cover every weight-loading mode, not just the cull.

`models/siglip2` (7.1 GB) is the **source** for the 3.5 GB ONNX loaded at runtime. It may well show up as a candidate. It is excluded from this task by name and left for the user to decide, because removing it means re-downloading to re-export ONNX.

- [ ] **Step 1: Trace a real Pro-tier cull**

```bash
SCRATCH="C:/Users/NICKYT~1/AppData/Local/Temp/claude/C--Users-Nicky-Tuason-Desktop-StreetPhotoEditor/271c1669-87ad-470a-a529-4f53410d5e47/scratchpad"
SIGLIP_TIER=high SIGLIP_MIN_FREE_RAM_GB=1.2 SIGLIP_HARD_MIN_RAM_GB=1.0 \
venv/Scripts/python.exe scripts/audit_model_refs.py \
  --trace "venv/Scripts/python.exe grade_runner.py $SCRATCH/grade_req_lx3.json $SCRATCH/audit_prog.jsonl" \
  --report | tee "$SCRATCH/audit_cull.txt"
```

Expected: a non-empty list of opened model files, and a candidate list. If **zero** files were traced as opened, the poller is broken — STOP and fix it, because every weight would look dead.

- [ ] **Step 2: Trace the other weight-loading modes**

The cull alone does not load the annotation, story, or competition weights. Each must be traced too, and the results unioned — a mode left untraced makes its weights look dead.

First find how each is invoked:

```bash
grep -n "story\|competition\|annotat" server.py | grep -i "def \|@app\|post\|get" | head -30
```

Then trace each mode the same way as Step 1, appending to one file:

```bash
: > "$SCRATCH/traced_paths.txt"
# Step 1's cull trace, plus one line per mode:
venv/Scripts/python.exe scripts/audit_model_refs.py \
  --trace "<mode invocation>" --report \
  | tee -a "$SCRATCH/audit_modes.txt"
```

`audit_model_refs.py` prints the candidate list but does not itself write `traced_paths.txt`. Collect the traced set per mode by calling `trace_opens()` directly and appending each absolute path on its own line:

```bash
venv/Scripts/python.exe - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import audit_model_refs as amr
cmd = sys.argv[1:] or ["venv/Scripts/python.exe", "-c", "pass"]
hits = amr.trace_opens(cmd, Path("models").resolve())
with open(os.environ["SCRATCH"] + "/traced_paths.txt", "a", encoding="utf-8") as fh:
    for p in sorted(hits):
        fh.write(str(p) + "\n")
print(f"traced {len(hits)} model files")
PY
```

If a mode's invocation is not obvious from `server.py`, ask the user rather than guessing — a mode you skip becomes a false positive that quarantines a live weight.

- [ ] **Step 3: Write the manifest and the move list**

Two files, because the human-readable record and the machine-readable input to Step 4 are different things:

- `$SCRATCH/quarantine_manifest.txt` — one line per candidate: path, size, and why it is a candidate. This is what the user reviews.
- `$SCRATCH/quarantine_rel_paths.txt` — one path per line, **relative to `models/`**, nothing else. This is what Step 4 reads.

```bash
venv/Scripts/python.exe - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import audit_model_refs as amr

S = Path(os.environ["SCRATCH"])
models = Path("models").resolve()
traced = {Path(l.strip()) for l in (S / "traced_paths.txt").read_text().splitlines() if l.strip()}
r = amr.audit(models, traced=traced)

cands = sorted(r["candidates"], key=lambda p: -p.stat().st_size)
with (S / "quarantine_manifest.txt").open("w", encoding="utf-8") as fh:
    for p in cands:
        fh.write(f"{p.relative_to(models)}\t{p.stat().st_size/1e9:.2f} GB\t"
                 f"unreferenced in source, never opened during trace\n")
with (S / "quarantine_rel_paths.txt").open("w", encoding="utf-8") as fh:
    for p in cands:
        fh.write(f"{p.relative_to(models).as_posix()}\n")
print(f"{len(cands)} candidates, {r['bytes_reclaimable']/1e9:.2f} GB")
PY
```

This depends on Step 2 having written the union of all traced paths to `$SCRATCH/traced_paths.txt`, one absolute path per line.

- [ ] **Step 4: Move candidates to quarantine**

Preserve relative paths so restoring is a single `mv` back. Exclude `models/siglip2`.

```bash
mkdir -p models/_quarantine
while read -r rel; do
  case "$rel" in siglip2/*|siglip2) continue;; esac
  mkdir -p "models/_quarantine/$(dirname "$rel")"
  mv "models/$rel" "models/_quarantine/$rel"
done < "$SCRATCH/quarantine_rel_paths.txt"
du -sh models models/_quarantine
```

- [ ] **Step 5: Prove nothing broke**

Run: `venv\Scripts\python.exe -m pytest tests -q`
Expected: `178 passed` (169 existing + 4 retention + 1 second-store guard + 4 audit).

Then a real Pro-tier cull against the quarantined tree:

```bash
SIGLIP_TIER=high SIGLIP_MIN_FREE_RAM_GB=1.2 SIGLIP_HARD_MIN_RAM_GB=1.0 \
PYTHONIOENCODING=utf-8 venv/Scripts/python.exe grade_runner.py \
  "$SCRATCH/grade_req_lx3.json" "$SCRATCH/quarantine_check.jsonl" \
  > "$SCRATCH/quarantine_check.log" 2>&1
grep -aE "SUMMARY|Encoder:" "$SCRATCH/quarantine_check.log"
```

Expected: `dim=1536` and `Strong=62  Mid=324  Weak=128`.

- [ ] **Step 6: Diff per-photo scores**

Bucket counts are not sufficient — a Balanced-tier run reproduces `62/324/128` exactly while 489 of 514 individual scores differ. Compare against a Pro baseline:

```bash
venv/Scripts/python.exe - <<'PY'
import re
from pathlib import Path
def parse(p):
    out={}
    for l in Path(p).read_text(encoding="utf-8",errors="replace").splitlines():
        if l.startswith("[v2]   "):
            m=re.match(r"^(.*?):\s*([0-9]*\.?[0-9]+)\s*\u2192\s*(.+)$", l[7:])
            if m: out[m.group(1).strip()]=(float(m.group(2)), m.group(3).strip())
    return out
import os
S=os.environ["SCRATCH"]
a=parse(f"{S}/seam_check.log"); b=parse(f"{S}/quarantine_check.log")
common=sorted(set(a)&set(b))
sd=[k for k in common if a[k][0]!=b[k][0]]
print(f"compared {len(common)}  score_diffs={len(sd)}  grade_diffs="
      f"{len([k for k in common if a[k][1]!=b[k][1]])}")
PY
```

Expected: `score_diffs=0  grade_diffs=0`. Any non-zero means a quarantined weight was in use — restore from `models/_quarantine/` and re-audit.

- [ ] **Step 7: Report to the user, delete nothing**

Present the manifest, the reclaimed total, and the verification evidence. **The user performs the deletion.** Also present the `models/siglip2` question separately with its trade-off: 7.1 GB back, at the cost of a re-download if ONNX ever needs re-exporting.

- [ ] **Step 8: Commit the audit artifacts**

`models/` is not tracked, so there is nothing to commit for the move itself.

```bash
git add docs/superpowers/plans/2026-08-03-lean-disk-footprint.md
git commit -m "docs: record model quarantine manifest and verification"
```

---

## Verification Summary

| Check | Expected |
|---|---|
| `pytest tests -q` | 178 passed (169 + 4 retention + 1 second-store guard + 4 audit) |
| `photos.lance` size | 859 MB → tens of MB |
| LanceDB row count | unchanged (1,745) |
| `cache/catalog.json` | still 1,745 photos |
| Pro-tier LX3 cull | `Strong=62 Mid=324 Weak=128`, `dim=1536` |
| Per-photo drift | 0 score diffs, 0 grade diffs |
| `models/` after quarantine | ~16.8 GB (or ~9.7 GB if `siglip2` also goes) |
