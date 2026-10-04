# Taste Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make FirstCut's Strong/Mid/Weak match the photographer's own judgement on photos it has never seen. Today it agrees 56% of the time on the 157 rated photos still on disk.

**Architecture:** Measure first (one scorecard every change must beat), then fix the data the learner starves on:
- ratings that lose their photo when files move,
- ratings that carry no features,
- ratings that are almost all keepers.

Then retrain the existing ridge **MasterJudge** on richer features, judged by 3-class agreement on held-out *shoots*, and let it keep improving as ratings arrive. The champion/challenger gate stays: nothing reaches grading unless it measurably beats the current grader.

**Tech Stack:** Python 3.12 venv (numpy only inside `master_judge`, since it is imported by the CUDA-free grade worker), FastAPI routers, React/TS frontend, pytest.

**Spec:** This conversation (2026-10-03). Measured facts it relies on:
- Agreement with stars on 157 on-disk rated photos: **0.561**. Spearman: **0.27**.
- Only **157 of 826** ratings still resolve to a file on disk.
- Rated set by stars: 5★ 44, 4★ 60, 3★ 48, 2★ 3, 1★ 2. That is almost no Mid/Weak examples.
- Shipped judge: n=791, holdout ρ 0.874 against 0.867 baseline (barely better than the plain machine score).
- Empty-scene penalty: tested and **rejected**, because it lowered agreement at every strength.

## Global Constraints

- `src/master_judge.py` must stay numpy-only at module scope. **No torch, no sklearn.** The grade worker imports it and must never initialise CUDA.
- Never call `torch.cuda.*` (not even `is_available()`) in a parent process.
- Fully offline at runtime. No network calls.
- Grades stay **absolute**. No per-batch or relative bucketing, and no floor ≥ 0.60.
- Any change that alters a photo's score bumps `GRADER_VERSION` in `src/pipeline_stages.py`.
- Star → class mapping must equal `ratings_store._STAR_GRADE`: 5,4 → Strong; 3,2 → Mid; 1 → Weak.
- A challenger that loses its exam must leave grading **unchanged** (it is recorded, not used).
- Run tests with `venv/Scripts/python.exe -m pytest …`. System Python lacks the ML stack.
- After touching any `.py`, run `venv/Scripts/python.exe -m pyflakes <files>`. The tests miss undefined names, and a NameError shipped today in `grade_worker.py`.
- **Task 6 changes a product rule** (CLAUDE.md "Ratings Are Never Ground Truth"). Do not start it without the user's explicit yes.

## Review Focus

1. **A rated photo is copied off the card or renamed.** Its rating must still apply, both to its grade and to training (Task 3: `test_rating_follows_a_copied_file`).
2. **Old ratings have no stored features or score.** Fitting must skip them, not crash or impute zeros (Task 2: `test_rows_without_features_are_skipped_not_zeroed`).
3. **Bursts leak across the train/test split.** Near-identical frames from one shoot on both sides inflate accuracy. Hold out whole folders (Task 5: `test_holdout_never_splits_a_folder`).
4. **The teach sample on a tiny or fully rated folder** must return what exists (possibly 0), never an error (Task 4: `test_sample_small_or_fully_rated_folder`).
5. **A challenger that ties or loses** must not change `active()`'s choice, even when the loop is on (Task 6: `test_lost_challenge_never_changes_grading`).

---

### Task 1: Accuracy scorecard

One number every later task must move: 3-class agreement with the photographer's stars, with a confidence interval, per shoot.

**Files:**
- Create: `src/accuracy.py`
- Create: `scripts/accuracy_report.py`
- Test: `tests/test_accuracy.py`

**Interfaces:**
- Produces:
  - `stars_to_class(stars:int) -> int` (0 Weak, 1 Mid, 2 Strong)
  - `score_to_class(score:float, strong_t:float, mid_t:float) -> int`
  - `agreement(pred:list[int], true:list[int]) -> float`
  - `confusion(pred, true) -> list[list[int]]`, as `[true][pred]` 3×3
  - `paired_bootstrap(true, pred_a, pred_b, groups, n=2000, seed=0) -> dict{delta, lo, hi}`
  - `shoot_of(path:str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_accuracy.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import accuracy as acc


def test_star_mapping_matches_ratings_store():
    import ratings_store as rs
    for s in range(1, 6):
        label = rs.grade_for_stars(s)
        want = 2 if "Strong" in label else 1 if "Mid" in label else 0
        assert acc.stars_to_class(s) == want


def test_score_to_class_uses_given_lines():
    assert acc.score_to_class(0.60, 0.58, 0.39) == 2
    assert acc.score_to_class(0.45, 0.58, 0.39) == 1
    assert acc.score_to_class(0.38, 0.58, 0.39) == 0


def test_agreement_and_confusion():
    true, pred = [2, 1, 0, 1], [2, 2, 0, 1]
    assert acc.agreement(pred, true) == 0.75
    cm = acc.confusion(pred, true)
    assert cm[1][2] == 1 and cm[2][2] == 1 and sum(map(sum, cm)) == 4


def test_paired_bootstrap_detects_a_real_gain_and_not_a_tie():
    true = [0, 1, 2] * 40
    groups = [f"s{i // 12}" for i in range(120)]
    better = list(true)
    worse = [1] * 120
    up = acc.paired_bootstrap(true, worse, better, groups, n=500, seed=1)
    assert up["lo"] > 0
    tie = acc.paired_bootstrap(true, better, better, groups, n=200, seed=1)
    assert tie["delta"] == 0 and tie["lo"] <= 0 <= tie["hi"]


def test_shoot_of_is_parent_folder():
    assert acc.shoot_of(r"F:\DCIM\100MSDCF\DSC1.ARW") == r"F:\DCIM\100MSDCF".lower()
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_accuracy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'accuracy'`

- [ ] **Step 3: Implement**

```python
# src/accuracy.py
"""Agreement between machine grades and the photographer's stars.

The scorecard every taste change must beat (2026-10-03 baseline: 0.561
3-class agreement on 157 on-disk rated photos). Pure numpy — importable from
the CUDA-free grade worker. Groups are SHOOTS (parent folders): resampling
photos instead would let near-identical burst frames inflate confidence.
"""
from __future__ import annotations

import os
import numpy as np

_STAR_CLASS = {5: 2, 4: 2, 3: 1, 2: 1, 1: 0}   # must equal ratings_store._STAR_GRADE


def stars_to_class(stars: int) -> int:
    return _STAR_CLASS[int(stars)]


def score_to_class(score: float, strong_t: float, mid_t: float) -> int:
    return 2 if score >= strong_t else 1 if score >= mid_t else 0


def agreement(pred, true) -> float:
    p, t = np.asarray(pred), np.asarray(true)
    return float((p == t).mean()) if t.size else float("nan")


def confusion(pred, true) -> list:
    cm = [[0, 0, 0] for _ in range(3)]
    for p, t in zip(pred, true):
        cm[int(t)][int(p)] += 1
    return cm


def shoot_of(path: str) -> str:
    return os.path.dirname(path.replace("/", "\\")).lower()


def paired_bootstrap(true, pred_a, pred_b, groups, n: int = 2000, seed: int = 0) -> dict:
    """Agreement(b) − agreement(a), resampling whole shoots. lo/hi = 95% CI."""
    t, a, b = map(np.asarray, (true, pred_a, pred_b))
    g = np.asarray(groups)
    ids = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in ids}
    hit_a, hit_b = (a == t).astype(float), (b == t).astype(float)
    delta = float(hit_b.mean() - hit_a.mean())
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(ids, size=len(ids), replace=True)])
        ds.append(hit_b[pick].mean() - hit_a[pick].mean())
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"delta": delta, "lo": float(lo), "hi": float(hi)}
```

```python
# scripts/accuracy_report.py
"""How well do grades match YOUR stars? One table, every change.

Usage: venv\\Scripts\\python.exe scripts/accuracy_report.py [--strong 0.58 --mid 0.39]
"""
import argparse, collections, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import accuracy as acc
import master_judge as mj

ap = argparse.ArgumentParser()
ap.add_argument("--strong", type=float, default=0.58)
ap.add_argument("--mid", type=float, default=0.39)
args = ap.parse_args()

rows = mj.collect_rows()
if not rows:
    sys.exit("no rated photos with a machine score — rate some photos first")
true = [acc.stars_to_class(r["stars"]) for r in rows]
pred = [acc.score_to_class(r["score"], args.strong, args.mid) for r in rows]
print(f"rated photos measured : {len(rows)}")
print(f"3-class agreement     : {acc.agreement(pred, true):.3f}")
print(f"spearman (score~stars): {mj.spearman([r['score'] for r in rows], [r['stars'] for r in rows]):.3f}")
cm = acc.confusion(pred, true)
print("confusion  (rows = YOUR grade, cols = machine)   Weak  Mid  Strong")
for name, row in zip(("Weak", "Mid", "Strong"), cm):
    print(f"  {name:6}{'':38}{row[0]:5}{row[1]:5}{row[2]:7}")
by = collections.defaultdict(list)
for r, t, p in zip(rows, true, pred):
    by[acc.shoot_of(r["path"])].append(t == p)
print("per shoot:")
for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):4}  {sum(v) / len(v):.2f}  {k}")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `venv/Scripts/python.exe -m pytest tests/test_accuracy.py -v`
Expected: 5 passed

- [ ] **Step 5: Record the baseline**

Run: `venv/Scripts/python.exe scripts/accuracy_report.py`
Expected: prints the table without error. Paste the output into the commit message body as the baseline. (Until Task 2 backfills features, it measures only rows with a live or snapshot score.)

- [ ] **Step 6: Commit**

```bash
git add src/accuracy.py scripts/accuracy_report.py tests/test_accuracy.py
git commit -m "feat(accuracy): 3-class agreement scorecard against the photographer's stars"
```

---

### Task 2: Ratings carry their own features

Retraining today needs the photo file *and* a live LanceDB row, and both vanish (catalog clears, files move). Snapshot the grade-time breakdown onto the rating itself, and backfill the photos still on disk.

**Files:**
- Modify: `src/ratings_store.py` (add `features` to `set_rating`, add `attach_features`, add `get_features`)
- Modify: `routers/grading.py:1117-1123` (the star endpoint's snapshot block)
- Modify: `src/master_judge.py:155-207` (`collect_rows` falls back to stored features)
- Create: `scripts/backfill_rating_features.py`
- Test: `tests/test_rating_features.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ratings_store.attach_features(path:str, features:dict, score:float|None=None) -> bool`. It keeps stars, `rated_at` and `source`, and returns False when the path is unrated.
  - `ratings_store.get_features(path:str) -> dict | None`
  - `ratings_store.slim_features(bd:dict) -> dict`, which keeps numeric/bool top-level keys plus `_arch_w` and drops `_grade_sig`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rating_features.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ratings_store as rs


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")


def test_attach_features_keeps_stars_and_source(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    rs.set_rating("C:/a.jpg", 2, source="tpe_master")
    before = rs._read_raw()["C:/a.jpg"]["rated_at"]
    assert rs.attach_features("C:/a.jpg", {"Technical": 0.5, "_grade_sig": "x",
                                           "_arch_w": {"geo": 1.0}, "label": "boat"}, score=0.44)
    e = rs._read_raw()["C:/a.jpg"]
    assert e["stars"] == 2 and e["source"] == "tpe_master" and e["rated_at"] == before
    assert e["score"] == 0.44
    assert rs.get_features("C:/a.jpg") == {"Technical": 0.5, "_arch_w": {"geo": 1.0}}


def test_attach_features_refuses_unrated(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    assert rs.attach_features("C:/none.jpg", {"Technical": 0.5}) is False
    assert "C:/none.jpg" not in rs._read_raw()


def test_rows_without_features_are_skipped_not_zeroed(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    import master_judge as mj
    rs.set_rating("C:/old.jpg", 4)                       # legacy: no score, no features
    rs.set_rating("C:/new.jpg", 2, score=0.5)
    rs.attach_features("C:/new.jpg", {k: 0.5 for k in mj.FEATURES} | {"_arch_w": {a: 0.2 for a in mj.ARCHES}})
    monkeypatch.setattr("lance_store.query_all", lambda min_score=0.0: [])
    rows = mj.collect_rows()
    assert [r["path"] for r in rows] == ["C:/new.jpg"]
    assert rows[0]["breakdown"]["Technical"] == 0.5
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_rating_features.py -v`
Expected: FAIL with `AttributeError: module 'ratings_store' has no attribute 'attach_features'`

- [ ] **Step 3: Implement in `src/ratings_store.py`** (add these after `set_rating`)

```python
def slim_features(bd: dict) -> dict:
    """The breakdown values the taste learner can use: numeric/bool scalars and
    the archetype weights. Strings, nested logs and the grade signature are
    dropped — they are not features and would bloat the store."""
    out: dict = {}
    for k, v in (bd or {}).items():
        if k == "_grade_sig":
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            out[k] = v
        elif k == "_arch_w" and isinstance(v, dict):
            out[k] = {a: float(x) for a, x in v.items() if isinstance(x, (int, float))}
    return out


def attach_features(path: str, features: dict, score: float | None = None) -> bool:
    """Store the grade-time features (and optionally the machine score) on an
    EXISTING rating without touching stars, rated_at or source. Returns False
    when the path is not rated — features never create a rating."""
    with _lock:
        cur = _read_raw()
        e = cur.get(path)
        if e is None:
            return False
        if not isinstance(e, dict):
            e = {"stars": int(e)}
        e["features"] = slim_features(features)
        if score is not None:
            e["score"] = float(score)
        cur[path] = e
        _atomic_write(_PATH, cur)
        try:
            _atomic_write(_BACKUP, cur)
        except Exception:
            pass
        return True


def get_features(path: str) -> dict | None:
    v = _read_raw().get(path)
    return v.get("features") if isinstance(v, dict) else None
```

- [ ] **Step 4: Fall back to stored features in `master_judge.collect_rows`**

In `src/master_judge.py` `collect_rows`, right after the `if row is not None:` block that fills `bd`, insert:

```python
        if not bd:
            bd = rs.get_features(path) or {}
```

- [ ] **Step 5: Snapshot features from the star endpoint**

In `routers/grading.py`, replace the block that calls `_rs.set_rating(path, stars, score=rows[0].get("score"), personal_score=...)` with:

```python
            if rows:
                _rs.set_rating(path, stars, score=rows[0].get("score"),
                                personal_score=rows[0].get("personal_score"))
                _bd_snap = rows[0].get("breakdown")
                if isinstance(_bd_snap, str):
                    try:
                        _bd_snap = json.loads(_bd_snap)
                    except Exception:
                        _bd_snap = None
                if isinstance(_bd_snap, dict) and _bd_snap:
                    _rs.attach_features(path, _bd_snap)
```

(Check that `json` is imported at the top of `routers/grading.py`; it is imported as `_json` in places. Use whichever name the module already imports.)

- [ ] **Step 6: Backfill script**

```python
# scripts/backfill_rating_features.py
"""Re-grade COPIES of every rated photo still on disk and attach the fresh
features + score to its rating. Copies keep the user's library untouched; the
LanceDB/catalog rows the throwaway grade writes are purged afterwards.

Usage: venv\\Scripts\\python.exe scripts/backfill_rating_features.py
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UNIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UNIT / "src"))
sys.path.insert(0, str(UNIT / ".claude" / "skills" / "run-framegrade"))
import ratings_store as rs
import driver   # _lance_purge: removes rows the grade wrote into the real store

stars = {p: s for p, s in rs.load().items() if os.path.exists(p)}
if not stars:
    sys.exit("no rated photos found on disk")
root = Path(tempfile.mkdtemp(prefix="fg_backfill_"))
copy_to_orig, folders = {}, set()
for p in stars:
    src = Path(p)
    d = root / f"{abs(hash(str(src.parent))) % 10**8}"
    d.mkdir(exist_ok=True)
    dst = d / src.name
    if dst.exists():
        continue
    shutil.copy2(src, dst)
    copy_to_orig[str(dst)] = p
    folders.add(str(d))
req, prog = root / "req.json", root / "p.jsonl"
prog.write_text("", encoding="utf-8")
req.write_text(json.dumps({"folders": sorted(folders), "preset": "Classic Street",
    "force_rescan": True, "scan_mode": False, "deep_grade": False,
    "catalog_path": str(root / "c.json"), "data_dir": str(root),
    "mogco_target": 0, "sample_limit": 0}), encoding="utf-8")
try:
    with open(root / "run.log", "w", encoding="utf-8", errors="replace") as lf:
        subprocess.run([str(UNIT / "venv" / "Scripts" / "python.exe"), "grade_runner.py",
                        str(req), str(prog)], cwd=str(UNIT), stdout=lf, stderr=subprocess.STDOUT,
                       timeout=7200, env=dict(os.environ, PYTHONIOENCODING="utf-8", SIGLIP_TIER="high"))
    import lance_store
    t = lance_store._connect_or_create().to_pandas()
    t = t[t["path"].str.startswith(str(root))]
    n = 0
    for _, r in t.iterrows():
        orig = copy_to_orig.get(r["path"])
        bd = r["breakdown"] if isinstance(r["breakdown"], dict) else json.loads(r["breakdown"] or "{}")
        if orig and rs.attach_features(orig, bd, score=float(r["score"])):
            n += 1
    print(f"attached features to {n}/{len(stars)} rated photos")
finally:
    driver._lance_purge(str(root))
    shutil.rmtree(root, ignore_errors=True)
```

- [ ] **Step 7: Run the tests and pyflakes**

Run: `venv/Scripts/python.exe -m pytest tests/test_rating_features.py tests/test_master_judge.py -v`
Expected: all pass.

Run: `venv/Scripts/python.exe -m pyflakes src/ratings_store.py src/master_judge.py routers/grading.py scripts/backfill_rating_features.py`
Expected: no `undefined name`.

- [ ] **Step 8: Backfill for real and re-run the scorecard**

Run: `venv/Scripts/python.exe scripts/backfill_rating_features.py`
Expected: `attached features to 157/157 rated photos` (± photos that fail to decode).

Run: `venv/Scripts/python.exe scripts/accuracy_report.py`
Expected: `rated photos measured : 157` or more.

- [ ] **Step 9: Commit**

```bash
git add src/ratings_store.py src/master_judge.py routers/grading.py scripts/backfill_rating_features.py tests/test_rating_features.py
git commit -m "feat(ratings): ratings carry grade-time features so retraining survives cleared caches"
```

---

### Task 3: Ratings follow the photo, not the path

669 of 826 ratings point at paths that no longer exist. Give each rating a content fingerprint, resolve stars by path then fingerprint, and relink the old ones.

**Files:**
- Create: `src/photo_identity.py`
- Modify: `src/ratings_store.py` (stamp `fp` in `set_rating`, add `stars_for_paths`)
- Modify: `src/grade_pipeline_v2.py:3844-3863` (use `stars_for_paths` instead of `load()` + `.get(path)`)
- Create: `scripts/relink_ratings.py`
- Test: `tests/test_photo_identity.py`

**Interfaces:**
- Consumes: `ratings_store.attach_features` (Task 2), whose entries keep any `fp` key.
- Produces:
  - `photo_identity.fingerprint(path:str) -> str | None`, giving `"fp1:<20 hex>"`, or None if unreadable.
  - `ratings_store.stars_for_paths(paths:list[str]) -> dict[str,int]`, which matches by path first and falls back to fingerprint, and only includes rated photos.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_photo_identity.py
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import photo_identity as pid
import ratings_store as rs


def _photo(p: Path, seed: int) -> Path:
    p.write_bytes(bytes([seed]) * 200_000)
    return p


def test_copy_has_same_fingerprint_edit_does_not(tmp_path):
    a = _photo(tmp_path / "DSC1.ARW", 7)
    b = tmp_path / "copy" / "renamed.ARW"; b.parent.mkdir(); shutil.copy2(a, b)
    c = _photo(tmp_path / "DSC2.ARW", 8)
    assert pid.fingerprint(str(a)) == pid.fingerprint(str(b))
    assert pid.fingerprint(str(a)) != pid.fingerprint(str(c))
    assert pid.fingerprint(str(tmp_path / "missing.ARW")) is None


def test_rating_follows_a_copied_file(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")
    a = _photo(tmp_path / "DSC1.ARW", 7)
    rs.set_rating(str(a), 1)
    moved = tmp_path / "laptop" / "DSC1.ARW"; moved.parent.mkdir(); shutil.move(a, moved)
    assert rs.stars_for_paths([str(moved)]) == {str(moved): 1}


def test_unrated_photos_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "_PATH", tmp_path / "r.json")
    monkeypatch.setattr(rs, "_BACKUP", tmp_path / "r.bak.json")
    x = _photo(tmp_path / "X.ARW", 3)
    assert rs.stars_for_paths([str(x)]) == {}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_photo_identity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'photo_identity'`

- [ ] **Step 3: Implement `src/photo_identity.py`**

```python
"""Content identity for a photo file — survives copy, move and rename.

size + the first 64 KB. That window holds the RAW/JPEG header with the EXIF
capture time and camera serial, so two different shots never collide in
practice, while a byte-identical copy (card -> laptop) always matches. An
edited export is a different file and correctly gets a new identity.
"""
from __future__ import annotations

import hashlib
import os

_HEAD = 65536


def fingerprint(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(_HEAD)
    except OSError:
        return None
    h = hashlib.sha1(str(size).encode() + b"|" + head).hexdigest()[:20]
    return f"fp1:{h}"
```

- [ ] **Step 4: Stamp and resolve fingerprints in `src/ratings_store.py`**

In `set_rating`, after `entry: dict = {...}` is built (stars > 0 branch), add:

```python
            try:
                from photo_identity import fingerprint as _fp
                fp = _fp(path) or (existing.get("fp") if isinstance(existing, dict) else None)
            except Exception:
                fp = None
            if fp:
                entry["fp"] = fp
            if isinstance(existing, dict):
                for keep in ("features",):
                    if keep in existing:
                        entry[keep] = existing[keep]
```

Then add this function:

```python
def stars_for_paths(paths: list) -> dict:
    """{path: stars} for every RATED photo among `paths` — exact path first,
    then content fingerprint, so a rating made on the SD card still applies
    after the photo is copied to the laptop or renamed."""
    raw = _read_raw()
    out, misses = {}, []
    for p in paths:
        s = _stars_of(raw.get(p))
        if s > 0:
            out[p] = s
        else:
            misses.append(p)
    if misses:
        by_fp = {v["fp"]: _stars_of(v) for v in raw.values()
                 if isinstance(v, dict) and v.get("fp") and _stars_of(v) > 0}
        if by_fp:
            from photo_identity import fingerprint
            for p in misses:
                fp = fingerprint(p)
                if fp and fp in by_fp:
                    out[p] = by_fp[fp]
    return out
```

- [ ] **Step 5: Use it in the pipeline**

In `src/grade_pipeline_v2.py` around line 3844, replace `_user_ratings = _ratings_store.load()` with:

```python
        _user_ratings = _ratings_store.stars_for_paths(list(paths))
```

Leave the existing `except` that sets `_user_ratings = {}` in place. `_user_ratings.get(path, 0)` below keeps working unchanged.

- [ ] **Step 6: Relink script for the 669 orphaned ratings**

```python
# scripts/relink_ratings.py
"""Re-point ratings whose file moved. A rating is relinked only when its
(parent-folder name, file name) pair matches EXACTLY ONE file under the
search roots — camera file names repeat across cards, so anything ambiguous
is reported, never guessed.

Usage: venv\\Scripts\\python.exe scripts/relink_ratings.py --search "D:\\Photos" "C:\\Users\\Nicky Tuason\\Desktop" [--apply]
"""
import argparse, collections, os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ratings_store as rs
from photo_identity import fingerprint

ap = argparse.ArgumentParser()
ap.add_argument("--search", nargs="+", required=True)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

raw = rs._read_raw()
orphans = {p: v for p, v in raw.items() if not os.path.exists(p)}
index = collections.defaultdict(list)
for root in a.search:
    for dp, _, files in os.walk(root):
        for f in files:
            index[(Path(dp).name.lower(), f.lower())].append(os.path.join(dp, f))

linked, ambiguous, missing = {}, 0, 0
for p, v in orphans.items():
    key = (Path(p).parent.name.lower(), Path(p).name.lower())
    hits = index.get(key, [])
    if len(hits) == 1 and hits[0] not in raw:
        linked[p] = hits[0]
    elif len(hits) > 1:
        ambiguous += 1
    else:
        missing += 1
print(f"orphaned ratings: {len(orphans)}  relinkable: {len(linked)}  ambiguous: {ambiguous}  not found: {missing}")
if a.apply and linked:
    with rs._lock:
        cur = rs._read_raw()
        for old, new in linked.items():
            e = cur.pop(old)
            e = e if isinstance(e, dict) else {"stars": int(e)}
            e["fp"] = fingerprint(new) or e.get("fp")
            e["relinked_from"] = old
            cur[new] = e
        rs._atomic_write(rs._PATH, cur)
        rs._atomic_write(rs._BACKUP, cur)
    print(f"relinked {len(linked)} — now run scripts/backfill_rating_features.py")
```

- [ ] **Step 7: Run tests, pyflakes and the pipeline tests**

Run: `venv/Scripts/python.exe -m pytest tests/test_photo_identity.py tests/test_rating_features.py tests/test_pipeline_stages.py -v`
Expected: all pass.

Run: `venv/Scripts/python.exe -m pyflakes src/photo_identity.py src/ratings_store.py scripts/relink_ratings.py`
Expected: no `undefined name`.

- [ ] **Step 8: Dry-run the relink against the user's drives**

Run: `venv/Scripts/python.exe scripts/relink_ratings.py --search "C:\Users\Nicky Tuason\Desktop" "E:\"`
Expected: a count line. **Show the counts to the user before running with `--apply`.**

- [ ] **Step 9: Commit**

```bash
git add src/photo_identity.py src/ratings_store.py src/grade_pipeline_v2.py scripts/relink_ratings.py tests/test_photo_identity.py
git commit -m "feat(ratings): ratings follow the photo by content fingerprint; relink orphaned ratings"
```

---

### Task 4: Teach session: 50 photos across the whole range

The learner has 5 photos rated 1–2★. One quick session must collect Weak/Mid/Strong examples. That means picking photos spread across the machine-score range, and making the Strong/Mid/Weak pills one-click ratings.

**Files:**
- Create: `src/teach_sampler.py`
- Modify: `routers/grading.py` (add `GET /api/teach/sample`)
- Modify: `frontend/src/components/views/AnalysisPanel.tsx:153-175` (pills become buttons)
- Modify: `frontend/src/App.tsx` (a "Teach (50)" header button that filters the gallery to the sample)
- Test: `tests/test_teach_sampler.py`

**Interfaces:**
- Consumes: `ratings_store.stars_for_paths` (Task 3).
- Produces:
  - `teach_sampler.pick(photos:list[dict], n:int=50) -> list[str]`. `photos` are catalog dicts with `path`, `score`, `cluster_id`, `sim_flag`, and the result is paths.
  - `GET /api/teach/sample?n=50` → `{"paths": [...], "available": int}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_teach_sampler.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import teach_sampler as ts


def _p(i, score, cluster=-1, flag=""):
    return {"path": f"C:/c/{i}.ARW", "score": score, "cluster_id": cluster, "sim_flag": flag}


def test_spans_the_score_range(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {})
    photos = [_p(i, i / 100) for i in range(100)]
    got = ts.pick(photos, n=10)
    scores = sorted(float(p.split("/")[-1][:-4]) / 100 for p in got)
    assert len(got) == 10 and scores[0] <= 0.05 and scores[-1] >= 0.94


def test_skips_rated_and_hidden_duplicates(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {"C:/c/1.ARW": 3})
    photos = [_p(1, .5), _p(2, .5, cluster=7, flag="🔁 Duplicate"), _p(3, .6, cluster=7, flag="★ Best of 2")]
    assert ts.pick(photos, n=5) == ["C:/c/3.ARW"]


def test_sample_small_or_fully_rated_folder(monkeypatch):
    monkeypatch.setattr(ts, "_rated", lambda paths: {p: 3 for p in paths})
    assert ts.pick([_p(1, .5)], n=50) == []
    monkeypatch.setattr(ts, "_rated", lambda paths: {})
    assert ts.pick([], n=50) == []
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_teach_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'teach_sampler'`

- [ ] **Step 3: Implement `src/teach_sampler.py`**

```python
"""Pick the photos most worth rating: evenly spread over the machine-score
range, unrated, and not hidden duplicates. Rating keepers only (the 2026-10-03
baseline: 5 of 157 rated photos were 1-2★) teaches the learner nothing about
Mid and Weak — spreading over the range fixes that in one short session."""
from __future__ import annotations


def _rated(paths: list) -> dict:
    import ratings_store as rs
    return rs.stars_for_paths(paths)


def pick(photos: list, n: int = 50) -> list:
    visible = [p for p in photos
               if not (int(p.get("cluster_id", -1)) >= 0 and "Best" not in (p.get("sim_flag") or ""))]
    rated = _rated([p["path"] for p in visible])
    pool = sorted((p for p in visible if p["path"] not in rated
                   and isinstance(p.get("score"), (int, float))), key=lambda p: p["score"])
    if len(pool) <= n:
        return [p["path"] for p in pool]
    step = (len(pool) - 1) / (n - 1)
    return [pool[round(i * step)]["path"] for i in range(n)]
```

- [ ] **Step 4: API endpoint in `routers/grading.py`**

```python
@router.get("/api/teach/sample")
async def teach_sample(n: int = 50):
    """Up to n unrated photos from the current catalog, spread across the
    machine-score range — the one-session taste calibration set."""
    from server_impl import _CATALOG_PATH
    import teach_sampler
    try:
        photos = json.loads(_CATALOG_PATH.read_text(encoding="utf-8")).get("photos", [])
    except Exception:
        photos = []
    paths = teach_sampler.pick(photos, n=max(1, min(int(n), 200)))
    return {"paths": paths, "available": len(photos)}
```

(Use the module's existing `json` import name.)

- [ ] **Step 5: Make the grade pills one-click ratings**

In `frontend/src/components/views/AnalysisPanel.tsx`, in the `(['Strong','Mid','Weak'] as const).map(g => { … })` block, change the `<div key={g} …>` into a `<button>`. Remove `pointerEvents:'none'`, keep the existing styles, and add:

```tsx
                          <button key={g} type="button"
                            title={`Rate as ${gradeLabel(g)} (${g === 'Strong' ? 5 : g === 'Mid' ? 3 : 1}★) — teaches the grader your taste`}
                            onClick={() => handleSetStars(sel.id, g === 'Strong' ? 5 : g === 'Mid' ? 3 : 1)}
                            style={{ /* existing style object, minus pointerEvents */ }}>
                            {gradeLabel(g)}
                          </button>
```

Update the comment above the block from "Grade display — read-only" to "Grade pills — one click rates the photo (Strong 5★, Mid 3★, Weak 1★)".

- [ ] **Step 6: "Teach (50)" button in `frontend/src/App.tsx`**

Next to the `Quick pass` button (same `{!isGrading && mainTab !== 'creative' && (…)}` pattern, shown only when `isDone`):

```tsx
        {!isGrading && isDone && mainTab !== 'creative' && (
          <Button variant="quiet"
            title="Show 50 unrated photos spread from best to worst. Click Strong / Mid / Weak on each — one short session teaches the grader your taste."
            onClick={async () => {
              try {
                const r = await axios.get(`${API}/api/teach/sample`, { params: { n: 50 } });
                const paths: string[] = r.data.paths || [];
                if (!paths.length) { notify('Nothing left to rate here — every visible photo already has your rating.', 'info'); return; }
                setSearchResults(new Set(paths));
                setMainTab('gallery');
                notify(`${paths.length} photos to rate. Use Strong / Mid / Weak on each.`, 'info');
              } catch { notify('Could not build the rating set — is the server running?', 'error'); }
            }}>
            Teach (50)
          </Button>
        )}
```

- [ ] **Step 7: Tests, typecheck/build, screenshot**

Run: `venv/Scripts/python.exe -m pytest tests/test_teach_sampler.py -v`
Expected: 3 passed

Run: `cd frontend && npm run build`
Expected: `✓ built`

Screenshot with an isolated profile, because the driver's `shot` hangs when the user's Chrome is open:
`"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --user-data-dir="%TEMP%\shotprof" --window-size=1500,950 --timeout=15000 --screenshot="%TEMP%\teach.png" http://127.0.0.1:8000/`
Expected: the header shows "Teach (50)" after a graded folder is open, and the pills look unchanged apart from a pointer cursor.

- [ ] **Step 8: Commit**

```bash
git add src/teach_sampler.py routers/grading.py frontend/src tests/test_teach_sampler.py
git commit -m "feat(teach): 50-photo rating session spread across the score range; grade pills rate in one click"
```

---

### Task 5: Learner v2: the features the judge can't see today, judged on unseen shoots

**Files:**
- Modify: `src/grade_pipeline_v2.py` (Step 6b: also write `_subject_streak` and `_living` into the breakdown)
- Modify: `src/subject_sharpness.py` (`score_paths`: add `living: bool` to each measure dict)
- Modify: `src/master_judge.py` (add `EXTRA` features, folder-grouped holdout, agreement-based promotion)
- Modify: `src/pipeline_stages.py` (bump `GRADER_VERSION`)
- Test: `tests/test_master_judge_v2.py`

**Interfaces:**
- Consumes: `accuracy.*` (Task 1), `ratings_store.get_features` (Task 2).
- Produces:
  - `master_judge.EXTRA: list[str]`
  - `master_judge.fit_from_rows(rows)`, now returning extra keys `agree_judge`, `agree_base`, `agree_ci` ([lo, hi]) and `holdout_shoots`.
  - `promoted` is True only when `agree_ci[0] > 0` **and** `rho_holdout >= rho_baseline`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_master_judge_v2.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import master_judge as mj


def _row(i, shoot, stars, score, smear=0.0):
    bd = {k: 0.5 for k in mj.FEATURES}
    bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}
    bd.update({"_subject_sharp": 0.7, "_subject_streak": smear, "_bg_streak": 0.1,
               "_living": 1.0})
    return {"path": f"C:/{shoot}/{i}.ARW", "stars": stars, "score": score,
            "breakdown": bd, "source": "", "from_snapshot": False}


def test_extra_features_are_in_the_design():
    for k in ("_subject_sharp", "_subject_streak", "_bg_streak", "_pan", "_smeared_subject", "_living"):
        assert k in mj.EXTRA and k in mj.DESIGN


def test_missing_flags_default_to_false_not_dropped():
    bd = {k: 0.5 for k in mj.FEATURES}; bd["_arch_w"] = {a: 0.2 for a in mj.ARCHES}
    bd.update({"_subject_sharp": 0.7, "_subject_streak": 0.0, "_bg_streak": 0.0, "_living": 0.0})
    assert np.isfinite(mj.feature_vector(bd)).all()      # _pan / _smeared_subject absent = False


def test_holdout_never_splits_a_folder():
    rows = [_row(i, f"s{i % 6}", 1 + i % 5, 0.4 + 0.1 * (i % 5)) for i in range(120)]
    tr, ho = mj._shoot_split(rows, frac=0.34, seed=1)
    tr_s = {rows[i]["path"].split("/")[1] for i in tr}
    ho_s = {rows[i]["path"].split("/")[1] for i in ho}
    assert tr_s and ho_s and not (tr_s & ho_s)


def test_judge_that_cannot_beat_baseline_is_not_promoted(tmp_path):
    rng = np.random.default_rng(0)
    rows = [_row(i, f"s{i % 6}", int(rng.integers(1, 6)), float(rng.random())) for i in range(150)]
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is False


def test_judge_learns_smear_signal_and_promotes(tmp_path):
    rows = []
    for i in range(180):
        smeared = i % 3 == 0
        stars = 1 if smeared else 4
        rows.append(_row(i, f"s{i % 6}", stars, 0.6, smear=0.3 if smeared else 0.02))
    out = mj.fit_from_rows(rows, weights_path=tmp_path / "j.json")
    assert out["promoted"] is True and out["agree_ci"][0] > 0
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `venv/Scripts/python.exe -m pytest tests/test_master_judge_v2.py -v`
Expected: FAIL with `AttributeError: module 'master_judge' has no attribute 'EXTRA'`

- [ ] **Step 3: Write the missing features at grade time**

In `src/subject_sharpness.py` `score_paths`, after `m["labels"] = …`, add:

```python
                m["living"] = any(d["label"] in _LIVING for d in boxes)
```

…and at module level:

```python
_LIVING = {"person", "cat", "dog", "bird", "horse", "cow", "sheep"}
```

In `src/grade_pipeline_v2.py` Step 6b, inside the per-photo loop right after `_subject_sharp` is written, add:

```python
            if _m.get("subject_streak") is not None:
                per_photo_breakdowns[i]["_subject_streak"] = round(float(_m["subject_streak"]), 3)
            if _m.get("living") is not None:
                per_photo_breakdowns[i]["_living"] = 1.0 if _m["living"] else 0.0
```

Bump in `src/pipeline_stages.py`:

```python
GRADER_VERSION = "2026-10-XX"  # breakdown carries _subject_streak/_living for the taste learner
```

(Use the real date.)

- [ ] **Step 4: Extend the judge's design** in `src/master_judge.py`

After `ARCHES = [...]`:

```python
# 2026-10 signals the incumbent judge never saw. Flags written only when true
# (_pan, _smeared_subject) default to 0 when absent; the measured ones must be
# present or the row is skipped (a missing measurement is not a zero).
EXTRA = ["_subject_sharp", "_subject_streak", "_bg_streak", "_living",
         "_pan", "_smeared_subject"]
_FLAG_DEFAULT_FALSE = {"_pan", "_smeared_subject"}
```

Change `DESIGN` to:

```python
DESIGN = (["(machine score)"] + FEATURES
          + [f"arch:{a}" for a in ARCHES] + EXTRA)
```

Replace `feature_vector` body's value list with:

```python
    vals = [bd.get(k) for k in FEATURES]
    arch = bd.get("_arch_w") or {}
    vals += [arch.get(a) for a in ARCHES]
    for k in EXTRA:
        v = bd.get(k)
        if v is None and k in _FLAG_DEFAULT_FALSE:
            v = 0.0
        vals.append(1.0 if v is True else 0.0 if v is False else v)
```

`_bd_value` handles plain keys already. Make it treat booleans the same way: `return 1.0 if v is True else 0.0 if v is False else float(v) …`.

- [ ] **Step 5: Hold out whole shoots and promote on agreement**

Add:

```python
def _shoot_split(rows: list, frac: float, seed: int):
    """Train/holdout indices with every shoot (parent folder) entirely on one
    side, so burst frames of the same moment never sit on both sides."""
    import accuracy as acc
    shoots = sorted({acc.shoot_of(r["path"]) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(shoots)
    target = max(1, int(round(len(rows) * frac)))
    ho_s, count = set(), 0
    for s in shoots:
        if count >= target or len(ho_s) >= len(shoots) - 1:
            break
        ho_s.add(s)
        count += sum(1 for r in rows if acc.shoot_of(r["path"]) == s)
    ho = np.array([i for i, r in enumerate(rows) if acc.shoot_of(r["path"]) in ho_s], dtype=np.intp)
    tr = np.array([i for i, r in enumerate(rows) if acc.shoot_of(r["path"]) not in ho_s], dtype=np.intp)
    return tr, ho
```

In `fit_from_rows`:

1. Replace `tr, ho = _stratified_split(y, holdout_frac, seed)` with `tr, ho = _shoot_split(rows, holdout_frac, seed)`.
2. After `rho_base` is computed, compute agreement on the **blended** score, exactly as grading uses it:

```python
    import accuracy as acc
    from rating_calibration import thresholds_from_records
    strong_t, mid_t, _ = thresholds_from_records(
        [{"path": rows[i]["path"], "stars": rows[i]["stars"], "score": rows[i]["score"],
          "rated_at": None} for i in tr])
    provisional = {"rho_holdout": rho_judge, "rho_baseline": rho_base}
    w = _weight_from(provisional) if not np.isnan(rho_judge) else 0.0
    judge01 = np.clip(pred_ho, 0.0, 1.0)
    blended = (1.0 - w) * machine[ho] + w * judge01
    true_c = [acc.stars_to_class(int(s)) for s in y[ho]]
    base_c = [acc.score_to_class(float(s), strong_t, mid_t) for s in machine[ho]]
    judge_c = [acc.score_to_class(float(s), strong_t, mid_t) for s in blended]
    groups = [acc.shoot_of(rows[i]["path"]) for i in ho]
    boot = acc.paired_bootstrap(true_c, base_c, judge_c, groups, n=2000, seed=seed)
```

3. Replace the `promoted = …` line with:

```python
    promoted = bool(not np.isnan(rho_judge) and not np.isnan(rho_base)
                    and rho_judge >= rho_base and boot["lo"] > 0)
```

4. Add to `out`: `"agree_base": round(acc.agreement(base_c, true_c), 4)`, `"agree_judge": round(acc.agreement(judge_c, true_c), 4)`, `"agree_ci": [round(boot["lo"], 4), round(boot["hi"], 4)]`, `"holdout_shoots": sorted(set(groups))`. Also put `agree_judge` and `agree_base` into the `history.append({...})` record.

(Verified 2026-10-03: `_weight_from` reads only `rho_holdout`/`rho_baseline` and returns the 0.10 floor when the judge does not beat the baseline. `rating_calibration.thresholds_from_records(records)` returns `(strong, mid, info)` and uses each record's `score` and `stars`.)

- [ ] **Step 6: Run all judge tests and pyflakes**

Run: `venv/Scripts/python.exe -m pytest tests/test_master_judge_v2.py tests/test_master_judge.py tests/test_master_judge_features.py tests/test_promote_master_judge_aadb_gate.py tests/test_subject_sharpness.py -v`
Expected: all pass. Old tests that asserted the stratified split or the old promotion rule must be updated to the new contract in this step. Do not delete them.

Run: `venv/Scripts/python.exe -m pyflakes src/master_judge.py src/subject_sharpness.py src/grade_pipeline_v2.py`

- [ ] **Step 7: Real retrain and measurement**

Run, in order:
1. Re-grade the rated folders with the new `GRADER_VERSION`: `venv/Scripts/python.exe scripts/backfill_rating_features.py`
2. `venv/Scripts/python.exe scripts/master_backtest.py --fit`
3. `venv/Scripts/python.exe scripts/accuracy_report.py`

Expected: a `cache/master_judge.json` record with `agree_base`, `agree_judge` and `agree_ci`. **Report all three numbers to the user, promoted or not.** A lost challenge is a valid outcome, and it means more teach ratings are needed (Task 4).

- [ ] **Step 8: Commit**

```bash
git add src/master_judge.py src/subject_sharpness.py src/grade_pipeline_v2.py src/pipeline_stages.py tests/test_master_judge_v2.py tests/test_master_judge*.py
git commit -m "feat(judge): learn from sharpness/pan/subject signals; hold out whole shoots; promote only on proven agreement gain"
```

---

### Task 6: Keep learning without anyone calibrating (DECISION GATE)

> **Approved by the user 2026-10-03** ("The star ratings are legit trained"). This reverses the documented product rule "ratings must never change how any image is graded" (CLAUDE.md, and the docstring of `POST /api/personal/star`); rewrite both as part of this task.

**Files:**
- Modify: `src/master_judge.py` (`active()` uses a promoted local judge without the opt-in flag; `maybe_autofit` no longer requires `FIRSTCUT_MASTER_JUDGE=1`)
- Modify: `routers/grading.py` (the star endpoint calls `maybe_autofit()` after writing; the docstring is updated)
- Modify: `CLAUDE.md` ("Ratings Are Never Ground Truth" section: describe the new contract)
- Modify: `frontend/src/App.tsx` (a small "Matches you: NN%" chip from `GET /api/accuracy`)
- Modify: `routers/grading.py` (add `GET /api/accuracy`)
- Test: `tests/test_master_judge_loop.py`

**Interfaces:**
- Consumes: `fit_from_rows` output keys from Task 5 (`promoted`, `agree_judge`, `agree_base`, `agree_ci`).
- Produces: `GET /api/accuracy` → `{"agree": float|null, "agree_base": float|null, "n": int, "promoted": bool, "history": [...]}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_master_judge_loop.py
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import master_judge as mj


def _record(promoted):
    return {"promoted": promoted, "features": list(mj.DESIGN), "coef": [0.0] * len(mj.DESIGN),
            "intercept": 0.5, "mean": [0.0] * len(mj.DESIGN), "std": [1.0] * len(mj.DESIGN),
            "feature_fingerprint": mj._feature_fingerprint(), "rho_holdout": 0.5,
            "rho_baseline": 0.4, "agree_judge": 0.7, "agree_base": 0.56, "agree_ci": [0.03, 0.2], "n": 200}


def test_promoted_local_judge_is_used_without_any_flag(tmp_path, monkeypatch):
    p = tmp_path / "j.json"; p.write_text(json.dumps(_record(True)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.delenv("FIRSTCUT_MASTER_JUDGE", raising=False)
    judge, w = mj.active()
    assert judge is not None and judge.get("_source") != "shipped" and w > 0


def test_lost_challenge_never_changes_grading(tmp_path, monkeypatch):
    p = tmp_path / "j.json"; p.write_text(json.dumps(_record(False)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    shipped = mj._load_shipped()
    judge, _ = mj.active()
    assert (judge or {}).get("coef") == (shipped or {}).get("coef")


def test_kill_switch_still_wins(tmp_path, monkeypatch):
    p = tmp_path / "j.json"; p.write_text(json.dumps(_record(True)))
    monkeypatch.setattr(mj, "_WEIGHTS_PATH", p)
    monkeypatch.setenv("FIRSTCUT_MASTER_JUDGE_OFF", "1")
    assert mj.active() == (None, 0.0)
```

- [ ] **Step 2: Run them and confirm the first fails**

Run: `venv/Scripts/python.exe -m pytest tests/test_master_judge_loop.py -v`
Expected: `test_promoted_local_judge_is_used_without_any_flag` FAILS (today a local judge needs `FIRSTCUT_MASTER_JUDGE=1`).

- [ ] **Step 3: Implement**

In `active()`, replace

```python
    if cache and os.environ.get("FIRSTCUT_MASTER_JUDGE", "").strip() == "1":
```

with

```python
    if cache and cache.get("agree_ci") and float(cache["agree_ci"][0]) > 0:
```

(Verified: `load()` returns only records with `promoted: true` and a matching `feature_fingerprint`, tagged `_source: "cache"`. An older record without `agree_ci` therefore keeps needing the opt-in flag, which is intended.)

In `maybe_autofit`, delete the `if os.environ.get("FIRSTCUT_MASTER_JUDGE", …) != "1": return …` block.

In the star endpoint (`routers/grading.py`), after the snapshot block, add:

```python
        try:
            import master_judge as _mj
            _mj.maybe_autofit()
        except Exception as _e_fit:
            print(f"[star] autofit check skipped: {_e_fit}")
```

…and rewrite its docstring's product-rule paragraph to: *"A star sets ground truth for THIS photo immediately. Every 25 new ratings a background refit runs; its result changes other photos' grades ONLY if it beats the current grader on held-out shoots (master_judge.fit_from_rows promotion rule)."*

Add the accuracy endpoint:

```python
@router.get("/api/accuracy")
async def accuracy_status():
    import master_judge as _mj
    d = _mj.load() or {}
    raw = {}
    try:
        raw = json.loads(_mj._WEIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"agree": raw.get("agree_judge") if raw.get("promoted") else raw.get("agree_base"),
            "agree_base": raw.get("agree_base"), "n": int(raw.get("n") or 0),
            "promoted": bool(d), "history": raw.get("history", [])}
```

Frontend: next to the grader `Chip` in the header, fetch `/api/accuracy` once after a grade completes, and render `<Chip label={`Matches you ${Math.round(agree*100)}%`} title="How often Strong/Mid/Weak matched your own ratings on shoots the grader was not trained on."/>` when `agree` is a number.

Update CLAUDE.md's "Ratings Are Never Ground Truth (2026-08-30)" heading and the rules list to the new contract, keeping the kill switch line.

- [ ] **Step 4: Tests, build, pyflakes**

Run: `venv/Scripts/python.exe -m pytest tests/test_master_judge_loop.py tests/test_master_judge.py tests/test_master_judge_v2.py tests/test_grade_singleflight.py -v`
Expected: all pass

Run: `cd frontend && npm run build`
Run: `venv/Scripts/python.exe -m pyflakes src/master_judge.py routers/grading.py`

- [ ] **Step 5: Commit**

```bash
git add src/master_judge.py routers/grading.py frontend/src CLAUDE.md tests/test_master_judge_loop.py
git commit -m "feat(judge): ratings keep improving the grader in the background, gated on proven held-out agreement"
```

---

## How we'll know it worked

| Checkpoint | Command | Pass condition |
|---|---|---|
| Baseline recorded | `scripts/accuracy_report.py` | Prints ~0.56 on the 157 on-disk photos |
| Data recovered | `scripts/relink_ratings.py … --apply` then backfill | Measured ratings ≫ 157 |
| Mid/Weak examples | Teach (50) on the SD card | ≥ 15 new 1–3★ ratings |
| Learner beats grader | `master_backtest.py --fit` | `agree_ci[0] > 0` on held-out shoots |
| User-visible | "Matches you NN%" chip | NN rises above 56 and stays there across new cards |
