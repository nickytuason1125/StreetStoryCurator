# AADB + Unsplash Human-Accuracy Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new, externally-validated features (`"AADB"` from an
AADB-trained aesthetic head, `"Exemplar"` from an Unsplash-curated
similarity bank) to the Master Judge's ridge regression, gated so neither
can touch live grading unless it wins the existing champion/challenger exam.

**Architecture:** Both features follow the same shape as everything already
in this codebase: a one-time `*_setup.py` script (operator-run, mirrors
`nima_setup.py`) produces a small artifact under `models/` (gitignored);
a thin `src/*_scorer.py` module loads that artifact and degrades to `None`
if it's absent (mirrors `nima_scorer.py`); `grade_pipeline_v2.run_v2()`
calls the scorer and writes into `per_photo_breakdowns`; `master_judge.py`'s
`FEATURES` list picks it up; `scripts/master_backtest.py` /
`promote_master_judge.py` decide whether it ships. AADB additionally gets
its own held-out ρ check, computed once at training time and stored
alongside the head.

**Tech Stack:** Python, numpy (ridge regression via closed-form normal
equations, matching `master_judge._ridge_fit`), the existing
`siglip2_encoder.SigLIP2Encoder` (already subprocess-isolated — safe to call
directly from a top-level setup script), pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-aadb-unsplash-human-accuracy-design.md`

## Global Constraints

- Never touch `torch.cuda` directly in any process that also spawns a grade
  worker — not a concern for the new setup scripts since they call
  `SigLIP2Encoder`, which already isolates all GPU work in its own
  subprocess (`src/siglip2_encoder.py:570-583`).
- `"Aesthetic"` is never a `FEATURES` entry — it's the base grader score,
  added to the stored blob only at Lance-write time
  (`src/master_judge.py:50-53`). The new features must be named `"AADB"` and
  `"Exemplar"`, never `"Aesthetic"` or `"aesthetic"`.
- `master_judge.feature_vector()` already emits `NaN` for a missing input
  rather than imputing zero (`master_judge.py:136`) — new scorers must
  return `None` (not `0.0`) when their model artifact is absent, so a photo
  graded without the new head lands in `per_photo_breakdowns` with the key
  simply missing, not a fabricated `0.0`.
- Raw dataset images (AADB, Unsplash) are never committed to the repo or
  shipped in the built app — only derived numeric artifacts leave the setup
  scripts. `models/` is already gitignored; keep everything there.
- `master_judge.DESIGN` order is part of the saved-weights fingerprint
  (`master_judge.py:53-55`) — appending to `FEATURES` deliberately
  invalidates old cached weights and forces a re-fit. This is intentional,
  not a bug to work around.
- No task may change what a live grade returns. Every new signal is
  additive input to a regression that only ships through the existing
  `promote_master_judge.py` gate.

---

### Task 1: Shared dataset-embedding helper

**Files:**
- Create: `src/dataset_embed.py`
- Test: `tests/test_dataset_embed.py`

**Interfaces:**
- Consumes: `siglip2_encoder.SigLIP2Encoder` (existing, `encode_images(paths: List[str], batch_size: int = 0, progress=None) -> np.ndarray`)
- Produces: `dataset_embed.encode_folder(image_paths: list[str], out_npz: Path, progress=None) -> np.ndarray` — used by both Task 3 (AADB) and Task 9 (Unsplash). Returns the same `(N, 1536)` array it writes to `out_npz` (keys `"paths"`, `"embeddings"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_embed.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def test_encode_folder_writes_npz_with_matching_order(tmp_path, monkeypatch):
    import dataset_embed

    fake_paths = ["a.jpg", "b.jpg", "c.jpg"]

    class _FakeEncoder:
        def __init__(self, device="auto"):
            pass

        def encode_images(self, paths, batch_size=0, progress=None):
            assert paths == fake_paths
            return np.arange(len(paths) * 4, dtype=np.float32).reshape(len(paths), 4)

    monkeypatch.setattr(dataset_embed, "SigLIP2Encoder", _FakeEncoder)

    out = tmp_path / "embs.npz"
    result = dataset_embed.encode_folder(fake_paths, out)

    assert result.shape == (3, 4)
    loaded = np.load(out, allow_pickle=False)
    assert list(loaded["paths"]) == fake_paths
    np.testing.assert_array_equal(loaded["embeddings"], result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_dataset_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dataset_embed'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dataset_embed.py
"""Encode an arbitrary folder of images through the production SigLIP-2
encoder and cache the result. Shared by aadb_setup.py and unsplash_setup.py
so both one-time dataset scripts use the exact same embedding space grading
already uses — never a foreign feature space.

SigLIP2Encoder already isolates all GPU work in its own subprocess
(src/siglip2_encoder.py), so this module — and any script that imports it —
never touches torch.cuda directly.
"""
from pathlib import Path
import numpy as np

from siglip2_encoder import SigLIP2Encoder


def encode_folder(image_paths: list, out_npz: "Path | str", progress=None) -> np.ndarray:
    """Encode image_paths, write {paths, embeddings} to out_npz, return the
    (N, D) embedding array. Re-encodes every call — callers that want caching
    across runs should check out_npz.exists() themselves before calling."""
    out_npz = Path(out_npz)
    enc = SigLIP2Encoder(device="auto", progress=progress)
    embeddings = enc.encode_images(list(image_paths), progress=progress)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, paths=np.array(image_paths, dtype=object), embeddings=embeddings)
    return embeddings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_dataset_embed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dataset_embed.py tests/test_dataset_embed.py
git commit -m "feat: shared SigLIP-2 folder-encoding helper for external datasets"
```

---

### Task 2: AADB acquisition + reproducible split

**Files:**
- Create: `aadb_setup.py` (repo root, mirrors `nima_setup.py`'s location and one-time-script pattern)
- Test: `tests/test_aadb_setup.py`

**Interfaces:**
- Consumes: nothing from earlier tasks yet (pure data-wrangling).
- Produces: `aadb_setup.load_labels(archive_dir: Path) -> list[tuple[str, float]]` (image path, aesthetic score in `[0, 1]`), `aadb_setup.split_labels(labels: list, seed: int = 20260911, frac_holdout: float = 0.2) -> tuple[list, list]` (train, holdout) — used by Task 3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aadb_setup.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_load_labels_parses_csv(tmp_path):
    import aadb_setup

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "ImageFile,score\n"
        "img001.jpg,0.72\n"
        "img002.jpg,0.31\n",
        encoding="utf-8",
    )
    (tmp_path / "img001.jpg").write_bytes(b"\xff\xd8\xff")  # not a real jpeg, just needs to exist
    (tmp_path / "img002.jpg").write_bytes(b"\xff\xd8\xff")

    labels = aadb_setup.load_labels(tmp_path)

    assert len(labels) == 2
    assert labels[0][0].endswith("img001.jpg")
    assert abs(labels[0][1] - 0.72) < 1e-9


def test_split_is_deterministic_and_proportional():
    import aadb_setup

    labels = [(f"img{i}.jpg", i / 100.0) for i in range(100)]

    train_a, holdout_a = aadb_setup.split_labels(labels, seed=20260911, frac_holdout=0.2)
    train_b, holdout_b = aadb_setup.split_labels(labels, seed=20260911, frac_holdout=0.2)

    assert len(holdout_a) == 20
    assert len(train_a) == 80
    assert [p for p, _ in train_a] == [p for p, _ in train_b]
    assert [p for p, _ in holdout_a] == [p for p, _ in holdout_b]
    assert set(p for p, _ in train_a).isdisjoint(set(p for p, _ in holdout_a))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aadb_setup'`

- [ ] **Step 3: Write minimal implementation**

```python
# aadb_setup.py (repo root)
"""One-time AADB acquisition + head trainer.

AADB (Aesthetics and Attributes Database, Kong et al., ECCV 2016) is
distributed by the paper authors — see https://ylkong.github.io/publication/aadb
for the current download location; it moves between hosts over time, which
is exactly why this script does NOT hardcode a URL. Download the archive
yourself, unzip it, and point AADB_ARCHIVE_DIR at the folder containing the
images and the score CSV.

Usage:
    set AADB_ARCHIVE_DIR=C:\path\to\aadb
    venv\Scripts\python.exe aadb_setup.py
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

_HEAD_OUT = _ROOT / "models" / "aadb_head.npz"
_METRICS_OUT = _ROOT / "cache" / "aadb_head_metrics.json"
_SPLIT_SEED = 20260911


def load_labels(archive_dir: "Path | str") -> list:
    """Parse the AADB score CSV into [(absolute_image_path, score_0_1), ...].
    Expects a CSV with an image-filename column and a score column; AADB's
    published labels are typically in [1, 10] or already normalised — this
    function normalises to [0, 1] if it detects a >1 max."""
    archive_dir = Path(archive_dir)
    csv_path = next(archive_dir.glob("*.csv"))
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fname_key = next(k for k in reader.fieldnames if "file" in k.lower() or "image" in k.lower())
        score_key = next(k for k in reader.fieldnames if "score" in k.lower())
        for row in reader:
            img_path = archive_dir / row[fname_key]
            if not img_path.exists():
                continue
            rows.append((str(img_path), float(row[score_key])))
    if not rows:
        raise RuntimeError(f"No labeled images found under {archive_dir} — check AADB_ARCHIVE_DIR")
    max_score = max(s for _, s in rows)
    if max_score > 1.0:
        rows = [(p, s / max_score) for p, s in rows]
    return rows


def split_labels(labels: list, seed: int = _SPLIT_SEED, frac_holdout: float = 0.2) -> tuple:
    """Deterministic (same seed -> same split) train/holdout partition."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n_holdout = int(round(len(labels) * frac_holdout))
    holdout_idx = set(idx[:n_holdout].tolist())
    train, holdout = [], []
    for i, item in enumerate(labels):
        (holdout if i in holdout_idx else train).append(item)
    return train, holdout


if __name__ == "__main__":
    archive = os.environ.get("AADB_ARCHIVE_DIR")
    if not archive:
        print("Set AADB_ARCHIVE_DIR to the unzipped AADB folder first. See this "
              "file's docstring for where to obtain the archive.")
        sys.exit(1)
    labels = load_labels(archive)
    train, holdout = split_labels(labels)
    print(f"[aadb_setup] {len(labels)} labeled images -> {len(train)} train / {len(holdout)} holdout")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_setup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aadb_setup.py tests/test_aadb_setup.py
git commit -m "feat: AADB label loading + deterministic train/holdout split"
```

---

### Task 3: Train the AADB head, report held-out ρ

**Files:**
- Modify: `aadb_setup.py` (add training + reporting to the `__main__` block, plus a testable `fit_head` function)
- Test: `tests/test_aadb_setup.py` (append)

**Interfaces:**
- Consumes: `dataset_embed.encode_folder` (Task 1), `aadb_setup.load_labels`/`split_labels` (Task 2), `master_judge.spearman` (existing, `src/master_judge.py:112`).
- Produces: `aadb_setup.fit_head(X_train, y_train, X_holdout, y_holdout, lam=1.0) -> dict` with keys `coef`, `intercept`, `mean`, `std`, `rho_holdout`, `n_train`, `n_holdout` — consumed by Task 4 (`aadb_scorer.py`) via the saved `.npz`/`.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aadb_setup.py (append)

def test_fit_head_recovers_linear_signal():
    import numpy as np
    import aadb_setup

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    true_w = rng.normal(size=8)
    y = X @ true_w
    y = (y - y.min()) / (y.max() - y.min())  # normalise to [0,1] like AADB scores

    X_train, X_hold = X[:160], X[160:]
    y_train, y_hold = y[:160], y[160:]

    result = aadb_setup.fit_head(X_train, y_train, X_hold, y_hold, lam=0.1)

    assert result["rho_holdout"] > 0.9   # near-perfect linear signal should be recovered
    assert result["n_train"] == 160
    assert result["n_holdout"] == 40


def test_fit_head_refuses_degenerate_holdout():
    import numpy as np
    import aadb_setup

    X_train = np.random.default_rng(1).normal(size=(50, 4))
    y_train = np.random.default_rng(1).normal(size=50)
    X_hold = np.random.default_rng(1).normal(size=(2, 4))  # too few for a real rho
    y_hold = np.array([0.5, 0.5])

    import math
    result = aadb_setup.fit_head(X_train, y_train, X_hold, y_hold, lam=1.0)
    assert math.isnan(result["rho_holdout"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_setup.py -v`
Expected: FAIL — `AttributeError: module 'aadb_setup' has no attribute 'fit_head'`

- [ ] **Step 3: Write minimal implementation**

```python
# aadb_setup.py — add near the top, after the existing imports:
sys.path.insert(0, str(_ROOT / "src"))  # already present from Task 2; keep single insert

# add these functions above the __main__ block:

def fit_head(X_train, y_train, X_holdout, y_holdout, lam: float = 1.0) -> dict:
    """Closed-form ridge fit (same math as master_judge._ridge_fit, kept
    self-contained here rather than importing a private helper across
    modules). Returns the fitted model plus its held-out Spearman rho."""
    import numpy as np
    from master_judge import spearman

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Xs = (X_train - mean) / std
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d)
    coef = np.linalg.solve(A, Xs.T @ y_train)
    intercept = float(y_train.mean() - Xs.mean(axis=0) @ coef)

    X_holdout = np.asarray(X_holdout, dtype=np.float64)
    y_holdout = np.asarray(y_holdout, dtype=np.float64)
    preds = (X_holdout - mean) / std @ coef + intercept
    rho = spearman(preds, y_holdout)  # NaN for n<3 or a constant side — see master_judge.spearman

    return {
        "coef": coef, "intercept": intercept, "mean": mean, "std": std,
        "rho_holdout": rho, "n_train": int(len(y_train)), "n_holdout": int(len(y_holdout)),
    }


def _save_head(result: dict) -> None:
    import json
    import numpy as np
    _HEAD_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_HEAD_OUT, coef=result["coef"], intercept=result["intercept"],
              mean=result["mean"], std=result["std"])
    _METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_OUT.write_text(json.dumps({
        "rho_holdout": None if np.isnan(result["rho_holdout"]) else round(float(result["rho_holdout"]), 4),
        "n_train": result["n_train"], "n_holdout": result["n_holdout"],
    }, indent=2), encoding="utf-8")
```

```python
# aadb_setup.py — replace the __main__ block from Task 2 with:

if __name__ == "__main__":
    import dataset_embed

    archive = os.environ.get("AADB_ARCHIVE_DIR")
    if not archive:
        print("Set AADB_ARCHIVE_DIR to the unzipped AADB folder first. See this "
              "file's docstring for where to obtain the archive.")
        sys.exit(1)

    labels = load_labels(archive)
    train, holdout = split_labels(labels)
    print(f"[aadb_setup] {len(labels)} labeled images -> {len(train)} train / {len(holdout)} holdout")

    train_paths = [p for p, _ in train]
    holdout_paths = [p for p, _ in holdout]
    X_train = dataset_embed.encode_folder(train_paths, _ROOT / "cache" / "aadb_train_embs.npz")
    X_holdout = dataset_embed.encode_folder(holdout_paths, _ROOT / "cache" / "aadb_holdout_embs.npz")
    y_train = np.array([s for _, s in train], dtype=np.float64)
    y_holdout = np.array([s for _, s in holdout], dtype=np.float64)

    result = fit_head(X_train, y_train, X_holdout, y_holdout, lam=1.0)
    print(f"[aadb_setup] held-out rho = {result['rho_holdout']:.4f} "
          f"(n_train={result['n_train']}, n_holdout={result['n_holdout']})")
    if result["rho_holdout"] != result["rho_holdout"] or result["rho_holdout"] < 0.15:
        print("[aadb_setup] rho is NaN or below the +0.149 chance-level floor "
              "(project_grading_measurement_bounds) — refusing to save a head "
              "that can't beat chance on its own labeled data.")
        sys.exit(1)
    _save_head(result)
    print(f"[aadb_setup] saved {_HEAD_OUT} and {_METRICS_OUT}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_setup.py -v`
Expected: PASS (all four tests in the file)

- [ ] **Step 5: Commit**

```bash
git add aadb_setup.py tests/test_aadb_setup.py
git commit -m "feat: fit AADB aesthetic head, refuse to save below chance-level rho"
```

---

### Task 4: AADB scorer module (degrade-gracefully)

**Files:**
- Create: `src/aadb_scorer.py`
- Test: `tests/test_aadb_scorer.py`

**Interfaces:**
- Consumes: `models/aadb_head.npz` (written by Task 3).
- Produces: `aadb_scorer.score(embeddings: np.ndarray) -> "np.ndarray | None"` — `None` whenever the model file is absent/unreadable (same contract as `nima_scorer.nima_scores`, `src/nima_scorer.py:26-47`). Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aadb_scorer.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_score_returns_none_when_model_absent(tmp_path, monkeypatch):
    import aadb_scorer
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", tmp_path / "missing.npz")
    result = aadb_scorer.score(np.zeros((3, 4), dtype=np.float32))
    assert result is None


def test_score_matches_manual_ridge_predict(tmp_path, monkeypatch):
    import aadb_scorer

    coef = np.array([0.5, -0.25], dtype=np.float64)
    intercept = 0.1
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 1.0])
    head_path = tmp_path / "aadb_head.npz"
    np.savez(head_path, coef=coef, intercept=intercept, mean=mean, std=std)
    monkeypatch.setattr(aadb_scorer, "_HEAD_PATH", head_path)

    embs = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    result = aadb_scorer.score(embs)

    expected = (embs - mean) / std @ coef + intercept
    np.testing.assert_allclose(result, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aadb_scorer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aadb_scorer.py
"""AADB-trained aesthetic head scorer.

Loads the tiny ridge model produced by aadb_setup.py and scores SigLIP-2
embeddings already resident in the grade worker's memory — no extra encode
pass, unlike NIMA which needs its own image decode/forward pass.

None is returned whenever models/aadb_head.npz is absent or unreadable:
callers must degrade to leaving the "AADB" breakdown key unset, never to a
fabricated 0.0 (see master_judge.feature_vector's NaN-not-zero contract).
"""
from pathlib import Path
import numpy as np

_HEAD_PATH = Path(__file__).resolve().parent.parent / "models" / "aadb_head.npz"


def score(embeddings: np.ndarray) -> "np.ndarray | None":
    if not _HEAD_PATH.exists():
        return None
    try:
        d = np.load(_HEAD_PATH, allow_pickle=False)
        coef, intercept, mean, std = d["coef"], float(d["intercept"]), d["mean"], d["std"]
    except Exception as e:
        print(f"[aadb_scorer] model unreadable ({e}) — keeping the AADB feature unset")
        return None
    embeddings = np.asarray(embeddings, dtype=np.float64)
    return (embeddings - mean) / std @ coef + intercept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/aadb_scorer.py tests/test_aadb_scorer.py
git commit -m "feat: aadb_scorer.py — degrade-gracefully scorer for the AADB head"
```

---

### Task 5: Wire the AADB feature into grading

**Files:**
- Modify: `src/grade_pipeline_v2.py` (add a new step immediately after the existing Step 4a-NIMA block)
- Test: `tests/test_grade_pipeline_aadb_step.py`

**Interfaces:**
- Consumes: `aadb_scorer.score` (Task 4), the existing `embs` array and `to_rate_indices`/`paths_to_rate` locals already present at the Step 4a-NIMA site, `per_photo_breakdowns` (existing list of dicts).
- Produces: `per_photo_breakdowns[idx]["AADB"]` populated for every rated photo whenever `aadb_scorer.score` returns non-`None`.

- [ ] **Step 1: Write the failing test**

Since `run_v2()` is a single very large function, this step is tested by
extracting the new block's logic into a small, independently-testable
function rather than invoking the whole pipeline.

```python
# tests/test_grade_pipeline_aadb_step.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_apply_aadb_step_writes_breakdown_key(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    to_rate_indices = [0, 1]
    breakdowns = [{}, {}]

    monkeypatch.setattr(
        "aadb_scorer.score",
        lambda e: np.array([0.7, 0.3], dtype=np.float32),
    )

    gp._apply_aadb_step(embs, to_rate_indices, breakdowns)

    assert breakdowns[0]["AADB"] == 0.7
    assert breakdowns[1]["AADB"] == 0.3


def test_apply_aadb_step_no_op_when_model_absent(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.zeros((2, 2), dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("aadb_scorer.score", lambda e: None)

    gp._apply_aadb_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert "AADB" not in breakdowns[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_grade_pipeline_aadb_step.py -v`
Expected: FAIL — `AttributeError: module 'grade_pipeline_v2' has no attribute '_apply_aadb_step'`

- [ ] **Step 3: Write minimal implementation**

Add this function near the top-level helpers in `grade_pipeline_v2.py`
(alongside other `_apply_*`/`_p`-style module-level helpers), then call it
from `run_v2()` right after the existing Step 4a-NIMA block.

```python
# src/grade_pipeline_v2.py — new module-level function

def _apply_aadb_step(embs, to_rate_indices, per_photo_breakdowns) -> None:
    """Step 4a-AADB: score already-computed SigLIP-2 embeddings through the
    AADB-trained head and write into per_photo_breakdowns. No-op (leaves the
    "AADB" key unset) whenever the model file is absent — see aadb_scorer's
    degrade-gracefully contract."""
    import numpy as np
    import aadb_scorer
    if not to_rate_indices:
        return
    idx_arr = np.asarray(to_rate_indices, dtype=np.intp)
    result = aadb_scorer.score(embs[idx_arr])
    if result is None:
        return
    for local_i, idx in enumerate(to_rate_indices):
        per_photo_breakdowns[idx]["AADB"] = round(float(result[local_i]), 3)
```

```python
# src/grade_pipeline_v2.py — inside run_v2(), immediately after the
# existing Step 4a-NIMA try/except block (the one ending with
# "keeping CLIP aesthetic"):

    _apply_aadb_step(embs, to_rate_indices, per_photo_breakdowns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_grade_pipeline_aadb_step.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to confirm nothing else broke, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: PASS (no new failures)

```bash
git add src/grade_pipeline_v2.py tests/test_grade_pipeline_aadb_step.py
git commit -m "feat: wire the AADB feature into run_v2 as Step 4a-AADB"
```

---

### Task 6: Add "AADB" to the Master Judge's FEATURES

**Files:**
- Modify: `src/master_judge.py:56-57`
- Test: `tests/test_master_judge_features.py`

**Interfaces:**
- Consumes: nothing new — this task only changes the existing `FEATURES` constant and confirms downstream effects.
- Produces: `master_judge.FEATURES` now includes `"AADB"`; `master_judge.feature_vector()` picks it up automatically since it already iterates `FEATURES` generically (`master_judge.py:133`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_master_judge_features.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np


def test_aadb_is_a_feature():
    import master_judge as mj
    assert "AADB" in mj.FEATURES


def test_feature_vector_reads_aadb_when_present():
    import master_judge as mj
    bd = {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
          "Narrative": 0.5, "Human/Culture": 0.5, "AADB": 0.8}
    vec = mj.feature_vector(bd)
    idx = mj.FEATURES.index("AADB")
    assert vec[idx] == 0.8


def test_feature_vector_is_nan_when_aadb_missing():
    import master_judge as mj
    bd = {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
          "Narrative": 0.5, "Human/Culture": 0.5}  # no "AADB" key
    vec = mj.feature_vector(bd)
    idx = mj.FEATURES.index("AADB")
    assert np.isnan(vec[idx])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_judge_features.py -v`
Expected: FAIL — `assert "AADB" in mj.FEATURES` fails (not yet added)

- [ ] **Step 3: Write minimal implementation**

```python
# src/master_judge.py:56-57 — change from:
FEATURES = ["Technical", "Composition", "Lighting",
            "Narrative", "Human/Culture"]
# to:
FEATURES = ["Technical", "Composition", "Lighting",
            "Narrative", "Human/Culture", "AADB"]
```

No other change needed — `feature_vector()` (line 127-137) and `DESIGN`
(line 73-74) already derive from `FEATURES` generically, and
`_feature_fingerprint()` (line 261-263) already hashes `DESIGN`, so this
one-line change automatically invalidates the old cached fingerprint and
forces the next `master_backtest.py --fit` to train a fresh model — exactly
the intended behavior per `master_judge.py`'s own comment at line 53-55.

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_judge_features.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: PASS

```bash
git add src/master_judge.py tests/test_master_judge_features.py
git commit -m "feat: add AADB to the Master Judge's FEATURES"
```

---

### Task 7: Surface the AADB held-out ρ in the backtest report

**Files:**
- Modify: `scripts/master_backtest.py`
- Test: `tests/test_master_backtest_aadb_report.py`

**Interfaces:**
- Consumes: `cache/aadb_head_metrics.json` (written by Task 3).
- Produces: `master_backtest.read_aadb_metrics() -> "dict | None"` — used by both this task's report printing and Task 8's promotion gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_master_backtest_aadb_report.py
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_read_aadb_metrics_returns_none_when_absent(tmp_path, monkeypatch):
    import importlib
    mb = importlib.import_module("scripts.master_backtest")
    monkeypatch.setattr(mb, "_AADB_METRICS_PATH", tmp_path / "missing.json")
    assert mb.read_aadb_metrics() is None


def test_read_aadb_metrics_parses_existing_file(tmp_path, monkeypatch):
    import importlib
    mb = importlib.import_module("scripts.master_backtest")
    metrics_path = tmp_path / "aadb_head_metrics.json"
    metrics_path.write_text(json.dumps({"rho_holdout": 0.42, "n_train": 8000, "n_holdout": 2000}), encoding="utf-8")
    monkeypatch.setattr(mb, "_AADB_METRICS_PATH", metrics_path)

    result = mb.read_aadb_metrics()
    assert result["rho_holdout"] == 0.42
    assert result["n_holdout"] == 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_backtest_aadb_report.py -v`
Expected: FAIL — `AttributeError: module 'scripts.master_backtest' has no attribute 'read_aadb_metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/master_backtest.py — add near the top, after the existing
# _ROOT/sys.path setup:

_AADB_METRICS_PATH = _ROOT / "cache" / "aadb_head_metrics.json"


def read_aadb_metrics() -> "dict | None":
    """The AADB-only held-out rho written by aadb_setup.py, or None if the
    head has never been trained."""
    import json
    if not _AADB_METRICS_PATH.exists():
        return None
    try:
        return json.loads(_AADB_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
```

```python
# scripts/master_backtest.py — inside main(), after the existing RANK
# AGREEMENT section prints its numbers, add:

    aadb = read_aadb_metrics()
    if aadb is not None:
        print(f"\nAADB (external human judgment):")
        print(f"  held-out rho = {aadb['rho_holdout']} "
              f"(n_train={aadb['n_train']}, n_holdout={aadb['n_holdout']})")
    else:
        print("\nAADB: no trained head found (run aadb_setup.py to add this check)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_backtest_aadb_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/master_backtest.py tests/test_master_backtest_aadb_report.py
git commit -m "feat: surface the AADB held-out rho in the master_backtest report"
```

---

### Task 8: Gate promotion on the AADB chance-level floor

**Files:**
- Modify: `scripts/promote_master_judge.py`
- Test: `tests/test_promote_master_judge_aadb_gate.py`

**Interfaces:**
- Consumes: `master_backtest.read_aadb_metrics` (Task 7).
- Produces: `promote_master_judge.check_aadb_gate(min_rho: float = 0.149) -> tuple[bool, str]` — `(passes, reason)`, called before the existing promotion logic proceeds.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_promote_master_judge_aadb_gate.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gate_fails_when_no_aadb_metrics(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics", lambda: None)

    passes, reason = pmj.check_aadb_gate()
    assert passes is False
    assert "no trained AADB head" in reason.lower()


def test_gate_fails_below_chance_floor(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics",
                         lambda: {"rho_holdout": 0.05, "n_train": 8000, "n_holdout": 2000})

    passes, reason = pmj.check_aadb_gate(min_rho=0.149)
    assert passes is False
    assert "0.05" in reason


def test_gate_passes_above_chance_floor(monkeypatch):
    import importlib
    pmj = importlib.import_module("scripts.promote_master_judge")
    monkeypatch.setattr("scripts.master_backtest.read_aadb_metrics",
                         lambda: {"rho_holdout": 0.42, "n_train": 8000, "n_holdout": 2000})

    passes, reason = pmj.check_aadb_gate(min_rho=0.149)
    assert passes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_promote_master_judge_aadb_gate.py -v`
Expected: FAIL — `AttributeError: module 'scripts.promote_master_judge' has no attribute 'check_aadb_gate'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/promote_master_judge.py — add near the top-level functions:

def check_aadb_gate(min_rho: float = 0.149) -> tuple:
    """(passes, reason). min_rho defaults to the chance-level rank agreement
    already measured for this project's baseline grader
    (project_grading_measurement_bounds: +0.149) — a head that can't beat
    chance on its own labeled AADB data is noise, not signal, regardless of
    what it does to the live baseline's rho_holdout."""
    import scripts.master_backtest as mb
    metrics = mb.read_aadb_metrics()
    if metrics is None:
        return False, "no trained AADB head found — run aadb_setup.py first"
    rho = metrics.get("rho_holdout")
    if rho is None or rho != rho:  # None or NaN
        return False, f"AADB held-out rho is undefined ({rho})"
    if rho < min_rho:
        return False, f"AADB held-out rho {rho} is below the chance-level floor {min_rho}"
    return True, f"AADB held-out rho {rho} clears the chance-level floor {min_rho}"
```

```python
# scripts/promote_master_judge.py — inside main(), before the existing
# promotion decision is made (wherever it currently checks
# cache/master_judge.json's "promoted" flag / rho comparison), add:

    if "AADB" in mj.FEATURES:  # only gate on this once the feature exists
        aadb_ok, aadb_reason = check_aadb_gate()
        print(f"[promote] AADB gate: {aadb_reason}")
        if not aadb_ok:
            print("[promote] REFUSING promotion — AADB gate not satisfied.")
            return 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_promote_master_judge_aadb_gate.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: PASS

```bash
git add scripts/promote_master_judge.py tests/test_promote_master_judge_aadb_gate.py
git commit -m "feat: refuse to promote a challenger unless AADB clears the chance-level floor"
```

This completes sub-project 1 end-to-end: the AADB head can be trained,
scored, wired into grading, added as a Master Judge feature, and is gated
from promotion by both the existing baseline exam and the new AADB check.
Nothing has changed in live grading yet — `models/aadb_head.npz` doesn't
exist until an operator runs `aadb_setup.py`, and `promote_master_judge.py`
still requires an explicit operator run to ship anything.

---

### Task 9: Unsplash API fetch + Strong/Weak split

**Files:**
- Create: `unsplash_setup.py` (repo root, same one-time-script pattern as `aadb_setup.py`)
- Test: `tests/test_unsplash_setup.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure data-wrangling, like Task 2).
- Produces: `unsplash_setup.fetch_collection_photos(collection_id: str, access_key: str, scratch_dir: Path, max_photos: int = 500) -> list[dict]` (downloads originals, returns `[{"path": local_jpg_path, "likes": int}, ...]`), `unsplash_setup.split_by_engagement(photos: list[dict], frac: float = 0.25) -> tuple[list[str], list[str]]` — both consumed by Task 10.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_unsplash_setup.py
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_split_by_engagement_top_and_bottom_quartile():
    import unsplash_setup

    photos = [{"path": f"p{i}.jpg", "likes": i} for i in range(100)]  # 0..99 likes
    strong, weak = unsplash_setup.split_by_engagement(photos, frac=0.25)

    assert len(strong) == 25
    assert len(weak) == 25
    assert "p99.jpg" in strong
    assert "p0.jpg" in weak
    assert set(strong).isdisjoint(set(weak))


def test_split_by_engagement_requires_same_collection_context():
    import unsplash_setup

    # Strong and weak must both come from the street/documentary pool passed
    # in — this function never mixes in an unrelated photo set.
    photos = [{"path": "a.jpg", "likes": 5}, {"path": "b.jpg", "likes": 1}]
    strong, weak = unsplash_setup.split_by_engagement(photos, frac=0.5)
    assert strong == ["a.jpg"]
    assert weak == ["b.jpg"]


def test_fetch_collection_photos_paginates_and_downloads(tmp_path, monkeypatch):
    import unsplash_setup

    page_1 = [{"id": f"id{i}", "likes": i, "urls": {"regular": f"https://example/{i}"}}
              for i in range(30)]
    page_2 = [{"id": f"id{i}", "likes": i, "urls": {"regular": f"https://example/{i}"}}
              for i in range(30, 45)]
    pages = {1: page_1, 2: page_2, 3: []}

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")
        def read(self):
            return self._payload
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=30):
        # req.full_url carries the page number as "...&page=N&..."
        import re
        m = re.search(r"page=(\d+)", req.full_url)
        page = int(m.group(1))
        return _FakeResponse(pages.get(page, []))

    downloaded = []

    def _fake_urlretrieve(url, filename):
        downloaded.append((url, filename))
        Path(filename).write_bytes(b"\xff\xd8\xff")  # stub jpeg bytes

    monkeypatch.setattr(unsplash_setup.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(unsplash_setup.urllib.request, "urlretrieve", _fake_urlretrieve)

    result = unsplash_setup.fetch_collection_photos(
        "collection123", "fake_key", tmp_path, max_photos=40)

    assert len(result) == 40
    assert result[0]["likes"] == 0
    assert Path(result[0]["path"]).exists()
    assert len(downloaded) == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_unsplash_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'unsplash_setup'`

- [ ] **Step 3: Write minimal implementation**

```python
# unsplash_setup.py (repo root)
"""One-time Unsplash exemplar-bank builder.

Pulls a curated Strong/Weak split from Street Photography / Documentary
collections via the Unsplash API (https://unsplash.com/developers) — an
access key is required; set UNSPLASH_ACCESS_KEY, and the collection(s) to
pull from via UNSPLASH_COLLECTION_IDS (comma-separated Unsplash collection
IDs — find these by browsing unsplash.com/collections and reading the ID
out of the collection's URL). Only derived embeddings are ever written to
this repo (models/exemplar_bank.npz); raw images are downloaded to a scratch
folder outside the repo and never committed, matching the same boundary
already drawn for AADB and the RAG PDFs.

Usage:
    set UNSPLASH_ACCESS_KEY=your_key_here
    set UNSPLASH_COLLECTION_IDS=317099,1114848
    venv\\Scripts\\python.exe unsplash_setup.py
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

_BANK_OUT = _ROOT / "models" / "exemplar_bank.npz"
_SCRATCH_DIR = _ROOT / "cache" / "unsplash_scratch"


def fetch_collection_photos(collection_id: str, access_key: str,
                            scratch_dir: Path, max_photos: int = 500) -> list:
    """Page through an Unsplash collection's photos (30/page, the API max),
    download each original to scratch_dir, and return
    [{"path": local_jpg_path, "likes": int}, ...]. Stops at max_photos or
    when the collection is exhausted, whichever comes first."""
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    photos = []
    page = 1
    while len(photos) < max_photos:
        url = (f"https://api.unsplash.com/collections/{collection_id}/photos"
               f"?page={page}&per_page=30&client_id={access_key}")
        req = urllib.request.Request(url, headers={"Accept-Version": "v1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        if not batch:
            break
        for item in batch:
            if len(photos) >= max_photos:
                break
            img_url = item["urls"]["regular"]
            local_path = scratch_dir / f"{item['id']}.jpg"
            if not local_path.exists():
                urllib.request.urlretrieve(img_url, str(local_path))
            photos.append({"path": str(local_path), "likes": int(item.get("likes", 0))})
        page += 1
    return photos


def split_by_engagement(photos: list, frac: float = 0.25) -> tuple:
    """photos: [{"path": str, "likes": int}, ...] all drawn from the SAME
    curated street/documentary collections (never mix in an unrelated pool —
    a weak exemplar must still be on-topic, just worse, see the spec).
    Returns (strong_paths, weak_paths): top/bottom frac by "likes"."""
    ordered = sorted(photos, key=lambda p: p["likes"], reverse=True)
    n = max(1, int(round(len(ordered) * frac)))
    strong = [p["path"] for p in ordered[:n]]
    weak = [p["path"] for p in ordered[-n:]]
    return strong, weak


if __name__ == "__main__":
    key = os.environ.get("UNSPLASH_ACCESS_KEY")
    collection_ids = os.environ.get("UNSPLASH_COLLECTION_IDS", "")
    if not key or not collection_ids:
        print("Set UNSPLASH_ACCESS_KEY and UNSPLASH_COLLECTION_IDS first. "
              "See this file's docstring.")
        sys.exit(1)

    all_photos = []
    for cid in collection_ids.split(","):
        all_photos.extend(fetch_collection_photos(cid.strip(), key, _SCRATCH_DIR))
    print(f"[unsplash_setup] fetched {len(all_photos)} photos across "
          f"{len(collection_ids.split(','))} collection(s)")

    strong_paths, weak_paths = split_by_engagement(all_photos)
    print(f"[unsplash_setup] {len(strong_paths)} strong / {len(weak_paths)} weak "
          f"(top/bottom quartile by engagement)")
    # Task 9b continues this script's __main__ block with encode + save + validate.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_unsplash_setup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add unsplash_setup.py tests/test_unsplash_setup.py
git commit -m "feat: Unsplash collection fetch + Strong/Weak engagement split"
```

---

### Task 9b: Encode the Strong pool, save the exemplar bank, validate the split

**Files:**
- Modify: `unsplash_setup.py` (complete the `__main__` block from Task 9)
- Test: `tests/test_unsplash_setup.py` (append)

**Interfaces:**
- Consumes: `dataset_embed.encode_folder` (Task 1), `unsplash_setup.split_by_engagement`/`fetch_collection_photos` (Task 9).
- Produces: `unsplash_setup.build_exemplar_bank(strong_paths: list[str], weak_paths: list[str], out_npz: Path) -> dict` with keys `"strong_self_sim"`, `"weak_self_sim"` (mean cosine similarity of each pool against the saved Strong bank) — the sanity check the spec requires (Strong must self-score higher than Weak does against the same bank). Writes `out_npz` with key `"embeddings"`, consumed by `exemplar_scorer.py` (Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_unsplash_setup.py (append)
import numpy as np


def test_build_exemplar_bank_saves_strong_embeddings_and_validates(tmp_path, monkeypatch):
    import unsplash_setup

    # Two well-separated clusters: "strong" near [1,0], "weak" near [0,1].
    fake_embeddings = {
        "s1.jpg": np.array([1.0, 0.05], dtype=np.float32),
        "s2.jpg": np.array([0.95, 0.1], dtype=np.float32),
        "w1.jpg": np.array([0.05, 1.0], dtype=np.float32),
        "w2.jpg": np.array([0.1, 0.95], dtype=np.float32),
    }

    def _fake_encode_folder(paths, out_npz, progress=None):
        return np.stack([fake_embeddings[p] for p in paths])

    monkeypatch.setattr(unsplash_setup.dataset_embed, "encode_folder", _fake_encode_folder)

    out_npz = tmp_path / "exemplar_bank.npz"
    result = unsplash_setup.build_exemplar_bank(
        strong_paths=["s1.jpg", "s2.jpg"], weak_paths=["w1.jpg", "w2.jpg"], out_npz=out_npz)

    assert out_npz.exists()
    saved = np.load(out_npz, allow_pickle=False)
    assert saved["embeddings"].shape == (2, 2)
    assert result["strong_self_sim"] > result["weak_self_sim"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_unsplash_setup.py -v`
Expected: FAIL — `AttributeError: module 'unsplash_setup' has no attribute 'build_exemplar_bank'`

- [ ] **Step 3: Write minimal implementation**

```python
# unsplash_setup.py — add near the top, after the existing imports:
import dataset_embed  # noqa: E402  (after sys.path.insert above)

# add this function above the __main__ block:

def build_exemplar_bank(strong_paths: list, weak_paths: list, out_npz: "Path | str") -> dict:
    """Encode the Strong pool, save it as the exemplar bank, and validate
    that Strong self-similarity beats Weak self-similarity against that
    bank — the sanity check the spec requires before trusting this feature.
    Does NOT save the Weak pool; it exists only for this validation."""
    import numpy as np
    out_npz = Path(out_npz)

    strong_embs = dataset_embed.encode_folder(strong_paths, out_npz)
    weak_embs = dataset_embed.encode_folder(weak_paths, out_npz.with_name("unsplash_weak_scratch.npz"))

    def _mean_self_sim(query_embs, bank_embs):
        bank_norm = bank_embs / np.linalg.norm(bank_embs, axis=1, keepdims=True)
        q_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
        return float((q_norm @ bank_norm.T).mean())

    return {
        "strong_self_sim": _mean_self_sim(strong_embs, strong_embs),
        "weak_self_sim": _mean_self_sim(weak_embs, strong_embs),
    }
```

```python
# unsplash_setup.py — replace the __main__ block's final comment
# ("# Task 9b continues...") with:

    result = build_exemplar_bank(strong_paths, weak_paths, _BANK_OUT)
    print(f"[unsplash_setup] strong self-sim={result['strong_self_sim']:.4f} "
          f"weak self-sim={result['weak_self_sim']:.4f}")
    if result["strong_self_sim"] <= result["weak_self_sim"]:
        print("[unsplash_setup] REFUSING to keep this bank — the Strong pool "
              "does not score higher than the Weak pool against its own "
              "bank, which means the engagement split isn't capturing a real "
              "quality signal for this collection.")
        _BANK_OUT.unlink(missing_ok=True)
        sys.exit(1)
    print(f"[unsplash_setup] saved {_BANK_OUT}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_unsplash_setup.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add unsplash_setup.py tests/test_unsplash_setup.py
git commit -m "feat: encode Strong pool into the exemplar bank, validate Strong > Weak self-similarity"
```

---

### Task 10: Exemplar scorer (kNN similarity, degrade-gracefully)

**Files:**
- Create: `src/exemplar_scorer.py`
- Test: `tests/test_exemplar_scorer.py`

**Interfaces:**
- Consumes: `models/exemplar_bank.npz` (Strong-pool embeddings, written by a future completion of Task 9's `__main__`).
- Produces: `exemplar_scorer.score(embeddings: np.ndarray, k: int = 8) -> "np.ndarray | None"` — mean cosine similarity to the k nearest Strong exemplars, in `[-1, 1]`. Consumed by Task 11.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exemplar_scorer.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_score_returns_none_when_bank_absent(tmp_path, monkeypatch):
    import exemplar_scorer
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", tmp_path / "missing.npz")
    assert exemplar_scorer.score(np.zeros((2, 4))) is None


def test_score_ranks_closer_vector_higher(tmp_path, monkeypatch):
    import exemplar_scorer

    bank = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    bank_path = tmp_path / "exemplar_bank.npz"
    np.savez(bank_path, embeddings=bank)
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", bank_path)

    close_to_bank = np.array([[0.9, 0.1]], dtype=np.float32)   # near [1,0] cluster
    far_from_bank = np.array([[-1.0, -1.0]], dtype=np.float32)  # near neither

    close_score = exemplar_scorer.score(close_to_bank, k=2)[0]
    far_score = exemplar_scorer.score(far_from_bank, k=2)[0]

    assert close_score > far_score


def test_strong_pool_scores_higher_than_weak_pool_against_own_bank(tmp_path, monkeypatch):
    """Sanity check from the spec: the Strong pool must score higher
    self-similarity against the Strong bank than the Weak pool does."""
    import exemplar_scorer

    strong_bank = np.array([[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]], dtype=np.float32)
    bank_path = tmp_path / "exemplar_bank.npz"
    np.savez(bank_path, embeddings=strong_bank)
    monkeypatch.setattr(exemplar_scorer, "_BANK_PATH", bank_path)

    weak_pool = np.array([[0.0, 1.0], [-0.1, 0.9]], dtype=np.float32)
    strong_pool_holdout = np.array([[0.92, 0.08], [0.97, 0.03]], dtype=np.float32)

    weak_scores = exemplar_scorer.score(weak_pool, k=2)
    strong_scores = exemplar_scorer.score(strong_pool_holdout, k=2)

    assert strong_scores.mean() > weak_scores.mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_exemplar_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'exemplar_scorer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/exemplar_scorer.py
"""Strong-exemplar kNN similarity scorer.

Scores SigLIP-2 embeddings by mean cosine similarity to the k nearest
neighbors in a curated Strong-pool bank (Unsplash-derived, see
unsplash_setup.py). Plays to SigLIP's actual strength — image-image
similarity — rather than the project's existing image-text zero-shot probes.

None is returned whenever models/exemplar_bank.npz is absent, matching the
same degrade-gracefully contract as nima_scorer.py and aadb_scorer.py.
"""
from pathlib import Path
import numpy as np

_BANK_PATH = Path(__file__).resolve().parent.parent / "models" / "exemplar_bank.npz"


def score(embeddings: np.ndarray, k: int = 8) -> "np.ndarray | None":
    if not _BANK_PATH.exists():
        return None
    try:
        bank = np.load(_BANK_PATH, allow_pickle=False)["embeddings"].astype(np.float64)
    except Exception as e:
        print(f"[exemplar_scorer] bank unreadable ({e}) — keeping the Exemplar feature unset")
        return None

    embeddings = np.asarray(embeddings, dtype=np.float64)
    bank_norms = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    emb_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = emb_norms @ bank_norms.T   # (N, bank_size) cosine similarity
    k_eff = min(k, sims.shape[1])
    top_k = np.sort(sims, axis=1)[:, -k_eff:]
    return top_k.mean(axis=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_exemplar_scorer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/exemplar_scorer.py tests/test_exemplar_scorer.py
git commit -m "feat: exemplar_scorer.py — Strong-pool kNN similarity, degrade-gracefully"
```

---

### Task 11: Wire the Exemplar feature into grading

**Files:**
- Modify: `src/grade_pipeline_v2.py` (new step immediately after `_apply_aadb_step`'s call site from Task 5)
- Test: `tests/test_grade_pipeline_exemplar_step.py`

**Interfaces:**
- Consumes: `exemplar_scorer.score` (Task 10), same `embs`/`to_rate_indices`/`per_photo_breakdowns` locals as Task 5.
- Produces: `per_photo_breakdowns[idx]["Exemplar"]` populated for every rated photo whenever `exemplar_scorer.score` returns non-`None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade_pipeline_exemplar_step.py
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_apply_exemplar_step_writes_breakdown_key(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: np.array([0.6, 0.2], dtype=np.float32))

    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert breakdowns[0]["Exemplar"] == 0.6
    assert breakdowns[1]["Exemplar"] == 0.2


def test_apply_exemplar_step_no_op_when_bank_absent(monkeypatch):
    import grade_pipeline_v2 as gp

    embs = np.zeros((2, 2), dtype=np.float32)
    breakdowns = [{}, {}]
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: None)

    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "Exemplar" not in breakdowns[0]
    assert "Exemplar" not in breakdowns[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_grade_pipeline_exemplar_step.py -v`
Expected: FAIL — `AttributeError: module 'grade_pipeline_v2' has no attribute '_apply_exemplar_step'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/grade_pipeline_v2.py — new module-level function, next to _apply_aadb_step:

def _apply_exemplar_step(embs, to_rate_indices, per_photo_breakdowns) -> None:
    """New step: score already-computed SigLIP-2 embeddings against the
    Unsplash-derived Strong-exemplar bank. No-op whenever the bank is
    absent — see exemplar_scorer's degrade-gracefully contract."""
    import numpy as np
    import exemplar_scorer
    if not to_rate_indices:
        return
    idx_arr = np.asarray(to_rate_indices, dtype=np.intp)
    result = exemplar_scorer.score(embs[idx_arr])
    if result is None:
        return
    for local_i, idx in enumerate(to_rate_indices):
        per_photo_breakdowns[idx]["Exemplar"] = round(float(result[local_i]), 3)
```

```python
# src/grade_pipeline_v2.py — inside run_v2(), immediately after the
# _apply_aadb_step(...) call added in Task 5:

    _apply_exemplar_step(embs, to_rate_indices, per_photo_breakdowns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_grade_pipeline_exemplar_step.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: PASS

```bash
git add src/grade_pipeline_v2.py tests/test_grade_pipeline_exemplar_step.py
git commit -m "feat: wire the Exemplar feature into run_v2"
```

---

### Task 12: Add "Exemplar" to the Master Judge's FEATURES

**Files:**
- Modify: `src/master_judge.py:56-57`
- Test: `tests/test_master_judge_features.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `master_judge.FEATURES` now includes both `"AADB"` and `"Exemplar"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_master_judge_features.py (append)

def test_exemplar_is_a_feature():
    import master_judge as mj
    assert "Exemplar" in mj.FEATURES


def test_feature_order_is_aadb_then_exemplar():
    """DESIGN order is part of the saved-weights fingerprint
    (master_judge.py:53-55) — pin the order explicitly so a future edit
    that silently reorders FEATURES is caught here, not by a confusing
    fingerprint mismatch somewhere else."""
    import master_judge as mj
    assert mj.FEATURES.index("AADB") < mj.FEATURES.index("Exemplar")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_judge_features.py -v`
Expected: FAIL — `assert "Exemplar" in mj.FEATURES` fails

- [ ] **Step 3: Write minimal implementation**

```python
# src/master_judge.py:56-57 — change from:
FEATURES = ["Technical", "Composition", "Lighting",
            "Narrative", "Human/Culture", "AADB"]
# to:
FEATURES = ["Technical", "Composition", "Lighting",
            "Narrative", "Human/Culture", "AADB", "Exemplar"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_master_judge_features.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite, then commit**

Run: `venv\Scripts\python.exe -m pytest tests/ -x -q`
Expected: PASS

```bash
git add src/master_judge.py tests/test_master_judge_features.py
git commit -m "feat: add Exemplar to the Master Judge's FEATURES"
```

---

### Task 13: End-to-end degrade-path integration test

**Files:**
- Create: `tests/test_aadb_unsplash_integration.py`

**Interfaces:**
- Consumes: `grade_pipeline_v2._apply_aadb_step`, `grade_pipeline_v2._apply_exemplar_step` (Tasks 5, 11), `master_judge.feature_vector` (existing).
- Produces: nothing new — this is a pure verification task confirming the two features compose correctly and the whole chain degrades safely with no model artifacts present (the state of a fresh clone).

- [ ] **Step 1: Write the test**

```python
# tests/test_aadb_unsplash_integration.py
"""Confirms the full chain — grading step -> breakdown key ->
master_judge.feature_vector -> ridge design matrix — behaves correctly both
with and without the new model artifacts present. This is the state of a
fresh clone (no models/aadb_head.npz, no models/exemplar_bank.npz) and must
grade exactly as it did before this plan, per the Global Constraints."""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_fresh_clone_state_degrades_to_prior_behavior(monkeypatch):
    import grade_pipeline_v2 as gp
    import master_judge as mj

    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    breakdowns = [
        {"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5, "Narrative": 0.5, "Human/Culture": 0.5},
        {"Technical": 0.6, "Composition": 0.6, "Lighting": 0.6, "Narrative": 0.6, "Human/Culture": 0.6},
    ]

    monkeypatch.setattr("aadb_scorer.score", lambda e: None)      # fresh-clone state
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: None)  # fresh-clone state

    gp._apply_aadb_step(embs, [0, 1], breakdowns)
    gp._apply_exemplar_step(embs, [0, 1], breakdowns)

    assert "AADB" not in breakdowns[0]
    assert "Exemplar" not in breakdowns[0]

    vec = mj.feature_vector(breakdowns[0])
    assert np.isnan(vec[mj.FEATURES.index("AADB")])
    assert np.isnan(vec[mj.FEATURES.index("Exemplar")])
    # the five original features are untouched
    for name in ["Technical", "Composition", "Lighting", "Narrative", "Human/Culture"]:
        assert not np.isnan(vec[mj.FEATURES.index(name)])


def test_both_models_present_populate_both_features(monkeypatch):
    import grade_pipeline_v2 as gp
    import master_judge as mj

    embs = np.array([[1.0, 0.0]], dtype=np.float32)
    breakdowns = [{"Technical": 0.5, "Composition": 0.5, "Lighting": 0.5,
                   "Narrative": 0.5, "Human/Culture": 0.5}]

    monkeypatch.setattr("aadb_scorer.score", lambda e: np.array([0.77], dtype=np.float32))
    monkeypatch.setattr("exemplar_scorer.score", lambda e, k=8: np.array([0.33], dtype=np.float32))

    gp._apply_aadb_step(embs, [0], breakdowns)
    gp._apply_exemplar_step(embs, [0], breakdowns)

    vec = mj.feature_vector(breakdowns[0])
    assert vec[mj.FEATURES.index("AADB")] == 0.77
    assert vec[mj.FEATURES.index("Exemplar")] == 0.33
```

- [ ] **Step 2: Run test to verify it passes**

Run: `venv\Scripts\python.exe -m pytest tests/test_aadb_unsplash_integration.py -v`
Expected: PASS (both tests — this task has no separate "make it fail first"
step since it composes only already-implemented pieces from Tasks 5-12)

- [ ] **Step 3: Run the entire test suite one final time**

Run: `venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero failures

- [ ] **Step 4: Commit**

```bash
git add tests/test_aadb_unsplash_integration.py
git commit -m "test: end-to-end degrade-path coverage for AADB + Exemplar features"
```

This closes both sub-projects. A fresh clone with no `models/aadb_head.npz`
or `models/exemplar_bank.npz` grades identically to before this plan. An
operator who runs `aadb_setup.py` and/or `unsplash_setup.py`, then
`scripts/master_backtest.py --fit`, gets a challenger scored against both
this project's rating baseline and (for AADB) an external human-judgment
benchmark — and `scripts/promote_master_judge.py` still refuses to ship it
unless it wins.
