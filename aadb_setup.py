r"""One-time AADB acquisition + head trainer.

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
    function normalises to [0, 1] if it detects a >1 max.

    Note: max_score is computed from ALL CSV rows before filtering by file
    existence, ensuring consistent normalization across different local archives."""
    archive_dir = Path(archive_dir)
    csv_path = next(archive_dir.glob("*.csv"))
    all_scores = []  # scores from all CSV rows (for normalization)
    rows = []  # only rows whose image files exist
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fname_key = next(k for k in reader.fieldnames if "file" in k.lower() or "image" in k.lower())
        score_key = next(k for k in reader.fieldnames if "score" in k.lower())
        for row in reader:
            score = float(row[score_key])
            all_scores.append(score)
            img_path = archive_dir / row[fname_key]
            if not img_path.exists():
                continue
            rows.append((str(img_path), score))
    if not rows:
        raise RuntimeError(f"No labeled images found under {archive_dir} — check AADB_ARCHIVE_DIR")
    max_score = max(all_scores)
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


def fit_head(X_train, y_train, X_holdout, y_holdout, lam: float = 1.0) -> dict:
    """Closed-form ridge fit (same math as master_judge._ridge_fit, kept
    self-contained here rather than importing a private helper across
    modules). Returns the fitted model plus its held-out Spearman rho."""
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
    _HEAD_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_HEAD_OUT, coef=result["coef"], intercept=result["intercept"],
              mean=result["mean"], std=result["std"])
    _METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_OUT.write_text(json.dumps({
        "rho_holdout": None if np.isnan(result["rho_holdout"]) else round(float(result["rho_holdout"]), 4),
        "n_train": result["n_train"], "n_holdout": result["n_holdout"],
    }, indent=2), encoding="utf-8")


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
