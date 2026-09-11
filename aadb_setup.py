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
