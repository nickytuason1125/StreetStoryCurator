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


def test_load_labels_normalizes_unnormalized_scores(tmp_path):
    """Test normalization of 1-10 range scores to [0, 1]."""
    import aadb_setup

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "ImageFile,score\n"
        "img001.jpg,5.5\n"
        "img002.jpg,8.0\n"
        "img003.jpg,2.1\n",
        encoding="utf-8",
    )
    (tmp_path / "img001.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "img002.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "img003.jpg").write_bytes(b"\xff\xd8\xff")

    labels = aadb_setup.load_labels(tmp_path)

    # max_score in CSV is 8.0, so scores should be divided by 8.0
    assert len(labels) == 3
    assert abs(labels[0][1] - 5.5 / 8.0) < 1e-9  # 0.6875
    assert abs(labels[1][1] - 8.0 / 8.0) < 1e-9  # 1.0
    assert abs(labels[2][1] - 2.1 / 8.0) < 1e-9  # 0.2625
    # All scores should be in [0, 1]
    for path, score in labels:
        assert 0.0 <= score <= 1.0


def test_load_labels_max_computed_before_filter(tmp_path):
    """Test that max_score is computed from all CSV rows before filtering by file existence.

    This ensures consistent normalization even if some images are missing locally."""
    import aadb_setup

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "ImageFile,score\n"
        "img001.jpg,5.0\n"
        "img_missing.jpg,9.0\n"
        "img003.jpg,3.0\n",
        encoding="utf-8",
    )
    # Only create two of three images; img_missing.jpg is not present
    (tmp_path / "img001.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "img003.jpg").write_bytes(b"\xff\xd8\xff")

    labels = aadb_setup.load_labels(tmp_path)

    # max_score from ALL CSV rows should be 9.0, not 5.0
    # So scores are normalized by 9.0, not 5.0
    assert len(labels) == 2  # only img001 and img003 exist
    assert abs(labels[0][1] - 5.0 / 9.0) < 1e-9  # 0.555...
    assert abs(labels[1][1] - 3.0 / 9.0) < 1e-9  # 0.333...


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
