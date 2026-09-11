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
