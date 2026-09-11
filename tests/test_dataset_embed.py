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
