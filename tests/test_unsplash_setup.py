import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


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


def test_split_by_engagement_disjoint_on_small_pools():
    import unsplash_setup

    # A single-photo pool can't yield non-overlapping non-empty Strong and
    # Weak lists — disjointness must win over forcing both sides non-empty.
    photos_1 = [{"path": "only.jpg", "likes": 5}]
    strong, weak = unsplash_setup.split_by_engagement(photos_1, frac=0.5)
    assert set(strong).isdisjoint(set(weak))

    # A 2-photo pool at frac=1.0 ("take everything" on each side) must still
    # not overlap.
    photos_2 = [{"path": "a.jpg", "likes": 2}, {"path": "b.jpg", "likes": 1}]
    strong, weak = unsplash_setup.split_by_engagement(photos_2, frac=1.0)
    assert set(strong).isdisjoint(set(weak))

    # Odd-sized pool with a frac that would round past half if unclamped.
    photos_5 = [{"path": f"p{i}.jpg", "likes": i} for i in range(5)]
    strong, weak = unsplash_setup.split_by_engagement(photos_5, frac=0.9)
    assert set(strong).isdisjoint(set(weak))


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

    seen_requests = []

    def _fake_urlopen(req, timeout=30):
        # req.full_url carries the page number as "...&page=N&..."
        import re
        seen_requests.append(req)
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

    # The access key must travel as an Authorization header, not as a URL
    # query param — keeps it out of any URL that might land in a proxy or
    # error log.
    assert seen_requests, "urlopen was never called"
    for req in seen_requests:
        assert "client_id" not in req.full_url
        assert req.get_header("Authorization") == "Client-ID fake_key"


def test_build_exemplar_bank_saves_strong_embeddings_and_validates(tmp_path, monkeypatch):
    import unsplash_setup

    # Two well-separated clusters: "strong" near [1,0], "weak" near [0,1].
    fake_embeddings = {
        "s1.jpg": np.array([1.0, 0.05], dtype=np.float32),
        "s2.jpg": np.array([0.95, 0.1], dtype=np.float32),
        "w1.jpg": np.array([0.05, 1.0], dtype=np.float32),
        "w2.jpg": np.array([0.1, 0.95], dtype=np.float32),
    }

    encode_calls = []

    def _fake_encode_folder(paths, out_npz, progress=None):
        encode_calls.append(Path(out_npz))
        return np.stack([fake_embeddings[p] for p in paths])

    monkeypatch.setattr(unsplash_setup.dataset_embed, "encode_folder", _fake_encode_folder)
    scratch_dir = tmp_path / "cache" / "unsplash_scratch"
    monkeypatch.setattr(unsplash_setup, "_SCRATCH_DIR", scratch_dir)

    out_npz = tmp_path / "models" / "exemplar_bank.npz"
    result = unsplash_setup.build_exemplar_bank(
        strong_paths=["s1.jpg", "s2.jpg"], weak_paths=["w1.jpg", "w2.jpg"], out_npz=out_npz)

    assert out_npz.exists()
    saved = np.load(out_npz, allow_pickle=False)
    assert saved["embeddings"].shape == (2, 2)
    assert result["strong_self_sim"] > result["weak_self_sim"]

    # Same-dimension-different-space guard: the bank carries an encoder
    # fingerprint exemplar_scorer.score() can compare against.
    assert "encoder_source" in saved.files
    assert "embed_dim" in saved.files
    assert str(saved["encoder_source"])
    assert int(saved["embed_dim"]) > 0

    # encode_folder's own (throwaway) writes must land in the gitignored
    # scratch dir already used for this exact purpose — not next to the
    # real, shipped bank in models/.
    assert len(encode_calls) == 2
    for p in encode_calls:
        assert p.parent == scratch_dir
    assert not (out_npz.parent / "unsplash_weak_scratch.npz").exists()
    assert not (out_npz.parent / "unsplash_strong_scratch.npz").exists()


def test_bank_is_valid_refuses_on_nan_self_sim():
    import unsplash_setup

    # Normal pass/fail cases still behave as a plain "strong > weak" check.
    assert unsplash_setup._bank_is_valid({"strong_self_sim": 0.9, "weak_self_sim": 0.3}) is True
    assert unsplash_setup._bank_is_valid({"strong_self_sim": 0.3, "weak_self_sim": 0.9}) is False
    assert unsplash_setup._bank_is_valid({"strong_self_sim": 0.5, "weak_self_sim": 0.5}) is False

    # A NaN on either side (e.g. an empty pool) must never be mistaken for a
    # pass — this is the exact bug an unguarded `<=` check misses, since
    # `nan <= x` and `x <= nan` are both False.
    nan = float("nan")
    assert unsplash_setup._bank_is_valid({"strong_self_sim": nan, "weak_self_sim": 0.5}) is False
    assert unsplash_setup._bank_is_valid({"strong_self_sim": 0.5, "weak_self_sim": nan}) is False
    assert unsplash_setup._bank_is_valid({"strong_self_sim": nan, "weak_self_sim": nan}) is False


def test_build_exemplar_bank_with_empty_weak_pool_is_refused(tmp_path, monkeypatch):
    import unsplash_setup

    fake_embeddings = {
        "s1.jpg": np.array([1.0, 0.05], dtype=np.float32),
        "s2.jpg": np.array([0.95, 0.1], dtype=np.float32),
    }

    def _fake_encode_folder(paths, out_npz, progress=None):
        return np.stack([fake_embeddings[p] for p in paths])

    monkeypatch.setattr(unsplash_setup.dataset_embed, "encode_folder", _fake_encode_folder)

    out_npz = tmp_path / "exemplar_bank.npz"
    # split_by_engagement can return an empty weak_paths for pathologically
    # tiny collections; build_exemplar_bank must not silently treat that as
    # a passing sanity check — it should come back as something the
    # __main__ refuse-logic (_bank_is_valid) correctly rejects.
    result = unsplash_setup.build_exemplar_bank(
        strong_paths=["s1.jpg", "s2.jpg"], weak_paths=[], out_npz=out_npz)

    assert unsplash_setup._bank_is_valid(result) is False
