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
