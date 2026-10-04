"""Gallery thumbnails must appear instantly, not re-load on every scroll.

2026-10-04, measured on a 4,745-photo catalog:
  1. Every /api/thumb response carried `Cache-Control: no-store` (the blanket
     /api/ rule), so the windowed grid re-downloaded a tile every time it
     scrolled back into view or the view switched — shimmer + fade each time.
  2. 10 of 12 random graded photos had no cached thumbnail at all (100-1200 ms
     cold render each): prewarm is dropped during a cull and only ever ran for
     the first 600 files of a folder-open. Nothing pre-built the gallery.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (mounts the routers)
from routers import misc  # noqa: E402


def _through_cache_middleware(path: str, status: int) -> Response:
    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "GET",
             "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [(b"host", b"127.0.0.1:8000")]}

    async def call_next(_req):
        return Response(b"x", status_code=status, media_type="image/webp")

    return asyncio.run(server_impl.cache_control_middleware(Request(scope), call_next))


def test_served_thumbnail_is_browser_cacheable():
    cc = _through_cache_middleware("/api/thumb", 200).headers["cache-control"]
    assert "no-store" not in cc
    assert "max-age=" in cc and int(cc.split("max-age=")[1].split(",")[0]) >= 3600


def test_failed_thumbnail_is_never_cached():
    # A 404 must stay retryable: the Thumb retry ladder depends on it.
    assert _through_cache_middleware("/api/thumb", 404).headers["cache-control"] == "no-store"


def test_other_api_responses_stay_uncached():
    assert _through_cache_middleware("/api/catalog", 200).headers["cache-control"] == "no-store"


def test_catalog_load_prewarms_gallery_thumbnails_once(monkeypatch):
    queued: list[list[str]] = []
    photos = [{"path": f"C:/shoot/{i}.jpg"} for i in range(5)]
    monkeypatch.setattr(misc, "_load_catalog_cached", lambda: {"photos": photos})
    monkeypatch.setattr(misc, "_prewarmed_catalog_key", {"v": None})
    import routers.library as lib
    monkeypatch.setattr(lib, "prewarm_thumbs", lambda ps: queued.append(list(ps)))
    monkeypatch.setattr(server_impl._grading_active, "is_set", lambda: False)

    misc._prewarm_catalog_thumbs(("catalog.json", 1.0, 10), wait=True)
    misc._prewarm_catalog_thumbs(("catalog.json", 1.0, 10), wait=True)   # same catalog
    assert queued == [[p["path"] for p in photos]]

    misc._prewarm_catalog_thumbs(("catalog.json", 2.0, 12), wait=True)   # new grade
    assert len(queued) == 2


def test_catalog_prewarm_waits_out_a_running_grade(monkeypatch):
    # Background thumbs are dropped while grading; marking the catalog done
    # then would mean it is never pre-built.
    queued: list = []
    monkeypatch.setattr(misc, "_load_catalog_cached", lambda: {"photos": [{"path": "a.jpg"}]})
    monkeypatch.setattr(misc, "_prewarmed_catalog_key", {"v": None})
    import routers.library as lib
    monkeypatch.setattr(lib, "prewarm_thumbs", lambda ps: queued.append(list(ps)))

    monkeypatch.setattr(server_impl._grading_active, "is_set", lambda: True)
    misc._prewarm_catalog_thumbs(("c", 1.0, 1), wait=True)
    assert queued == []

    monkeypatch.setattr(server_impl._grading_active, "is_set", lambda: False)
    misc._prewarm_catalog_thumbs(("c", 1.0, 1), wait=True)
    assert queued == [["a.jpg"]]
