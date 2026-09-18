"""_in_process_pull_stream_locked's 416 (Range Not Satisfiable) recovery.

A stale/oversized .part file makes the model-download server answer 416 to a
Range request. r.raise_for_status() used to run BEFORE the 416 check, so it
raised HTTPError on the very response the code below it exists to recover
from — the "drop the stale partial, restart clean" branch was unreachable.
Fixed by moving raise_for_status() after the 416 handling.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (must load fully first — routers.extras <-> server_impl is circular)
import routers.extras as extras  # noqa: E402
import model_registry as _mr  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code, content=b"", headers=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {"Content-Length": str(len(content))}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size):
        yield self._content

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_416_response_recovers_instead_of_raising(monkeypatch, tmp_path):
    dest = tmp_path / "model.gguf"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"X" * 999)   # an oversized/stale partial

    fake_model = SimpleNamespace(
        key="vision", dest=dest, repo="fake/repo", filename="model.gguf",
        size_gb=0.001, purpose="test", present=lambda: False,
    )
    # keys always starts as ["vision", "vision_mmproj", "text"] regardless of
    # model_name (unrelated pre-existing behaviour) — only "vision" should
    # actually attempt a download here; the other two report already-present
    # so the mocked HTTP call count stays predictable.
    already_present = SimpleNamespace(
        key="skip", dest=tmp_path / "already.gguf", repo="x", filename="x",
        size_gb=0.0, purpose="", present=lambda: True,
    )
    monkeypatch.setattr(_mr, "gguf",
                        lambda key: fake_model if key == "vision" else already_present)
    monkeypatch.setattr(_mr, "GGUF_MODELS", [fake_model])

    calls = []

    def fake_get(url, headers=None, stream=True, timeout=None, allow_redirects=True):
        calls.append(dict(headers or {}))
        if len(calls) == 1:
            # First attempt: Range beyond EOF -> the server answers 416.
            return _FakeResponse(416)
        # Retry without Range: full, fresh content.
        body = b"hello world"
        return _FakeResponse(200, content=body,
                              headers={"Content-Length": str(len(body))})

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    events = [__import__("json").loads(line) for line in
              extras._in_process_pull_stream_locked("vision")]

    assert len(calls) == 2, "expected an initial 416 then a clean retry, got: " + repr(calls)
    assert calls[0].get("Range"), "first attempt should have sent a Range header"
    assert "Range" not in calls[1], "retry after 416 must drop the Range header"
    statuses = [e.get("status") for e in events]
    assert "fail" not in statuses, f"416 should recover, not fail: {events}"
    assert dest.exists() and dest.read_bytes() == b"hello world"
    assert not part.exists(), "the .part file should be finalized away"
