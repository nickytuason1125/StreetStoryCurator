"""
Middleware accept/reject matrix for _security_isolation (M1/L1 hardening).

Fail-closed contract on /api/*:
  - Host header must be present AND localhost (L1: an empty Host used to
    skip the check entirely — fail-open).
  - Sec-Fetch-Site cross-site/cross-origin → 403 (browser-only header,
    forbidden-header class, page JS cannot forge it).
  - sec-fetch-mode navigate → 404 (phishing-link navigation to an API URL).
  - POST/PUT/DELETE/PATCH need a legitimate browser context: a well-formed
    Sec-Fetch-Site OR X-Requested-With: FirstCut (M1: a browser with
    Sec-Fetch-* stripped no longer restores CSRF reach, because simple
    cross-origin requests cannot set custom headers; scripts get a
    one-header escape hatch).

The middleware is driven directly as an ASGI callable with a hand-built
scope — no HTTP client. (The installed starlette 0.36 TestClient is
incompatible with httpx 0.28, which dropped the `app=` kwarg; driving the
bare coroutine avoids the dependency fight entirely.) `call_next` is a
sentinel returning 200: any response the middleware produced itself is a
rejection, any 200 means the request reached the app.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import server_impl  # noqa: E402  (mounts the routers)


def _probe(method: str, path: str, headers: dict | None = None):
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if not any(k == b"host" for k, _ in hdrs):
        hdrs.append((b"host", b"127.0.0.1:8001"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": hdrs,
        "server": ("127.0.0.1", 8001),
        "scheme": "http",
    }
    request = Request(scope)

    async def call_next(_req):  # sentinel: "reached the app"
        return JSONResponse({"reached_app": True})

    return asyncio.run(server_impl._security_isolation(request, call_next))


def _reached(resp) -> bool:
    body = getattr(resp, "body", b"")
    return resp.status_code == 200 and b"reached_app" in body


# ── L1: Host pinning, now fail-closed ────────────────────────────────────────
def test_localhost_get_passes():
    assert _reached(_probe("GET", "/api/health/engine"))


def test_forbidden_host_is_403():
    assert _probe("GET", "/api/health/engine", {"Host": "evil.com"}).status_code == 403


def test_empty_host_is_403_not_skipped():
    resp = _probe("GET", "/api/health/engine", {"Host": ""})
    assert resp.status_code == 403, "empty Host must fail CLOSED, not skip the check"


def test_missing_host_is_403():
    scope_hdrs = {}  # _probe always appends a default host — build one without it
    hdrs = []
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "method": "GET",
        "path": "/api/health/engine", "raw_path": b"/api/health/engine",
        "query_string": b"", "headers": hdrs,
        "server": ("127.0.0.1", 8001), "scheme": "http",
    }
    request = Request(scope)

    async def call_next(_req):
        return JSONResponse({"reached_app": True})

    resp = asyncio.run(server_impl._security_isolation(request, call_next))
    assert resp.status_code == 403


# ── Fetch-Metadata rules (behaviour pinned here so it can't regress) ─────────
def test_cross_site_get_is_403():
    assert _probe("GET", "/api/health/engine", {"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_navigation_mode_is_404():
    resp = _probe("GET", "/api/health/engine",
                  {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Site": "none"})
    assert resp.status_code == 404


# ── M1: state-changing methods need a browser-legitimate context ─────────────
def test_post_without_any_context_is_403():
    resp = _probe("POST", "/api/watch/start")
    assert resp.status_code == 403, (
        "a POST with neither Sec-Fetch-Site nor the custom header is exactly "
        "the stripped-header CSRF hole this gate exists to close"
    )


def test_post_with_token_passes_middleware():
    resp = _probe("POST", "/api/watch/start", {"X-Requested-With": "FirstCut"})
    assert _reached(resp)


def test_post_same_origin_passes_middleware():
    resp = _probe("POST", "/api/watch/start", {"Sec-Fetch-Site": "same-origin"})
    assert _reached(resp)


def test_post_same_site_passes_middleware():
    """Vite dev server (different port, same host) is same-site."""
    resp = _probe("POST", "/api/watch/start", {"Sec-Fetch-Site": "same-site"})
    assert _reached(resp)


def test_post_cross_site_rejected_even_with_token():
    resp = _probe("POST", "/api/watch/start",
                  {"Sec-Fetch-Site": "cross-site", "X-Requested-With": "FirstCut"})
    assert resp.status_code == 403, "explicit cross-site beats any header forgery"


def test_delete_also_gated():
    assert _probe("DELETE", "/api/catalog/clear").status_code == 403
    resp = _probe("DELETE", "/api/catalog/clear", {"X-Requested-With": "FirstCut"})
    assert _reached(resp)