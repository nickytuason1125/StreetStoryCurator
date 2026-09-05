/**
 * lib/api.ts — the single place FirstCut's frontend talks about the backend.
 *
 * Every network path, its base URL and path sanitisation live here so a
 * route change is a one-file edit, not a grep across a 5k-line component.
 */

export const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

// Block any accidental external analytics / tracking calls — this is a fully offline app.
// The wrapper also stamps X-Requested-With: FirstCut onto every request: the
// backend's security middleware requires it (or a browser-legitimate
// Sec-Fetch-Site) on state-changing methods as a CSRF second layer, because a
// simple cross-origin request (form/img) cannot set custom headers.
import axios from "axios";

if (typeof window !== "undefined") {
  const _origFetch = window.fetch.bind(window);
  const _BLOCKED   = ["googleapis.com", "analytics", "sentry.io", "segment.io", "mixpanel", "hotjar"];
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input.toString();
    if (_BLOCKED.some(h => url.includes(h))) return Promise.reject(new Error(`Blocked external request: ${url}`));
    const headers = new Headers(
      init?.headers ?? (input instanceof Request ? input.headers : undefined),
    );
    if (!headers.has("X-Requested-With")) headers.set("X-Requested-With", "FirstCut");
    return _origFetch(input, { ...init, headers });
  };
}

// Same token for the axios code paths (axios uses XHR, not window.fetch).
axios.defaults.headers.common["X-Requested-With"] = "FirstCut";

// The API base follows the serving origin when served over http(s): the SPA is
// same-origin with its backend by design, so hardwiring the port silently
// broke the app whenever it was served from any other origin (a second dev
// instance on :8001, a future port change) — the readiness probe went
// cross-origin and the app sat on "Starting FirstCut…" forever. Tauri keeps
// the explicit default because its webview origin is a custom protocol.
/**
 * API base — follows the serving origin over http(s). The SPA is same-origin
 * with its backend by design; a hardwired port silently broke the app whenever
 * it was served from any other origin (a second dev instance on :8001, a
 * future port change): the readiness probe went cross-origin and the app sat
 * on "Starting FirstCut…" forever. Note the env override CANNOT win here —
 * .env.production pins VITE_API_URL, which vite statically substitutes and
 * dead-code-eliminates anything guarded behind `env || ...`. Tauri's
 * custom-protocol webview (and any non-http origin) keeps the explicit
 * default, which is why it is checked first.
 */
export const API = isTauri() || !window.location.origin.startsWith("http")
  ? (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000")
  : window.location.origin;
export const thumbUrl = (p: string) => `${API}/api/thumb?path=${encodeURIComponent(p)}`;
export const photoUrl = (p: string) => `${API}/api/photo?path=${encodeURIComponent(p)}`;

/** Strip traversal sequences and normalise separators before sending paths to the API. */
export const sanitizePath = (raw: string): string =>
  raw.trim()
    .replace(/[\/\\]+/g, "/")   // normalise separators
    .split("/")
    .filter(seg => seg !== "..")  // drop traversal segments
    .join("/")
    .replace(/^\//, match => match); // preserve leading slash (absolute paths)
