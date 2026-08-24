/**
 * lib/api.ts — the single place FrameGrade's frontend talks about the backend.
 *
 * Every network path, its base URL and path sanitisation live here so a
 * route change is a one-file edit, not a grep across a 5k-line component.
 */

export const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

// Block any accidental external analytics / tracking calls — this is a fully offline app.
if (typeof window !== "undefined") {
  const _origFetch = window.fetch.bind(window);
  const _BLOCKED   = ["googleapis.com", "analytics", "sentry.io", "segment.io", "mixpanel", "hotjar"];
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input.toString();
    if (_BLOCKED.some(h => url.includes(h))) return Promise.reject(new Error(`Blocked external request: ${url}`));
    return _origFetch(input, init);
  };
}

export const API = import.meta.env.VITE_API_URL || (isTauri() ? "http://127.0.0.1:8000" : "http://127.0.0.1:8000");
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
