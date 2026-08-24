import { useEffect } from "react";

/**
 * A setInterval that runs `fn` immediately, skips ticks while the tab is
 * hidden, and refreshes once when the tab becomes visible again. Keeps
 * background polling from running (and flooding the server log) when the app
 * isn't on screen.
 */
export function useGuardedInterval(fn: () => void, ms: number, deps: any[]) {
  useEffect(() => {
    fn();
    const id = setInterval(() => { if (!document.hidden) fn(); }, ms);
    const onVis = () => { if (!document.hidden) fn(); };
    document.addEventListener("visibilitychange", onVis);
    return () => { clearInterval(id); document.removeEventListener("visibilitychange", onVis); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
