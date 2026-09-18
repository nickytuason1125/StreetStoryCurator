"""stage_runner — per-stage timing for the creative pipeline (2026-09-16).

The pipeline used to be a black box: a run printed progress strings but
nothing recorded WHERE the time went. StageTimer is deliberately dumb —
`mark(name)` at each existing stage boundary, `snapshot()` at the end —
so stages stay inline in run_creative_direction (no re-architecture) while
every run produces a timing profile that lands in the done payload and
crash.log. The profile is what makes "why was this run slow" a lookup
instead of an investigation."""

import time


class StageTimer:
    def __init__(self):
        self._stages = []   # [{"stage": name, "seconds": float}]
        self._open_since = None

    def mark(self, name: str) -> None:
        """Close the currently open stage (if any) and open `name`."""
        now = time.perf_counter()
        if self._open_since is not None:
            if self._stages:
                self._stages[-1]["seconds"] = round(now - self._open_since, 2)
            else:
                # First mark closes the implicit "setup" segment.
                self._stages.append({"stage": "setup", "seconds": round(now - self._open_since, 2)})
        self._stages.append({"stage": name, "seconds": 0.0})
        self._open_since = now

    def snapshot(self) -> dict:
        """Close the open stage and return {stages: [...], total: s}."""
        now = time.perf_counter()
        if self._open_since is not None:
            if self._stages:
                self._stages[-1]["seconds"] = round(now - self._open_since, 2)
            self._open_since = None
        total = round(sum(s["seconds"] for s in self._stages), 2)
        return {"stages": self._stages, "total": total}
