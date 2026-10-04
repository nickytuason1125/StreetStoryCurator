"""The encode ticker must never hide the encoder's real progress.

2026-10-04: a 4300-photo cull sat at 46% for ~13 minutes. The SigLIP encode
reported real chunk progress (photos 841-4300, "~9 min left", ...), but the
_ProgressTicker wrapped around it used a fixed 30 s time constant over the
same 0.07-0.46 span: it reached ~0.455 within two minutes, the monotonic clamp
then pinned every real chunk emit to that floor, and the ticker overwrote the
chunk text with a generic label every 1.5 s. The ticker now creeps from the
latest REAL emit toward the next expected one instead of toward the end.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import grade_pipeline_v2 as gp  # noqa: E402


class _Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.emits: list[tuple[float, str]] = []

    def __call__(self, frac, desc=""):
        with self.lock:
            self.emits.append((float(frac), desc))

    def last(self):
        with self.lock:
            return self.emits[-1]


def test_ticker_stays_near_real_progress_after_chunk_emit():
    rec = _Recorder()
    with gp._ProgressTicker(rec, 0.07, 0.46, "Analyzing 4300 photos…",
                            tau=0.05, interval=0.02) as tk:
        tk.progress(0.07, "Analyzing photos 1–4300 of 4300")
        time.sleep(0.3)   # >> tau: the old ticker would be pinned at ~0.46 by now
        frac, desc = rec.last()
        assert frac < 0.20, f"ticker raced past real progress: {frac}"
        assert desc.startswith("Analyzing photos"), "real chunk text was overwritten"

        tk.progress(0.146, "Analyzing photos 841–4300 of 4300 · ~9 min left")
        time.sleep(0.3)
        frac, desc = rec.last()
        # Next real emit lands at ~0.222; the creep must not overshoot it.
        assert 0.146 <= frac < 0.222, frac
        assert "~9 min left" in desc


def test_ticker_without_real_emits_still_spans_the_phase():
    rec = _Recorder()
    with gp._ProgressTicker(rec, 0.07, 0.46, "Analyzing 10 photos…",
                            tau=0.05, interval=0.02):
        time.sleep(0.3)
    fracs = [f for f, _ in rec.emits]
    assert fracs and fracs == sorted(fracs)
    assert 0.40 < fracs[-1] < 0.46


def test_ticker_never_emits_after_exit_and_never_reaches_end():
    rec = _Recorder()
    with gp._ProgressTicker(rec, 0.66, 0.83, "Scoring…", tau=0.05, interval=0.02) as tk:
        for k in range(1, 4):
            tk.progress(0.66 + 0.04 * k, f"Scored {k}")
            time.sleep(0.1)
    n = len(rec.emits)
    time.sleep(0.1)
    assert len(rec.emits) == n
    assert all(f < 0.83 for f, _ in rec.emits)
