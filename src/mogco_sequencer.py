"""
Public API wrapper for the MOGCO beam-search sequencer.

Resolves the DuckDB path from the PhotoCache singleton so callers
only need to pass algorithmic parameters.
"""
import numpy as np
from mogco_engine import run_mogco_beam
from photo_cache import get_photo_cache


def run_mogco_sequence(
    vibe_vec: np.ndarray | None = None,
    vibe_thresh: float = 0.60,
    target: int = 5,
    beam_width: int = 4,
    min_score: float = 0.45,
) -> dict:
    """
    Select a narrative sequence from the DuckDB photo cache.

    Parameters
    ----------
    vibe_vec    : Optional reference embedding for style-similarity filtering.
                  When None the vibe filter is disabled.
    vibe_thresh : Minimum cosine similarity to the vibe reference (default 0.60).
    target      : Number of frames to select (default 5).
    beam_width  : Parallel beam paths (default 4).
    min_score   : Hard quality floor applied in the DB query (default 0.45).

    Returns
    -------
    dict with keys: paths, slots, global_score, beam_objectives, [error]
    """
    db = get_photo_cache()
    # Use the singleton's existing connection to avoid a second DuckDB file-lock
    # on Windows (opening read_only=True while the write connection is still open
    # causes an OS error on re-grade).
    with db._lock:
        rows = db._connect().execute(
            "SELECT path, score, embedding, breakdown, exif_ts FROM photos WHERE score >= ?",
            [min_score],
        ).fetchall()

    return run_mogco_beam(
        db._path,
        vibe_vec=vibe_vec,
        vibe_thresh=vibe_thresh,
        target=target,
        beam_width=beam_width,
        min_score=min_score,
        _rows=rows,
    )
