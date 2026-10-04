"""Pick the photos most worth rating: evenly spread over the machine-score
range, unrated, and not hidden duplicates. Rating keepers only (the 2026-10-03
baseline: 5 of 157 on-disk rated photos were 1-2★) teaches the learner nothing
about Mid and Weak — spreading over the range fixes that in one short session."""
from __future__ import annotations


def _rated(paths: list) -> dict:
    import ratings_store as rs
    return rs.stars_for_paths(paths)


def pick(photos: list, n: int = 50) -> list:
    visible = [p for p in photos
               if not (int(p.get("cluster_id", -1)) >= 0 and "Best" not in (p.get("sim_flag") or ""))]
    rated = _rated([p["path"] for p in visible])
    pool = sorted((p for p in visible if p["path"] not in rated
                   and isinstance(p.get("score"), (int, float))), key=lambda p: p["score"])
    if len(pool) <= n:
        return [p["path"] for p in pool]
    step = (len(pool) - 1) / (n - 1)
    return [pool[round(i * step)]["path"] for i in range(n)]
