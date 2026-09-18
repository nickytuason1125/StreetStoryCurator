"""The library's quality tier, PERSISTED (2026-09-13).

Why this exists
---------------
The embedding tables are PER TIER (lance_store: one table per tier, so 1536-d
Pro embeddings never mix with 768-d Fast ones), and the text-probe cache is
tier-suffixed (_tier_cache_name). That partitioning is correct — but the TIER
itself was re-derived from scratch on every run from a free-RAM measurement
taken at an arbitrary moment:

    launch_hidden.vbs pinned SIGLIP_TIER=low  -> Fast table
    a grade on a quieter machine auto-picked  -> high table
    the next grade, 1 GB less free            -> mid table

Each flip makes the incremental encode cache look EMPTY (the previous tier's
table holds the embeddings), so the whole library re-encodes, the job balloons,
and a machine that barely funded the first pass now faces a bigger one. That
is the "adding a new thing konks out the grade" spiral.

Fix: the tier is LIBRARY STATE, not a per-run measurement. Chosen once —
explicitly, or auto-selected on the very first grade — and persisted here.
Later grades inherit it regardless of how much RAM is free today, so adding
photos encodes only the new photos. SIGLIP_TIER in the environment remains an
explicit per-run override (tests, A/B, debugging); setting it does NOT rewrite
this file.

Resolution order (see tier_select.apply / run_profile.current):
    1. SIGLIP_TIER env          (explicit override, wins, not persisted)
    2. this file                (library state)
    3. auto-select by RAM       (first grade only; result is persisted)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_TIERS = ("high", "mid", "low")


def _path() -> Path:
    """Where the tier file lives. FIRSTCUT_LIB_TIER_FILE exists so tests can
    point this at a scratch dir (conftest) instead of the real cache."""
    env = os.environ.get("FIRSTCUT_LIB_TIER_FILE", "").strip()
    if env:
        return Path(env)
    data = os.environ.get("FIRSTCUT_DATA_DIR", "").strip()
    root = Path(data) if data else Path(__file__).resolve().parent.parent / "cache"
    return root / "library_tier.json"


def get() -> "str | None":
    """The persisted library tier, or None when never chosen (or corrupt).

    A corrupt/garbage file is treated as "no tier" — a broken cache file must
    never stop a grade; the next successful grade simply re-persists it.
    """
    try:
        doc = json.loads(_path().read_text(encoding="utf-8"))
        t = str(doc.get("tier", "")).strip().lower()
        return t if t in _TIERS else None
    except Exception:
        return None


def set(tier: str, reason: str = "") -> bool:
    """Persist the library tier. Refuses (returns False) anything not a tier,
    so a caller bug cannot poison the file."""
    t = (tier or "").strip().lower()
    if t not in _TIERS:
        return False
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"tier": t, "ts": time.time(), "reason": reason[:200]}),
            encoding="utf-8")
        return True
    except Exception:
        return False


def clear() -> None:
    """Back to 'never chosen' (next grade auto-selects and persists again)."""
    try:
        _path().unlink()
    except Exception:
        pass