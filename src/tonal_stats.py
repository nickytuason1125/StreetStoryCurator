"""
tonal_stats.py — does this frame LOOK like it belongs with those frames?

Why this exists
---------------
A real Story run scored 0.891 cohesion and still returned warm night colour,
black-and-white, and a bright daytime frame in one six-image sequence. Every
frame genuinely matched the brief ("quiet streets, people at a distance"),
because SigLIP embeddings encode SUBJECT AND SCENE. They do not encode
tonality, colour or time of day — which is most of what makes a set of
photographs read as one body of work.

So cosine cohesion was measuring the wrong kind of similarity, exactly as the
design doc warned: "looks similar" is not "belongs together". This module adds
the missing axis, cheaply.

Two numbers per photo, both 0..1:

    luma    mean brightness — separates night from daylight
    chroma  mean saturation — separates monochrome from colour

That is deliberately not a colour science model. It is the smallest pair that
distinguishes the three failures actually observed in one real sequence.

Cost
----
Read from `cache/thumbs`, which already holds 7,791 webp thumbnails of about
4 KB, named `{stem}_{md5(path)[:10]}.webp` by `app.get_or_create_thumb`.
Decoding those is effectively free. Decoding full frames — some of these are
40-megapixel — would not be, and this runs over a candidate shortlist on every
Story request.

Unknown is never treated as clashing. A photo with no thumbnail scores neutral,
so it is neither preferred nor pushed out for lacking one.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

THUMB_DIR = Path("cache/thumbs")

# Returned when a frame cannot be measured. Deliberately mid-scale: an
# unmeasured photo competes on its other merits rather than being penalised for
# a missing thumbnail.
NEUTRAL_FIT = 0.5


def thumb_path_for(img_path: str) -> Path:
    """Where app.get_or_create_thumb would have written this photo's thumbnail.

    Mirrors that function exactly — hash the FULL path so identical filenames in
    different folders cannot collide, and replace spaces in the stem.
    """
    h = hashlib.md5(str(img_path).encode()).hexdigest()[:10]
    stem = Path(str(img_path)).stem.replace(" ", "_")
    return THUMB_DIR / f"{stem}_{h}.webp"


def stats_from_thumb(path) -> "Optional[tuple]":
    """(luma, chroma) in 0..1, or None when the thumbnail cannot be read.

    None rather than a default: a guessed tonality would quietly place a frame
    in or out of a set on evidence that does not exist.
    """
    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            im.thumbnail((32, 32))
            a = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None
    if a.size == 0:
        return None

    luma = float(a.mean())
    # Saturation as max-minus-min per pixel: 0 for any grey, high for a strong
    # hue. Cheaper than an HSV conversion and enough to separate mono from
    # colour, which is the distinction that was actually visible.
    chroma = float((a.max(axis=2) - a.min(axis=2)).mean())
    return luma, chroma


def stats_from_original(img_path) -> "Optional[tuple]":
    """Same two numbers, read straight from the photograph via draft-mode decode.

    Thumbnails are created ON DEMAND, so only about 7,800 of 38,000 catalogued
    photos have one; without this, tone was unmeasurable for most candidates and
    the term did nothing.

    draft() asks the JPEG decoder to scale down while decoding, using the DCT
    coefficients, so a 40-megapixel frame never becomes a 40-megapixel array.
    The same trick the cull path uses for RAM. RAW files have no draft mode and
    are skipped rather than fully decoded — a Story request must not turn into a
    RAW decode of the whole shortlist.
    """
    try:
        from PIL import Image
        with Image.open(str(img_path)) as im:
            if getattr(im, "format", "") not in ("JPEG", "WEBP", "PNG"):
                return None
            try:
                im.draft("RGB", (64, 64))
            except Exception:
                pass
            im = im.convert("RGB")
            im.thumbnail((32, 32))
            a = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None
    if a.size == 0:
        return None
    return float(a.mean()), float((a.max(axis=2) - a.min(axis=2)).mean())


_CACHE_PATH = Path("cache/tonal_stats.json")
_cache: "Optional[dict]" = None


def _load_cache() -> dict:
    """Tonal stats never change for a given file, so they are worth keeping.

    Measured cost without this: 339 ms per 40-megapixel frame even with
    draft-mode decode, which is 100 seconds over a 300-candidate shortlist on
    every single Story request.
    """
    global _cache
    if _cache is None:
        try:
            import json
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        import json, os
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_load_cache()), encoding="utf-8")
        os.replace(str(tmp), str(_CACHE_PATH))   # atomic: never a half file
    except Exception as e:
        print(f"[tone] cache not saved ({e})")


def stats_for_paths(paths: "list[str]", budget: "Optional[int]" = None) -> np.ndarray:
    """(N, 2) of (luma, chroma), NaN where a photo could not be measured.

    `budget` caps how many UNCACHED photos are decoded in one call. Beyond it
    the rest stay NaN and score neutral, so a first run on a large shortlist
    stays responsive and later runs fill in from cache.
    """
    cache = _load_cache()
    out = np.full((len(paths), 2), np.nan, dtype=np.float32)
    decoded = 0
    dirty = False
    for i, p in enumerate(paths):
        key = str(p)
        hit = cache.get(key)
        if hit is not None:
            if hit:                        # [] marks "known unmeasurable"
                out[i] = hit
            continue
        if budget is not None and decoded >= budget:
            continue
        st = stats_from_thumb(thumb_path_for(p))
        if st is None:
            st = stats_from_original(p)    # thumbs are on-demand; most miss
            decoded += 1
        cache[key] = list(st) if st is not None else []
        dirty = True
        if st is not None:
            out[i] = st
    if dirty:
        _save_cache()
    return out


def tonal_fit(candidate, chosen) -> float:
    """How well one frame's tone matches a set's, 0..1. Higher fits better.

    NEUTRAL_FIT when the candidate is unmeasured or the set is empty — the
    first pick has nothing to match, and an unknown frame must not be pushed
    out for being unknown.
    """
    if candidate is None:
        return NEUTRAL_FIT
    cand = np.asarray(candidate, dtype=np.float32)
    if cand.size < 2 or not np.all(np.isfinite(cand)):
        return NEUTRAL_FIT

    ref = np.asarray(chosen, dtype=np.float32)
    if ref.ndim != 2 or ref.shape[0] == 0:
        return NEUTRAL_FIT
    ref = ref[np.all(np.isfinite(ref), axis=1)]
    if ref.shape[0] == 0:
        return NEUTRAL_FIT

    centre = ref.mean(axis=0)
    # Both axes are already 0..1, so a plain mean absolute difference is
    # comparable across them without weighting one over the other.
    dist = float(np.abs(cand - centre).mean())
    return float(max(0.0, 1.0 - dist))
