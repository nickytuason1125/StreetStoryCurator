"""Sequence routes — moved verbatim from server_impl.py (Milestone 4 split).

Decorators retargeted app -> router; every bare name that used to live in
server_impl resolves lazily through the module __getattr__ below (PEP 562),
so request-time access always sees the fully-initialised app without
circular imports. FastAPI names are imported eagerly because decorator-time
evaluation (parameter defaults like Query(...)) runs at import.
"""
from fastapi import (
    APIRouter, Body, Depends, File, Form, HTTPException, Query, Request,
    Response, UploadFile,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, validator, model_validator

from server_impl import (  # shared state & helpers
    GLOBAL_CLUSTER_CACHE, LAST_SEQUENCE, MAX_HISTORY, Path, RECENTLY_GENERATED, _DATA_DIR, analyzer, asyncio, get_analyzer, time,
)

router = APIRouter()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.post("/api/detect_niches")
async def detect_niches(payload: dict):
    photos = payload.get("photos", [])
    if not photos:
        return []
    input_data = [(p["path"], {"breakdown": p.get("breakdown", {}), "faces": p.get("faces", 0)}) for p in photos]
    return analyzer._detect_top_niches(input_data, top_n=5)


@router.post("/api/niches/build-anchors")
async def build_niche_anchors():
    """(Re)build NicheClassifier visual prototypes from the current cache."""
    import asyncio
    loop = asyncio.get_running_loop()
    built = await loop.run_in_executor(None, get_analyzer()._build_niche_anchors)
    clf   = get_analyzer()._niche_clf
    return {
        "built":   built,
        "anchors": clf.anchor_info if clf else {},
    }


# ---------------------------------------------------------------------------
# Generate sequence
# ---------------------------------------------------------------------------

@router.post("/api/generate")
async def generate_carousel(payload: dict):
    global RECENTLY_GENERATED, LAST_SEQUENCE
    try:
        photos = payload.get("photos", [])
        seed   = payload.get("seed") or int(time.time() * 1000) % (2**31)
        if not photos or len(photos) < 5:
            raise ValueError("Need at least 5 photos to generate a sequence.")

        # DEBUG LOG: Capture state before filtering
        total_photos = len(photos)
        before_filter_count = len(photos)
        before_filter_available = [p for p in photos if p["path"] not in RECENTLY_GENERATED]
        before_filter_count_available = len(before_filter_available)
        
        # Filter out recently generated paths to guarantee unique regenerations
        # (user-marked avoid_paths are handled by the sequencer, not global history)
        available = [p for p in photos if p["path"] not in RECENTLY_GENERATED]
        if len(available) < 5:
            print(f"[DEBUG generate] POOL EXHAUSTED - RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, available={len(available)}, resetting...")
            available = photos          # pool exhausted — reset and start fresh
            RECENTLY_GENERATED.clear()
            LAST_SEQUENCE = []          # stale avoidances would starve the fresh pool
        else:
            print(f"[DEBUG generate] Pool stats: total={total_photos}, RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, available={len(available)}")

        input_data = [
            (p["path"], {
                "score":    p["score"],
                "grade":    p["grade"],
                "embedding": analyzer.cache.get(p["path"], {}).get("embedding", p.get("embedding", [])),
                "breakdown": p.get("breakdown", {}),
                "sim_flag":  p.get("sim_flag", ""),
                "exif_ts":   analyzer.cache.get(p["path"], {}).get("exif_ts") or p.get("exif_ts") or 0.0,
            })
            for p in available
        ]
        _override = payload.get("subject_type") or payload.get("genre")
        _valid    = {"street", "nature", "portrait", "architecture"}
        # user_genre is None when the user chose "Any" — no genre filter applied in sequencer.
        # auto-detected type is used only for the response label.
        user_genre    = _override if (_override in _valid) else None
        detected_type = user_genre or analyzer.detect_subject_type(available)
        subject_type  = user_genre   # None = "Any" → sequence_story skips genre thresholds
        _pacing_valid  = {"Classic Street", "Travel / Documentary", "Minimalist / Art", "Custom"}
        pacing_preset  = payload.get("pacing_preset") if payload.get("pacing_preset") in _pacing_valid else None

        # Inject pre-computed cluster labels if the cache is warm for this folder
        _cache        = GLOBAL_CLUSTER_CACHE
        _folder       = payload.get("folder", "")
        cached_labels = None
        if _cache.get("folder") == _folder and _cache.get("labels") is not None:
            # Build a path→label lookup and align to input_data order
            import numpy as _np
            _lbl_map = dict(zip(_cache["paths"], _cache["labels"].tolist()))
            cached_labels = _np.array(
                [_lbl_map.get(r[0], -1) for r in input_data], dtype=_np.int32
            )

        # Merge server-tracked last sequence with any paths the user manually
        # marked as "used" in the frontend — both are excluded from the next pick.
        _user_avoid  = payload.get("avoid_paths", [])
        _avoid_set   = list(dict.fromkeys(LAST_SEQUENCE + _user_avoid))
        
        # DEBUG LOG: Capture avoid list state
        avoid_set_size = len(_avoid_set)
        last_seq_size = len(LAST_SEQUENCE)
        user_avoid_size = len(_user_avoid)

        # locked_slots: {slot_index_str: path} — positions that must not change
        _locked_slots = payload.get("locked_slots") or {}

        # DEBUG LOG: Print state before calling sequencer
        print(f"[DEBUG generate] total={total_photos}, RECENTLY_GENERATED={len(RECENTLY_GENERATED)}, LAST_SEQUENCE={last_seq_size}, user_avoid={user_avoid_size}, avoid_set={avoid_set_size}, available_for_sequencer={len(input_data)}")
        print(f"[DEBUG generate] avoid_paths: {len(_avoid_set)} unique paths to avoid")

        seq_paths, rationale, seq_type = analyzer.sequence_story(
            input_data, target=5, seed=seed, subject_type=subject_type,
            avoid_paths=_avoid_set, pacing_preset=pacing_preset,
            cached_labels=cached_labels, locked_slots=_locked_slots,
        )

        # Surface error when not enough photos passed genre thresholds
        if not seq_paths:
            err = rationale[0] if rationale else "Not enough qualifying images."
            return JSONResponse({"sequence": [], "subject_type": detected_type, "error": err})

        # Track the last sequence so the next Regenerate avoids the same picks
        LAST_SEQUENCE = list(seq_paths)
        
        # DEBUG LOG: Print generated sequence info
        print(f"[DEBUG generate] Generated sequence: {len(seq_paths)} photos")
        print(f"[DEBUG generate] LAST_SEQUENCE updated to: {LAST_SEQUENCE}")
        print(f"[DEBUG generate] RECENTLY_GENERATED now has {len(RECENTLY_GENERATED)} paths")

        # Record generated paths; trim history to MAX_HISTORY
        RECENTLY_GENERATED.update(seq_paths)
        if len(RECENTLY_GENERATED) > MAX_HISTORY:
            trimmed = list(RECENTLY_GENERATED)[-MAX_HISTORY:]
            RECENTLY_GENERATED.clear()
            RECENTLY_GENERATED.update(trimmed)
            print(f"[DEBUG generate] RECENTLY_GENERATED trimmed to {MAX_HISTORY} paths")

        carousel = []
        for i, path in enumerate(seq_paths):
            info = next((p for p in photos if p["path"] == path), {})
            carousel.append({
                **info,
                "rationale": rationale[i] if i < len(rationale) else "Strong candidate.",
            })
        return JSONResponse({"sequence": carousel, "subject_type": detected_type})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/sequence")
async def mogco_sequence_simple(payload: dict):
    """
    Clean single-endpoint MOGCO sequencer for Tauri IPC and external callers.

    Payload fields (all optional):
        vibe_prompt : str   – reserved for future text-to-vector vibe encoding
        target      : int   – frames to select (default 5)
        min_score   : float – quality floor for DuckDB query (default 0.45)
        beam_width  : int   – beam paths (default 4)

    Returns raw beam result: { paths, slots, global_score, beam_objectives }
    """
    import asyncio
    try:
        target     = int(payload.get("target", 5))
        min_score  = float(payload.get("min_score", 0.45))
        beam_width = int(payload.get("beam_width", 4))
        # vibe_prompt reserved — encode to vector here when text encoder is added
        from mogco_sequencer import run_mogco_sequence
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_mogco_sequence(
                vibe_vec=None,
                target=target,
                min_score=min_score,
                beam_width=beam_width,
            ),
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/sequence/album")
async def generate_album_sequence(payload: dict):
    """
    Build a multi-event album sequence from all graded images in the cache.

    Groups photos into temporal events (default 15-min gap), consolidates burst
    clusters within each event, then applies role-based pacing (SHOT_ROLES) to
    select the best 'frames' photos per event.

    Payload (all optional):
        gap_threshold : int   – seconds between events (default 900)
        frames        : int   – photos per event (default 5)
    """
    try:
        from sequence_engine import segment_events, consolidate_bursts, assign_roles

        gap_threshold = int(payload.get("gap_threshold", 900))
        frames        = int(payload.get("frames", 5))

        # Build flat list of records from the graded cache, injecting path
        records = []
        for path, data in analyzer.cache.items():
            if not data.get("embedding") or data.get("grade", "") == "Error ❌":
                continue
            records.append({
                "path":      path,
                "score":     float(data.get("score", 0)),
                "grade":     data.get("grade", ""),
                "embedding": data.get("embedding", []),
                "breakdown": data.get("breakdown", {}),
                "exif_ts":   float(data.get("exif_ts") or 0.0),
                "sim_flag":  data.get("sim_flag", ""),
            })

        if len(records) < frames:
            return JSONResponse({"error": f"Need at least {frames} graded images, got {len(records)}."})

        events      = segment_events(records, gap_threshold=gap_threshold)
        album       = []
        for i, event_group in enumerate(events):
            heroes   = consolidate_bursts(event_group)
            sequence = assign_roles(heroes, target=frames)
            album.append({
                "event_id":  f"evt_{i}",
                "start_ts":  event_group[0].get("exif_ts"),
                "frames": [
                    {
                        "path":       h["path"],
                        "score":      h.get("score", 0),
                        "grade":      h.get("grade", ""),
                        "burst_size": h.get("burst_size", 1),
                    }
                    for h in sequence
                ],
                "pacing": "Role-constrained + diversity-enforced",
            })

        return JSONResponse({"album": album, "events_detected": len(events)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# MOGCO sequencer — multi-objective Pareto selection backed by DuckDB
# ---------------------------------------------------------------------------

@router.post("/api/sequence/mogco")
async def mogco_sequence_endpoint(payload: dict):
    """
    MOGCO sequencer — two modes selectable via ``mode`` field:

    mode="beam"  (default)
        Queries DuckDB directly, applies an optional vibe/style filter, then runs
        beam search (width configurable) scoring quality + role_fit + visual_flow.
        Fastest for large libraries; DuckDB does the heavy lifting.

    mode="pareto"
        Greedy per-slot Pareto-front selection across 5 objectives.
        Requires embeddings from the frontend payload or JSON cache.

    Shared payload fields:
        photos        list  – photo records from the graded gallery
        target        int   – frames to select (default 5)
        subject_type  str   – genre hint passed to Pareto mode (default 'street')
        seed          int   – RNG seed (Pareto mode only, default 42)

    Beam-mode extra fields:
        mode          str   – "beam" | "pareto" (default "beam")
        vibe_path     str   – path of a reference photo; its DuckDB embedding is
                              used to filter candidates by style similarity
        vibe_thresh   float – minimum cosine similarity to the vibe photo (default 0.60)
        beam_width    int   – parallel beam paths (default 4)
        min_score     float – hard quality floor for DB query (default 0.45)
    """
    import asyncio
    import numpy as np

    try:
        photos      = payload.get("photos", [])
        target      = int(payload.get("target", 5))
        stype       = payload.get("subject_type") or "street"
        seed        = int(payload.get("seed", 42))
        mode        = payload.get("mode", "beam")
        vibe_path   = payload.get("vibe_path")
        vibe_thresh = float(payload.get("vibe_thresh", 0.60))
        beam_width  = int(payload.get("beam_width", 4))
        min_score   = float(payload.get("min_score", 0.45))

        if len(photos) < target:
            return JSONResponse({"sequence": [], "error": f"Need at least {target} photos."})

        from photo_cache import get_photo_cache
        db_cache = get_photo_cache()

        # ── Beam mode — DuckDB does the query, beam search does the rest ──────
        if mode == "beam":
            # Resolve vibe embedding from DuckDB or JSON cache
            vibe_vec = None
            if vibe_path:
                vibe_rows = db_cache.get_by_paths([vibe_path])
                if vibe_rows and len(vibe_rows[0]["embedding"]) > 0:
                    vibe_vec = vibe_rows[0]["embedding"]
                else:
                    raw = analyzer.cache.get(vibe_path, {}).get("embedding", [])
                    if raw:
                        vibe_vec = np.array(raw, dtype=np.float64)

            from mogco_sequencer import run_mogco_sequence
            beam_result = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: run_mogco_sequence(
                    vibe_vec=vibe_vec,
                    vibe_thresh=vibe_thresh,
                    target=target,
                    beam_width=beam_width,
                    min_score=min_score,
                ),
            )

            if "error" in beam_result and not beam_result.get("paths"):
                return JSONResponse({"sequence": [], **beam_result})

            # Merge with frontend photo data for the carousel
            info_by_path = {p["path"]: p for p in photos}
            carousel = []
            for path, slot, obj in zip(
                beam_result["paths"],
                beam_result["slots"],
                beam_result["beam_objectives"],
            ):
                base = dict(info_by_path.get(path, {"path": path}))
                base["slot"]             = slot
                base["mogco_objectives"] = {
                    "flow":     obj.get("flow", 0),
                    "quality":  obj.get("quality", 0),
                    "role_fit": obj.get("role_fit", 0),
                }
                base["engine"] = "mogco-beam"
                carousel.append(base)

            return JSONResponse({
                "sequence":      carousel,
                "subject_type":  stype,
                "engine":        "mogco-beam",
                "global_score":  beam_result.get("global_score"),
                "vibe_active":   vibe_vec is not None,
            })

        # ── Pareto mode — pre-fetch embeddings, greedy Pareto selection ───────
        db_records = db_cache.get_by_paths([p["path"] for p in photos])
        db_by_path = {r["path"]: r for r in db_records}

        candidates: list[dict] = []
        for p in photos:
            path = p["path"]
            db   = db_by_path.get(path)
            if db is not None and len(db["embedding"]) > 0:
                emb = db["embedding"]
            else:
                raw = analyzer.cache.get(path, {}).get("embedding", [])
                emb = np.array(raw, dtype=np.float64)
            if len(emb) == 0 or np.linalg.norm(emb) < 1e-6:
                continue
            candidates.append({
                "path":      path,
                "score":     float(p.get("score", db["score"] if db else 0.0)),
                "grade":     p.get("grade", ""),
                "breakdown": p.get("breakdown", db["breakdown"] if db else {}),
                "embedding": emb,
                "exif_ts":   float(p.get("exif_ts") or (db["exif_ts"] if db else 0.0)),
                "sim_flag":  p.get("sim_flag", ""),
            })

        if len(candidates) < target:
            return JSONResponse({
                "sequence": [],
                "error": f"Only {len(candidates)} photos have valid embeddings (need {target}).",
            })

        from mogco_engine import mogco_sequence
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: mogco_sequence(candidates, target=target, stype=stype, rng_seed=seed),
        )

        info_by_path = {p["path"]: p for p in photos}
        carousel = []
        for frame in result:
            base = dict(info_by_path.get(frame["path"], {}))
            base["slot"]             = frame.get("slot", "")
            base["mogco_objectives"] = frame.get("mogco_objectives", {})
            base["engine"]           = "mogco-pareto"
            carousel.append(base)

        return JSONResponse({
            "sequence":     carousel,
            "subject_type": stype,
            "engine":       "mogco-pareto",
            "db_hits":      len(db_records),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Art Director — prompt-driven sequence generation
# ---------------------------------------------------------------------------

_DIRECTOR_GENRE_MAP: dict = {
    "portrait":     ["portrait", "face", "person", "people", "character", "subject"],
    "street":       ["street", "urban", "city", "candid", "reportage", "bystander"],
    "architecture": ["architecture", "building", "geometric", "structure", "facade"],
    "nature":       ["nature", "landscape", "outdoor", "wildlife", "scenic"],
}

_DIRECTOR_MOOD_MAP: dict = {
    "melancholic":  ["melancholic", "melancholy", "somber", "dark", "brooding", "quiet", "lonely"],
    "dramatic":     ["dramatic", "intense", "powerful", "bold", "charged", "striking"],
    "minimalist":   ["minimalist", "minimal", "simple", "clean", "sparse", "austere", "zen"],
    "humanist":     ["humanist", "intimate", "candid", "emotional", "empathetic", "warm", "tender"],
    "cinematic":    ["cinematic", "film", "noir", "atmospheric", "moody", "filmic"],
    "documentary":  ["documentary", "reportage", "journalistic", "photojournalism", "real", "raw"],
    "editorial":    ["editorial", "magazine", "commercial", "polished", "professional", "fashion"],
    "competition":  ["competition", "award", "submit", "contest", "prize", "jury", "festival"],
}

# Per-mood weight deltas applied to each slot's (comp, human, mood/light) weights
_MOOD_DELTA: dict = {
    "melancholic":  {"comp": -0.05, "human": -0.10, "mood": +0.15},
    "dramatic":     {"comp": +0.00, "human": +0.15, "mood": +0.05},
    "minimalist":   {"comp": +0.20, "human": -0.15, "mood": -0.05},
    "humanist":     {"comp": -0.05, "human": +0.20, "mood": -0.10},
    "cinematic":    {"comp": +0.05, "human": -0.10, "mood": +0.20},
    "documentary":  {"comp": -0.05, "human": +0.10, "mood": +0.00},
    "editorial":    {"comp": +0.15, "human": -0.05, "mood": +0.00},
    "competition":  {"comp": +0.10, "human": +0.05, "mood": +0.05},
}


class _CLIPTextSearch:
    """
    CLIP text-to-image search engine.

    Encodes a text query into the CLIP embedding space (ViT-B/32, trained on
    400M image-text pairs) and ranks candidate photos by cosine similarity.
    Image embeddings are computed on-demand and cached in memory for the
    lifetime of the server process — subsequent calls for the same photo path
    are instant.

    This is the standard industry approach for semantic photo search:
    text and image share the same 512-dim embedding space so similarity
    directly reflects how well a photo matches the description.
    """

    def __init__(self) -> None:
        import clip as _clip
        import torch as _torch
        self._device     = _torch.device("cpu")
        self._model, self._prep = _clip.load(
            "ViT-B/32", device=self._device, download_root="./models"
        )
        self._model.eval()
        # path → normalised (512,) CLIP image embedding
        self._img_cache: dict[str, "np.ndarray"] = {}

    def _img_emb(self, path: str) -> "np.ndarray | None":
        if path in self._img_cache:
            return self._img_cache[path]
        try:
            import torch as _torch
            from PIL import Image as _PImage
            img    = _PImage.open(path).convert("RGB")
            tensor = self._prep(img).unsqueeze(0).to(self._device)
            with _torch.no_grad():
                emb = self._model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            arr = emb.cpu().numpy()[0]
            self._img_cache[path] = arr
            return arr
        except Exception:
            return None

    def rank(self, paths: list[str], query: str) -> list[tuple[str, float]]:
        """Return [(path, similarity)] sorted by descending cosine similarity."""
        import clip as _clip
        import torch as _torch
        tokens = _clip.tokenize([query], truncate=True).to(self._device)
        with _torch.no_grad():
            txt = self._model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        txt_arr = txt.cpu().numpy()[0]

        results: list[tuple[str, float]] = []
        for p in paths:
            emb = self._img_emb(p)
            sim = float(np.dot(emb, txt_arr)) if emb is not None else 0.0
            results.append((p, sim))
        return sorted(results, key=lambda x: -x[1])


_clip_searcher: "_CLIPTextSearch | None" = None


def _get_clip_searcher() -> "_CLIPTextSearch":
    global _clip_searcher
    if _clip_searcher is None:
        _clip_searcher = _CLIPTextSearch()
    return _clip_searcher


def _clip_rank_by_brief(input_data: list, brief: str) -> list:
    """
    Re-rank (path, data) pairs by CLIP text-image similarity to the brief.
    Falls back to original order silently if CLIP is unavailable.
    """
    try:
        searcher = _get_clip_searcher()
        paths    = [d[0] for d in input_data]
        ranked   = searcher.rank(paths, brief)          # [(path, sim), …]
        order    = {path: i for i, (path, _) in enumerate(ranked)}
        return sorted(input_data, key=lambda x: order.get(x[0], len(ranked)))
    except Exception:
        return input_data


def _parse_director_prompt(prompt: str) -> dict:
    """
    Extract genre, mood biases, and frame count from a natural language brief.
    Returns custom SHOT_ROLES weights + director's note.
    """
    import re, copy
    from collections import OrderedDict

    text = prompt.lower()

    # Genre detection — first match wins
    genre = None
    for g, kws in _DIRECTOR_GENRE_MAP.items():
        if any(kw in text for kw in kws):
            genre = g
            break

    # Mood detection — can be multiple
    moods = [m for m, kws in _DIRECTOR_MOOD_MAP.items() if any(kw in text for kw in kws)]

    # Frame count extraction
    target = 5
    m = re.search(r'\b(\d+)\s*(?:photo|frame|image|shot|picture)', text)
    if m:
        target = max(3, min(10, int(m.group(1))))

    # Start from base SHOT_ROLES weights
    base = OrderedDict([
        ("opener",   {"comp_weight": 0.4, "human_weight": 0.1, "mood_weight": 0.3, "diversity_penalty": 0.2}),
        ("subject",  {"comp_weight": 0.2, "human_weight": 0.5, "mood_weight": 0.1, "diversity_penalty": 0.2}),
        ("detail",   {"comp_weight": 0.5, "human_weight": 0.1, "mood_weight": 0.2, "diversity_penalty": 0.2}),
        ("contrast", {"comp_weight": 0.2, "human_weight": 0.2, "mood_weight": 0.4, "diversity_penalty": 0.2}),
        ("closer",   {"comp_weight": 0.3, "human_weight": 0.1, "mood_weight": 0.5, "diversity_penalty": 0.2}),
    ])

    # Accumulate mood deltas across all detected moods
    for mood in moods:
        delta = _MOOD_DELTA.get(mood, {})
        dc, dh, dm = delta.get("comp", 0), delta.get("human", 0), delta.get("mood", 0)
        for role in base.values():
            role["comp_weight"]  = max(0.05, role["comp_weight"]  + dc)
            role["human_weight"] = max(0.05, role["human_weight"] + dh)
            role["mood_weight"]  = max(0.05, role["mood_weight"]  + dm)
            # Re-normalise so weights sum to (1 - diversity_penalty)
            total = role["comp_weight"] + role["human_weight"] + role["mood_weight"]
            budget = 1.0 - role["diversity_penalty"]
            scale  = budget / total if total > 0 else 1.0
            role["comp_weight"]  *= scale
            role["human_weight"] *= scale
            role["mood_weight"]  *= scale

    # Compose director's note
    mood_desc  = " + ".join(moods) if moods else "balanced"
    genre_desc = genre or "auto-detected genre"
    note = f"Reading brief as **{mood_desc}** with **{genre_desc}** focus."
    if "competition" in moods:
        note += " Competition mode: maximising technical quality and compositional impact."
    note += f" Selecting {target} frames calibrated for this narrative arc."

    return {
        "genre":            genre,
        "target":           target,
        "custom_shot_roles": base,
        "director_note":    note,
        "style_tags":       moods,
    }


_DIRECTOR_POOL_DIR = _DATA_DIR / "cache" / "director_pool"


@router.post("/api/director")
async def director_sequence(payload: dict):
    """
    Art Director: parse a natural language brief and generate a curated sequence
    from the graded photo pool (or uploaded competition photos).
    """
    import asyncio as _aio
    try:
        prompt = str(payload.get("prompt", "")).strip()
        photos = payload.get("photos", [])

        if not prompt:
            return JSONResponse({"error": "Please describe the sequence you want."})

        parsed = _parse_director_prompt(prompt)
        target = int(payload.get("target") or parsed["target"])

        if not photos or len(photos) < target:
            return JSONResponse({
                "error": f"Need at least {target} graded photos. "
                         "Grade your folder first or upload competition photos."
            })

        input_data = [
            (p["path"], {
                "score":     float(p.get("score", 0)),
                "grade":     p.get("grade", ""),
                "embedding": analyzer.cache.get(p["path"], {}).get("embedding", p.get("embedding", [])),
                "breakdown": p.get("breakdown", {}),
                "sim_flag":  p.get("sim_flag", ""),
                "exif_ts":   float(analyzer.cache.get(p["path"], {}).get("exif_ts") or p.get("exif_ts") or 0.0),
            })
            for p in photos
        ]

        # CLIP text-image ranking: encodes brief → 512-dim embedding, ranks photos
        # by cosine similarity (CLIP trained on 400M image-text pairs).
        if prompt:
            input_data = await _aio.get_running_loop().run_in_executor(
                None, lambda: _clip_rank_by_brief(input_data, prompt)
            )

        loop = _aio.get_running_loop()
        seq_paths, rationale, seq_type = await loop.run_in_executor(
            None,
            lambda: analyzer.sequence_story(
                input_data,
                target=target,
                subject_type=parsed["genre"],
                avoid_paths=[],
                seed=int(time.time() * 1000) % (2 ** 31),
                custom_shot_roles=parsed["custom_shot_roles"],
            ),
        )

        if not seq_paths:
            err = rationale[0] if rationale else "No qualifying images for this brief."
            return JSONResponse({"error": err})

        sequence = []
        for i, path in enumerate(seq_paths):
            info = next((p for p in photos if p["path"] == path), {})
            sequence.append({**info, "slot_label": rationale[i] if i < len(rationale) else f"Frame {i+1}"})

        return JSONResponse({
            "sequence":      sequence,
            "director_note": parsed["director_note"],
            "genre":         seq_type,
            "style_tags":    parsed["style_tags"],
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/director/upload-grade")
async def director_upload_grade(
    files: list[UploadFile] = File(...),
    preset: str = Form("Classic Street"),
):
    """
    Upload image files for competition use, grade them, and return scored results.
    Files are saved to cache/director_pool/ so thumbnails remain accessible.
    """
    import asyncio as _aio, shutil

    if not files:
        raise HTTPException(400, "No files provided.")

    # Clear previous pool and create fresh batch folder
    batch_dir = _DIRECTOR_POOL_DIR
    if batch_dir.exists():
        shutil.rmtree(batch_dir, ignore_errors=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    for f in files:
        safe = Path(f.filename or "upload").name
        dest = batch_dir / safe
        dest.write_bytes(await f.read())

    # Grade using existing pipeline
    loop = _aio.get_running_loop()
    results = await loop.run_in_executor(
        None,
        lambda: analyzer.analyze_folder(str(batch_dir), preset=preset),
    )

    gallery = [
        {
            "path":       r[0],
            "grade":      r[1]["grade"],
            "score":      r[1]["score"],
            "critique":   r[1].get("critique", ""),
            "breakdown":  r[1]["breakdown"],
            "sim_flag":   r[1].get("sim_flag", ""),
            "cluster_id": r[1].get("cluster_id", -1),
            "embedding":  r[1].get("embedding", []),
        }
        for r in results
    ]

    return JSONResponse({"photos": gallery, "total": len(gallery)})


@router.post("/api/director/clear-pool")
async def director_clear_pool():
    """Delete uploaded competition photos from cache/director_pool/."""
    import shutil
    if _DIRECTOR_POOL_DIR.exists():
        shutil.rmtree(_DIRECTOR_POOL_DIR, ignore_errors=True)
    return JSONResponse({"cleared": True})


