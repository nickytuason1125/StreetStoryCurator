"""
V2 grading pipeline — Frontier Edition.

Step 1  Discover images in the folder.
Step 2  SigLIP-2 ViT-g/14 NaFlex → 1536-d embeddings.
Step 3  Detect duplicates via cosine similarity (0.92 threshold).
Step 4  CLIP scoring + SpecVLM Draft-and-Verify reasoning.
            4a  SpecVLMPipeline: calibrated CLIP scores + per-aspect breakdown.
            4b  SpecVLM (DeepSeek-R1): structured metadata → chain-of-thought
                reasoning_log + refined score (±0.12 from CLIP base).
                Draft: 1.5B INT4. Verify: 7B INT4 (uncertain images only).
                Fallback: Qwen2.5-VL-3B critique if DeepSeek unavailable.
Step 5  PersonalHead adjusts scores by learned user preference (if weights present).
Step 6  Grade buckets: ≥0.60 Strong ✅ / 0.41-0.59 Mid ⚠️ / ≤0.40 Weak ❌
Step 7  Write to LanceDB (1536-d IVF-PQ schema).
Step 8  Build gallery response (V1-compatible keys + reasoning_log).
Step 9  NSGA-III multi-objective sequence: Reasoning_Accuracy × Semantic_Vibe
            × Portfolio_Diversity × Aspect_Ratio_Balance.

VRAM Protocol (4-6 GB cards):
    SigLIP-2 (~1.8 GB) → purge_vram() → DeepSeek 1.5B (~1.0 GB) → purge_vram()
    → DeepSeek 7B (~3.5 GB, uncertain only) → purge_vram()
    Models never overlap.  Peak ≤ 5.5 GB on RTX 3060.
"""
from __future__ import annotations

import os
import gc
import json
import threading
import numpy as np
from pathlib import Path
from typing import Callable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

STRONG_THRESH = 0.60
MID_THRESH    = 0.41

GRADE_STRONG = "Strong ✅"
GRADE_MID    = "Mid ⚠️"
GRADE_WEAK   = "Weak ❌"

_EXIF_LOCK = threading.Lock()


# ── EXIF timestamp ─────────────────────────────────────────────────────────────

def _exif_ts(path: str) -> float:
    try:
        import piexif
        exif = piexif.load(path)
        raw = (
            exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            or exif.get("0th",  {}).get(piexif.ImageIFD.DateTime)
        )
        if raw:
            from datetime import datetime
            return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return 0.0


# ── VRAM helper ────────────────────────────────────────────────────────────────

def _vram_clear():
    """Release CUDA caches between pipeline phases."""
    try:
        from vram_manager import VRAMManager
        VRAMManager.purge_vram()
    except Exception:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_v2(
    folder_path: str,
    preset: str = "Classic Street",
    force_rescan: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
    mogco_target: int = 5,
    scan_mode: bool = False,
) -> dict:
    """
    Run the full V2 SpecVLM pipeline on `folder_path`.

    Returns:
        gallery         list[dict]   per-photo result (V1-compatible keys + reasoning_log)
        mogco_sequence  list[dict]   NSGA-III output
        strong / mid / weak int      counts
        total           int
        pipeline        "v2"
    """
    _p = progress or (lambda f, d: None)

    # ── Step 1: Discover images ───────────────────────────────────────────────
    _p(0.01, "Scanning folder…")
    folder = Path(folder_path)
    all_paths  = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not all_paths:
        return {"error": "No images found in folder.", "gallery": [], "total": 0}

    # ── Incremental: skip already-graded images when force_rescan=False ───────
    cached_rows: dict[str, dict] = {}
    if not force_rescan:
        try:
            import lance_store as _ls
            fp_str = str(Path(folder_path).resolve())
            for row in _ls.query_all(min_score=0.0):
                rp = row.get("path", "")
                # Normalise separators so Windows backslash paths and frontend
                # forward-slash paths both map to the same key.
                rp_norm = str(Path(rp)) if rp else ""
                if rp_norm.startswith(fp_str) and float(row.get("score", 0)) > 0:
                    cached_rows[rp_norm] = row
        except Exception as _ce:
            print(f"[v2] LanceDB cache check failed: {_ce}")

    paths = [p for p in all_paths if p not in cached_rows]
    n     = len(paths)

    def _cached_to_gallery(row: dict) -> dict:
        bd = row.get("breakdown", {})
        if isinstance(bd, str):
            try: bd = json.loads(bd)
            except Exception: bd = {}
        return {
            "id": row["path"], "path": row["path"],
            "filename": Path(row["path"]).name,
            "grade": row.get("grade", GRADE_MID),
            "score": round(float(row.get("score", 0.5)), 3),
            "human_perception": round(float(row.get("personal_score", 0.5)), 3),
            "personal_score":   round(float(row.get("personal_score", 0.5)), 3),
            "embedding": row.get("embedding", []),
            "breakdown": bd,
            "critique": row.get("reasoning_log", "")[:120],
            "reasoning_log": row.get("reasoning_log", ""),
            "is_verified": False,
            "exif_ts": float(row.get("exif_ts", 0.0)),
            "stars": 0, "reject": False, "sim_flag": "", "cluster_id": -1,
        }

    if cached_rows and not paths:
        # All images already graded — return cached data immediately
        _p(1.0, f"All {len(cached_rows)} images already graded (use Re-grade to force rescan)")
        gallery = [_cached_to_gallery(cached_rows[p]) for p in all_paths if p in cached_rows]
        grades  = [g["grade"] for g in gallery]
        return {
            "gallery": gallery,
            "mogco_sequence": [],
            "strong": sum(1 for g in grades if g == GRADE_STRONG),
            "mid":    sum(1 for g in grades if g == GRADE_MID),
            "weak":   sum(1 for g in grades if g == GRADE_WEAK),
            "total":  len(gallery),
            "pipeline": "v2_cached",
        }

    if cached_rows:
        _p(0.02, f"Found {n} new images to grade ({len(cached_rows)} already graded)")
    else:
        _p(0.02, f"Found {n} images")

    # ── Step 2: Bulk encoding ─────────────────────────────────────────────────
    # Try SigLIP-2 ViT-g/14 NaFlex (FP8, 1536-d).  Falls back to So400M (1152-d).
    embs            = None
    embed_dim       = 1152
    siglip_ok       = False
    _pos_text_embs  = None
    _neg_text_embs  = None
    _aspect_pos     = None
    _aspect_neg     = None
    _aspect_names   = None

    import traceback as _tb
    for _attempt, _kwargs in enumerate([
        {"device": "auto", "quantize": True},   # 1st: GPU INT8/FP16
        {"device": "cpu",  "quantize": False},  # 2nd: CPU FP16 (slow but correct)
    ]):
        try:
            from siglip2_encoder import SigLIP2Encoder
            from specvlm_pipeline import _POS_PROMPTS, _NEG_PROMPTS, _ASPECT_PROMPTS
            enc  = SigLIP2Encoder(**_kwargs, progress=_p)
            embs = enc.encode_images(paths, progress=_p)   # (N, 1536)

            # Encode aesthetic text references before unloading the encoder
            _p(0.48, "Encoding aesthetic reference prompts…")
            _pos_text_embs = enc.encode_text(_POS_PROMPTS)              # (P, 1536)
            _neg_text_embs = enc.encode_text(_NEG_PROMPTS)              # (Q, 1536)
            _aspect_names  = list(_ASPECT_PROMPTS.keys())
            _aspect_pos    = enc.encode_text(
                [v[0] for v in _ASPECT_PROMPTS.values()]                # (A, 1536)
            )
            _aspect_neg    = enc.encode_text(
                [v[1] for v in _ASPECT_PROMPTS.values()]                # (A, 1536)
            )

            # Cache "people" concept embedding for empty-brief creative direction (Task 1)
            _PEOPLE_PROMPTS = [
                "people", "crowds", "pedestrians", "human figure", "faces",
            ]
            _ppl_raw = enc.encode_text(_PEOPLE_PROMPTS)   # (5, 1536)
            _ppl_mean = _ppl_raw.mean(axis=0)
            _ppl_mean /= (np.linalg.norm(_ppl_mean) + 1e-9)
            try:
                _cache_dir = Path("cache")
                _cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(_cache_dir / "people_emb.npy"), _ppl_mean.astype(np.float32))
                print("[v2] people_emb.npy saved for empty-brief CD gate")
            except Exception as _e_ppl:
                print(f"[v2] people_emb save skipped: {_e_ppl}")

            enc.unload()
            del enc
            embed_dim = 1536
            siglip_ok = True
            _tag = "GPU" if _kwargs["device"] == "auto" else "CPU fallback"
            _p(0.50, "SigLIP-2 done — clearing VRAM…")
            print(f"[v2] Encoder: SigLIP-2 NaFlex ({_tag})  dim={embed_dim}")
            break
        except Exception as e_siglip2:
            print(f"[v2] SigLIP-2 attempt {_attempt+1} failed: {e_siglip2}")
            if _attempt == 0:
                print("[v2] Retrying SigLIP-2 on CPU…")
            else:
                print("[v2] SigLIP-2 unavailable after all attempts.")
                print(_tb.format_exc())

    # SigLIP-2 (1536-d) is required — all legacy encoders removed in Frontier 2026.
    if embed_dim != 1536:
        raise RuntimeError(
            f"Frontier 2026 requires SigLIP-2 NaFlex (1536-d). "
            f"Got {embed_dim}-d — SigLIP-2 failed to load on both GPU and CPU.\n"
            "Check backend.log for the full error traceback."
        )

    # VRAM guard — free encoder memory before loading reasoning models
    _vram_clear()

    # ── Step 3: Duplicate detection ───────────────────────────────────────────
    _p(0.50, "Detecting duplicates…")
    cluster_ids:     list[int] = [-1] * n
    sim_flags:       list[str] = [""] * n
    to_rate_indices: list[int] = list(range(n))

    if siglip_ok and n >= 2:
        try:
            from collections import defaultdict as _dd
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-9)
            sims   = normed @ normed.T

            # SigLIP-2 spaces are tighter — 0.92 still works for both dims
            SIM_THRESH = 0.92

            parent = list(range(n))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(n):
                for j in range(i + 1, n):
                    if sims[i, j] > SIM_THRESH:
                        ri, rj = _find(i), _find(j)
                        if ri != rj:
                            parent[ri] = rj

            groups_d: dict = _dd(list)
            for i in range(n):
                groups_d[_find(i)].append(i)

            # Populate cluster_ids for all photos in duplicate groups (size >= 2)
            for root, members in groups_d.items():
                if len(members) >= 2:
                    for i in members:
                        cluster_ids[i] = root

            to_rate_indices = []
            for members in groups_d.values():
                to_rate_indices.extend(members)

        except Exception as e:
            print(f"[v2] Duplicate detection failed: {e}")
            to_rate_indices = list(range(n))

    # ── Step 4: Qwen2.5-VL-3B-Instruct per-photo grading ─────────────────────
    _p(0.51, "Loading Qwen2.5-VL-3B-Instruct…")
    scores                = np.full(n, 0.5, dtype=np.float32)
    draft_scores:         list[float] = [0.5] * n
    reasoning_logs:       list[str]  = [""] * n
    is_verified:          list[bool] = [False] * n
    per_photo_breakdowns: list[dict] = [{}] * n
    per_photo_critiques:  list[str]  = [""] * n

    paths_to_rate = [paths[i] for i in to_rate_indices]

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _props = _torch.cuda.get_device_properties(0)
            _free  = (_props.total_memory - _torch.cuda.memory_reserved(0)) / 1e9
            print(f"[v2] VRAM before VLM load: {_free:.2f} GB free / {_props.total_memory/1e9:.2f} GB total")
        del _torch
    except Exception:
        pass

    # ── Step 4a: CLIP scoring (instant — reuses SigLIP-2 embeddings) ─────────
    # Always runs.  Gives calibrated 0-1 scores + per-aspect breakdown.
    _p(0.51, "CLIP scoring…")
    clip_scores_rated  = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    clip_aspects_rated: list[dict] = [{}] * len(paths_to_rate)

    try:
        from specvlm_pipeline import SpecVLMPipeline
        pipeline        = SpecVLMPipeline()
        specvlm_results = pipeline.grade_images(
            paths_to_rate,
            progress        = _p,
            scan_mode       = scan_mode,
            embeddings      = embs[to_rate_indices],
            pos_text_embs   = _pos_text_embs,
            neg_text_embs   = _neg_text_embs,
            aspect_pos_embs = _aspect_pos,
            aspect_neg_embs = _aspect_neg,
            aspect_names    = _aspect_names,
        )
        pipeline.unload()
        del pipeline

        clip_map = {r.path: r for r in specvlm_results}
        for local_i, idx in enumerate(to_rate_indices):
            r = clip_map.get(paths[idx])
            if r:
                clip_scores_rated[local_i]  = float(r.score)
                clip_aspects_rated[local_i] = r.breakdown or {}
                # Populate arrays now so CLIP results survive if VLM fails
                scores[idx]               = float(r.score)
                draft_scores[idx]         = float(r.draft_score or r.score)
                reasoning_logs[idx]       = r.reasoning_log or ""
                is_verified[idx]          = bool(r.is_verified)
                per_photo_breakdowns[idx] = r.breakdown or {}

        _p(0.55, "CLIP scoring done — loading VLM for reasoning…")
        print(f"[v2] CLIP scores: min={clip_scores_rated.min():.3f}  max={clip_scores_rated.max():.3f}")

    except Exception as e_clip:
        print(f"[v2] CLIP scoring failed: {e_clip}")

    # ── Step 4b: SpecVLM Draft+Verify reasoning ──────────────────────────────
    # DeepSeek-R1 1.5B draft → 7B verify for uncertain images.
    # Scores come from CLIP (Step 4a); DeepSeek refines ±0.12 and writes
    # the full chain-of-thought reasoning_log.
    # Qwen2.5-VL-3B is used as a fallback if DeepSeek fails to load.
    _p(0.56, "Building visual metadata…")
    try:
        from specvlm_pipeline import SpecVLM, build_visual_metadata

        # Build VisualMetadata for every photo being rated
        metadatas = []
        for local_i, idx in enumerate(to_rate_indices):
            metadatas.append(build_visual_metadata(
                path         = paths[idx],
                clip_score   = float(clip_scores_rated[local_i]),
                aspect_scores= clip_aspects_rated[local_i],
            ))

        specvlm = SpecVLM()
        specvlm_results = specvlm.process_metadata_batch(metadatas, progress=_p)
        specvlm.unload()
        del specvlm

        result_map = {r.path: r for r in specvlm_results}
        for local_i, idx in enumerate(to_rate_indices):
            r = result_map.get(paths[idx])
            if r:
                # Refined score replaces CLIP score
                scores[idx]             = float(r.score)
                draft_scores[idx]       = float(r.draft_score or r.score)
                reasoning_logs[idx]     = r.reasoning_log or reasoning_logs[idx]
                is_verified[idx]        = bool(r.is_verified)
                per_photo_critiques[idx]= r.reasoning_log or ""

        _p(0.86, "SpecVLM reasoning done — clearing VRAM…")
        _vram_clear()
        print(f"[v2] SpecVLM Draft+Verify: {len(metadatas)} photos  "
              f"verified={sum(1 for r in specvlm_results if r.is_verified)}")

    except Exception as e_specvlm:
        _p(0.72, f"SpecVLM unavailable ({type(e_specvlm).__name__}) — falling back to Qwen…")
        print(f"[v2] SpecVLM failed ({e_specvlm}) — trying Qwen2.5-VL-3B fallback")
        try:
            from qwen_vlm_grader import QwenVLMGrader
            grader      = QwenVLMGrader(progress=_p)
            vlm_results = grader.grade_images(
                paths_to_rate,
                clip_scores  = clip_scores_rated,
                clip_aspects = clip_aspects_rated,
                progress     = _p,
            )
            grader.unload()
            del grader
            result_map = {r.path: r for r in vlm_results}
            for idx in to_rate_indices:
                r = result_map.get(paths[idx])
                if r:
                    reasoning_logs[idx]      = r.reasoning or reasoning_logs[idx]
                    is_verified[idx]         = True
                    per_photo_critiques[idx] = r.critique or ""
            _p(0.86, "Qwen fallback done — clearing VRAM…")
        except Exception as e_qwen:
            _p(0.86, "VLM unavailable — using CLIP reasoning…")
            print(f"[v2] Qwen fallback also failed ({e_qwen}) — CLIP-only mode")
        _vram_clear()

    # ── Step 4c: Weighted aspect blend ───────────────────────────────────────
    # Regardless of which path (CLIP-only, DeepSeek draft, DeepSeek+verify) set
    # scores[idx], fold in the 5 CLIP aspect dimensions here so the final score
    # is always: 0.60 × current_score + 0.40 × weighted_aspect_avg.
    # This is the single authoritative blend point — grade_images() also does
    # this blend but Step 4b overwrites it; we redo it here so no path can skip it.
    _W_DEFAULT  = {"Technical":1.0, "Composition":1.2, "Lighting":1.0, "Narrative":1.0, "Human/Culture":1.0}
    _W_ARCH     = {"Technical":1.0, "Composition":1.2, "Lighting":1.0, "Narrative":0.6, "Human/Culture":0.15}
    for idx in range(n):
        aspects = per_photo_breakdowns[idx]
        if not aspects:
            continue
        is_arch = (aspects.get("Human/Culture", 0.5) < 0.38
                   and aspects.get("Composition", 0.5) > 0.52)
        _W = _W_ARCH if is_arch else _W_DEFAULT
        total_w = sum(_W.get(k, 1.0) for k in aspects)
        if total_w <= 0:
            continue
        aspect_avg   = sum(v * _W.get(k, 1.0) for k, v in aspects.items()) / total_w
        scores[idx]  = float(np.clip(0.60 * scores[idx] + 0.40 * aspect_avg, 0.0, 1.0))

    scores_arr = np.array(scores, dtype=np.float32)
    print(
        f"[v2] grader scores — min={scores_arr.min():.3f}  "
        f"max={scores_arr.max():.3f}  mean={scores_arr.mean():.3f}  "
        f"median={float(np.median(scores_arr)):.3f}"
    )

    # ── Step 5: PersonalHead adjustment ──────────────────────────────────────
    _p(0.87, "Applying personal preference…")
    pers         = np.full(n, 0.5, dtype=np.float32)
    final_scores = scores_arr
    _ph_weights  = Path("cache/personal_head.pt")
    if _ph_weights.exists():
        print("[v2] PersonalHead weights found — blending 80/20")
        try:
            import personal_head as ph
            pers         = ph.score(embs)
            final_scores = 0.80 * scores_arr + 0.20 * pers
        except Exception as _e:
            print(f"[v2] PersonalHead blend failed: {_e}")
    else:
        print("[v2] PersonalHead weights absent — using raw grader scores")

    # ── Step 5b: Duplicate sim-flag assignment based on final_scores ──────────
    _p(0.88, "Flagging duplicates…")
    try:
        from collections import defaultdict as _dd2
        groups2: dict = _dd2(list)
        for i, cid in enumerate(cluster_ids):
            if cid >= 0:
                groups2[cid].append(i)
        for members in groups2.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda i: float(final_scores[i]), reverse=True)
            best_fn = Path(paths[members[0]]).name
            best_sc = float(final_scores[members[0]])
            for rank, idx in enumerate(members):
                if rank == 0:
                    sim_flags[idx] = f"★ Best of {len(members)} similar shots (score {best_sc:.2f})"
                else:
                    diff = best_sc - float(final_scores[idx])
                    sim_flags[idx] = (
                        f"\U0001f501 Duplicate — {best_fn} is better: higher overall score (+{diff:.2f})"
                    )
    except Exception as e:
        print(f"[v2] Sim-flag assignment failed: {e}")

    # ── Step 5c: Genre-aware score recalibration ──────────────────────────────
    # Architectural and Liminal photos score on different raw-discriminant ranges
    # than street photos.  The batch IQR in _calibrate() anchors to the street
    # majority, systematically under-scoring non-street genres.  Recalibrate
    # each non-street genre within its own peer group and blend 55 / 45 with the
    # already-calibrated batch score so they compete fairly without losing
    # cross-genre comparability.
    try:
        from specvlm_pipeline import _detect_genre
        genres = [_detect_genre(per_photo_breakdowns[i]) for i in range(n)]
        for target_genre in ("Architectural", "Liminal"):
            members = [i for i, g in enumerate(genres) if g == target_genre]
            if len(members) < 2:
                if members:
                    print(f"[v2] {target_genre}: only 1 photo — skipping genre recal")
                continue
            genre_raw = np.array([float(final_scores[i]) for i in members])
            lo = float(np.percentile(genre_raw, 25))
            hi = float(np.percentile(genre_raw, 75))
            span = max(hi - lo, 1e-4)
            genre_cal = np.clip((genre_raw - lo) / span * 0.19 + 0.41, 0.0, 1.0)
            for j, i in enumerate(members):
                blended = 0.75 * float(final_scores[i]) + 0.25 * float(genre_cal[j])
                final_scores[i] = float(np.clip(blended, 0.0, 1.0))
            print(
                f"[v2] {target_genre} genre recal: {len(members)} photos  "
                f"raw=[{genre_raw.min():.3f}–{genre_raw.max():.3f}]  "
                f"cal=[{genre_cal.min():.3f}–{genre_cal.max():.3f}]"
            )
    except Exception as e:
        print(f"[v2] Genre recalibration failed: {e}")

    # ── Step 5d: Draft-model hard cap (unverified photos only) ───────────────
    # The 1.5B draft model can't move scores much (±0.07 max drift, 80/20 blend)
    # but it CAN detect confidently-bad photos.  When draft score ≤ 0.33, the
    # model is strongly signalling "weak" — prevent calibration from pushing
    # that photo into Strong regardless of batch ranking.
    # Symmetric floor: draft ≥ 0.67 protects genuinely-strong photos from
    # being buried in Weak by a weak-majority batch.
    # Verified photos (7B) are trusted fully — no cap applied.
    capped = 0
    for i in range(n):
        if is_verified[i]:
            continue
        ds = draft_scores[i]
        if ds <= 0.33 and final_scores[i] >= STRONG_THRESH:
            final_scores[i] = STRONG_THRESH - 0.01   # push just below Strong
            capped += 1
        elif ds >= 0.67 and final_scores[i] <= MID_THRESH:
            final_scores[i] = MID_THRESH + 0.01      # lift just above Weak
            capped += 1
    if capped:
        print(f"[v2] Draft cap applied to {capped} photos")

    # ── Step 6: Grade buckets ─────────────────────────────────────────────────
    # Snap to 2 decimal places BEFORE bucketing so that the comparison and the
    # badge displayed in the frontend (Math.round(score*100)) always agree.
    # e.g. raw 0.5996 → snapped 0.60 → Strong AND badge shows 60. No mismatch.
    final_scores = np.round(final_scores, 2)

    _p(0.89, "Bucketing Strong / Mid / Weak…")
    print(
        f"[v2] final scores — min={final_scores.min():.2f}  "
        f"max={final_scores.max():.2f}  mean={final_scores.mean():.2f}  "
        f"median={float(np.median(final_scores)):.2f}"
    )
    grades = []
    for i, s in enumerate(final_scores):
        if s >= STRONG_THRESH:
            g = GRADE_STRONG
        elif s >= MID_THRESH:
            g = GRADE_MID
        else:
            g = GRADE_WEAK
        grades.append(g)
        print(f"[v2]   {Path(paths[i]).name}: {s:.2f} → {g}")

    # ── Step 6b: Patch reasoning_log headers to match final grades ────────────
    # reasoning_logs were written from intermediate CLIP draft scores (before
    # Steps 4c/5c/5d/PersonalHead).  Replace the structured header so the
    # Reasoning tab never shows "Mid  47%" for a photo graded Strong.
    # Format matched: "Strong  63%\n<tier description>\n..." → replaced with
    # the correct tier/score while leaving the aspect bars unchanged.
    _TIER_DESCS_V2 = {
        GRADE_STRONG: "Strong visual intent — decisive moment, bold geometry, or atmospheric power.",
        GRADE_MID:    "Some strong elements but inconsistent execution or missing visual tension.",
        GRADE_WEAK:   "Blurry, poorly framed, flat light, or no clear visual subject or intent.",
    }
    import re as _re_hdr
    _HDR_LINE = _re_hdr.compile(
        r'^(Strong|Mid|Weak)\s+\d+%[^\n]*\n?(?:[^\n]+\n?)?',
        _re_hdr.IGNORECASE,
    )
    for i in range(n):
        log = reasoning_logs[i]
        if not log:
            continue
        tier_str = ('Strong' if grades[i] == GRADE_STRONG
                    else 'Mid' if grades[i] == GRADE_MID
                    else 'Weak')
        pct      = int(round(float(final_scores[i]) * 100))
        new_hdr  = f"{tier_str}  {pct}%\n{_TIER_DESCS_V2[grades[i]]}\n"
        reasoning_logs[i] = _HDR_LINE.sub(new_hdr, log, count=1)

    # ── Step 7: EXIF + LanceDB ────────────────────────────────────────────────
    _p(0.90, "Reading EXIF…")
    timestamps = [_exif_ts(p) for p in paths]

    _p(0.92, "Writing to LanceDB (1536-d IVF-PQ)…")
    try:
        import lance_store as ls
        records = [
            {
                "path":           paths[i],
                "embedding":      embs[i].tolist(),
                "score":          float(final_scores[i]),
                "personal_score": float(pers[i]),
                "grade":          grades[i],
                "reasoning_log":  reasoning_logs[i],
                "breakdown":      {"aesthetic": round(float(scores_arr[i]), 3),
                                   "personal":  round(float(pers[i]),       3)},
                "exif_ts":        timestamps[i],
            }
            for i in range(n)
        ]
        ls.upsert_batch(records)
        lance_ok = True
    except Exception as e:
        print(f"[v2] LanceDB write failed: {e}")
        lance_ok = False

    # ── Step 8: Gallery response ──────────────────────────────────────────────
    _p(0.94, "Building gallery…")
    gallery = []
    for i, path in enumerate(paths):
        fn        = Path(path).name
        breakdown = {
            "Aesthetic": round(float(scores_arr[i]), 3),
            "Personal":  round(float(pers[i]),       3),
        }
        if per_photo_breakdowns[i]:
            breakdown.update(per_photo_breakdowns[i])
        gallery.append({
            "id":              path,
            "path":            path,
            "filename":        fn,
            "grade":           grades[i],
            "score":           round(float(final_scores[i]), 3),
            "human_perception":round(float(pers[i]),         3),
            "personal_score":  round(float(pers[i]),         3),
            "embedding":       embs[i].tolist(),
            "breakdown":       breakdown,
            "critique":        per_photo_critiques[i],
            "reasoning_log":   reasoning_logs[i],
            "is_verified":     is_verified[i],
            "exif_ts":         timestamps[i],
            "stars":           0,
            "reject":          cluster_ids[i] >= 0 and not sim_flags[i].startswith("★"),
            "sim_flag":        sim_flags[i],
            "cluster_id":      cluster_ids[i],
        })

    # Merge cached images back into the gallery (preserving folder sort order)
    if cached_rows:
        gallery_by_path = {g["path"]: g for g in gallery}
        gallery = [
            gallery_by_path[p] if p in gallery_by_path else _cached_to_gallery(cached_rows[p])
            for p in all_paths
            if p in gallery_by_path or p in cached_rows
        ]

    # ── Step 9: NSGA-III multi-objective sequencing ───────────────────────────
    _p(0.96, "Running NSGA-III…")
    mogco_seq: list[dict] = []
    if siglip_ok and lance_ok:
        try:
            from nsga3_sequencer import run_nsga3_sequence_with_vlm

            # Pass Strong + Mid candidates with embeddings and reasoning logs
            seq_candidates = [
                {
                    "path":          g["path"],
                    "score":         g["score"],
                    "embedding":     np.array(g["embedding"], dtype=np.float32),
                    "reasoning_log": g["reasoning_log"],
                }
                for g in gallery
                if g["grade"] in (GRADE_STRONG, GRADE_MID)
            ]

            selected = run_nsga3_sequence_with_vlm(
                seq_candidates, target=mogco_target, progress=_p
            )

            info_by_path = {g["path"]: g for g in gallery}
            for rank, frame in enumerate(selected):
                base = {
                    k: v for k, v in
                    info_by_path.get(frame["path"], {"path": frame["path"]}).items()
                    if k != "embedding"
                }
                base.update({
                    "slot":             _SEQUENCE_SLOTS[rank % len(_SEQUENCE_SLOTS)],
                    "mogco_objectives": frame.get("nsga3_objectives", {}),
                    "engine":           "nsga3",
                })
                mogco_seq.append(base)

        except Exception as e:
            print(f"[v2] NSGA-III sequencing failed: {e}")

    _p(1.0, "Done")

    all_grades = [g["grade"] for g in gallery]
    strong = sum(1 for g in all_grades if g == GRADE_STRONG)
    mid    = sum(1 for g in all_grades if g == GRADE_MID)
    weak   = sum(1 for g in all_grades if g == GRADE_WEAK)
    print(f"[v2] SUMMARY: {len(gallery)} photos → Strong={strong}  Mid={mid}  Weak={weak}  (new={n}  cached={len(cached_rows)})")

    return {
        "gallery":        gallery,
        "mogco_sequence": mogco_seq,
        "strong":         strong,
        "mid":            mid,
        "weak":           weak,
        "total":          len(gallery),
        "pipeline":       "v2",
        "scan_mode":      scan_mode,
    }


_SEQUENCE_SLOTS = [
    "Opening",
    "Act 1",
    "Act 2",
    "Climax",
    "Resolution",
    "Coda",
    "Epilogue",
]


# ── File management (no-op stubs) ─────────────────────────────────────────────

def sort_files(folder_path: str, gallery: list[dict], copy: bool = False) -> dict:
    """Photos stay in their original folder. Returns a summary only."""
    strong = sum(1 for g in gallery if g.get("grade") == GRADE_STRONG)
    mid    = sum(1 for g in gallery if g.get("grade") == GRADE_MID)
    weak   = sum(1 for g in gallery if g.get("grade") == GRADE_WEAK)
    return {
        "moved":   0,
        "errors":  [],
        "dirs":    {},
        "summary": {"strong": strong, "mid": mid, "weak": weak},
        "message": "Photos remain in original folder (file moving disabled)",
    }
