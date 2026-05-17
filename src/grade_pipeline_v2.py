"""
V2 grading pipeline — Frontier 2026 (Pure Vision Regression Stack).

Step 1  Discover images in the folder.
Step 2  SigLIP-2 ViT-g/14 NaFlex → 1536-d embeddings + brief prompt embedding.
Step 3  Detect duplicates via cosine similarity (0.88 threshold).
Step 4  Vision Regression Stack.
            4a  SpecVLMPipeline: per-aspect breakdown (Composition, Lighting, Narrative…).
            4b  UniQAHead: pyiqa 'uniqa' unified backbone with YOLO11s-seg routing.
                           Route 1 (empty scene): sqrt(comp × light) from SpecVLM.
                           Route 2 (layered frame): UniQA on subject crop.
            4c  Semantic anchor: SigLIP-2 dot-product vs. user-brief embedding.
            4d  Score fusion: q * 0.75 + fa * 0.25 (VLP) or q (standard).
Step 5  PersonalHead adjusts scores by learned user preference (if weights present).
Step 6  Relative quantile buckets: top 25% → Strong / bottom 20% → Weak / rest → Mid.
Step 7  Write to LanceDB (1536-d IVF-PQ schema).
Step 8  Build gallery response (V1-compatible keys).
Step 9  NSGA-III multi-objective sequence: Score × Semantic_Vibe
            × Portfolio_Diversity × Aspect_Ratio_Balance.

VRAM Protocol (4-6 GB cards):
    SigLIP-2 FP16 (~4.5 GB singleton) + IQA heads (~2 GB peak during scoring)
    → release_iqa_models() after Step 4b → ~4.5 GB for LanceDB + NSGA-III.
    IQA singletons released after each run; SigLIP-2 persists for fast repeat runs.
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


def _generate_brief_variants(brief: str) -> list[str]:
    """
    Generate 3–5 semantically equivalent phrasings of the CD brief.

    Encoding all variants and averaging their embeddings produces a more robust
    semantic anchor than a single-text encoding — reduces sensitivity to exact
    wording and covers both noun-phrase and descriptive-sentence formulations.
    """
    t = brief.strip()
    candidates = [
        t,
        f"street photography: {t}",
        f"photographic mood and visual atmosphere: {t}",
        f"a photograph that captures {t}",
        f"visual style and aesthetic: {t}",
    ]
    seen: set[str] = set()
    out:  list[str] = []
    for v in candidates:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:5]


# ── Low-contrast genre reference prompts for TOPIQ bias correction ────────────
# SigLIP-2 cosine similarity to these is computed per-photo; if max > 0.70,
# TOPIQ NR's flat-texture penalty is partially reversed (up to ×1.20 correction).
_GENRE_REF_PROMPTS: list[str] = [
    "minimalist photography with clean empty surfaces and geometric simplicity",
    "fine art architectural photography with stark geometric structure and symmetry",
    "liminal space photograph — empty corridor, atmospheric, quietly unsettling",
]

# Fine-art pictorialism anchor — 3-prompt ensemble averaged and L2-normalised.
# Used for Vintage Lens Protocol weight adjustment and Soft-Focus Protection Gate.
_FINE_ART_PROMPTS: list[str] = [
    "A fine-art street photograph with cinematic low-light chiaroscuro.",
    "Intentional vintage lens softness, pictorialism aesthetic.",
    "Atmospheric moody street scene, artistic analog film texture, deep shadows.",
]

_EXIF_LOCK = threading.Lock()

# Module-level grader status — updated each run, read by /api/models/status.
_grader_status: dict = {
    "mode":        "idle",   # "idle" | "iqa_heads" | "clip_only"
    "verify_used": False,
    "photos_last": 0,
    "error":       None,
}

# ── SigLIP-2 singleton ────────────────────────────────────────────────────────
# Persists between grading runs — avoids 15-30 s weight-load overhead every run.
# VRAM budget: SigLIP-2 INT8 (~1.8 GB) + TOPIQ (~0.5 GB) = ~2.3 GB peak (safe).
# Released by release_grading_models() before Creative Mode loads LLMs.
_enc_singleton = None       # SigLIP2Encoder instance, or None
_text_emb_cache: dict = {}  # POS / NEG / ASPECT embeddings — static across runs


def release_grading_models() -> None:
    """Evict all grading singletons (SigLIP-2 + IQA heads) before Creative Mode loads LLMs."""
    global _enc_singleton, _text_emb_cache
    if _enc_singleton is not None:
        try:
            _enc_singleton.unload()
        except Exception:
            pass
        _enc_singleton = None
    _text_emb_cache.clear()
    try:
        from vision_grading_heads import release_iqa_models
        release_iqa_models()
    except Exception as _e_iqa:
        print(f"[v2] IQA singleton release skipped: {_e_iqa}")
    _vram_clear()
    print("[v2] All grading singletons released — VRAM freed for Creative Mode")


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
    Run the full V2 Vision Regression pipeline on `folder_path`.

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
    import lance_store as _ls_diag
    print(f"[v2] RUN START  folder={folder_path}  force_rescan={force_rescan}")
    print(f"[v2] LanceDB    path={_ls_diag._DB_DIR}  table={_ls_diag._TBL_NAME}")

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
                if rp_norm.startswith(fp_str) and float(row.get("score", 0)) >= 0.10:
                    cached_rows[rp_norm] = row
        except Exception as _ce:
            print(f"[v2] LanceDB cache check failed: {_ce}")

    paths = [p for p in all_paths if p not in cached_rows]
    n     = len(paths)
    print(f"[v2] Images     total={len(all_paths)}  cached(skipped)={len(cached_rows)}  to_grade={n}")

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
            "critique": "",
            "reasoning_log": "",
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

    # ── Step 1b: Cascaded Early-Exit Gate ────────────────────────────────────
    # Fastest checks first — CPU Laplacian blur, then brief-conditional YOLO gate.
    # Disqualified images get score 0.00 written to LanceDB; all downstream GPU
    # models see only the survivors, eliminating wasted compute.
    _blur_disqualified:   set[str] = set()
    _yolo_disqualified:   set[str] = set()
    _yolo_soft_penalized: set[str] = set()
    try:
        from early_exit_gate import run_early_exit_gate
        try:
            from specvlm_pipeline import _cd_brief_implies_empty as _implies_empty
            _run_yolo = _implies_empty()
        except Exception:
            _run_yolo = False

        _p(0.015, "Early-exit gate: Laplacian blur check…")
        _survivors, _blur_disqualified, _yolo_disqualified, _yolo_soft_penalized = (
            run_early_exit_gate(paths, run_yolo=_run_yolo)
        )

        _n_early_fail = len(_blur_disqualified) + len(_yolo_disqualified)
        if _n_early_fail:
            _p(0.025, f"Early-exit: {_n_early_fail} images disqualified → score 0.00")
            print(
                f"[v2] Early-exit gate: {len(_blur_disqualified)} blur-failed, "
                f"{len(_yolo_disqualified)} YOLO-failed → score 0.00, IQA skipped"
            )
    except Exception as _ee_err:
        print(f"[v2] Early-exit gate skipped ({_ee_err})")

    # ── Pre-flush: commit fail records before GPU stages begin ────────────────
    # Persists disqualified images to LanceDB immediately so that if the GPU
    # pipeline aborts mid-run, score=0.00 records are already in the store and
    # won't re-enter the processing queue on the next run.
    _prefail_paths = list(_blur_disqualified | _yolo_disqualified)
    if _prefail_paths:
        _p(0.027, f"Pre-flushing {len(_prefail_paths)} fail records to LanceDB…")
        try:
            import lance_store as _ls_pf
            _ls_pf.upsert_batch([{
                "path":           p,
                "embedding":      [0.0] * 1536,
                "score":          0.00,
                "personal_score": 0.5,
                "grade":          GRADE_WEAK,
                "reasoning_log":  "",
                "breakdown":      {
                    "disqualified": True,
                    "reason": "blur" if p in _blur_disqualified else "yolo",
                },
                "exif_ts":        0.0,
            } for p in _prefail_paths])
            print(f"[v2] Pre-flushed {len(_prefail_paths)} fail records to LanceDB")
        except Exception as _e_pf:
            print(f"[v2] Fail record pre-flush skipped: {_e_pf}")

    # ── Step 2: Bulk encoding ─────────────────────────────────────────────────
    # Singleton path (repeat runs): reuse encoder already in VRAM — no reload.
    # Cold path (first run): load, encode, cache static text embeddings, keep
    # encoder in VRAM as _enc_singleton for subsequent runs.
    global _enc_singleton, _text_emb_cache
    embs            = None
    embed_dim       = 1152
    siglip_ok       = False
    _pos_text_embs  = None
    _neg_text_embs  = None
    _aspect_pos     = None
    _aspect_neg     = None
    _aspect_names   = None
    _prompt_emb      = None   # (1536,) L2-normalised brief ensemble embedding for SemanticHead
    _genre_ref_embs  = None   # (3, 1536) low-contrast genre refs for TOPIQ bias correction
    _fine_art_anchor = None   # (1536,) averaged fine-art pictorialism anchor
    _enc_reused      = False

    if _enc_singleton is not None:
        _p(0.03, "SigLIP-2 cached — encoding images directly…")
        try:
            embs = _enc_singleton.encode_images(paths, progress=_p)
            if _text_emb_cache:
                _pos_text_embs  = _text_emb_cache["pos"]
                _neg_text_embs  = _text_emb_cache["neg"]
                _aspect_names   = _text_emb_cache["aspect_names"]
                _aspect_pos     = _text_emb_cache["aspect_pos"]
                _aspect_neg     = _text_emb_cache["aspect_neg"]
                _genre_ref_embs  = _text_emb_cache.get("genre_ref_embs")
                _fine_art_anchor = _text_emb_cache.get("fine_art_anchor")
            try:
                from specvlm_pipeline import _CD_BRIEF as _brief_text
                if _brief_text and _brief_text.strip():
                    _p(0.49, "Encoding brief ensemble for semantic alignment…")
                    _brief_variants = _generate_brief_variants(_brief_text)
                    _brief_raw  = _enc_singleton.encode_text(_brief_variants)  # (V, 1536)
                    _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                    _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                    _prompt_emb  = _prompt_emb.astype(np.float32)
                    print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
            except Exception as _e_brief:
                print(f"[v2] Brief embedding skipped: {_e_brief}")
            embed_dim   = 1536
            siglip_ok   = True
            _enc_reused = True
            print("[v2] Encoder: SigLIP-2 singleton reused — no VRAM reload")
        except Exception as _e_reuse:
            print(f"[v2] Singleton reuse failed ({_e_reuse}) — reloading encoder")
            try:
                _enc_singleton.unload()
            except Exception:
                pass
            _enc_singleton = None
            _text_emb_cache.clear()

    if not _enc_reused:
        import traceback as _tb
        _siglip_last_err: str = ""
        for _attempt, _kwargs in enumerate([
            {"device": "auto", "quantize": True},   # 1st: GPU INT8/FP16
            {"device": "cpu",  "quantize": False},  # 2nd: CPU FP16 (slow but correct)
        ]):
            try:
                from siglip2_encoder import SigLIP2Encoder
                from specvlm_pipeline import _POS_PROMPTS, _NEG_PROMPTS, _ASPECT_PROMPTS
                enc  = SigLIP2Encoder(**_kwargs, progress=_p)
                embs = enc.encode_images(paths, progress=_p)   # (N, 1536)

                # Encode aesthetic text references and cache for subsequent runs
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

                _text_emb_cache.update({
                    "pos":          _pos_text_embs,
                    "neg":          _neg_text_embs,
                    "aspect_names": _aspect_names,
                    "aspect_pos":   _aspect_pos,
                    "aspect_neg":   _aspect_neg,
                })

                # Cache "people" concept embedding for empty-brief creative direction
                _PEOPLE_PROMPTS = [
                    "people", "crowds", "pedestrians", "human figure", "faces",
                ]
                _ppl_raw  = enc.encode_text(_PEOPLE_PROMPTS)   # (5, 1536)
                _ppl_mean = _ppl_raw.mean(axis=0)
                _ppl_mean /= (np.linalg.norm(_ppl_mean) + 1e-9)
                try:
                    _cache_dir = Path("cache")
                    _cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(str(_cache_dir / "people_emb.npy"), _ppl_mean.astype(np.float32))
                    print("[v2] people_emb.npy saved for empty-brief CD gate")
                except Exception as _e_ppl:
                    print(f"[v2] people_emb save skipped: {_e_ppl}")

                # Encode low-contrast genre references for TOPIQ bias correction
                try:
                    _genre_raw   = enc.encode_text(_GENRE_REF_PROMPTS)          # (3, 1536)
                    _gnorms      = np.linalg.norm(_genre_raw, axis=1, keepdims=True)
                    _genre_ref_embs = (_genre_raw / (_gnorms + 1e-9)).astype(np.float32)
                    _text_emb_cache["genre_ref_embs"] = _genre_ref_embs
                    print("[v2] Genre reference embeddings cached for TOPIQ bias correction")
                except Exception as _e_genre_enc:
                    print(f"[v2] Genre ref encoding skipped: {_e_genre_enc}")

                # Encode fine-art pictorialism anchor (3-prompt ensemble, averaged + L2-norm)
                # Used for Vintage Lens Protocol and Soft-Focus Protection Gate.
                try:
                    _fa_raw   = enc.encode_text(_FINE_ART_PROMPTS)              # (3, 1536)
                    _fa_mean  = _fa_raw.mean(axis=0).astype(np.float64)
                    _fa_mean /= (np.linalg.norm(_fa_mean) + 1e-9)
                    _fine_art_anchor = _fa_mean.astype(np.float32)              # (1536,)
                    _text_emb_cache["fine_art_anchor"] = _fine_art_anchor
                    print("[v2] Fine-art anchor encoded and cached (3-prompt pictorialism ensemble)")
                except Exception as _e_fa:
                    print(f"[v2] Fine-art anchor encoding skipped: {_e_fa}")

                # Encode CD brief with prompt ensembling
                try:
                    from specvlm_pipeline import _CD_BRIEF as _brief_text
                    if _brief_text and _brief_text.strip():
                        _p(0.49, "Encoding brief ensemble for semantic alignment…")
                        _brief_variants = _generate_brief_variants(_brief_text)
                        _brief_raw  = enc.encode_text(_brief_variants)           # (V, 1536)
                        _prompt_emb = _brief_raw.mean(axis=0).astype(np.float64)
                        _prompt_emb /= (np.linalg.norm(_prompt_emb) + 1e-9)
                        _prompt_emb  = _prompt_emb.astype(np.float32)
                        print(f"[v2] Brief ensemble ({len(_brief_variants)} variants): '{_brief_text[:60]}'")
                except Exception as _e_brief:
                    print(f"[v2] Brief embedding skipped: {_e_brief}")

                _enc_singleton = enc   # keep in VRAM — evicted by release_grading_models()
                embed_dim = 1536
                siglip_ok = True
                _tag = "GPU" if _kwargs["device"] == "auto" else "CPU fallback"
                _p(0.50, "SigLIP-2 done — cached as singleton…")
                print(f"[v2] Encoder: SigLIP-2 NaFlex ({_tag})  dim={embed_dim}")
                break
            except Exception as e_siglip2:
                _siglip_last_err = str(e_siglip2)
                print(f"[v2] SigLIP-2 attempt {_attempt+1} failed: {e_siglip2}")
                if _attempt == 0:
                    print("[v2] Retrying SigLIP-2 on CPU…")
                else:
                    print("[v2] SigLIP-2 unavailable after all attempts.")
                    print(_tb.format_exc())

        # SigLIP-2 (1536-d) is required — all legacy encoders removed in Frontier 2026.
        if embed_dim != 1536:
            raise RuntimeError(
                f"SigLIP-2 failed to load on both GPU and CPU.\n"
                f"Reason: {_siglip_last_err}"
            )

    # Flush caching allocator — singleton weights remain resident in VRAM
    _vram_clear()

    # ── Step 3: Duplicate detection ───────────────────────────────────────────
    _p(0.50, "Detecting duplicates…")
    cluster_ids:     list[int] = [-1] * n
    sim_flags:       list[str] = [""] * n
    to_rate_indices: list[int] = list(range(n))
    _comp_eligible:  set[str]  = set(paths)   # default: all paths eligible for composition

    if siglip_ok and n >= 2:
        try:
            from collections import defaultdict as _dd
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-9)
            sims   = normed @ normed.T

            SIM_THRESH = 0.96   # true burst duplicates only (same frame ±ms)

            parent = list(range(n))
            def _find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            # Vectorized: find all above-threshold pairs in one numpy call
            dup_i, dup_j = np.where(np.triu(sims > SIM_THRESH, k=1))
            for i, j in zip(dup_i.tolist(), dup_j.tolist()):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj

            groups_d: dict = _dd(list)
            for i in range(n):
                groups_d[_find(i)].append(i)

            # Populate cluster_ids for all photos in duplicate groups (size >= 2)
            _comp_eligible: set[str] = set()
            for root, members in groups_d.items():
                if len(members) >= 2:
                    for i in members:
                        cluster_ids[i] = root
                    _comp_eligible.add(paths[root])   # only representative gets composition
                else:
                    _comp_eligible.add(paths[members[0]])   # singleton: always eligible

            n_clustered = sum(1 for c in cluster_ids if c >= 0)
            n_reps      = sum(1 for root, members in groups_d.items() if len(members) >= 2)
            if n_clustered:
                print(f"[v2] Duplicate detection: {n_clustered} images in clusters"
                      f" ({n_reps} representatives) — burst dedup saves depth/seg/chiaroscuro"
                      f" for {n_clustered - n_reps} cluster members")

        except Exception as e:
            print(f"[v2] Duplicate detection failed: {e}")
            to_rate_indices = list(range(n))

    # Step 3b: YOLO gate handled by early-exit gate (Step 1b) before SigLIP-2.
    # _yolo_disqualified and _yolo_soft_penalized are already populated above.

    # ── Step 4: Vision Regression Stack ──────────────────────────────────────
    scores                = np.full(n, 0.5, dtype=np.float32)
    per_photo_breakdowns: list[dict] = [{} for _ in range(n)]

    # Stamp all early-exit disqualified photos (score 0.00, skip IQA)
    _all_disqualified = _blur_disqualified | _yolo_disqualified
    for i, p in enumerate(paths):
        if p in _all_disqualified:
            scores[i] = 0.00

    # Exclude disqualified images from IQA scoring
    to_rate_indices = [i for i in to_rate_indices if paths[i] not in _all_disqualified]
    paths_to_rate = [paths[i] for i in to_rate_indices]

    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _props = _torch.cuda.get_device_properties(0)
            _free  = (_props.total_memory - _torch.cuda.memory_reserved(0)) / 1e9
            print(f"[v2] VRAM before IQA heads: {_free:.2f} GB free / {_props.total_memory/1e9:.2f} GB total")
        del _torch
    except Exception:
        pass

    # ── Step 4a: SpecVLM scoring (instant — reuses SigLIP-2 embeddings) ─────────
    # r.score = weighted CLIP score across ALL aspects (Composition, Lighting,
    # Narrative, Human/Culture).  This is the primary semantic signal.
    # comp_scores_rated keeps the Composition sub-dimension for Step 4d formula display.
    _p(0.51, "SpecVLM scoring…")
    vlm_scores_rated  = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    comp_scores_rated = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    _raw_comp_by_path: dict[str, float] = {}  # SpecVLM Composition before any override

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
                _raw_comp = float(r.breakdown.get("Composition", r.score))
                vlm_scores_rated[local_i]    = float(r.score)
                comp_scores_rated[local_i]   = _raw_comp
                _raw_comp_by_path[paths[idx]] = _raw_comp   # track before any override
                per_photo_breakdowns[idx]     = r.breakdown or {}
                scores[idx]                   = float(r.score)   # survive IQA failure

        _p(0.55, "SpecVLM done — running IQA heads…")
        print(
            f"[v2] SpecVLM scores: min={vlm_scores_rated.min():.3f}  "
            f"max={vlm_scores_rated.max():.3f}  mean={vlm_scores_rated.mean():.3f}"
        )

    except Exception as e_clip:
        print(f"[v2] Composition scoring failed: {e_clip}")

    _vram_clear()  # Free SpecVLM VRAM before IQA heads load

    # Pre-compute luminance stats — shared by composition analysis, ChiaroscuroHead,
    # and Vintage Lens Protocol in Step 4d.
    _p(0.555, "Computing luminance stats…")

    def _lum_stats(path: str):
        try:
            from PIL import Image as _PILI
            with _PILI.open(path) as _raw:
                img = _raw.convert("RGB")
            img.thumbnail((128, 128), _PILI.LANCZOS)   # 10-100× faster; lum stats are invariant to scale
            _arr = np.array(img, dtype=np.float32)
            _Y   = 0.299 * _arr[:, :, 0] + 0.587 * _arr[:, :, 1] + 0.114 * _arr[:, :, 2]
            return float(_Y.mean()), float(_Y.std())
        except Exception:
            return 128.0, 60.0   # neutral defaults — no VLP trigger

    from concurrent.futures import ThreadPoolExecutor as _TPELUM
    with _TPELUM(max_workers=min(8, len(paths_to_rate) or 1)) as _lpool:
        lum_stats_rated = list(_lpool.map(_lum_stats, paths_to_rate))

    # ── Step 4b: Vision IQA Head (UniQA unified backbone) ───────────────────────
    # scan_mode bypasses IQA — Scan uses composition scores (already set in Step 4a).
    # Full grading runs:
    #   1. run_composition_analysis (Depth → Seg → Chiaroscuro) — within run_vision_heads
    #   2. UniQAHead with YOLO11s-seg routing (empty-scene / layered-frame / standard)
    composition_overrides:    dict[str, float] = {}
    chiaroscuro_flags:        dict[str, bool]  = {}
    person_detected_dict:     dict[str, bool]  = {}
    framing_obstruction_dict: dict[str, bool]  = {}

    if scan_mode:
        _p(0.84, "Scan mode — IQA heads skipped, using SpecVLM scores…")
        print(f"[v2] Scan mode: IQA skipped, {len(paths_to_rate)} photos at CLIP speed")
        tech_scores_rated      = vlm_scores_rated.copy()
        aesthetic_scores_rated = vlm_scores_rated.copy()
    else:
        _p(0.56, f"IQA heads — scoring {len(paths_to_rate)} images…")
        try:
            from vision_grading_heads import run_vision_heads

            iqa_embs  = embs[np.array(to_rate_indices)]   # (M, 1536)
            _vlm_bds  = [per_photo_breakdowns[idx] for idx in to_rate_indices]

            iqa_out = run_vision_heads(
                image_paths         = paths_to_rate,
                image_embeddings    = iqa_embs,
                prompt_embedding    = _prompt_emb,
                clip_scores         = vlm_scores_rated,
                genre_ref_embs      = _genre_ref_embs,
                lum_stats           = lum_stats_rated,
                progress            = _p,
                comp_eligible_paths = _comp_eligible,
                vlm_breakdowns      = _vlm_bds,
            )

            tech_scores_rated        = iqa_out["quality"]                      # (M,) UniQA
            aesthetic_scores_rated   = iqa_out["quality"]                      # (M,) UniQA
            iqa_breakdowns           = iqa_out["breakdowns"]                   # list[dict]
            composition_overrides    = iqa_out.get("composition_overrides",  {})
            chiaroscuro_flags        = iqa_out.get("chiaroscuro_flags",      {})
            person_detected_dict     = iqa_out.get("person_detected",        {})
            framing_obstruction_dict = iqa_out.get("framing_obstruction",    {})

            for local_i, idx in enumerate(to_rate_indices):
                per_photo_breakdowns[idx].update(iqa_breakdowns[local_i])
                # Apply over-the-shoulder portrait composition override
                _opath = paths[idx]
                if _opath in composition_overrides:
                    per_photo_breakdowns[idx]["Composition"] = composition_overrides[_opath]

            _p(0.84, "IQA heads done — releasing singletons…")
            try:
                from vision_grading_heads import release_iqa_models as _rel_iqa
                _rel_iqa()
            except Exception as _e_rel:
                print(f"[v2] IQA singleton release skipped: {_e_rel}")
            _vram_clear()
            print(f"[v2] IQA heads: {len(paths_to_rate)} photos scored")
            if composition_overrides:
                print(f"[v2] Composition overrides: {len(composition_overrides)} images "
                      f"(over-the-shoulder portrait → 0.85)")
            _n_ch = sum(1 for v in chiaroscuro_flags.values() if v)
            if _n_ch:
                print(f"[v2] Chiaroscuro: {_n_ch}/{len(chiaroscuro_flags)} images flagged "
                      f"(VLP forced, YOLO soft penalty waived)")

        except Exception as e_iqa:
            _p(0.84, f"IQA heads failed ({type(e_iqa).__name__}) — using SpecVLM scores…")
            print(f"[v2] IQA heads failed ({e_iqa}) — SpecVLM-only mode")
            tech_scores_rated      = vlm_scores_rated.copy()
            aesthetic_scores_rated = vlm_scores_rated.copy()
            _vram_clear()

    _grader_status.update({"mode": "iqa_heads" if not scan_mode else "clip_only",
                           "verify_used": False, "photos_last": len(paths_to_rate), "error": None})

    # ── Step 4c: Fine-art anchor similarity + Min-Max stretch ────────────────
    # Raw cosine sims cluster in a narrow band (e.g., 0.28–0.42) because all street
    # photos share some similarity to the anchor. A naive (sim+1)/2 map compresses
    # everything into 0.64–0.71 — useless for differentiation.
    # Min-Max normalization stretches the batch distribution to full [0,1] range,
    # giving fine-art semantic alignment equal mathematical weight to technical metrics.
    _fine_art_sims_all = np.zeros(n, dtype=np.float32)
    if _fine_art_anchor is not None:
        _fine_art_sims_all = (embs @ _fine_art_anchor).astype(np.float32)  # (n,) raw cosine

    fine_art_scores_rated = np.full(len(paths_to_rate), 0.5, dtype=np.float32)
    if _fine_art_anchor is not None and len(to_rate_indices) > 0:
        _fa_sims_rated = _fine_art_sims_all[np.array(to_rate_indices)]
        _fa_lo   = float(_fa_sims_rated.min())
        _fa_hi   = float(_fa_sims_rated.max())
        _fa_span = max(_fa_hi - _fa_lo, 1e-4)
        # Stretch rated-batch sims to [0,1]
        fine_art_scores_rated = np.clip(
            (_fa_sims_rated - _fa_lo) / _fa_span, 0.0, 1.0
        ).astype(np.float32)
        # Normalise all-image sims using same batch statistics so the Soft-Focus Gate
        # threshold is consistent with the per-image fine-art scores used in Step 4d.
        _fine_art_sims_all = np.clip(
            (_fine_art_sims_all - _fa_lo) / _fa_span, 0.0, 1.0
        ).astype(np.float32)
        print(
            f"[v2] Fine-art sims (raw): min={_fa_lo:.3f}  max={_fa_hi:.3f}  "
            f"→ stretched to [0,1]  rated mean={fine_art_scores_rated.mean():.3f}"
        )

    # ── Step 4d: Score fusion with Vintage Lens Protocol + Anchor Floor ────────
    # Base formula: q * 0.35 + q * 0.65 = q  (t == a == UniQA quality score)
    #
    # Vintage Lens Protocol fires when BOTH conditions hold:
    #   (a) Image is low-light or low-contrast: mean_lum < 40 (0-255) OR std < 30
    #   (b) UniQA quality ≥ 0.556 — confirms intentional quality above neutral.
    # On trigger: quality weight drops to 0.75; freed 0.25 reallocates to fine-art sem.
    #   Triggered formula: q * 0.10 + q * 0.65 + fine_art_sem * 0.25 = q * 0.75 + fa * 0.25
    #
    # YOLO Soft Penalty: silhouette-in-dark-scene photos stay in IQA but receive
    # -0.15 to their fused score before Anchor Floor evaluation.
    #
    # Creative Director Anchor Floor: if UniQA quality ≥ 0.611 OR fine-art normalised
    # similarity ≥ 0.75 → enforce overall_score = max(score, 0.65).
    # This guarantees compositionally elite or fine-art-aligned photos can never drop
    # below the Strong bucket threshold due to IQA penalties.
    _AES_VLP_THRESHOLD    = 0.556   # UniQA quality threshold for VLP trigger
    _AES_ANCHOR_THRESHOLD = 0.611   # UniQA quality threshold for Anchor Floor
    _ANCHOR_FLOOR         = 0.65
    _FA_ANCHOR_THRESHOLD  = 0.75                 # normalised fine-art sim threshold

    _p(0.86, "Fusing scores (Dynamic Routing + VLP + Anchor Floor)…")
    _vlp_count       = 0
    _chiaroscuro_vlp = 0
    _anchor_count    = 0
    _penalty_count   = 0
    _route1_count    = 0
    _route2_count    = 0
    _fo_count        = 0   # Framing Obstruction sub-count

    for local_i, idx in enumerate(to_rate_indices):
        t  = float(tech_scores_rated[local_i])
        a  = float(aesthetic_scores_rated[local_i])
        fa = float(fine_art_scores_rated[local_i])
        mean_lum, std_lum = lum_stats_rated[local_i]
        _path = paths[idx]

        _comp_score  = float(per_photo_breakdowns[idx].get("Composition",   0.5))
        _light_score = float(per_photo_breakdowns[idx].get("Lighting",      0.5))
        _hc_score    = float(per_photo_breakdowns[idx].get("Human/Culture", 0.5))
        _narr_score  = float(per_photo_breakdowns[idx].get("Narrative",     0.5))
        _raw_comp    = _raw_comp_by_path.get(_path, _comp_score)
        _has_person  = person_detected_dict.get(_path, True)
        _is_ots      = _path in composition_overrides
        _is_fo       = bool(framing_obstruction_dict.get(_path, False))

        _is_chiaroscuro = bool(chiaroscuro_flags.get(_path, False))

        # ── Route 2A: Framing Obstruction ─────────────────────────────────────
        # Fires when a large off-centre person (>30% frame area, center_x <0.35
        # or >0.65) was identified as intentional framing.  UniQA already ran on
        # the isolated subject crop.  Override breakdowns with protected values:
        #   Composition = 0.85  (intentional compositional choice)
        #   Lighting    = 0.78  (protects low-key chiaroscuro atmosphere)
        #   Technical   = max(uniqa_crop, 0.75)  (subject-crop sharpness floor)
        # Formula: fused = (tech * 0.30) + (comp * 0.40) + (lighting * 0.30)
        if _is_fo:
            _tech_fo  = max(t, 0.75)
            _comp_fo  = 0.85
            _light_fo = 0.78
            per_photo_breakdowns[idx]["Composition"] = _comp_fo
            per_photo_breakdowns[idx]["Lighting"]    = _light_fo
            fused = _tech_fo * 0.30 + _comp_fo * 0.40 + _light_fo * 0.30
            scores[idx] = float(np.clip(fused, 0.0, 1.0))
            _route2_count += 1
            _fo_count     += 1
            print(
                f"[v2] Route 2A Framing Obstruction: {Path(_path).name}  "
                f"tech={_tech_fo:.2f} comp={_comp_fo} light={_light_fo} → fused={fused:.3f}"
            )
            continue

        # ── Route 2B: Intentional Layered Frame Portrait (OTS) ────────────────
        # Fires when: strong human presence (HC ≥ 0.65) + SpecVLM raw composition
        # was penalized below 0.35 by a foreground obstruction + OTS portrait
        # detected (foreground OOF person + sharp midground subject).
        # Formula: subject-crop quality (40%) + Human/Culture (30%) + Narrative (30%)
        # so that high-context documentary work (shopkeeper, vendor, elder) is not
        # buried by a flat technical crop score.
        # Internal floor: max(fused, 0.65) when storytelling is strong.
        if _hc_score >= 0.65 and _raw_comp < 0.35 and _is_ots:
            per_photo_breakdowns[idx]["Composition"] = 0.82
            fused = a * 0.40 + _hc_score * 0.30 + _narr_score * 0.30
            if _hc_score >= 0.70 and _narr_score >= 0.60:
                if fused < _ANCHOR_FLOOR:
                    fused = _ANCHOR_FLOOR
                    _anchor_count += 1
            scores[idx] = float(np.clip(fused, 0.0, 1.0))
            _route2_count += 1
            print(
                f"[v2] Route 2B Layered Frame: {Path(_path).name}  "
                f"HC={_hc_score:.2f} Narr={_narr_score:.2f} UniQA={a:.2f} → fused={fused:.3f}"
            )
            continue

        # ── Route 1: Empty Scene ──────────────────────────────────────────────
        # Fires when YOLO finds zero human instances (class 0, area ≥ 0.5%).
        # Formula: comp * 0.40 + light * 0.30 + uniqa * 0.30
        # Linear split so a strong geometric composition (high comp, moderate light)
        # is not penalised by the geometric mean's multiplicative collapse.
        if not _has_person:
            fused = _comp_score * 0.40 + _light_score * 0.30 + a * 0.30
            scores[idx] = float(np.clip(fused, 0.0, 1.0))
            _route1_count += 1
            print(
                f"[v2] Route 1 Empty Scene: {Path(_path).name}  "
                f"Comp={_comp_score:.2f} Light={_light_score:.2f} UniQA={a:.2f} → fused={fused:.3f}"
            )
            continue

        # ── Standard routing: VLP + Anchor Floor ─────────────────────────────
        # Chiaroscuro flag forces VLP — intentional dramatic contrast is fine art.
        _vlp = _is_chiaroscuro or (
            (mean_lum < 40.0 or std_lum < 30.0) and (a >= _AES_VLP_THRESHOLD)
        )
        if _vlp:
            fused = t * 0.10 + a * 0.65 + fa * 0.25
            _vlp_count += 1
            if _is_chiaroscuro:
                _chiaroscuro_vlp += 1
        else:
            fused = t * 0.35 + a * 0.65

        # YOLO soft penalty for dark-scene silhouettes (waived for chiaroscuro).
        if _path in _yolo_soft_penalized and not _is_chiaroscuro:
            fused = max(0.0, fused - 0.15)
            _penalty_count += 1

        # Anchor Floor — fired AFTER soft penalty so elite photos can override it.
        # Triggers on: strong UniQA quality OR fine-art alignment OR documentary
        # street work with exceptional Human/Culture (≥ 0.70) + Narrative (≥ 0.60).
        _doc_strong = _hc_score >= 0.70 and _narr_score >= 0.60
        if a >= _AES_ANCHOR_THRESHOLD or fa >= _FA_ANCHOR_THRESHOLD or _doc_strong:
            if fused < _ANCHOR_FLOOR:
                fused = _ANCHOR_FLOOR
                _anchor_count += 1

        scores[idx] = float(np.clip(fused, 0.0, 1.0))

    if _route1_count:
        print(f"[v2] Route 1 (Empty Scene): {_route1_count} images — geometric formula (no HC drag)")
    if _route2_count:
        print(f"[v2] Route 2 (Layered Frame): {_route2_count} images — Composition protected @ 0.82")
    if _vlp_count:
        print(
            f"[v2] Vintage Lens Protocol: {_vlp_count}/{len(to_rate_indices)} triggered "
            f"(low-light/low-contrast + UniQA ≥ {_AES_VLP_THRESHOLD:.3f}"
            + (f"; {_chiaroscuro_vlp} via chiaroscuro flag" if _chiaroscuro_vlp else "")
            + ")"
        )
    if _penalty_count:
        print(f"[v2] YOLO soft penalty: {_penalty_count} images penalised -0.15 (dark silhouette)")
    if _anchor_count:
        print(
            f"[v2] Anchor Floor: {_anchor_count} images floored to {_ANCHOR_FLOOR} "
            f"(UniQA ≥ {_AES_ANCHOR_THRESHOLD:.3f} OR fine-art sim ≥ 0.75)"
        )

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

    # ── Step 5c: Soft-Focus Protection Gate ──────────────────────────────────
    # Images with high cosine similarity to the fine-art anchor (> 0.68) receive
    # a flat +0.15 score boost applied before quantile bucketing. This prevents
    # atmospheric, low-contrast, or soft-focus fine-art frames from being dropped
    # to Weak purely because pixel-sharpness metrics ranked them lower.
    # The gate is additive, not multiplicative — it shifts the score up the
    # distribution without changing relative ordering within the fine-art cohort.
    _sfpg_count = 0
    if _fine_art_anchor is not None:
        for i in range(n):
            if float(_fine_art_sims_all[i]) > 0.68:
                final_scores[i] = float(np.clip(float(final_scores[i]) + 0.15, 0.0, 1.0))
                _sfpg_count += 1
        if _sfpg_count:
            print(
                f"[v2] Soft-Focus Gate: {_sfpg_count} images boosted +0.15 "
                f"(fine_art_sim > 0.68)"
            )

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

    # ── Step 6: Absolute grade thresholds ────────────────────────────────────
    # Strong ≥ 0.60  |  Mid 0.41–0.59  |  Weak ≤ 0.40
    final_scores = np.clip(np.nan_to_num(final_scores, nan=0.15), 0.10, 1.0)
    final_scores = np.round(final_scores, 2)

    _p(0.89, "Applying grade thresholds…")
    print(
        f"[v2] final scores — min={final_scores.min():.2f}  "
        f"max={final_scores.max():.2f}  mean={final_scores.mean():.2f}  "
        f"median={float(np.median(final_scores)):.2f}"
    )
    print(f"[v2] Thresholds — Weak < 0.41  |  Mid 0.41–0.59  |  Strong ≥ 0.60")

    grades = []
    for i, s in enumerate(final_scores):
        if s >= 0.60:
            g = GRADE_STRONG
        elif s >= 0.41:
            g = GRADE_MID
        else:
            g = GRADE_WEAK
        grades.append(g)
        print(f"[v2]   {Path(paths[i]).name}: {s:.2f} → {g}")

    # ── Step 7: EXIF + LanceDB ────────────────────────────────────────────────
    _p(0.90, "Reading EXIF…")
    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(16, len(paths) or 1)) as _pool:
        timestamps = list(_pool.map(_exif_ts, paths))

    _p(0.92, "Writing to LanceDB (bulk upsert)…")
    lance_ok = False
    try:
        import lance_store as ls
        import traceback as _tb_lance
        print(f"[v2] LanceDB WRITE START — {n} records → {ls._DB_DIR}")
        # Build all records in memory first, then a single vectorised upsert.
        # Per-photo breakdown includes all CLIP aspect dimensions, not just
        # the high-level aesthetic/personal summary.
        lance_records: list[dict] = []
        for i in range(n):
            bd = {"aesthetic": round(float(scores_arr[i]), 3),
                  "personal":  round(float(pers[i]),       3)}
            if per_photo_breakdowns[i]:
                bd.update(per_photo_breakdowns[i])
            lance_records.append({
                "path":           paths[i],
                "embedding":      embs[i].tolist(),
                "score":          float(final_scores[i]),
                "personal_score": float(pers[i]),
                "grade":          grades[i],
                "reasoning_log":  "",          # LLM layer removed; field kept for schema compat
                "breakdown":      bd,
                "exif_ts":        timestamps[i],
            })
        ls.upsert_batch(lance_records)
        ls.compact_after_write()
        lance_ok = True
        print(f"[v2] LanceDB WRITE OK — {len(lance_records)} records committed")
    except Exception as _e_lance:
        import traceback as _tb_lance
        print(f"[v2] !!! LanceDB WRITE FAILED: {_e_lance}")
        _tb_lance.print_exc()
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
            "critique":        "",
            "reasoning_log":   "",
            "is_verified":     False,
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

    # ── Step 8b: Atomic server-side catalog.json write ───────────────────────
    # Keeps catalog.json in sync immediately after LanceDB upsert, regardless of
    # whether the frontend later calls POST /api/catalog/save.  Atomic rename
    # prevents a partially-written file from corrupting the next app load.
    try:
        import time as _cat_time
        _cat_dir  = Path(__file__).resolve().parent.parent / "cache"
        _cat_dir.mkdir(parents=True, exist_ok=True)
        _cat_path = _cat_dir / "catalog.json"
        _cat_photos  = [{k: v for k, v in g.items() if k != "embedding"} for g in gallery]
        _cat_folders = list(dict.fromkeys(str(Path(g["path"]).parent) for g in gallery))
        _cat_payload = json.dumps({
            "photos":   _cat_photos,
            "folders":  _cat_folders,
            "saved_at": _cat_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False)
        _cat_tmp = _cat_path.with_suffix(".json.tmp")
        _cat_tmp.write_text(_cat_payload, encoding="utf-8")
        _cat_tmp.replace(_cat_path)
        print(f"[v2] catalog.json → {len(_cat_photos)} photos (atomic write)")
    except Exception as _e_cat:
        print(f"[v2] catalog.json write skipped: {_e_cat}")

    # ── Step 9: NSGA-III multi-objective sequencing ───────────────────────────
    _p(0.96, "Running NSGA-III (strict literal constraints)…")
    mogco_seq:   list[dict] = []
    mogco_error: str        = ""
    if siglip_ok and lance_ok:
        try:
            from nsga3_sequencer import run_nsga3_sequence_with_vlm, SequencerConstraintError

            # Pass brief so the sequencer can apply literal pre-filter
            try:
                from specvlm_pipeline import _CD_BRIEF as _seq_brief
            except Exception:
                _seq_brief = ""

            # Pass Strong + Mid candidates with embeddings, reasoning logs, and breakdown
            seq_candidates = [
                {
                    "path":          g["path"],
                    "score":         g["score"],
                    "embedding":     np.array(g["embedding"], dtype=np.float32),
                    "reasoning_log": g["reasoning_log"],
                    "breakdown":     g.get("breakdown", {}),
                }
                for g in gallery
                if g["grade"] in (GRADE_STRONG, GRADE_MID)
            ]

            selected = run_nsga3_sequence_with_vlm(
                seq_candidates,
                target     = mogco_target,
                progress   = _p,
                brief      = _seq_brief,
            )

            info_by_path = {g["path"]: g for g in gallery}
            for rank, frame in enumerate(selected):
                base = {
                    k: v for k, v in
                    info_by_path.get(frame["path"], {"path": frame["path"]}).items()
                    if k != "embedding"
                }
                base.update({
                    "slot":             frame.get("slot", _SEQUENCE_SLOTS[rank % len(_SEQUENCE_SLOTS)]),
                    "slot_role":        frame.get("slot_role", ""),
                    "slot_score":       frame.get("slot_score", 0.0),
                    "mogco_objectives": frame.get("nsga3_objectives", {}),
                    "engine":           "nsga3",
                })
                mogco_seq.append(base)

        except SequencerConstraintError as e:
            mogco_error = str(e)
            print(f"[v2] NSGA-III constraint error: {e}")
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
        "mogco_error":    mogco_error,
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
