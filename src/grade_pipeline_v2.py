"""
V2 grading pipeline orchestrator.

Dump → Grade → MOGCO-II
─────────────────────────
Step 1  Discover images in the folder.
Step 2  SigLIP So400M encodes images to 1152-d embeddings (INT8, GPU/CPU).
Step 3  SigLIP.unload() clears VRAM.
Step 4  Q-Align grades each image aesthetically (4-bit NF4, GPU).
Step 5  Q-Align.unload() clears VRAM.
Step 6  PersonalHead adjusts scores by learned user preference.
Step 7  Threshold → grade buckets:  ≥ 0.70 Strong ✅ / 0.40–0.70 Mid ⚠️ / < 0.40 Weak ❌
Step 8  Write to LanceDB.
Step 9  Auto-run MOGCO-II for a pre-built sequence.
Step 10 Return results in the same format as V1 (compatible with existing frontend).

Progress callback: progress(0.0–1.0, desc=str)
"""
from __future__ import annotations

import os
import json
import shutil
import threading
import numpy as np
from pathlib import Path
from typing import Callable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}

STRONG_THRESH = 0.70
MID_THRESH    = 0.40

GRADE_STRONG = "Strong ✅"
GRADE_MID    = "Mid ⚠️"
GRADE_WEAK   = "Weak ❌"

_EXIF_LOCK = threading.Lock()


# ── EXIF timestamp ─────────────────────────────────────────────────────────────

def _exif_ts(path: str) -> float:
    try:
        import piexif
        exif = piexif.load(path)
        for ifd in (piexif.ImageIFD.DateTime, piexif.ExifIFD.DateTimeOriginal):
            try:
                raw = exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal) or \
                      exif.get("0th",  {}).get(piexif.ImageIFD.DateTime)
                if raw:
                    from datetime import datetime
                    return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S").timestamp()
            except Exception:
                pass
    except Exception:
        pass
    return 0.0


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_v2(
    folder_path: str,
    preset: str = "Classic Street",
    force_rescan: bool = True,
    progress: Optional[Callable[[float, str], None]] = None,
    mogco_target: int = 5,
) -> dict:
    """
    Run the full V2 pipeline on `folder_path`.

    Returns a dict:
        gallery         list[dict]   per-photo result (V1-compatible keys)
        mogco_sequence  list[dict]   MOGCO-II output
        strong / mid / weak int      counts
        total           int
        pipeline        "v2"
    """
    _p = progress or (lambda f, d: None)

    # ── Step 1: Discover images ───────────────────────────────────────────────
    _p(0.01, "Scanning folder…")
    folder = Path(folder_path)
    paths  = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        return {"error": "No images found in folder.", "gallery": [], "total": 0}

    n = len(paths)
    _p(0.02, f"Found {n} images")

    # ── Step 2: SigLIP encoding ───────────────────────────────────────────────
    _p(0.03, "Loading SigLIP encoder…")
    try:
        from siglip_encoder import SigLIPEncoder
        enc  = SigLIPEncoder()
        embs = enc.encode_images(paths, progress=_p)      # (N, 1152)
        _p(0.50, "SigLIP done — releasing GPU…")
        enc.unload()
        del enc
        siglip_ok = True
    except Exception as e:
        print(f"[v2] SigLIP failed: {e} — using zero embeddings")
        embs      = np.zeros((n, 1152), dtype=np.float32)
        siglip_ok = False

    # ── Step 3: Q-Align scoring ───────────────────────────────────────────────
    _p(0.51, "Loading Q-Align grader…")
    per_photo_breakdowns: list[dict] = [{}] * n
    per_photo_critiques:  list[str]  = [""] * n
    try:
        from qalign_grader import get_grader
        grader = get_grader()
        if hasattr(grader, "score_detailed"):
            detailed = grader.score_detailed(paths, progress=_p)
            scores              = [d["score"]               for d in detailed]
            per_photo_breakdowns = [d.get("breakdown", {})  for d in detailed]
            per_photo_critiques  = [d.get("critique",  "")  for d in detailed]
        else:
            scores = grader.score(paths, progress=_p)
        grader.unload()
        del grader
    except Exception as e:
        print(f"[v2] Grader failed: {e} — using fallback scores")
        scores = [0.5] * n

    scores_arr = np.array(scores, dtype=np.float32)
    _p(0.86, "Scoring done")
    print(
        f"[v2] grader scores — min={scores_arr.min():.3f}  "
        f"max={scores_arr.max():.3f}  mean={scores_arr.mean():.3f}  "
        f"median={float(np.median(scores_arr)):.3f}"
    )

    # ── Step 4: PersonalHead adjustment ──────────────────────────────────────
    _p(0.87, "Applying personal preference…")
    pers = np.full(n, 0.5, dtype=np.float32)
    final_scores = scores_arr
    # Only blend when the model has actually been trained (weights file present).
    # An untrained PersonalHead outputs random ~0.5 values that pull Strong
    # scores below the 0.70 threshold, causing systematic mis-grading.
    _ph_weights = Path("cache/personal_head.pt")
    if _ph_weights.exists():
        print("[v2] PersonalHead weights found — blending 80/20")
        try:
            import personal_head as ph
            pers         = ph.score(embs)
            final_scores = 0.80 * scores_arr + 0.20 * pers
            print(
                f"[v2] PersonalHead pers — min={pers.min():.3f}  "
                f"max={pers.max():.3f}  mean={pers.mean():.3f}"
            )
        except Exception as _e:
            print(f"[v2] PersonalHead blend failed: {_e}")
    else:
        print("[v2] PersonalHead weights absent — using raw grader scores")

    # ── Step 5: Duplicate detection via SigLIP cosine similarity ─────────────
    _p(0.88, "Detecting duplicates…")
    cluster_ids: list[int] = [-1] * n
    sim_flags:   list[str] = [""]  * n

    if siglip_ok and n >= 2:
        try:
            from collections import defaultdict as _dd
            norms  = np.linalg.norm(embs, axis=1, keepdims=True)
            normed = embs / (norms + 1e-9)
            sims   = normed @ normed.T          # (N, N) cosine sims

            SIM_THRESH = 0.92                   # tuned for SigLIP-SO400M embeddings

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

            cid = 0
            for members in groups_d.values():
                if len(members) < 2:
                    continue
                members.sort(key=lambda i: float(final_scores[i]), reverse=True)
                best_i  = members[0]
                best_fn = Path(paths[best_i]).name
                best_sc = float(final_scores[best_i])
                for rank, idx in enumerate(members):
                    cluster_ids[idx] = cid
                    if rank == 0:
                        sim_flags[idx] = (
                            f"★ Best of {len(members)} similar shots "
                            f"(score {best_sc:.2f})"
                        )
                    else:
                        diff = best_sc - float(final_scores[idx])
                        sim_flags[idx] = (
                            f"\U0001f501 Duplicate — {best_fn} is better: "
                            f"higher overall score (+{diff:.2f})"
                        )
                cid += 1
        except Exception as e:
            print(f"[v2] Duplicate detection failed: {e}")

    # ── Step 6: Threshold → grades ───────────────────────────────────────────
    _p(0.89, "Bucketing into Strong / Mid / Weak…")
    print(
        f"[v2] final scores — min={final_scores.min():.3f}  "
        f"max={final_scores.max():.3f}  mean={final_scores.mean():.3f}  "
        f"median={float(np.median(final_scores)):.3f}"
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
        print(f"[v2]   {Path(paths[i]).name}: {s:.3f} → {g}")

    # ── Step 6: Collect EXIF timestamps ──────────────────────────────────────
    _p(0.90, "Reading EXIF…")
    timestamps = [_exif_ts(p) for p in paths]

    # ── Step 7: Write to LanceDB ──────────────────────────────────────────────
    _p(0.92, "Writing to LanceDB…")
    try:
        import lance_store as ls
        records = [
            {
                "path":           paths[i],
                "embedding":      embs[i].tolist(),
                "score":          float(final_scores[i]),
                "personal_score": float(pers[i]),
                "grade":          grades[i],
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

    # ── Step 8: Build gallery response (V1-compatible) ───────────────────────
    _p(0.94, "Building gallery…")
    gallery = []
    for i, path in enumerate(paths):
        fn = Path(path).name
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
            "exif_ts":         timestamps[i],
            "stars":           0,
            "reject":          grades[i] == GRADE_WEAK or (
                cluster_ids[i] >= 0 and not sim_flags[i].startswith("★")
            ),
            "sim_flag":        sim_flags[i],
            "cluster_id":      cluster_ids[i],
        })

    # ── Step 9: MOGCO-II sequence ─────────────────────────────────────────────
    _p(0.96, "Running MOGCO-II…")
    mogco_seq: list[dict] = []
    if siglip_ok and lance_ok:
        try:
            from mogco2 import run_mogco2
            result = run_mogco2(target=mogco_target, min_score=0.55)
            if result.get("sequence"):
                info_by_path = {g["path"]: g for g in gallery}
                for frame in result["sequence"]:
                    base = {
                        k: v for k, v in
                        info_by_path.get(frame["path"], {"path": frame["path"]}).items()
                        if k != "embedding"
                    }
                    base.update({
                        "slot":             frame.get("slot", ""),
                        "mogco_objectives": frame.get("mogco2_objectives", {}),
                        "engine":           "mogco2",
                    })
                    mogco_seq.append(base)
        except Exception as e:
            print(f"[v2] MOGCO-II failed: {e}")

    _p(1.0, "Done")

    strong = sum(1 for g in grades if g == GRADE_STRONG)
    mid    = sum(1 for g in grades if g == GRADE_MID)
    weak   = sum(1 for g in grades if g == GRADE_WEAK)
    print(f"[v2] SUMMARY: {n} photos → Strong={strong}  Mid={mid}  Weak={weak}")

    return {
        "gallery":        gallery,
        "mogco_sequence": mogco_seq,
        "strong":         strong,
        "mid":            mid,
        "weak":           weak,
        "total":          n,
        "pipeline":       "v2",
    }


# ── Step 5: File management ───────────────────────────────────────────────────

def sort_files(folder_path: str, gallery: list[dict], copy: bool = False) -> dict:
    """
    Move (or copy) graded photos into subfolders:
        <folder>/Strong/
        <folder>/Mid/
        <folder>/Weak/

    Returns a summary dict with counts and any errors.
    """
    root    = Path(folder_path)
    buckets = {GRADE_STRONG: root / "Strong",
               GRADE_MID:    root / "Mid",
               GRADE_WEAK:   root / "Weak"}
    for d in buckets.values():
        d.mkdir(exist_ok=True)

    moved, errors = 0, []
    for photo in gallery:
        src   = Path(photo["path"])
        grade = photo.get("grade", GRADE_MID)
        dest  = buckets.get(grade, buckets[GRADE_MID]) / src.name
        try:
            if copy:
                shutil.copy2(src, dest)
            else:
                shutil.move(str(src), dest)
            moved += 1
        except Exception as e:
            errors.append({"path": str(src), "error": str(e)})

    return {
        "moved":  moved,
        "errors": errors,
        "dirs":   {g: str(d) for g, d in buckets.items()},
    }
