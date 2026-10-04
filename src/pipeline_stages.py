"""
pipeline_stages.py — run_v2's stages, as functions with declared interfaces.

Why
---
``grade_pipeline_v2.run_v2`` is 2,411 lines in one function: 69% of its file,
sharing roughly a hundred locals. Two concrete costs, neither cosmetic:

  * Static analysis has given up. Pylance reports hundreds of spurious
    "not accessed" warnings for that function because it cannot follow the
    scope, which means no IDE navigation, no rename refactoring, and no
    unused-variable signal on the one function where it would matter most.
  * Nothing has a testable boundary. Every behaviour is reachable only by
    running a whole cull, so the 150-odd tests around it verify the parts that
    were already extracted and almost nothing inside it.

Approach
--------
Extracted incrementally, seam by seam, verifying after each that a real 514-photo
cull produces a byte-identical grade distribution. Stages taken in order of how
narrow their interface is, not how big they are — a stage with two inputs and one
output is safe to move; the 440-line encoder block with a dozen live outputs is
not, and is deliberately still in place.

Each stage takes what it needs and returns what it produces. No stage reaches
back into run_v2's locals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


@dataclass
class GateResult:
    """Outcome of the cascaded early-exit gate."""
    survivors: list = field(default_factory=list)
    blur_disqualified: set = field(default_factory=set)
    yolo_disqualified: set = field(default_factory=set)
    yolo_soft_penalized: set = field(default_factory=set)
    technical_disq: dict = field(default_factory=dict)   # path -> "flat"|"unreadable"

    @property
    def n_failed(self) -> int:
        return (len(self.blur_disqualified) + len(self.yolo_disqualified)
                + len(self.technical_disq))


def run_gate_stage(paths: list, progress: Callable) -> GateResult:
    """Step 1b — cheapest checks first, so GPU stages only ever see survivors.

    CPU Laplacian blur, then a brief-conditional person gate. Disqualified images
    are recorded with score 0.00 downstream; they never reach a model.

    Degrades to "everything survives" if the gate itself fails: a broken filter
    must not be able to empty a photographer's cull. That is why the exception
    handler returns an empty GateResult rather than propagating — the caller
    treats no-disqualifications as no-op.
    """
    out = GateResult()
    try:
        from early_exit_gate import run_early_exit_gate
        try:
            from specvlm_pipeline import _cd_brief_implies_empty as _implies_empty
            run_yolo = _implies_empty()
        except Exception:
            run_yolo = False

        progress(0.015, "Checking image files…")

        # This gate decodes every image, so on a real folder it owns minutes of
        # wall time. Reporting only before and after left the bar frozen at 2%
        # for the whole pass — the single worst "is it stuck?" moment in a cull.
        # 0.015 -> 0.030 is its lane; spend it proportionally.
        def _gate_progress(done: int, total: int) -> None:
            frac = 0.015 + 0.015 * (done / max(1, total))
            progress(frac, f"Checking image files… {done}/{total}")

        (out.survivors, out.blur_disqualified, out.yolo_disqualified,
         out.yolo_soft_penalized, out.technical_disq) = run_early_exit_gate(
            paths, run_yolo=run_yolo, on_progress=_gate_progress)

        if out.n_failed:
            progress(0.025, f"{out.n_failed} unusable images set aside")
            print(f"[v2] Early-exit gate: {len(out.technical_disq)} technical-failed "
                  f"(flat/unreadable), {len(out.blur_disqualified)} blur-failed, "
                  f"{len(out.yolo_disqualified)} YOLO-failed → score 0.00, "
                  f"models skipped")
    except Exception as err:
        print(f"[v2] Early-exit gate skipped ({err})")
    return out


def flush_gate_failures(gate: GateResult, embed_dim: int, weak_grade: str,
                        progress: Callable) -> int:
    """Commit disqualified images to LanceDB BEFORE any GPU stage runs.

    If the GPU pipeline later aborts mid-run, these score=0.00 records are
    already stored, so the next run does not re-queue files that were already
    judged unusable.

    Technical failures (flat / unreadable) never reached a model, so they are
    marked ``technical_pass=False`` with an explicit note — otherwise the UI
    shows an empty breakdown and it looks like the grade is broken rather than
    like the file is.
    """
    failed = list(gate.blur_disqualified | gate.yolo_disqualified
                  | set(gate.technical_disq))
    if not failed:
        return 0
    progress(0.027, f"Saving {len(failed)} results…")

    def _breakdown(path: str) -> dict:
        if path in gate.technical_disq:
            reason = gate.technical_disq[path]          # "flat" | "unreadable"
            note = ("Technical Failure: Invalid or Empty Image Frame"
                    if reason == "flat" else
                    "Technical Failure: Unreadable or Locked Image File")
            return {"disqualified": True, "reason": reason,
                    "technical_pass": False, "notes": note}
        reason = "blur" if path in gate.blur_disqualified else "yolo"
        return {"disqualified": True, "reason": reason,
                "technical_pass": True, "notes": ""}

    try:
        import lance_store
        lance_store.upsert_batch([{
            "path":           p,
            "embedding":      [0.0] * embed_dim,
            "score":          0.00,
            "personal_score": 0.5,
            "grade":          weak_grade,
            "reasoning_log":  "",
            "breakdown":      _breakdown(p),
            "exif_ts":        0.0,
        } for p in failed])
        print(f"[v2] Pre-flushed {len(failed)} fail records to LanceDB")
    except Exception as err:
        print(f"[v2] Fail record pre-flush skipped: {err}")
    return len(failed)


def drop_unreadable_rows(paths: list, embs, np_mod):
    """Remove zero-vector rows — RAW files whose preview could not be read.

    encode_worker emits a zero vector rather than failing the batch. Those rows
    must leave the pipeline entirely, before dedup/scoring/gallery: judged in
    place they become 0.00 "Weak" and sit in the results as though the
    photograph were bad, rather than unread. Returns (paths, embs, n).
    """
    if embs is None or len(embs) != len(paths) or not paths:
        return paths, embs, len(paths)
    norms = np_mod.linalg.norm(embs, axis=1)
    keep = [i for i in range(len(paths)) if float(norms[i]) >= 1e-6]
    if len(keep) == len(paths):
        return paths, embs, len(paths)
    dropped = [paths[i] for i in range(len(paths)) if float(norms[i]) < 1e-6]
    print(f"[v2] RAW read error — dropped {len(dropped)} unreadable file(s) "
          f"from grading: " + ", ".join(Path(p).name for p in dropped[:10]))
    paths = [paths[i] for i in keep]
    embs = embs[np_mod.asarray(keep, dtype=np_mod.intp)]
    return paths, embs, len(paths)


def write_catalog(gallery: list) -> None:
    """Sync catalog.json right after the LanceDB upsert.

    MERGES rather than replaces. Writing only the folder just graded erased
    every other folder from the gallery even though their grades were safe in
    LanceDB — 1,745 photos were hidden behind 135 that way. A catalog must
    accumulate. catalog_store does the atomic tmp+rename.
    """
    try:
        import catalog_store
        catalog_store.merge_write(
            [{k: v for k, v in g.items() if k != "embedding"} for g in gallery],
            tag="(pipeline)")
    except Exception as err:
        import traceback
        print(f"[v2] catalog.json write skipped: {err}")
        traceback.print_exc()


def duplicate_groups(embs, sim_thresh: float = 0.96, chunk: int = 512) -> list:
    """Near-duplicate groups (lists of indices, size >= 2) — COMPLETE linkage.

    Every pair inside a group is >= sim_thresh. The old union-find was single
    linkage: A~B and B~C put A and C together however different they were, and
    a walk along one harbour chained into ONE 207-photo "duplicate" group whose
    least-similar pair was 0.75 (2026-10-03, 445-photo SD card). Every non-best
    member of a group is hidden in the gallery, so chaining hid 380 of 445
    photos. Complete linkage keeps real bursts together and stops the chain.

    Pairs are merged strongest-first with a deterministic tie-break, so the
    same embeddings always give the same groups. Memory stays O(chunk x n).
    """
    import numpy as np
    n = len(embs) if embs is not None else 0
    if embs is None or n < 2:
        return []
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = (embs / (norms + 1e-9)).astype(np.float32)

    # Candidate pairs as compact arrays (12 bytes/pair), not Python tuples —
    # a folder of near-identical frames yields O(n^2) pairs.
    ps, pi, pj = [], [], []
    for r0 in range(0, n, chunk):
        r1 = min(r0 + chunk, n)
        block = normed[r0:r1] @ normed.T
        for li in range(r1 - r0):
            block[li, : r0 + li + 1] = -1.0
        bi, bj = np.nonzero(block > sim_thresh)
        ps.append(block[bi, bj].astype(np.float32))
        pi.append((bi + r0).astype(np.int32))
        pj.append(bj.astype(np.int32))
        del block
    if not ps:
        return []
    s_all, i_all, j_all = np.concatenate(ps), np.concatenate(pi), np.concatenate(pj)
    del ps, pi, pj
    order = np.lexsort((j_all, i_all, -s_all))      # strongest first, index tie-break

    gid = list(range(n))
    members = {i: [i] for i in range(n)}
    for i, j in zip(i_all[order].tolist(), j_all[order].tolist()):
        gi, gj = gid[i], gid[j]
        if gi == gj:
            continue
        a, b = members[gi], members[gj]
        if len(a) * len(b) > 1:
            if float((normed[a] @ normed[b].T).min()) <= sim_thresh:
                continue
        if len(a) < len(b):
            gi, gj, a, b = gj, gi, b, a
        a.extend(b)
        for k in b:
            gid[k] = gi
        del members[gj]
    return [sorted(m) for m in members.values() if len(m) >= 2]


def cluster_similar(embs, sim_thresh: float = 0.96) -> list:
    """Cluster embeddings into near-duplicate groups; returns per-item cluster ids.

    Same union-find, same strict threshold (0.96 = true burst duplicates,
    same frame ±ms) and same row-blocked memory shape as the live grade path
    in grade_pipeline_v2 Step 3 — factored out so the ALL-CACHED fast path
    (which skips every GPU stage) can still mark duplicates. Without this,
    a re-run of an already-encoded folder returned zero clusters and an
    empty Duplicates tab even when bursts existed (2026-09-10: the 2,754-photo
    re-run wrote a catalog with no duplicate groups at all).

    Returns a fresh list of ints (−1 for photos not in a cluster of 2+).
    Degrades to all-−1 on failure: advisory UI data, never worth the grade.
    """
    import numpy as np
    n = len(embs) if embs is not None else 0
    cluster_ids = [-1] * n
    if embs is None or n < 2:
        return cluster_ids
    try:
        for members in duplicate_groups(np.asarray(embs), sim_thresh):
            for i in members:
                cluster_ids[i] = members[0]
    except Exception as err:
        import traceback
        print(f"[stages] cluster_similar failed: {err}")
        traceback.print_exc()
    return cluster_ids


# Bump whenever a change alters the score a photo receives (fusion formula,
# gates, thresholds, models). Stored grades carry it in breakdown["_grade_sig"];
# a stale version is re-graded on the next cull instead of served for ever.
# Before 2026-09-28 ANY stored row with score >= 0.10 counted as "already
# graded", so grader fixes never reached existing photos and Scan-quality
# rows were served as final by later full-quality culls.
GRADER_VERSION = "2026-10-04b"  # camera JPEGs analysed from their MPF preview; RAW+JPEG pairs graded once. Was 04a: shake rule (directional smear >= 0.20 -> Weak); thresholds back on rating-time scores


def grade_mode(scan_mode: bool, deep_grade: bool) -> str:
    return "scan" if scan_mode else ("deep" if deep_grade else "full")


def grade_signature(scan_mode: bool, deep_grade: bool) -> str:
    return f"{GRADER_VERSION}|{grade_mode(scan_mode, deep_grade)}"


def row_is_reusable(row: dict, *, scan_mode: bool, deep_grade: bool) -> bool:
    """May this stored grade be served instead of re-grading the photo?

    Only if it was made by the CURRENT grader version, and in the mode this
    run asks for — a full cull never serves a Scan grade (Scan skips the
    image-quality stage, so its scores are systematically different). A Scan
    the user explicitly asked for may reuse any current-version grade: a
    richer grade is fine for a quick pass. Unsigned rows (written before
    signatures existed) are never reusable — they are re-graded once.
    """
    bd = row.get("breakdown") or {}
    sig = bd.get("_grade_sig") if isinstance(bd, dict) else None
    if not isinstance(sig, str) or "|" not in sig:
        return False
    version, mode = sig.split("|", 1)
    if version != GRADER_VERSION:
        return False
    if scan_mode:
        return True
    return mode == grade_mode(False, deep_grade)


def soft_focus_gate(final_scores, raw_sims, stretched_sims, *, absolute: bool,
                    raw_thresh: float) -> tuple:
    """Step 5c — +0.15 for fine-art-like frames already scoring >= 0.50.

    ``absolute`` (the calibrated Pro encoder) tests each photo's RAW cosine to
    the fine-art anchor against one fixed threshold, so a photo's boost never
    depends on which other photos share its batch. The legacy test on the
    batch-stretched similarity (> 0.75) remains only for encoder tiers with no
    rated photos to calibrate on. Returns (new scores, number boosted); the
    input array is not modified.
    """
    import numpy as np
    scores = np.array(final_scores, dtype=np.float64, copy=True)
    signal = np.asarray(raw_sims if absolute else stretched_sims, dtype=np.float64)
    fires = (signal >= raw_thresh) if absolute else (signal > 0.75)
    hit = fires & (scores >= 0.50)
    scores[hit] = np.clip(scores[hit] + 0.15, 0.0, 1.0)
    return scores, int(hit.sum())


def mark_duplicate_groups(cluster_ids: list, final_scores, paths: list) -> list:
    """Step 5b — label each near-duplicate cluster with a winner and its losers.

    Ranked on the FINAL score, not the pre-taste one, so the frame the pipeline
    actually rates highest is the one kept. Returns a fresh ``sim_flags`` list
    (one entry per photo, "" for anything not in a cluster of 2+) rather than
    mutating a caller list — the caller reads it in exactly one place.

    Degrades to all-empty flags on failure: the flags are advisory UI text, and
    losing them must not cost the grades computed above.
    """
    flags = [""] * len(cluster_ids)
    try:
        groups: dict = {}
        for i, cid in enumerate(cluster_ids):
            if cid >= 0:
                groups.setdefault(cid, []).append(i)
        for members in groups.values():
            if len(members) < 2:
                continue
            members.sort(key=lambda i: float(final_scores[i]), reverse=True)
            best_fn = Path(paths[members[0]]).name
            best_sc = float(final_scores[members[0]])
            for rank, idx in enumerate(members):
                if rank == 0:
                    flags[idx] = (f"★ Best of {len(members)} similar shots "
                                  f"(score {best_sc:.2f})")
                else:
                    diff = best_sc - float(final_scores[idx])
                    flags[idx] = (f"\U0001f501 Duplicate — {best_fn} is better: "
                                  f"higher overall score (+{diff:.2f})")
    except Exception as err:
        import traceback
        print(f"[v2] Sim-flag assignment failed: {err}")
        traceback.print_exc()
    return flags


def assign_grades(final_scores, paths: list, np_mod, *,
                  strong_thresh: float, mid_thresh: float,
                  strong_label: str, mid_label: str, weak_label: str):
    """Step 6 — clamp, round, then bucket on ABSOLUTE thresholds.

    The grade reflects the photo itself, not its rank within the batch. Quantile
    calibration (75th/20th percentile + historical anchor) forced a ~25% Strong /
    20% Weak split every run, so a uniformly excellent shoot still got "Weak"
    tails and a poor one still got "Strong". Do not reintroduce relative buckets.

    Thresholds and labels are parameters, not constants declared here:
    ``grade_pipeline_v2`` stays their single declaration. Returns
    (final_scores, grades) — scores come back because they are clamped in place.
    """
    final_scores = np_mod.clip(np_mod.nan_to_num(final_scores, nan=0.15), 0.10, 1.0)
    final_scores = np_mod.round(final_scores, 2)

    print(f"[v2] final scores — min={final_scores.min():.2f}  "
          f"max={final_scores.max():.2f}  mean={final_scores.mean():.2f}  "
          f"median={float(np_mod.median(final_scores)):.2f}")
    print(f"[v2] Thresholds (absolute) — Weak < {mid_thresh:.2f}  |  "
          f"Mid {mid_thresh:.2f}–{strong_thresh - 0.01:.2f}  |  "
          f"Strong ≥ {strong_thresh:.2f}")

    grades = []
    for i, s in enumerate(final_scores):
        if s >= strong_thresh:
            g = strong_label
        elif s >= mid_thresh:
            g = mid_label
        else:
            g = weak_label
        grades.append(g)
        print(f"[v2]   {Path(paths[i]).name}: {s:.2f} → {g}")
    return final_scores, grades


# EXIF lives in the file HEADER, but piexif.load(<path>) slurps the entire file
# before parsing it. On a RAW shoot that is ~12 MB read per frame — across a
# free-RAM-derived thread pool it was the second-largest stage transient in the
# 514-image run (0.83 GB) purely to read a timestamp. Reading a bounded prefix
# and handing piexif the BYTES gives identical results: verified byte-identical
# timestamps at a 64 KB prefix on both 11.6 MB RW2 and JPEG files. The full-file
# read stays as a fallback so an unusual layout (EXIF past the prefix) still
# resolves exactly as before — worst case is the old behaviour.
_EXIF_PREFIX_BYTES = 262144        # 256 KB — 4x the verified-sufficient 64 KB


def exif_timestamp(path: str) -> float:
    """One file's DateTimeOriginal as a POSIX timestamp, 0.0 if absent."""
    def _parse(src) -> float:
        import piexif
        exif = piexif.load(src)
        raw = (exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
               or exif.get("0th", {}).get(piexif.ImageIFD.DateTime))
        if raw:
            from datetime import datetime
            return datetime.strptime(raw.decode(), "%Y:%m:%d %H:%M:%S").timestamp()
        return 0.0

    try:
        with open(path, "rb") as fh:
            head = fh.read(_EXIF_PREFIX_BYTES)
        ts = _parse(head)
        if ts:
            return ts
    except Exception:
        pass
    try:
        return _parse(path)        # fallback: whole file, as before
    except Exception:
        return 0.0


def read_exif_timestamps(paths: list) -> list:
    """Step 7 — timestamps for every path, in order.

    Thread count is bounded by free RAM rather than fixed, because each worker
    holds a whole prefix (and, on the fallback path, a whole file). Order is
    preserved by ``pool.map``, so the count has no effect on the result.
    """
    from concurrent.futures import ThreadPoolExecutor
    try:
        from early_exit_gate import decode_workers
        workers = decode_workers(len(paths), mb_per_worker=60.0, hi=16)
    except Exception:
        workers = min(8, len(paths) or 1)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths) or 1))) as pool:
        return list(pool.map(exif_timestamp, paths))


def attach_face_signals(gallery: list, person_detected: dict) -> int:
    """Face + subject-focus signals. REPORT ONLY — touches no score.

    Aesthetic grading has no notion of whether the subject is actually sharp,
    which is a fact about the photo rather than a matter of taste: a portrait
    focused on the ear behind the eye is a reject no aesthetic model catches.

    Gated on the person detector, which has already run. Measured at ~430
    ms/photo, so running it folder-wide would add ~3.7 min to a 514-photo cull
    to answer a question that only applies where there is a person — on street
    work only 8 of 60 frames have a face at all.

    Returns the number of photos annotated.
    """
    try:
        import face_signals
    except Exception as err:
        print(f"[v2] Face signals skipped: {err}")
        return 0
    if not face_signals.available():
        return 0

    # Decode AND detect on a thread pool (2026-10-04). One photo at a time this
    # cost ~90 ms each — ~20 s for 218 photos in a 600-RAW cull. face_signals
    # keeps one YuNet detector per thread, so concurrent detection is safe;
    # map() keeps order. Same functions, same model: identical results.
    todo = [e for e in gallery if person_detected.get(e["path"], False)]

    def _load(path):
        try:
            return face_signals._load_small(path)
        except Exception:
            return None, "unreadable"

    def _metrics(loaded):
        img, _src = loaded
        if img is None:
            out = dict(face_signals.face_metrics(None))
            out["unreadable"] = True
            return out
        return face_signals.face_metrics(img)

    from concurrent.futures import ThreadPoolExecutor
    import os as _os_fs
    n = 0
    with ThreadPoolExecutor(max_workers=min(8, _os_fs.cpu_count() or 4),
                            thread_name_prefix="face-decode") as _pool:
        def _one(e):
            try:
                return _metrics(_load(e["path"]))
            except Exception:
                return None

        for entry, metrics in zip(todo, _pool.map(_one, todo)):
            if metrics is None:
                continue
            _attach_one_face(entry, metrics)
            n += 1 if entry.get("face") else 0
    if n:
        print(f"[v2] Face/focus signals computed for {n} photos")
    return n


def _attach_one_face(entry: dict, metrics: dict) -> None:
    """Copy one photo's face metrics (and normalized boxes) onto its entry."""
    if metrics.get("faces_detected"):
        entry["face"] = {k: v for k, v in metrics.items() if k != "faces"}
        # Persist normalized boxes (grade-time path) — the Close-Ups panel
        # recomputes crops on demand via /api/photo-faces for rows graded
        # before this field existed, but fresh grades carry geometry so
        # face filters and any future offline UI work without re-detect.
        boxes = [{"norm": f.get("norm"), "confidence": round(f["confidence"], 3),
                  "area_frac": round(f["area_frac"], 5)}
                 for f in (metrics.get("faces") or [])[:12] if f.get("norm")]
        if boxes:
            entry["face"]["boxes"] = boxes
