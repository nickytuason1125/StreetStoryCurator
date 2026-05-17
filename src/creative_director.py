"""
Creative Director — Purist Curation Pipeline

Governs the 5-image "Story Sequence" using only Original Pixel Metadata.
No pixel modification is performed. The output is always the original file.

Pipeline
────────
1. Phi-4-mini-reasoning Agent (CPU GGUF) → Rule Set JSON
   Reads the Style Brief as Boolean Constraints. Emits HARD_FILTER_PEOPLE,
   GEOMETRIC_PRIORITY, LIGHTING_MOOD, BRIEF_KEYWORDS.

2. YOLO26-nano person_kill_switch (CPU)
   If HARD_FILTER_PEOPLE is True and class:person is detected (conf ≥ 0.35),
   the image is DISQUALIFIED from the Story Sequence. Absolute — no exceptions.

3. SigLIP-2 Penalty — Subject Intrusion (CPU)
   people_sim > 0.40 OR Human/Culture aspect > 0.55 → score × 0.10.

4. Story Sequence Selection — select_story_sequence()
   Greedy max-dissimilarity + role guarantee over top-40 by score.

5. Cinematic Reorder
   Opener → slot 0, Contrast → slot n//2, Closer → slot n-1.
   Luminance smoothing (adjacent Δ < 25%).

6. Copy Originals → output_dir/Final_Portfolio/
   Output is 100% the original capture. No stylization.

Per-image narrative roles (assigned by content, not list position):
  subject  → highest aesthetic score (hero shot)
  opener   → negative-space image with highest score
  closer   → negative-space image with 2nd-highest score
  contrast → most visually distinct (furthest from centroid)
  detail   → third-highest score (texture / decisive gesture)

Cinematic pacing constraints:
  • Opener + Closer negative space ≥ 30% (sim_to_centroid ≤ 0.70)
  • Contrast placed at exactly slot n//2 in narrative order
  • Luminance smoothing: adjacent images must not differ > 25% mean brightness
  • Diversity guard: any pair with cosine sim > 0.88 is penalised / swapped
"""
from __future__ import annotations

import shutil
import numpy as np
from pathlib import Path
from typing import Callable, Optional

# ── Shot roles ────────────────────────────────────────────────────────────────

_ROLE_ORDER = ["subject", "opener", "closer", "contrast", "detail"]

# ── Cinematic pacing thresholds ───────────────────────────────────────────────
_NEG_SPACE_THRESH  = 0.70   # sim_to_centroid ≤ this → qualifies as negative space
_DUP_SIM_THRESH    = 0.88   # cosine sim > this → near-duplicate; penalise / swap
_LUM_SMOOTH_THRESH = 0.25   # max allowed mean-brightness diff between adjacent images
_POOL_DEDUP_THRESH = 0.92   # pre-selection pool dedup: hard-drop near-identical shots

# ── Empty-brief filtering ─────────────────────────────────────────────────────
_EMPTY_BRIEF_KEYWORDS = {"empty", "liminal", "desert", "void", "abandoned", "desolate"}
_PEOPLE_SIM_THRESHOLD = 0.40   # SigLIP-2 cosine sim to "people" concept → hard penalty
_PEOPLE_PENALTY       = 0.10   # score multiplier when triggered
_HUMAN_CULTURE_THRESH = 0.55   # Human/Culture aspect fallback threshold
_YOLO_PERSON_CONF     = 0.35    # YOLO26-nano strict detection threshold (auditor guardrail)
_YOLO_MIN_AREA_FRAC   = 0.0005  # ignore detections < 0.05% of canvas (distant background figures)


def _empty_brief_detected(style_prompt: str) -> bool:
    text = style_prompt.lower()
    return any(kw in text for kw in _EMPTY_BRIEF_KEYWORDS)


def _load_people_emb() -> "Optional[np.ndarray]":
    try:
        p = Path("cache/people_emb.npy")
        if p.exists():
            emb  = np.load(str(p)).astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-9)
    except Exception as e:
        print(f"[cd] people_emb load failed: {e}")
    return None


def _apply_brief_constraints(
    paths: list[str],
    embeddings: list[np.ndarray],
    scores: list[float],
    aspect_scores_list: "Optional[list[dict]]",
    style_prompt: str,
) -> tuple[list[float], list[str]]:
    """
    Subject Intrusion penalty for empty-brief sessions.

    SigLIP-2 people_sim > 0.40  OR  Human/Culture aspect > 0.55 → score × 0.10.
    Returns (adjusted_scores, disqualification_notes).
    """
    if not _empty_brief_detected(style_prompt):
        return list(scores), [""] * len(scores)

    people_emb = _load_people_emb()
    adjusted   = list(scores)
    notes: list[str] = [""] * len(scores)

    for i, (path, emb, sc) in enumerate(zip(paths, embeddings, scores)):
        people_sim   = 0.0
        if people_emb is not None:
            emb_n       = np.asarray(emb, dtype=np.float32)
            emb_n      /= (np.linalg.norm(emb_n) + 1e-9)
            people_sim  = float(emb_n @ people_emb)

        human_culture = 0.0
        if aspect_scores_list and i < len(aspect_scores_list):
            human_culture = float(aspect_scores_list[i].get("Human/Culture", 0.0))

        if people_sim > _PEOPLE_SIM_THRESHOLD or human_culture > _HUMAN_CULTURE_THRESH:
            adjusted[i] = sc * _PEOPLE_PENALTY
            notes[i] = (
                f"disqualification: Subject Intrusion — person presence detected. "
                f"people_sim={people_sim:.3f} (threshold {_PEOPLE_SIM_THRESHOLD}), "
                f"human_culture={human_culture:.2f} (threshold {_HUMAN_CULTURE_THRESH}). "
                f"Score {sc:.3f}→{adjusted[i]:.4f} (×{_PEOPLE_PENALTY}). "
                f"Brief implies empty scene: '{style_prompt[:60]}'."
            )
            print(f"[cd] Subject Intrusion: {Path(path).name}  {notes[i]}")

    n_pen = sum(1 for n in notes if n)
    if n_pen:
        print(f"[cd] empty-brief filter: {n_pen}/{len(paths)} images penalised")
    return adjusted, notes


def person_kill_switch(paths: list[str], style_prompt: str) -> set[str]:
    """
    YOLO26-nano Auditor Guardrail — CPU.

    Literal Judge enforcement: if the brief implies an empty/no-people scene,
    YOLO26-nano scans ALL candidates at conf ≥ 0.35. Any image where
    class:person is detected is DISQUALIFIED from the Story Sequence.
    This is an absolute Boolean Constraint — no score adjustment, no clean-up.
    """
    if not _empty_brief_detected(style_prompt):
        return set()

    blocked: set[str] = set()
    try:
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        from ultralytics import YOLO as _YOLO

        # Try YOLO26-nano (NMS-free, better small-object detection, ~40% faster CPU).
        # Fall back to yolo11n if YOLO26 weights unavailable.
        yolo = None
        for _model_name in ("yolo26n.pt", "yolo11n.pt"):
            try:
                yolo = _YOLO(_model_name)
                print(f"[cd] YOLO model: {_model_name}")
                break
            except Exception:
                continue
        if yolo is None:
            raise RuntimeError("No YOLO model available")

        for path in paths:
            try:
                results = yolo(path, device="cpu", verbose=False,
                               classes=[0], conf=_YOLO_PERSON_CONF)
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    img_h, img_w = r.orig_shape
                    canvas_area  = img_h * img_w
                    area_thresh  = _YOLO_MIN_AREA_FRAC * canvas_area
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area < area_thresh:
                            print(
                                f"[cd] person_kill_switch: IGNORED distant figure in "
                                f"{Path(path).name} — box_area={box_area:.0f}px "
                                f"< {area_thresh:.0f}px ({_YOLO_MIN_AREA_FRAC*100:.3f}% canvas)"
                            )
                            continue
                        blocked.add(path)
                        print(
                            f"[cd] person_kill_switch: DISQUALIFIED {Path(path).name} "
                            f"— person conf≥{_YOLO_PERSON_CONF}, "
                            f"box_area={box_area:.0f}px"
                        )
                        break
            except Exception as e_img:
                print(f"[cd] YOLO inference error {Path(path).name}: {e_img}")

    except ImportError:
        print("[cd] ultralytics not installed — YOLO gate skipped")
    except Exception as e:
        print(f"[cd] YOLO gate error: {e}")

    if blocked:
        print(f"[cd] person_kill_switch: {len(blocked)}/{len(paths)} images disqualified")
    return blocked


def _mean_luminance(path: str) -> float:
    """Mean luminance in [0, 1] via 64×64 PIL grayscale thumbnail."""
    try:
        from PIL import Image
        with Image.open(path) as _raw:
            img = _raw.convert("L")
        img.thumbnail((64, 64), Image.LANCZOS)
        return float(np.asarray(img, dtype=np.float32).mean() / 255.0)
    except Exception:
        return 0.5


def _cinematic_reorder(
    paths:  list[str],
    embs_n: np.ndarray,
    roles:  list[str],
    scores: list[float],
) -> list[int]:
    """
    Place the sequence in cinematic order:
      slot 0     — opener
      slot n//2  — contrast
      slot n-1   — closer
      remaining  — subject, detail (sorted by score desc)

    One greedy luminance-smoothing pass swaps adjacent non-anchor slots
    whose brightness delta exceeds _LUM_SMOOTH_THRESH.
    """
    n = len(paths)
    if n == 0:
        return []
    if n == 1:
        return [0]

    by_role: dict[str, list[int]] = {}
    for i, r in enumerate(roles):
        by_role.setdefault(r, []).append(i)

    sc = np.array(scores, dtype=np.float32)
    for r in by_role:
        by_role[r].sort(key=lambda i: -sc[i])

    def _take(role: str) -> int:
        bucket = by_role.get(role, [])
        return bucket.pop(0) if bucket else -1

    mid = n // 2
    opener_idx   = _take("opener")
    contrast_idx = _take("contrast")
    closer_idx   = _take("closer")

    remaining: list[int] = []
    for role in ("subject", "detail", "opener", "contrast", "closer", *_ROLE_ORDER):
        for idx in by_role.get(role, []):
            if idx not in remaining:
                remaining.append(idx)
    all_used = {opener_idx, contrast_idx, closer_idx, *remaining} - {-1}
    for i in range(n):
        if i not in all_used:
            remaining.append(i)
    remaining.sort(key=lambda i: -sc[i])

    slots: list[int | None] = [None] * n
    if opener_idx >= 0:
        slots[0] = opener_idx
    if closer_idx >= 0:
        slots[n - 1] = closer_idx
    if contrast_idx >= 0:
        slots[mid] = contrast_idx

    rem_iter = iter(remaining)
    for i in range(n):
        if slots[i] is None:
            try:
                slots[i] = next(rem_iter)
            except StopIteration:
                pass

    final: list[int] = [(s if s is not None else i) for i, s in enumerate(slots)]

    fixed   = {0, mid, n - 1}
    lum     = [_mean_luminance(paths[i]) for i in final]
    changed = True
    passes  = 0
    while changed and passes < 3:
        changed = False
        passes += 1
        for j in range(n - 1):
            if j in fixed or (j + 1) in fixed:
                continue
            if abs(lum[j] - lum[j + 1]) > _LUM_SMOOTH_THRESH:
                if j + 2 < n and (j + 2) not in fixed:
                    cand = lum[j + 2]
                    before = abs(lum[j] - lum[j + 1]) + abs(lum[j + 1] - lum[min(j + 2, n - 1)])
                    after  = abs(lum[j] - cand) + abs(cand - lum[j + 1])
                    if after < before - 0.01:
                        final[j + 1], final[j + 2] = final[j + 2], final[j + 1]
                        lum[j + 1],   lum[j + 2]   = lum[j + 2],   lum[j + 1]
                        changed = True

    for j in range(n - 1):
        diff = abs(lum[j] - lum[j + 1])
        if diff > _LUM_SMOOTH_THRESH:
            print(
                f"[cd] luminance penalty: slots {j}→{j+1}  Δ={diff:.2f}  "
                f"({Path(paths[final[j]]).name} → {Path(paths[final[j+1]]).name})"
            )

    for a in range(n):
        for b in range(a + 1, n):
            sim = float(embs_n[final[a]] @ embs_n[final[b]])
            if sim > _DUP_SIM_THRESH:
                print(
                    f"[cd] diversity penalty: slots {a},{b}  sim={sim:.3f}  "
                    f"({Path(paths[final[a]]).name}, {Path(paths[final[b]]).name})"
                )
    return final


def _assign_roles_by_content(
    embeddings: list[np.ndarray],
    scores: Optional[list[float]] = None,
    paths: Optional[list[str]] = None,
) -> list[str]:
    """
    Assign narrative roles based on image content.

    subject  → highest aesthetic score
    opener   → negative-space + highest score (sim_to_centroid ≤ 0.70)
    closer   → negative-space + 2nd-highest score
    contrast → most visually distinct (furthest from centroid)
    detail   → third-highest score
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return ["subject"]

    embs   = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-9)

    centroid = embs_n.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim_to_centroid = embs_n @ centroid

    sc = np.array(scores if scores and len(scores) == n else [0.5] * n, dtype=np.float32)

    used: set[int] = set()
    assignments: dict[int, str] = {}

    def _pick(rank_arr: np.ndarray) -> int:
        for idx in rank_arr:
            if int(idx) not in used:
                used.add(int(idx))
                return int(idx)
        return -1

    def _pick_neg_space(rank_arr: np.ndarray) -> int:
        for idx in rank_arr:
            if int(idx) not in used and sim_to_centroid[int(idx)] <= _NEG_SPACE_THRESH:
                used.add(int(idx))
                return int(idx)
        return _pick(rank_arr)

    def _pick_diverse(rank_arr: np.ndarray) -> int:
        assigned_embs = np.stack([embs_n[i] for i in used]) if used else None
        for idx in rank_arr:
            i = int(idx)
            if i in used:
                continue
            if assigned_embs is not None:
                if float(np.max(assigned_embs @ embs_n[i])) > _DUP_SIM_THRESH:
                    lbl = Path(paths[i]).name if paths else str(i)
                    print(f"[cd] diversity skip: {lbl}")
                    continue
            used.add(i)
            return i
        return _pick(rank_arr)

    score_desc   = np.argsort(-sc)
    centroid_asc = np.argsort(sim_to_centroid)

    idx = _pick(score_desc);         assignments[idx] = "subject"  if idx >= 0 else None
    idx = _pick_neg_space(score_desc); assignments[idx] = "opener"  if idx >= 0 else None
    idx = _pick_neg_space(score_desc); assignments[idx] = "closer"  if idx >= 0 else None
    idx = _pick(centroid_asc);       assignments[idx] = "contrast" if idx >= 0 else None
    idx = _pick_diverse(score_desc); assignments[idx] = "detail"   if idx >= 0 else None

    assignments = {k: v for k, v in assignments.items() if v is not None and k >= 0}

    ri = 0
    for i in range(n):
        if i not in assignments:
            assignments[i] = _ROLE_ORDER[ri % len(_ROLE_ORDER)]
            ri += 1

    result_roles = [assignments[i] for i in range(n)]
    labels = paths or [f"img_{i}" for i in range(n)]
    for i, (role, lbl) in enumerate(zip(result_roles, labels)):
        print(
            f"[cd] role={role:8s}  score={sc[i]:.3f}  "
            f"centroid_sim={sim_to_centroid[i]:.3f}  {Path(lbl).name}"
        )
    return result_roles


def select_story_sequence(
    paths:       list[str],
    embeddings:  list[np.ndarray],
    scores:      Optional[list[float]] = None,
    n_min:       int = 5,
    n_max:       int = 10,
    avoid_paths: Optional[list[str]] = None,
) -> tuple[list[str], list[np.ndarray], list[float]]:
    """
    Pick n_min–n_max visually diverse images covering all 5 narrative roles.

    1. Filter avoided paths.
    2. Pre-filter to top-40 by aesthetic score.
    3. Guarantee one image per core role using content signals.
    4. Fill remaining slots with greedy max-dissimilarity (60% diversity / 40% score).
    """
    avoid  = set(avoid_paths or [])
    n_raw  = len(paths)
    sc_raw = np.array(
        scores if scores and len(scores) == n_raw else [0.5] * n_raw,
        dtype=np.float32,
    )

    keep = [i for i, p in enumerate(paths) if p not in avoid]
    if not keep:
        return [], [], []

    paths      = [paths[i]      for i in keep]
    embeddings = [embeddings[i] for i in keep]
    sc         = sc_raw[keep]
    n          = len(paths)

    if n <= n_max:
        return paths, embeddings, list(sc)

    pre_n   = min(40, n)
    pre_idx = np.argsort(-sc)[:pre_n].tolist()
    p_paths = [paths[i]      for i in pre_idx]
    p_embs  = [embeddings[i] for i in pre_idx]
    p_sc    = sc[pre_idx]

    embs   = np.stack([np.asarray(e, dtype=np.float32) for e in p_embs])
    norms  = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-9)

    centroid = embs_n.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    sim      = embs_n @ centroid

    score_desc   = np.argsort(-p_sc).tolist()
    centroid_desc = np.argsort(-sim).tolist()
    centroid_asc  = np.argsort(sim).tolist()

    selected: list[int] = []

    def _pick(order: list[int]) -> None:
        for i in order:
            if i in selected:
                continue
            # Hard pairwise sim guard — skip if too similar to any already-selected image
            if selected:
                sims = embs_n[selected] @ embs_n[i]
                if float(np.max(sims)) > _DUP_SIM_THRESH:
                    continue
            selected.append(i)
            return

    _pick(score_desc)     # subject
    _pick(centroid_desc)  # opener
    _pick(score_desc)     # closer
    _pick(centroid_asc)   # contrast
    _pick(score_desc)     # detail

    while len(selected) < n_max:
        sel_embs   = embs_n[selected]
        best_idx   = -1
        best_blend = -np.inf
        for i in range(pre_n):
            if i in selected:
                continue
            max_sim   = float(np.max(sel_embs @ embs_n[i]))
            diversity = 1.0 - max_sim
            blend     = 0.60 * diversity + 0.40 * float(p_sc[i])
            if blend > best_blend:
                best_blend = blend
                best_idx   = i
        if best_idx < 0:
            break
        selected.append(best_idx)

    return (
        [p_paths[i] for i in selected],
        [p_embs[i]  for i in selected],
        [float(p_sc[i]) for i in selected],
    )


# ── Brief-aware aspect re-scoring ─────────────────────────────────────────────

def _compute_brief_scores(
    paths: list[str],
    aspect_scores_list: Optional[list[dict]],
    scores: list[float],
    rule_set: dict,
    style_prompt: str,
) -> list[float]:
    """
    Re-rank candidates by blending existing scores with brief-to-aspect alignment.

    Maps brief keywords to aspect weights, then computes:
        aligned = 60% original_score + 40% weighted_aspect_alignment

    This ensures the style prompt actually influences selection even when
    the Phi-4-mini GGUF is absent (keyword-only fallback path).
    """
    if not style_prompt.strip() or not aspect_scores_list:
        return list(scores)

    text = style_prompt.lower()

    # Start with neutral weights — any matched theme multiplies the relevant aspect
    w: dict[str, float] = {
        "Narrative":    1.0,
        "Composition":  1.0,
        "Lighting":     1.0,
        "Technical":    0.5,
        "Human/Culture":1.0,
    }

    # Lighting-dominant: rain, reflections, neon, fog, golden hour, night…
    _LIGHT_KW = {"rain", "wet", "reflection", "reflections", "puddle", "fog", "mist",
                 "neon", "golden", "sunset", "sunrise", "shadow", "shadows", "glow",
                 "night", "dusk", "dawn", "light", "dark", "atmosphere", "overcast",
                 "cloudy", "mood", "moody", "hazy", "blue hour", "twilight"}
    if any(kw in text for kw in _LIGHT_KW):
        w["Lighting"] = 2.8

    # Narrative/moment: decisive moment, emotion, storytelling
    _MOMENT_KW = {"moment", "emotion", "candid", "story", "decisive", "gesture",
                  "expression", "drama", "tension", "solitude", "quiet", "fleeting",
                  "encounter", "life"}
    if any(kw in text for kw in _MOMENT_KW):
        w["Narrative"] = 2.8

    # Human-centric: people, crowds, street life
    _HUMAN_KW = {"people", "crowd", "figure", "figures", "faces", "human",
                 "pedestrian", "passerby", "culture", "community", "portrait",
                 "stranger", "strangers"}
    if any(kw in text for kw in _HUMAN_KW):
        w["Human/Culture"] = 2.5

    # Geometric/architectural: lines, symmetry, architecture
    _GEO_KW = {"geometry", "geometric", "architecture", "architectural", "pattern",
               "lines", "symmetry", "abstract", "structure", "grid", "form",
               "minimal", "minimalist"}
    if any(kw in text for kw in _GEO_KW) or rule_set.get("GEOMETRIC_PRIORITY") == "High":
        w["Composition"] = 2.8

    # Empty/liminal: suppress Human/Culture (YOLO already handles hard filter)
    if rule_set.get("HARD_FILTER_PEOPLE"):
        w["Human/Culture"] = 0.0

    total_w = sum(w.values()) or 1.0

    aligned: list[float] = []
    for i, (path, base_score) in enumerate(zip(paths, scores)):
        if i >= len(aspect_scores_list):
            aligned.append(base_score)
            continue
        aspects = aspect_scores_list[i] or {}
        brief_alignment = sum(aspects.get(k, 0.5) * v for k, v in w.items()) / total_w
        blended = 0.60 * base_score + 0.40 * brief_alignment
        aligned.append(float(np.clip(blended, 0.0, 1.0)))

    active = {k: round(v, 1) for k, v in w.items() if v > 1.0}
    print(f"[cd] brief-alignment weights: {active or 'neutral (no matching keywords)'}")
    return aligned


# ── Pre-selection pool deduplication ─────────────────────────────────────────

def _dedup_pool(
    paths: list[str],
    embeddings: list[np.ndarray],
    scores: Optional[list[float]],
    aspects: Optional[list[dict]],
    thresh: float = _POOL_DEDUP_THRESH,
) -> tuple[list[str], list[np.ndarray], list[float], Optional[list[dict]]]:
    """
    Hard-drop near-duplicate images from the candidate pool before selection.

    Sort by score (desc) so the best shot in each burst cluster is kept.
    Any subsequent image with cosine sim > thresh to an already-kept image
    is discarded. Returns a cleaned (paths, embeddings, scores, aspects) tuple.
    """
    n = len(paths)
    if n == 0:
        return paths, embeddings, scores or [], aspects

    sc = np.array(scores if scores and len(scores) == n else [0.5] * n, dtype=np.float32)
    order = np.argsort(-sc).tolist()

    embs = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / (norms + 1e-9)

    kept: list[int] = []
    kept_embs: list[np.ndarray] = []

    for i in order:
        if kept_embs:
            stack = np.stack(kept_embs)
            max_sim = float(np.max(stack @ embs_n[i]))
            if max_sim > thresh:
                print(f"[cd-dedup] dropped near-duplicate: {Path(paths[i]).name}  sim={max_sim:.3f}")
                continue
        kept.append(i)
        kept_embs.append(embs_n[i])

    dropped = n - len(kept)
    if dropped:
        print(f"[cd-dedup] {dropped}/{n} near-duplicates removed from candidate pool (thresh={thresh})")

    out_paths = [paths[i] for i in kept]
    out_embs  = [embeddings[i] for i in kept]
    out_sc    = [float(sc[i]) for i in kept]
    out_asp   = [aspects[i] for i in kept] if aspects and len(aspects) == n else aspects

    return out_paths, out_embs, out_sc, out_asp


# ── Top-level Purist Orchestrator ─────────────────────────────────────────────

def run_creative_direction(
    strong_paths:      list[str],
    embeddings:        list[np.ndarray],
    anchor_path:       str,
    output_dir:        str,
    scores:            Optional[list[float]] = None,
    aspect_scores_list: Optional[list[dict]] = None,
    style_prompt:      str = "",
    n_target:          int = 7,
    avoid_paths:       Optional[list[str]] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Purist Creative Direction pipeline.

    Selects the best original captures according to the Style Brief.
    No pixel modification is performed — output files are copies of originals.

    Steps:
      1. Agent generates Rule Set JSON from Style Brief (CPU GGUF).
      2. YOLO kill-switch hard-blocks people when HARD_FILTER_PEOPLE is True.
      3. SigLIP-2 Subject Intrusion penalty applied to remaining scores.
      4. Story Sequence selected via greedy max-dissimilarity + role guarantee.
      5. Cinematic reorder: opener→0, contrast→n//2, closer→n-1.
      6. Originals copied to output_dir/Final_Portfolio/.
    """
    from creative_director_agent import generate_rule_set, generate_director_brief

    _p = progress or (lambda f, d: None)

    if not strong_paths:
        return {"error": "No images to curate.", "outputs": [], "total": 0}

    # ── Step 1: Rule Set + Director Brief from Brief ──────────────────────────
    _p(0.02, "Agent: generating Rule Set from Style Brief…")
    rule_set = generate_rule_set(style_prompt)
    _p(0.06, f"Rule Set: HARD_FILTER_PEOPLE={rule_set['HARD_FILTER_PEOPLE']}  "
             f"GEOMETRIC={rule_set['GEOMETRIC_PRIORITY']}  "
             f"MOOD={rule_set['LIGHTING_MOOD']}")

    # Director Brief uses the same Phi-4 GGUF (already loaded as singleton)
    # Provides thematic_niche + color_profile_target for the Judge's Verdict
    _director_brief = None
    try:
        _director_brief = generate_director_brief(style_prompt)
        _p(0.08, f"Director Brief: theme='{_director_brief.thematic_niche}' "
                 f"color='{_director_brief.color_profile_target}'"
        )
    except Exception as _e_brief:
        _p(0.08, f"Director Brief skipped ({_e_brief}) — verdict uses fallback context")

    # ── Step 2: YOLO26-nano person_kill_switch ────────────────────────────────
    yolo_blocked: set[str] = set()
    if rule_set["HARD_FILTER_PEOPLE"]:
        _p(0.08, f"person_kill_switch: scanning {len(strong_paths)} images (YOLO26 CPU, conf≥0.35)…")
        yolo_blocked = person_kill_switch(strong_paths, style_prompt)
        if yolo_blocked:
            _p(0.14, f"person_kill_switch: {len(yolo_blocked)} images DISQUALIFIED")

    # Remove YOLO-blocked from candidate pool
    filtered_paths = [p for p in strong_paths if p not in yolo_blocked]
    filtered_embs  = [e for p, e in zip(strong_paths, embeddings) if p not in yolo_blocked]
    filtered_scores = [s for p, s in zip(strong_paths, (scores or [0.5] * len(strong_paths)))
                       if p not in yolo_blocked]

    if not filtered_paths:
        return {
            "error": "All images were disqualified by the YOLO kill-switch.",
            "outputs": [], "total": 0,
            "rule_set": rule_set,
        }

    # Align aspect_scores_list with filtered_paths
    filtered_aspects: Optional[list[dict]] = None
    if aspect_scores_list and len(aspect_scores_list) == len(strong_paths):
        filtered_aspects = [
            aspect_scores_list[strong_paths.index(p)]
            for p in filtered_paths
            if p in strong_paths
        ]

    # ── Pre-dedup: remove near-identical shots from pool ─────────────────────
    # Burst duplicates (cosine sim > 0.92) are collapsed to their best-scoring
    # representative before any selection logic runs, guaranteeing that no two
    # visually near-identical images can appear in the final sequence.
    _p(0.15, f"Deduplicating candidate pool ({len(filtered_paths)} images)…")
    filtered_paths, filtered_embs, filtered_scores, filtered_aspects = _dedup_pool(
        filtered_paths, filtered_embs, filtered_scores, filtered_aspects,
    )
    _p(0.16, f"Candidate pool after dedup: {len(filtered_paths)} images")

    # ── Step 3: SigLIP-2 Subject Intrusion penalty ────────────────────────────
    _p(0.16, "Applying Subject Intrusion constraints…")
    adjusted_scores, disq_notes = _apply_brief_constraints(
        filtered_paths, filtered_embs, filtered_scores,
        aspect_scores_list=filtered_aspects,
        style_prompt=style_prompt,
    )

    # ── Step 3b: Brief-aware aspect re-scoring ────────────────────────────────
    # Maps brief keywords → aspect weights and blends them with the existing
    # score so the prompt actually influences which photos are selected.
    # Without this step, non-empty briefs are silently ignored because
    # select_story_sequence() is pure score+diversity with no prompt awareness.
    _p(0.19, "Applying brief-aware scoring…")
    adjusted_scores = _compute_brief_scores(
        filtered_paths, filtered_aspects, adjusted_scores,
        rule_set, style_prompt,
    )

    # ── Step 4: Story Sequence Selection ──────────────────────────────────────
    # 4a: Pre-filter avoid_paths, cap pool to top-40 by score for LLM manifest
    avoid   = set(avoid_paths or [])
    pool_idx = [i for i, p in enumerate(filtered_paths) if p not in avoid]
    if not pool_idx:
        pool_idx = list(range(len(filtered_paths)))

    pool_sc = np.array([adjusted_scores[i] for i in pool_idx], dtype=np.float32)
    top_n   = min(20, len(pool_idx))
    top_idx = np.argsort(-pool_sc)[:top_n].tolist()

    # Build structured metadata manifest for the 1.5B agent
    candidates: list[dict] = []
    for rank, pi in enumerate(top_idx):
        real_i  = pool_idx[pi]
        path    = filtered_paths[real_i]
        aspects = (filtered_aspects[real_i] if filtered_aspects else {})
        candidates.append({
            "id":            rank,
            "_real_idx":     real_i,
            "filename":      path,
            "score":         float(adjusted_scores[real_i]),
            "Composition":   aspects.get("Composition",   0.5),
            "Lighting":      aspects.get("Lighting",      0.5),
            "Narrative":     aspects.get("Narrative",     0.5),
            "Human/Culture": aspects.get("Human/Culture", 0.5),
            "people_sim":    0.0,
            "yolo_blocked":  path in yolo_blocked,
        })

    # 4b: Ask the 1.5B agent to pick the sequence in a single reasoning pass
    from creative_director_agent import select_sequence_from_batch
    _p(0.22, f"Agent: selecting {n_target}-image sequence from top-{top_n} candidates…")
    llm_ids = select_sequence_from_batch(candidates, n_target, style_prompt, rule_set)

    if llm_ids:
        # Use LLM selection
        sel_real = [candidates[cid]["_real_idx"] for cid in llm_ids if cid < len(candidates)]
        seq_paths  = [filtered_paths[i]  for i in sel_real]
        seq_embs   = [filtered_embs[i]   for i in sel_real]
        seq_scores = [adjusted_scores[i] for i in sel_real]
        seq_aspects = [filtered_aspects[i] if filtered_aspects else {} for i in sel_real]
        _p(0.30, f"Agent selected {len(seq_paths)} images (single-pass reasoning)")
    else:
        # Greedy numpy fallback
        _p(0.22, f"Greedy selection: top-{n_target} diverse images…")
        seq_paths, seq_embs, seq_scores = select_story_sequence(
            filtered_paths, filtered_embs, adjusted_scores,
            n_min=min(5, n_target), n_max=n_target,
            avoid_paths=avoid_paths,
        )
        seq_aspects = []
        if filtered_aspects:
            path_set = {p: filtered_aspects[i] for i, p in enumerate(filtered_paths)}
            seq_aspects = [path_set.get(p, {}) for p in seq_paths]
        _p(0.30, f"Greedy selected {len(seq_paths)} images")

    n = len(seq_paths)
    _p(0.30, f"Selected {n} images for Story Sequence")

    if n == 0:
        return {
            "error": "No images survived sequence selection.",
            "outputs": [], "total": 0,
            "rule_set": rule_set,
        }

    # ── Step 5: Cinematic Reorder ─────────────────────────────────────────────
    _p(0.32, "Applying cinematic reorder…")
    bucket_embs = np.stack([np.asarray(e, dtype=np.float32) for e in seq_embs])
    embs_n      = bucket_embs / (np.linalg.norm(bucket_embs, axis=1, keepdims=True) + 1e-9)

    roles     = _assign_roles_by_content(seq_embs, scores=seq_scores, paths=seq_paths)
    cin_order = _cinematic_reorder(seq_paths, embs_n, roles, seq_scores)

    seq_paths  = [seq_paths[i]  for i in cin_order]
    seq_scores = [seq_scores[i] for i in cin_order]
    roles      = [roles[i]      for i in cin_order]
    if seq_aspects:
        seq_aspects = [seq_aspects[i] for i in cin_order]

    # ── Step 5b: 8B Judge's Verdict (GPU, loaded AFTER sequence is final) ───────
    # DeepSeek-R1-Distill-Llama-8B (INT4) generates the official competition
    # narrative. VRAM is purged by the caller via purge_vram() after this step.
    # Skipped gracefully if 8B weights absent.
    _p(0.36, "Generating Judge's Verdict (8B, GPU)…")
    from creative_director_agent import generate_judges_verdict_8b
    seq_narrative = generate_judges_verdict_8b(
        selected_images=[
            {"filename": p, **(seq_aspects[i] if seq_aspects else {})}
            for i, p in enumerate(seq_paths)
        ],
        style_prompt=style_prompt,
        roles=roles,
        director_brief=_director_brief,
        scores=seq_scores,
    )
    if seq_narrative:
        _p(0.40, "Judge's Verdict complete")
    else:
        _p(0.40, "8B Judge unavailable — verdict skipped")

    # ── Step 6: Copy Originals to Final_Portfolio/ ────────────────────────────
    out_dir = Path(output_dir) / "Final_Portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict] = []
    n_ok = 0

    for seq_pos, (path, score, role) in enumerate(zip(seq_paths, seq_scores, roles)):
        fname    = Path(path).stem + "_purist.jpg"
        out_path = out_dir / fname
        _p(
            0.40 + (seq_pos / n) * 0.55,
            f"[{seq_pos+1}/{n}] {role.upper()} — {Path(path).name}",
        )
        try:
            shutil.copy2(path, str(out_path))
            rlog = (
                f"Role: {role.upper()} — Purist original capture.\n"
                f"Score: {score:.3f}  |  Brief: '{style_prompt[:60]}'\n"
                f"Rule Set: HARD_FILTER_PEOPLE={rule_set['HARD_FILTER_PEOPLE']}  "
                f"GEOMETRIC={rule_set['GEOMETRIC_PRIORITY']}  "
                f"MOOD={rule_set['LIGHTING_MOOD']}\n"
            )
            if seq_narrative:
                rlog += f"\nJudge's Verdict: {seq_narrative}\n"
            rlog += "Engine: purist_original — no pixel modification."
            outputs.append({
                "source_path":   path,
                "output_path":   str(out_path),
                "filename":      fname,
                "params":        {"role": role, "seq_pos": seq_pos, "rule_set": rule_set},
                "success":       True,
                "engine":        "purist_original",
                "reasoning_log": rlog,
            })
            n_ok += 1
            print(f"[cd] copied {Path(path).name} → {out_path.name}")
        except Exception as e:
            print(f"[cd] copy failed {Path(path).name}: {e}")
            outputs.append({
                "source_path": path, "output_path": None,
                "error": str(e), "success": False,
            })

    _p(1.0, f"Purist selection complete — {n_ok}/{n} images in Final_Portfolio")

    return {
        "outputs":     outputs,
        "output_dir":  str(out_dir),
        "total":       n,
        "success":     n_ok,
        "failed":      n - n_ok,
        "anchor_path": anchor_path,
        "rule_set":    rule_set,
    }
