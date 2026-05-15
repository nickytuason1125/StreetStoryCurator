"""
Creative Director — Purist Curation Pipeline

Governs the 5-image "Story Sequence" using only Original Pixel Metadata.
No pixel modification is performed. The output is always the original file.

Pipeline
────────
1. DeepSeek-R1-1.5B Agent (CPU GGUF) → Rule Set JSON
   Reads the Style Brief and emits HARD_FILTER_PEOPLE, GEOMETRIC_PRIORITY, etc.

2. Binary Kill-Switch — YOLOv11-nano (CPU)
   If HARD_FILTER_PEOPLE is True and class:person is detected (conf > 0.40),
   the image is hard-blocked from the Story Sequence. No clean-up. Disqualified.

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

# ── Empty-brief filtering ─────────────────────────────────────────────────────
_EMPTY_BRIEF_KEYWORDS = {"empty", "liminal", "desert", "void", "abandoned", "desolate"}
_PEOPLE_SIM_THRESHOLD = 0.40   # SigLIP-2 cosine sim to "people" concept → hard penalty
_PEOPLE_PENALTY       = 0.10   # score multiplier when triggered
_HUMAN_CULTURE_THRESH = 0.55   # Human/Culture aspect fallback threshold
_YOLO_PERSON_CONF     = 0.40   # YOLOv11-nano confidence threshold (binary kill-switch)


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


def _yolo_person_gate(paths: list[str], style_prompt: str) -> set[str]:
    """
    Binary kill-switch — YOLOv11-nano CPU gate.

    If the brief implies an empty scene and YOLO detects class:person
    with confidence > 0.40, the image is hard-blocked from the Story
    Sequence. No clean-up is permitted — it is a disqualification.
    """
    if not _empty_brief_detected(style_prompt):
        return set()

    blocked: set[str] = set()
    try:
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        from ultralytics import YOLO as _YOLO

        yolo = _YOLO("yolo11n.pt")

        for path in paths:
            try:
                results = yolo(path, device="cpu", verbose=False,
                               classes=[0], conf=_YOLO_PERSON_CONF)
                for r in results:
                    if len(r.boxes) > 0:
                        blocked.add(path)
                        print(
                            f"[cd] YOLO kill-switch: BLOCKED {Path(path).name} "
                            f"— person conf>{_YOLO_PERSON_CONF}"
                        )
                        break
            except Exception as e_img:
                print(f"[cd] YOLO inference error {Path(path).name}: {e_img}")

    except ImportError:
        print("[cd] ultralytics not installed — YOLO gate skipped")
    except Exception as e:
        print(f"[cd] YOLO gate error: {e}")

    if blocked:
        print(f"[cd] YOLO kill-switch: {len(blocked)}/{len(paths)} images disqualified")
    return blocked


def _mean_luminance(path: str) -> float:
    """Mean luminance in [0, 1] via 64×64 PIL grayscale thumbnail."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
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
            if i not in selected:
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


# ── Top-level Purist Orchestrator ─────────────────────────────────────────────

def run_creative_direction(
    strong_paths:   list[str],
    embeddings:     list[np.ndarray],
    anchor_path:    str,
    output_dir:     str,
    scores:         Optional[list[float]] = None,
    style_prompt:   str = "",
    n_target:       int = 7,
    avoid_paths:    Optional[list[str]] = None,
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
    from creative_director_agent import generate_rule_set

    _p = progress or (lambda f, d: None)

    if not strong_paths:
        return {"error": "No images to curate.", "outputs": [], "total": 0}

    # ── Step 1: Rule Set from Brief ───────────────────────────────────────────
    _p(0.02, "Agent: generating Rule Set from Style Brief…")
    rule_set = generate_rule_set(style_prompt)
    _p(0.06, f"Rule Set: HARD_FILTER_PEOPLE={rule_set['HARD_FILTER_PEOPLE']}  "
             f"GEOMETRIC={rule_set['GEOMETRIC_PRIORITY']}  "
             f"MOOD={rule_set['LIGHTING_MOOD']}")

    # ── Step 2: YOLO Binary Kill-Switch ───────────────────────────────────────
    yolo_blocked: set[str] = set()
    if rule_set["HARD_FILTER_PEOPLE"]:
        _p(0.08, f"YOLO kill-switch: scanning {len(strong_paths)} images (CPU)…")
        yolo_blocked = _yolo_person_gate(strong_paths, style_prompt)
        if yolo_blocked:
            _p(0.14, f"YOLO: {len(yolo_blocked)} images disqualified (person detected)")

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

    # ── Step 3: SigLIP-2 Subject Intrusion penalty ────────────────────────────
    _p(0.16, "Applying Subject Intrusion constraints…")
    adjusted_scores, disq_notes = _apply_brief_constraints(
        filtered_paths, filtered_embs, filtered_scores,
        aspect_scores_list=None,
        style_prompt=style_prompt,
    )

    # ── Step 4: Story Sequence Selection ──────────────────────────────────────
    _p(0.22, f"Selecting {n_target}-image Story Sequence…")
    seq_paths, seq_embs, seq_scores = select_story_sequence(
        filtered_paths, filtered_embs, adjusted_scores,
        n_min=min(5, n_target), n_max=n_target,
        avoid_paths=avoid_paths,
    )
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
                f"Engine: purist_original — no pixel modification."
            )
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
