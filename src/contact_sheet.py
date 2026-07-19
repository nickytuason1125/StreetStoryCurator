"""
Contact-sheet compositor + agentic self-revision loop for Story Mode.

Story Mode's original design only ever produced ONE candidate sequence per
run — role assignment + cinematic reorder, then straight to the Judge's
Verdict. This module adds a bounded propose -> render -> critique -> revise
loop on top of that: render the current sequence as a labeled contact
sheet, ask a vision-capable critic whether any slot looks wrong for its
role, and if so swap it for a leftover pool candidate — then re-run role
assignment and cinematic reorder from scratch so the negative-space/
luminance/dedup guards in creative_director.py stay enforced every
iteration, not just the first.

Gated to Story Mode only — see run_revision_loop's docstring. Competition
Mode's brief is about independent standout images, not narrative pacing;
its single-pass flow is untouched by this module.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional

_MAX_ITERS = 3
_THUMB_PX  = 320
_GRID_COLS = 4


def build_contact_sheet(
    seq_paths: list[str],
    roles: list[str],
    out_path: str,
    thumb_px: int = _THUMB_PX,
) -> str:
    """
    Render seq_paths as a labeled grid image (role + 1-based slot per cell)
    and save to out_path. Returns out_path. Reuses editorial_renderer's
    _crop_and_resize thumbnail helper for consistent framing.
    """
    from PIL import Image, ImageDraw, ImageFont
    from editorial_renderer import _crop_and_resize

    n = len(seq_paths)
    cols = min(_GRID_COLS, max(n, 1))
    rows = (n + cols - 1) // cols
    pad = 6
    label_h = 28

    sheet = Image.new(
        "RGB",
        (cols * thumb_px + (cols + 1) * pad, rows * (thumb_px + label_h) + (rows + 1) * pad),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for i, (path, role) in enumerate(zip(seq_paths, roles)):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_px + pad)
        y = pad + r * (thumb_px + label_h + pad)
        try:
            img = Image.open(path).convert("RGB")
            thumb = _crop_and_resize(img, thumb_px, thumb_px)
        except Exception:
            thumb = Image.new("RGB", (thumb_px, thumb_px), (60, 60, 60))
        sheet.paste(thumb, (x, y))
        label = f"{i + 1}. {role.upper()}"
        draw.rectangle([x, y + thumb_px, x + thumb_px, y + thumb_px + label_h], fill=(0, 0, 0))
        draw.text((x + 4, y + thumb_px + 4), label, fill=(255, 255, 255), font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="JPEG", quality=85)
    return out_path


def critique_contact_sheet(
    sheet_path: str,
    roles: list[str],
    seq_aspects: list[dict],
    style_prompt: str,
) -> dict:
    """
    One multimodal critique call — reuses critique_engine.py's existing
    Qwen2.5-VL-2B GGUF loader (free if already warmed by run_jury_critique/
    run_audit_annotation this session). Returns
    {"action": "accept"|"swap", "swap_slot": int|None, "reason": str}.

    Phase 2: loose JSON parsing in critique_engine.run_contact_sheet_critique
    (no grammar yet — a later phase upgrades this to grammar-constrained
    decoding via LlamaGrammar, following vlm_niche_detector.py's pattern).
    Never raises — on any failure returns action="accept" so the loop
    always terminates safely.
    """
    import critique_engine as ce

    slot_summaries = [
        {
            "slot": i,
            "role": role,
            **{
                k: round(float(v), 2)
                for k, v in (seq_aspects[i] if i < len(seq_aspects) else {}).items()
                if isinstance(v, (int, float))
            },
        }
        for i, role in enumerate(roles)
    ]
    try:
        return ce.run_contact_sheet_critique(sheet_path, slot_summaries, style_prompt)
    except Exception as e:
        return {"action": "accept", "swap_slot": None, "reason": f"critique failed: {e}"}


def run_revision_loop(
    seq_paths: list[str],
    seq_embs: list[np.ndarray],
    seq_scores: list[float],
    seq_aspects: list[dict],
    roles: list[str],
    art_pool: list[dict],
    style_prompt: str,
    output_dir: str,
    max_iters: int = _MAX_ITERS,
    progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[list[str], list[float], list[str], list[dict], list[np.ndarray], list[dict]]:
    """
    Bounded propose -> render -> critique -> revise loop for Story Mode.

    Terminates when the critic accepts, max_iters is reached, no valid
    replacement remains in the leftover pool, or the same image is swapped
    out twice (prevents oscillation/thrashing). On any critique failure,
    treats it as an accept and falls through — this loop is never a hard
    dependency for Story Mode to complete.

    art_pool entries must carry "_embedding" (creative_director.py's pool
    construction adds this field). Candidates without one are skipped as
    swap targets.

    Returns (seq_paths, seq_scores, roles, seq_aspects, seq_embs,
    revision_log). revision_log is a list of per-iteration dicts recording
    action/reason, used both for the reasoning_log and (a later phase's)
    LanceDB persistence.
    """
    _p = progress or (lambda f, d: None)
    revision_log: list[dict] = []
    removed_paths: set[str] = set()

    seq_paths  = list(seq_paths)
    seq_scores = list(seq_scores)
    seq_aspects = list(seq_aspects)
    seq_embs   = list(seq_embs)
    roles      = list(roles)

    # Leftover pool (candidates not currently in the sequence), keyed by path.
    pool_by_path: dict[str, dict] = {
        c["path"]: c for c in art_pool
        if c["path"] not in seq_paths and c.get("_embedding") is not None
    }

    from creative_director import _assign_roles_by_content, _cinematic_reorder

    for iteration in range(max_iters):
        cache_dir  = Path(output_dir) / "cache" / "contact_sheets"
        sheet_path = str(cache_dir / f"contact_sheet_iter{iteration}.jpg")
        try:
            build_contact_sheet(seq_paths, roles, sheet_path)
        except Exception as e:
            revision_log.append({"iteration": iteration, "action": "accept",
                                  "reason": f"render failed: {e}", "swap_slot": None})
            break

        _p_frac = 0.33 + iteration * 0.02
        _p(_p_frac, f"Revision {iteration + 1}/{max_iters}: viewing contact sheet…")

        result = critique_contact_sheet(sheet_path, roles, seq_aspects, style_prompt)
        action = result.get("action", "accept")
        reason = result.get("reason", "")
        swap_slot = result.get("swap_slot")
        revision_log.append({"iteration": iteration, "action": action,
                              "reason": reason, "swap_slot": swap_slot})

        if action != "swap":
            _p(_p_frac + 0.01, f"Revision {iteration + 1}/{max_iters}: accepted — {reason[:60]}")
            break

        if swap_slot is None or not (0 <= swap_slot < len(seq_paths)):
            break
        current_path_at_slot = seq_paths[swap_slot]
        if current_path_at_slot in removed_paths:
            # This exact image was already swapped out once, came back via
            # the pool's soft-return, and is now being kicked out again —
            # genuine thrashing. A raw slot-index guard would be fooled by
            # reordering (role assignment is content-based, not position-
            # preserving, so "slot 2" can hold a different image each
            # iteration) — tracking by image path is reorder-proof.
            break
        if not pool_by_path:
            break

        # Rank leftover candidates: 0.6*score + 0.4*(1 - max_cosine_sim_to_current_seq)
        seq_stack = np.stack([np.asarray(e, dtype=np.float32) for e in seq_embs])
        seq_stack /= (np.linalg.norm(seq_stack, axis=1, keepdims=True) + 1e-9)

        best_path, best_val = None, -1e9
        for cand_path, cand in pool_by_path.items():
            c = np.asarray(cand["_embedding"], dtype=np.float32)
            c /= (np.linalg.norm(c) + 1e-9)
            max_sim = float((seq_stack @ c).max()) if len(seq_stack) else 0.0
            val = 0.6 * float(cand.get("score", 0.5)) + 0.4 * (1.0 - max_sim)
            if val > best_val:
                best_val, best_path = val, cand_path

        if best_path is None:
            break

        # Perform the swap.
        old_path    = current_path_at_slot
        old_score   = seq_scores[swap_slot]
        old_aspects = seq_aspects[swap_slot] if swap_slot < len(seq_aspects) else {}
        old_emb     = seq_embs[swap_slot]
        new_cand    = pool_by_path.pop(best_path)
        removed_paths.add(old_path)

        seq_paths[swap_slot]  = best_path
        seq_scores[swap_slot] = float(new_cand.get("score", 0.5))
        if swap_slot < len(seq_aspects):
            seq_aspects[swap_slot] = new_cand.get("breakdown", {})
        seq_embs[swap_slot] = np.asarray(new_cand["_embedding"], dtype=np.float32)

        # Soft-return the displaced image — a later iteration could still re-pick it.
        pool_by_path[old_path] = {
            "path": old_path, "score": old_score,
            "breakdown": old_aspects, "_embedding": old_emb,
        }

        _p(_p_frac + 0.01,
           f"Revision {iteration + 1}/{max_iters}: swapped slot {swap_slot + 1} "
           f"({Path(old_path).name} -> {Path(best_path).name})")

        # Re-run role assignment and cinematic reorder from scratch so the
        # negative-space/luminance/dedup guards stay enforced every
        # iteration — never hand-splice order.
        bucket_embs = np.stack([np.asarray(e, dtype=np.float32) for e in seq_embs])
        embs_n      = bucket_embs / (np.linalg.norm(bucket_embs, axis=1, keepdims=True) + 1e-9)
        roles       = _assign_roles_by_content(seq_embs, scores=seq_scores, paths=seq_paths)
        cin_order   = _cinematic_reorder(seq_paths, embs_n, roles, seq_scores)

        seq_paths  = [seq_paths[i]  for i in cin_order]
        seq_scores = [seq_scores[i] for i in cin_order]
        seq_embs   = [seq_embs[i]   for i in cin_order]
        roles      = [roles[i]      for i in cin_order]
        if seq_aspects:
            seq_aspects = [seq_aspects[i] for i in cin_order]

    # Unload the critique GGUF before returning control — Story Mode's next
    # phase (Judge's Verdict) must not find a GPU model still resident.
    try:
        import critique_engine as ce
        ce.unload()
    except Exception:
        pass
    try:
        from vram_manager import VRAMManager
        VRAMManager.purge_vram()
    except Exception:
        pass

    return seq_paths, seq_scores, roles, seq_aspects, seq_embs, revision_log
