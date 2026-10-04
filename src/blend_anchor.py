"""
blend_anchor.py — ONE shared brief / peg / reference-book blender.

2026-09-22: the Creative Director's text rerank (brief embedding + RAG book
ensemble) and the peg ranking existed only inside run_creative_direction.
Every other pipeline that builds a semantic target — the JUDGE grading
anchor (grade_pipeline_v2), the mogco story sequencer (grade_worker), and
the sequence vibe search (routers/sequence.py) — either ignored the user's
brief and peg entirely, or (grading) read specvlm_pipeline._CD_BRIEF, which
NOBODY ever set: set_cd_brief() existed but had zero callers. The brief you
typed influenced the CD selection and nothing else — a silent no-op of the
same class as the Path-vs-str comparison bug.

This module is now the single source of truth for:

    * session context      — set_session_context() stores the brief + peg
                             hash each Creative Direction run received, so
                             grading / sequencing / vibe search reuse the
                             SAME targets (a JUDGE pass after a CD run grades
                             against the same brief the story was built on)
    * peg resolution       — resolve_peg_embedding() finds the uploaded
                             reference photo's LanceDB embedding by exact/
                             boundary stem match (never a blind substring)
    * RAG book phrases     — rag_phrase_vectors() with the npz cache and the
                             pdf_rag opt-in gate, MMR phrase selection
    * blending             — blend_query_vector(): 70/30 brief+books, then
                             30% peg, L2-normalised. Identical weights to
                             the CD pipeline's inline maths.

Every consumer logs a [blend] line with what was blended, so a silent no-op
can never hide again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# ── Session context (set by routers/creative.py on every CD run) ─────────────
_SESSION_BRIEF: str = ""
_SESSION_PEG_HASH: str = ""


def set_session_context(brief: str = "", peg_hash: str = "") -> None:
    """Store the brief + peg the current session is working against.

    Called by the creative-direction router; consumed by grading, the story
    sequencer and vibe search. Empty strings clear the respective part.
    """
    global _SESSION_BRIEF, _SESSION_PEG_HASH
    _SESSION_BRIEF = (brief or "").strip()
    _SESSION_PEG_HASH = (peg_hash or "").strip()


def get_session_context() -> dict:
    return {"brief": _SESSION_BRIEF, "peg_image_hash": _SESSION_PEG_HASH}


def clear_session_context() -> None:
    set_session_context("", "")


# ── Peg resolution ───────────────────────────────────────────────────────────

def peg_stem_match(stem: str, image_hash: str) -> bool:
    """Exact/boundary peg match — never a blind substring.

    'TPE26-1' must match 'TPE26-1' or 'TPE26-1_xyz' but NOT 'TPE26-105'
    (substring matching silently picked the wrong reference photo).
    """
    if stem == image_hash:
        return True
    # The hash may sit mid-stem after a folder prefix (carousel_01_TPE26-1).
    tail = stem.rsplit("_", 1)[-1]
    if tail == image_hash:
        return True
    if tail.startswith(image_hash):
        rest = tail[len(image_hash):]
        # Only accept if what follows is not more of the same number.
        return not (rest and rest[0].isdigit())
    return False


def _peg_stem_match(stem: str, image_hash: str) -> bool:  # back-compat alias
    return peg_stem_match(stem, image_hash)


def resolve_peg_embedding(peg_image_hash: Optional[str] = None) -> Optional[np.ndarray]:
    """Embedding of the uploaded reference photo, from LanceDB.

    Returns None (callers log loudly) when no hash is given, the table has
    no matching row, or the row carries no embedding — the caller then blends
    nothing and behaviour is identical to before this module existed.
    """
    h = (peg_image_hash or _SESSION_PEG_HASH or "").strip()
    if not h:
        return None
    try:
        import lance_store as _ls
        rows = _ls.query_all(min_score=0.0)
        hit = next(
            (r for r in rows if peg_stem_match(Path(r["path"]).stem, h)),
            None,
        )
        if hit is None:
            return None
        emb = hit.get("embedding")
        if emb is None or len(emb) == 0:
            return None
        vec = np.asarray(emb, dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)
    except Exception as _e:
        print(f"[blend] peg lookup failed ({h[:12]}…): {_e}", flush=True)
        return None


# ── RAG book phrases (opt-in gated inside pdf_rag) ───────────────────────────

_RAG_VEC_CACHE: dict = {}


def rag_phrases() -> list[str]:
    """Book concept phrases ([] when the user has not opted in)."""
    try:
        from pdf_rag import load_concepts
        return list(load_concepts() or [])
    except Exception:
        return []


def rag_phrase_vectors(phrases: list[str]) -> "Optional[np.ndarray]":
    """(P, 1536) normalised phrase embeddings, npz-cached by content hash
    (same convention as creative_director's rag_phrase_embs.npz)."""
    if not phrases:
        return None
    import hashlib
    key = hashlib.sha1("\x00".join(phrases).encode("utf-8")).hexdigest()[:16]
    hit = _RAG_VEC_CACHE.get(key)
    if hit is not None and hit.shape[0] == len(phrases):
        return hit
    cache_path = (Path(__file__).resolve().parent.parent /
                  "cache" / "rag_phrase_embs.npz")
    vecs = None
    if cache_path.exists():
        try:
            _z = np.load(cache_path)
            if str(_z["key"]) == key:
                vecs = _z["vecs"].astype(np.float32)
        except Exception:
            vecs = None
    if vecs is None or vecs.shape[0] != len(phrases):
        try:
            from siglip2_encoder import embed_text_query
        except Exception:
            return None
        try:
            vecs = np.stack([np.asarray(embed_text_query(p), dtype=np.float32)
                             for p in phrases])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, key=key, vecs=vecs)
        except Exception:
            return None
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    _RAG_VEC_CACHE[key] = vecs
    return vecs


def mmr_pick(ph_vecs: np.ndarray, qvec: np.ndarray, k: int = 8,
             lam: float = 0.72) -> list[int]:
    """Greedy MMR over phrase vectors: relevance first, then redundancy.
    Identical to the selector creative_director used inline."""
    sims = ph_vecs @ qvec
    order = list(np.argsort(-sims))
    if not order:
        return []
    picked = [order[0]]
    while len(picked) < min(k, len(order)):
        rest = [i for i in order if i not in picked]
        redun = (ph_vecs[rest] @ ph_vecs[picked].T).max(axis=1)
        mmr = lam * sims[rest] - (1.0 - lam) * redun
        picked.append(rest[int(np.argmax(mmr))])
    return picked


# ── The blend ────────────────────────────────────────────────────────────────

RAG_WEIGHT = 0.30      # 70% brief / 30% book-phrase centroid (CD parity)
PEG_WEIGHT = 0.30      # 30% peg — the same share the books get


def blend_query_vector(
    base_vec: "Optional[np.ndarray]",
    brief_text: str = "",
    peg_vec: "Optional[np.ndarray]" = None,
    use_rag: bool = True,
    phrases: "Optional[list[str]]" = None,
) -> tuple:
    """
    Blend a semantic query vector from its sources, in priority order:

        1. base_vec (the caller's own anchor — e.g. the grading brief
           ensemble) or, absent, an embedding of brief_text
        2. + 30% book-phrase centroid (MMR-picked, similarity-softmax
           weighted) when there is a brief and the books are opted in
        3. + 30% peg embedding when a reference peg is in play

    Returns (vector, diag). diag records what actually blended — callers
    print blend_diag_line(diag) so a silent no-op is impossible.
    """
    diag: dict = {"brief": (brief_text or "").strip()[:60], "rag": 0,
                  "peg": False, "sources": []}
    q = None
    if base_vec is not None:
        q = np.asarray(base_vec, dtype=np.float32).ravel()
        diag["sources"].append("base")
    if q is None and (brief_text or "").strip():
        try:
            from siglip2_encoder import embed_text_query
            q = np.asarray(embed_text_query(brief_text.strip()),
                           dtype=np.float32).ravel()
            diag["sources"].append("brief")
        except Exception as _e:
            print(f"[blend] brief embedding failed: {_e}", flush=True)
    if q is None:
        return None, diag
    q = q / (np.linalg.norm(q) + 1e-9)

    # Books (only meaningful when there is a brief to match against)
    if use_rag and (brief_text or "").strip():
        ph = list(phrases) if phrases is not None else rag_phrases()
        if ph:
            pv = rag_phrase_vectors(ph)
            if pv is not None and pv.shape[0] == len(ph):
                pv = pv / (np.linalg.norm(pv, axis=1, keepdims=True) + 1e-9)
                picked = mmr_pick(pv, q)
                if picked:
                    sel_sims = pv[picked] @ q
                    w = np.exp((sel_sims - sel_sims.max()) / 0.10)
                    w /= w.sum()
                    ens = (w[:, None] * pv[picked]).sum(axis=0)
                    ens /= np.linalg.norm(ens) + 1e-9
                    q = RAG_WEIGHT * ens + (1.0 - RAG_WEIGHT) * q
                    q /= np.linalg.norm(q) + 1e-9
                    diag["rag"] = len(picked)
                    diag["sources"].append("books")

    # Peg (a positive anchor, never a filter)
    if peg_vec is not None:
        pv2 = np.asarray(peg_vec, dtype=np.float32).ravel()
        if pv2.size == q.size:
            pv2 = pv2 / (np.linalg.norm(pv2) + 1e-9)
            q = PEG_WEIGHT * pv2 + (1.0 - PEG_WEIGHT) * q
            q /= np.linalg.norm(q) + 1e-9
            diag["peg"] = True
            diag["sources"].append("peg")

    return (q / (np.linalg.norm(q) + 1e-9)), diag


def blend_diag_line(diag: dict) -> str:
    """One log line: [blend] brief='…' rag=8 peg=yes sources=base,books,peg"""
    src = ",".join(diag.get("sources", [])) or "none"
    return (f"[blend] brief='{diag.get('brief', '')}' rag={diag.get('rag', 0)} "
            f"peg={'yes' if diag.get('peg') else 'no'} sources={src}")


def context_line() -> str:
    """Human-readable session context for LLM prompts (judge / critique).

    Empty string when nothing is set — callers append only when non-empty,
    so prompts stay byte-identical to before when no CD run has happened.
    """
    ctx = get_session_context()
    parts = []
    if ctx["brief"]:
        parts.append(f"creative brief: {ctx['brief']}")
    if ctx["peg_image_hash"]:
        parts.append("a user-uploaded reference photo anchors the visual style")
    return "; ".join(parts)

