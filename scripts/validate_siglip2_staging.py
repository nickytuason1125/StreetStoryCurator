"""
Validate (and optionally promote) the staged SigLIP-2 fp16 checkpoint WITHOUT
ever loading the open_clip model.

Why this exists
---------------
setup_siglip2_hf.py's stage 3 proves the staged HF checkpoint matches the live
open_clip one by encoding the same inputs through BOTH loaders. That is correct,
but the open_clip leg alone peaks at ~10.3 GB — the exact footprint the whole
migration is meant to eliminate. On a 16 GB machine that is unrunnable without
closing everything else, so the promotion never happens and every grade keeps
paying the 10.3 GB fallback.

The open_clip side, however, is ALREADY ON DISK from previous runs:

  cache/probe_embs.npz   text embeddings for the pos/neg/aspect/genre/street
                         prompt lists, in list order
  cache/lance.db         one image embedding per graded photo
  cache/encoder_source.txt   names the loader that produced both

So this script uses those stored vectors as the reference side and only ever
loads the LEAN staged model (~3.5 GB, in a disposable encode_worker subprocess,
one at a time). Same >= 0.98 cosine gate as the official script, one third of
the peak RAM, and no network.

Two guards make the stored reference trustworthy rather than assumed:
  * encoder_source.txt must name the open_clip loader, otherwise the cached
    vectors are not the space we are comparing against.
  * probe_embs.hash must equal the md5 recomputed from TODAY's prompt lists —
    that is what proves row i of probe_embs.npz is still prompt i. (This is the
    same key grade_pipeline_v2 uses to invalidate the cache.)

Run:
    venv\\Scripts\\python.exe scripts/validate_siglip2_staging.py
    venv\\Scripts\\python.exe scripts/validate_siglip2_staging.py --promote

Validation is read-only. Promotion happens only with --promote AND a pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_DEST    = _ROOT / "models" / "siglip2_hf_fp16"
_STAGING = _ROOT / "models" / "_siglip2_hf_fp16_staging"
_WORKER  = _ROOT / "src" / "encode_worker.py"
_CACHE   = _ROOT / "cache"

_MIN_COSINE = 0.98
_EXPECT_DIM = 1536

_FAIL: list[str] = []


def _fail(msg: str) -> None:
    _FAIL.append(msg)
    print(f"  FAIL  {msg}", flush=True)


def _ok(msg: str) -> None:
    print(f"  ok    {msg}", flush=True)


# ── Reference side: stored open_clip vectors ─────────────────────────────────

def _prompt_groups() -> "tuple[list[tuple[str, list[str]]], str]":
    """Row-aligned (group_name, prompts) pairs + the probe-cache key hash.

    Order and content mirror grade_pipeline_v2's encode_text_groups() call
    exactly; the hash is built the same way it builds _probe_key_hash.
    """
    from specvlm_pipeline import _POS_PROMPTS, _NEG_PROMPTS, _ASPECT_PROMPTS
    import grade_pipeline_v2 as G

    rag: list[str] = []
    rag_path = _CACHE / "rag_concepts.json"
    if rag_path.exists():
        try:
            rag = json.loads(rag_path.read_text(encoding="utf-8")).get("phrases", [])
        except Exception:
            rag = []
    pos_aug = list(_POS_PROMPTS) + rag

    key = repr((
        pos_aug, _NEG_PROMPTS,
        list(_ASPECT_PROMPTS.keys()),
        [v for pair in _ASPECT_PROMPTS.values() for v in pair],
        G._GENRE_REF_PROMPTS, G._FINE_ART_PROMPTS,
        G._STREET_POS_PROBES, G._STREET_NEG_PROBES,
    ))
    key_hash = hashlib.md5(key.encode()).hexdigest()

    # Only groups stored ROW-PER-PROMPT are comparable element-wise. 'ppl' and
    # 'fine_art' are stored as a single averaged vector, so they are excluded.
    groups = [
        ("pos",        pos_aug),
        ("neg",        list(_NEG_PROMPTS)),
        ("aspect_pos", [v[0] for v in _ASPECT_PROMPTS.values()]),
        ("aspect_neg", [v[1] for v in _ASPECT_PROMPTS.values()]),
        ("genre_ref",  list(G._GENRE_REF_PROMPTS)),
        ("sp",         list(G._STREET_POS_PROBES)),
        ("sn",         list(G._STREET_NEG_PROBES)),
    ]
    return groups, key_hash


def _sample_image_refs(n_images: int) -> "dict[str, np.ndarray]":
    """Evenly-spaced sample of graded photos → their stored open_clip embeddings."""
    import lance_store as ls
    tbl = ls._open_table()
    rows = tbl.search().select(["path"]).limit(100000).to_list()
    paths = sorted({r["path"] for r in rows if r.get("path")})
    alive = [p for p in paths if os.path.exists(p)]
    if not alive:
        return {}
    if len(alive) > n_images:                       # evenly spaced, not the first N
        step = len(alive) / n_images
        alive = [alive[int(i * step)] for i in range(n_images)]
    return ls.query_embeddings_by_paths(alive)


# ── Candidate side: the staged checkpoint, in its own subprocess ─────────────

def _encode_via_staging(mode: str, items: list, batch: int = 2) -> np.ndarray:
    """One disposable encode_worker run against the STAGING checkpoint.

    SIGLIP_HF_DIR is the documented hook for exercising a staging checkpoint
    through the real runtime path. SIGLIP_ENC_USE_OC=0 forbids any open_clip
    fallback, so this can never silently load the 10 GB path: if the staged
    checkpoint is unusable the subprocess fails and we report it.
    """
    fd, in_json = tempfile.mkstemp(suffix=".json"); os.close(fd)
    out_npy = in_json + ".npy"
    try:
        Path(in_json).write_text(json.dumps(list(items)), encoding="utf-8")
        env = dict(os.environ)
        env["SIGLIP_HF_DIR"]     = str(_STAGING)
        env["SIGLIP_ENC_USE_OC"] = "0"
        env["SIGLIP_TIER"]       = "high"
        env["SIGLIP_ENC_BATCH"]  = str(batch)
        env["PYTHONIOENCODING"]  = "utf-8"
        env["HF_HUB_OFFLINE"]    = "1"
        print(f"      encoding {len(items)} {mode} via staged fp16 checkpoint…", flush=True)
        r = subprocess.run(
            [sys.executable, str(_WORKER), mode, in_json, out_npy],
            cwd=str(_ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        for line in (r.stdout or "").splitlines():
            if any(t in line for t in ("loader", "peak_wset", "FATAL", "Traceback",
                                       "Error", "error")):
                print(f"        | {line}", flush=True)
        if r.returncode != 0 or not os.path.exists(out_npy):
            print((r.stderr or "")[-2000:], flush=True)
            raise RuntimeError(f"staged {mode} encode failed (exit {r.returncode})")
        return np.load(out_npy)
    finally:
        for f in (in_json, out_npy):
            try: os.unlink(f)
            except Exception: pass


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a * b).sum(axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=24,
                    help="how many graded photos to re-encode for the image check")
    ap.add_argument("--promote", action="store_true",
                    help="rename staging -> live if (and only if) validation passes")
    ap.add_argument("--force", action="store_true",
                    help="allow replacing a NON-EMPTY live checkpoint dir")
    args = ap.parse_args()

    print("=" * 68)
    print("SigLIP-2 staged-checkpoint validation (no open_clip load)")
    print("=" * 68)

    # ── 0. Preflight ─────────────────────────────────────────────────────────
    print("\n[0/4] preflight")
    if not (_STAGING / "config.json").exists():
        _fail(f"no staged checkpoint at {_STAGING}")
        return 1
    _shards = sorted(_STAGING.glob("*.safetensors"))
    _bytes = sum(f.stat().st_size for f in _shards)
    _ok(f"staging present: {len(_shards)} shards, {_bytes/1e9:.2f} GB")

    if (_DEST / "config.json").exists() and not args.force:
        _fail(f"{_DEST} is already populated — nothing to promote (use --force to replace)")
        return 1

    src_file = _CACHE / "encoder_source.txt"
    src_tag = src_file.read_text(encoding="utf-8").strip() if src_file.exists() else ""
    if not src_tag.startswith("openclip"):
        _fail(f"encoder_source.txt is '{src_tag or 'missing'}', not an open_clip tag — "
              f"the cached vectors are not the reference space")
        return 1
    _ok(f"cached vectors were produced by: {src_tag}")

    groups, key_hash = _prompt_groups()
    hash_file = _CACHE / "probe_embs.hash"
    npz_file  = _CACHE / "probe_embs.npz"
    if not (hash_file.exists() and npz_file.exists()):
        _fail("cache/probe_embs.npz|.hash missing — no stored text reference")
        return 1
    if hash_file.read_text().strip() != key_hash:
        _fail("probe_embs.hash does not match today's prompt lists — stored rows are "
              "no longer prompt-aligned; re-grade once to refresh, then re-run")
        return 1
    _ok(f"probe cache is row-aligned with current prompts (md5 {key_hash[:12]}…)")

    ref_txt = np.load(str(npz_file))
    flat_prompts: list[str] = []
    ref_rows:     list[np.ndarray] = []
    labels:       list[str] = []
    for name, prompts in groups:
        if name not in ref_txt.files:
            _fail(f"probe cache has no '{name}' group")
            return 1
        arr = ref_txt[name]
        if arr.shape[0] != len(prompts):
            _fail(f"group '{name}': cache has {arr.shape[0]} rows, prompts have {len(prompts)}")
            return 1
        flat_prompts.extend(prompts)
        ref_rows.append(arr)
        labels.extend([name] * len(prompts))
    ref_text = np.concatenate(ref_rows, axis=0)
    _ok(f"text reference: {ref_text.shape[0]} prompts across {len(groups)} groups")

    ref_img = _sample_image_refs(args.images)
    if not ref_img:
        _fail("no graded photos with existing files — cannot run the image check")
        return 1
    img_paths = sorted(ref_img)
    _ok(f"image reference: {len(img_paths)} graded photos sampled from LanceDB")

    # ── 1. Text: staged checkpoint vs stored open_clip text vectors ──────────
    print("\n[1/4] text embeddings")
    hf_text = _encode_via_staging("text", flat_prompts)
    if hf_text.shape != ref_text.shape:
        _fail(f"shape mismatch: staged {hf_text.shape} vs reference {ref_text.shape}")
        return 1
    txt_cos = _rowwise_cosine(hf_text, ref_text)
    print(f"      cosine vs open_clip:  mean {txt_cos.mean():.4f}   "
          f"min {txt_cos.min():.4f}   p05 {np.percentile(txt_cos, 5):.4f}")
    worst = int(np.argmin(txt_cos))
    print(f"      worst prompt [{labels[worst]}]: {txt_cos[worst]:.4f}  "
          f"{flat_prompts[worst][:64]!r}")

    # ── 2. Images: staged checkpoint vs stored open_clip image vectors ───────
    print("\n[2/4] image embeddings")
    hf_img = _encode_via_staging("images", img_paths)
    ref_img_arr = np.stack([ref_img[p] for p in img_paths], axis=0)
    if hf_img.shape != ref_img_arr.shape:
        _fail(f"shape mismatch: staged {hf_img.shape} vs reference {ref_img_arr.shape}")
        return 1
    img_cos = _rowwise_cosine(hf_img, ref_img_arr)
    print(f"      cosine vs open_clip:  mean {img_cos.mean():.4f}   "
          f"min {img_cos.min():.4f}   p05 {np.percentile(img_cos, 5):.4f}")
    w = int(np.argmin(img_cos))
    print(f"      worst image: {img_cos[w]:.4f}  {Path(img_paths[w]).name}")

    # ── 3. Standalone sanity (independent of the reference) ──────────────────
    print("\n[3/4] standalone sanity of the staged checkpoint")
    if hf_img.shape[1] != _EXPECT_DIM:
        _fail(f"embedding dim {hf_img.shape[1]} != {_EXPECT_DIM} — wrong model")
    else:
        _ok(f"embedding dim {hf_img.shape[1]}")

    if not np.all(np.isfinite(hf_img)) or not np.all(np.isfinite(hf_text)):
        _fail("non-finite values in staged embeddings (NaN/Inf)")
    else:
        _ok("all embeddings finite")

    norms = np.linalg.norm(hf_img, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-2):
        _fail(f"image embeddings not unit-norm (min {norms.min():.4f} max {norms.max():.4f})")
    else:
        _ok(f"unit-norm (min {norms.min():.4f} max {norms.max():.4f})")

    # Discriminative: distinct photos must not collapse onto one vector.
    if len(hf_img) >= 2:
        gram = hf_img @ hf_img.T
        off = gram[~np.eye(len(gram), dtype=bool)]
        if off.max() > 0.999:
            _fail(f"distinct images produce near-identical vectors (max off-diag {off.max():.4f})")
        else:
            _ok(f"discriminative across photos (max off-diag sim {off.max():.4f}, "
                f"mean {off.mean():.4f})")

    # ── 4. Verdict ───────────────────────────────────────────────────────────
    print("\n[4/4] verdict")
    gate_txt = float(txt_cos.mean()) >= _MIN_COSINE
    gate_img = float(img_cos.mean()) >= _MIN_COSINE
    print(f"      text  mean cosine {txt_cos.mean():.4f}  "
          f"{'PASS' if gate_txt else 'FAIL'}  (gate >= {_MIN_COSINE})")
    print(f"      image mean cosine {img_cos.mean():.4f}  "
          f"{'PASS' if gate_img else 'FAIL'}  (gate >= {_MIN_COSINE})")
    if not gate_txt:
        _fail("text cosine below gate")
    if not gate_img:
        _fail("image cosine below gate")

    if _FAIL:
        print("\n" + "=" * 68)
        print("VALIDATION FAILED — nothing was moved; open_clip stays in place.")
        for m in _FAIL:
            print(f"  - {m}")
        print("=" * 68)
        return 1

    print("\n" + "=" * 68)
    print("VALIDATION PASSED — the staged checkpoint matches the live embedding space.")
    if not args.promote:
        print("Dry run (no --promote): nothing was moved.")
        print("To promote:  venv\\Scripts\\python.exe scripts/validate_siglip2_staging.py --promote")
        print("=" * 68)
        return 0

    if _DEST.exists():
        leftovers = list(_DEST.iterdir())
        if leftovers and not args.force:
            _fail(f"{_DEST} is not empty — refusing to replace without --force")
            return 1
        shutil.rmtree(_DEST)
    _STAGING.rename(_DEST)
    print(f"PROMOTED — {_DEST}")
    print("encode_worker.py will now use the lean fp16 loader (~3.5 GB) instead of")
    print("the open_clip fallback (measured 10.3 GB peak).")
    print("")
    print("NOTE: this changes ENCODER_SOURCE, so grade_pipeline_v2's source-change")
    print("guard will clear the probe cache and RE-ENCODE every cached embedding")
    print("once on the next grade. That is intended — the two loaders are separate")
    print("vector spaces and must not be mixed.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
