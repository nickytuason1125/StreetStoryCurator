"""
Measure the ruler once, so grades stop being a curve.

specvlm_pipeline._calibrate used to stretch each batch onto [0.10, 0.95] from
its own min and max, which meant a photo's grade depended on what it was culled
alongside. The fix anchors that stretch to a fixed pair of values instead. This
script derives them.

The anchors are the p1 and p99 of the raw discriminant
(max(img·pos) - max(img·neg)) over a reference corpus. Percentiles rather than
min/max so a single freak frame cannot set the scale for everything graded
afterwards.

The corpus must be DIVERSE — the whole library, not one shoot. Anchors derived
from a single folder reproduce the original bug with extra steps.

Usage:
    venv\\Scripts\\python.exe scripts/derive_calibration_anchors.py
    SIGLIP_TIER=mid venv\\Scripts\\python.exe scripts/derive_calibration_anchors.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo-pct", type=float, default=1.0)
    ap.add_argument("--hi-pct", type=float, default=99.0)
    ap.add_argument("--min-photos", type=int, default=200,
                    help="refuse to derive from a corpus smaller than this")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import lance_store as ls
    import run_profile as rp
    from specvlm_pipeline import probe_fingerprint, _calibrate

    prof = rp.current()
    print(f"[anchors] tier={prof.tier} dim={prof.embed_dim} table={prof.lance_table}")

    # ── the library ───────────────────────────────────────────────────────────
    arr = ls._open_table().to_arrow()
    embs = np.asarray(arr["embedding"].to_pylist(), dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] < args.min_photos:
        print(f"[anchors] REFUSING: only {embs.shape[0]} photos in the store; "
              f"need >= {args.min_photos} for a representative scale.")
        return 1
    print(f"[anchors] corpus: {embs.shape[0]} photos, {embs.shape[1]}-d")

    # ── the probes actually used by the discriminant ──────────────────────────
    # NOT the 307 street probes: _raw_discriminant receives the RAG-augmented
    # "pos" group and "neg", which is a different and much smaller set. Deriving
    # against the wrong array would produce confidently wrong grades.
    # Use grade_pipeline_v2's own naming, not a reimplementation of it. Deriving
    # "probe_embs.npz" here independently ignored the tier suffix entirely and
    # looked for the high-tier file while running at mid — the same
    # re-derive-tier-state-in-a-second-place bug RunProfile exists to prevent.
    from grade_pipeline_v2 import _tier_cache_name
    cache = _ROOT / "cache" / _tier_cache_name("probe_embs", ".npz")
    if not cache.exists():
        print(f"[anchors] no probe cache at {cache}. Run one cull first so the "
              f"probes are computed and cached, then re-run this.")
        return 1
    d = np.load(cache)
    pos, neg = d["pos"], d["neg"]
    if pos.shape[1] != embs.shape[1]:
        print(f"[anchors] REFUSING: probe cache is {pos.shape[1]}-d but the "
              f"store is {embs.shape[1]}-d — different tiers.")
        return 1
    print(f"[anchors] probes: pos {pos.shape}, neg {neg.shape}")

    # ── the distribution ──────────────────────────────────────────────────────
    raw = (embs @ pos.T).max(axis=1) - (embs @ neg.T).max(axis=1)
    lo = float(np.percentile(raw, args.lo_pct))
    hi = float(np.percentile(raw, args.hi_pct))
    print("\n[anchors] raw discriminant distribution:")
    for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        print(f"    p{q:<3} {np.percentile(raw, q): .5f}")

    if hi - lo < 1e-5:
        print("[anchors] REFUSING: the p1-p99 span is degenerate.")
        return 1

    # ── what the new scale implies ────────────────────────────────────────────
    scored = _calibrate(raw, anchors=(lo, hi))
    strong = int((scored >= 0.60).sum())
    weak = int((scored < 0.41).sum())
    mid = len(scored) - strong - weak
    print(f"\n[anchors] lo(p{args.lo_pct:g})={lo:.5f}  hi(p{args.hi_pct:g})={hi:.5f}")
    print(f"[anchors] implied over the corpus: "
          f"Strong={strong} ({strong/len(scored):.0%})  "
          f"Mid={mid} ({mid/len(scored):.0%})  "
          f"Weak={weak} ({weak/len(scored):.0%})")
    print("[anchors] NOTE: this is the CLIP term alone, before TOPIQ and the "
          "taste blend, so it is not the final grade distribution.")
    if strong == 0 or weak == 0:
        print("[anchors] WARNING: a degenerate bucket suggests the anchors or "
              "the probe set are wrong, not the photographs.")

    if args.dry_run:
        print("\n[anchors] --dry-run: nothing written.")
        return 0

    # ── per-aspect anchors ────────────────────────────────────────────────────
    # Each aspect needs its OWN pair. Composition and Technical do not share a
    # distribution, so one scale for all five would systematically flatter
    # whichever sits highest and depress the rest.
    aspects_out: dict = {}
    asp_fp = None
    if "aspect_pos" in d and "aspect_neg" in d:
        from specvlm_pipeline import aspect_fingerprint
        ap, an = d["aspect_pos"], d["aspect_neg"]
        # _ASPECT_PROMPTS lives in specvlm_pipeline; grade_pipeline_v2 imports
        # it from there. Importing from the consumer rather than the definition
        # is what made this silently yield zero names.
        from specvlm_pipeline import _ASPECT_PROMPTS
        names = list(_ASPECT_PROMPTS.keys())
        if len(names) == ap.shape[0]:
            raw_asp = (embs @ ap.T) - (embs @ an.T)          # (N, A)
            print("\n[anchors] per-aspect p1/p99:")
            for j, nm in enumerate(names):
                a_lo = float(np.percentile(raw_asp[:, j], args.lo_pct))
                a_hi = float(np.percentile(raw_asp[:, j], args.hi_pct))
                if a_hi - a_lo < 1e-6:
                    print(f"    {nm:<16} DEGENERATE span — skipped")
                    continue
                aspects_out[nm] = [a_lo, a_hi]
                print(f"    {nm:<16} {a_lo: .5f} .. {a_hi: .5f}")
            asp_fp = aspect_fingerprint(ap, an)
        else:
            print(f"\n[anchors] aspect prompts ({len(names)}) do not match the "
                  f"cached aspect embeddings ({ap.shape[0]}) — skipping aspects")
    else:
        print("\n[anchors] no aspect embeddings in the probe cache — "
              "per-aspect scores will stay batch-relative")

    out = {
        "fingerprint": probe_fingerprint(pos, neg),
        "aspect_fingerprint": asp_fp,
        "aspects":     aspects_out,
        "tier":        prof.tier,
        "dim":         int(embs.shape[1]),
        "n_photos":    int(embs.shape[0]),
        "lo_pct":      args.lo_pct,
        "hi_pct":      args.hi_pct,
        "lo":          lo,
        "hi":          hi,
        "generated":   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = _ROOT / "cache" / "calibration_anchors.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[anchors] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
