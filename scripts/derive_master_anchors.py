"""Derive the HUMAN-anchored calibration ruler from your master stars.

specvlm_pipeline._calibrate stretches the raw discriminant onto the absolute
[0.10, 0.95] scale using (lo, hi) anchors. Those anchors currently come from
the p1/p99 of the whole LIBRARY — the ruler measures how many photos exist,
not how good they are. This script re-anchors the same stretch at the
photographer's own quality distribution:

    hi = median(4-5★ discriminant) + ¼·IQR     your strong work tops the scale
    lo = median(1-2★ discriminant) − ¼·IQR     your rejects bottom it

…computed ONLY over master-rated photos (the tpe_master baseline), with the
same probe-fingerprint guard as the corpus anchors, plus a monotonicity
refusal: if the 3★ median doesn't sit between the anchored ends, your stars
are not being measured coherently by the discriminant and a scale must not
be written.

The output (cache/master_anchors.json) is picked up by
specvlm_pipeline.load_anchors() ahead of the library-percentile anchors.
Dry-run with --dry-run; refuse-guarded like derive_calibration_anchors.py.

Usage:
    venv\\Scripts\\python.exe scripts/derive_master_anchors.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252, which cannot encode ★/ρ — a crash in a
# print would abort a derivation that was otherwise fine. UTF-8 everywhere.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import ratings_store as rs
    import lance_store as ls
    from grade_pipeline_v2 import _tier_cache_name
    from specvlm_pipeline import probe_fingerprint
    import master_judge as mj

    # ── master paths (the human ground truth) ─────────────────────────────────
    ratings = rs.load()
    master = {p: s for p, s in ratings.items()
              if rs.get_source(p) == "tpe_master"}
    if len(master) < 30:
        print(f"[master-anchors] REFUSING: only {len(master)} master-tagged "
              f"ratings; need >= 30 for a human-anchored scale.")
        return 1

    # ── embeddings for exactly those paths ────────────────────────────────────
    arr = ls._open_table().to_arrow()
    paths_col = arr["path"].to_pylist()
    _col = arr["embedding"].combine_chunks()
    embs = (_col.flatten().to_numpy(zero_copy_only=False)
            .astype(np.float32, copy=False)
            .reshape(len(_col), -1))
    norms = np.linalg.norm(embs, axis=1)
    keep = np.abs(norms - 1.0) <= 0.05          # same contamination filter as
    embs = embs[keep]                            # derive_calibration_anchors
    paths_kept = [p for p, k in zip(paths_col, keep) if k]
    want = {p.strip().lower(): s for p, s in master.items()}
    sel, sel_stars = [], []
    for p, e in zip(paths_kept, embs):
        s = want.get(p.strip().lower())
        if s is not None:
            sel.append(e)
            sel_stars.append(int(s))
    if len(sel) < 30:
        print(f"[master-anchors] REFUSING: only {len(sel)} master-rated photos "
              f"have embeddings in the store (store may have been re-graded "
              f"since rating).")
        return 1
    embs = np.stack(sel)
    print(f"[master-anchors] master photos with embeddings: {len(embs)}")

    # ── the probes actually used by the discriminant ──────────────────────────
    cache = _ROOT / "cache" / _tier_cache_name("probe_embs", ".npz")
    if not cache.exists():
        print(f"[master-anchors] no probe cache at {cache}. Run one cull "
              f"first, then re-run this.")
        return 1
    d = np.load(cache)
    pos, neg = d["pos"], d["neg"]
    if pos.shape[1] != embs.shape[1]:
        print(f"[master-anchors] REFUSING: probe cache is {pos.shape[1]}-d but "
              f"the store is {embs.shape[1]}-d — different tiers.")
        return 1

    raw = (embs @ pos.T).max(axis=1) - (embs @ neg.T).max(axis=1)

    disc_by_star: dict = {}
    for s, v in zip(sel_stars, raw):
        disc_by_star.setdefault(s, []).append(float(v))
    for s in sorted(disc_by_star):
        v = np.asarray(disc_by_star[s])
        print(f"[master-anchors] {s}★  n={len(v):<4} median={np.median(v): .5f}")

    try:
        lo, hi, info = mj.human_anchor_lo_hi(disc_by_star)
    except ValueError as err:
        print(f"[master-anchors] REFUSING: {err}")
        return 1

    print(f"\n[master-anchors] human-anchored scale: "
          f"lo={lo:.5f} (1-2★)  hi={hi:.5f} (4-5★)")

    if args.dry_run:
        print("[master-anchors] --dry-run: nothing written.")
        return 0

    out = {
        "note": ("Human-anchored calibration: lo/hi derived from the "
                 "tpe_master star baseline (photographer 1-2★ / 4-5★ "
                 "discriminant quartiles), NOT library volume percentiles. "
                 "Preferred by specvlm_pipeline.load_anchors when the "
                 "fingerprint is fresh."),
        "fingerprint": probe_fingerprint(pos, neg),
        "lo": lo, "hi": hi,
        "master_photos": int(len(embs)),
        "stats": info,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = _ROOT / "cache" / "master_anchors.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[master-anchors] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
