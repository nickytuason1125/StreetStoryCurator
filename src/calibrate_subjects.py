"""calibrate_subjects — measured thresholds for the subject filter.

The subject filter ("vehicles only") splits the pool by cosine similarity
between a SigLIP text query and image embeddings. Absolute similarity values
vary per query phrase (measured: "vehicles" p75≈0.035, "a car or motorcycle…"
p75≈0.017), so a hardcoded threshold is wrong for every phrase but one.

This module measures the ACTUAL distribution of each subject query across the
graded catalog and stores per-query quantiles in data/subject_calibration.json.
The pipeline prefers the calibrated threshold and falls back to adaptive
in-pool quantiles for uncalibrated queries.

Run after every grading pass:  venv\\Scripts\\python.exe src\\calibrate_subjects.py
"""

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CAL_PATH = _ROOT / "data" / "subject_calibration.json"

QUANTILES = {"p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90}


def compute_thresholds(sims) -> dict:
    """Pure: quantiles + max of one similarity array → the calibration entry."""
    import numpy as np
    sims = np.asarray(sims, dtype=float)
    d = {k: float(np.quantile(sims, q)) for k, q in QUANTILES.items()}
    d["max"] = float(sims.max())
    return d


def calibrate(subject_queries: dict, embed_texts, image_embs, image_paths=None) -> dict:
    """Core: measure every subject query against the catalog's image
    embeddings. `embed_texts(list[str]) -> (n,1536)`; `image_embs` is the
    (m,1536) matrix. Returns the calibration dict ready to persist."""
    import numpy as np
    out = {"calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "catalog_size": int(len(image_embs)), "subjects": {}}
    queries = list(subject_queries.items())
    vecs = embed_texts([q for _, q in queries]).astype(float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    embs = np.asarray(image_embs, dtype=float)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    for (name, query), v in zip(queries, vecs):
        sims = embs @ v
        entry = compute_thresholds(sims)
        if image_paths is not None:
            order = np.argsort(-sims)
            entry["top_examples"] = [Path(image_paths[i]).name for i in order[:5]]
        out["subjects"][query] = entry
    return out


def load_calibration() -> dict:
    try:
        return json.loads(_CAL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def threshold_for(query: str):
    """Calibrated p75 threshold for this exact query phrase, or None."""
    cal = load_calibration()
    e = cal.get("subjects", {}).get(query)
    return e.get("p75") if e else None


def main() -> int:
    sys.path.insert(0, str(_ROOT / "src"))
    import numpy as np
    import lance_store
    from photo_brief import SUBJECT_SYNONYMS
    from creative_director import _embed_texts

    t0 = time.time()
    rows = lance_store.query_all(min_score=0.0)
    embs = np.stack([np.asarray(r["embedding"], dtype=float) for r in rows])
    paths = [r["path"] for r in rows]
    print(f"[calibrate] {len(embs)} catalog embeddings", flush=True)

    cal = calibrate(dict(SUBJECT_SYNONYMS), _embed_texts, embs, paths)
    _CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CAL_PATH.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(f"[calibrate] wrote {_CAL_PATH} "
          f"({len(cal['subjects'])} subjects) in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    import json
    sys.exit(main())
