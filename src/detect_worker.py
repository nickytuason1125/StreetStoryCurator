"""Phase-A detection subprocess — runs WHILE the GPU encode runs (2026-10-04).

Person detection (D-FINE) and subject sharpness need only the file list, not
the image embeddings, yet they ran after the encode inside iqa_worker with
the GPU otherwise idle. This worker runs the SAME functions on the SAME photo
list in the SAME slices iqa_worker will use, so iqa_worker can reuse the
results verbatim (grade_pipeline_v2 only hands them over on an exact slice
match — anything else falls back to computing them as before).

Isolated like iqa_worker/encode_worker: the grade runner never touches CUDA.

Usage:
    python detect_worker.py <in_json> <out_json>

in_json : {"slices": [[path, ...], ...]}
out_json: {"slices": [{"paths": [...], "person_detected": {...},
                       "subject_bboxes": {...}, "subject_sharpness": {...}}]}
"""
import sys, os, json
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> int:
    try:
        import work_counters as _wc
        _wc.install_exit_flush("detect")
        import proc_qos as _pq
        _pq.opt_out_power_throttling()
        _wc.bump("worker.detect")
    except Exception:
        pass
    import time as _time
    in_json, out_json = sys.argv[1], sys.argv[2]
    with open(in_json, encoding="utf-8") as f:
        slices = json.load(f)["slices"]

    # Same pre-warm as iqa_worker: cold imports from a fresh process can hit
    # WinError 6714 on Windows; retry them here in the main thread.
    for _mod in ("torch", "cv2", "fast_ingestion"):
        for _attempt in range(6):
            try:
                __import__(_mod)
                break
            except OSError:
                _time.sleep(0.25)
            except Exception:
                break

    from vision_grading_heads import _run_yolo_seg
    from subject_sharpness import score_paths

    out = []
    for sl in slices:
        t0 = _time.monotonic()
        person_detected, subject_bboxes = _run_yolo_seg(sl)
        t1 = _time.monotonic()
        sharp = {}
        if os.environ.get("FIRSTCUT_SUBJECT_SHARP", "1").strip() != "0":
            sharp = score_paths(sl)
        print(f"[detect_worker] {len(sl)} photos: detection {t1 - t0:.1f}s, "
              f"sharpness {_time.monotonic() - t1:.1f}s", flush=True)
        out.append({"paths": list(sl), "person_detected": person_detected,
                    "subject_bboxes": subject_bboxes, "subject_sharpness": sharp})

    tmp = out_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"slices": out}, f, default=_json_default)
    os.replace(tmp, out_json)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
