"""Isolated IQA / TOPIQ subprocess.

Loads the pyiqa 'topiq_nr' quality model + YOLO routing + composition heads in a
CLEAN process, scores the images, writes the results, and EXITS — so the
multiprocessing grade worker NEVER loads a GPU model itself. That is the whole
point: running GPU work directly inside the windowed grade worker is what caused
the recurring 0xC0000005 ACCESS_VIOLATION crash; the isolated-subprocess pattern
(same one src/encode_worker.py uses for SigLIP) makes it structurally impossible.

Usage:
    python iqa_worker.py <in_npz> <in_json> <out_npz> <out_json>

in_npz  : image_embeddings, clip_scores, [prompt_embedding], [genre_ref_embs]
in_json : image_paths, lum_stats, comp_eligible_paths, vlm_breakdowns
out_npz : quality (M,) float32
out_json: breakdowns, composition_overrides, chiaroscuro_flags, person_detected, subject_bboxes
"""
import sys, os, json
import numpy as np


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()                      # numpy scalar -> python
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    import traceback as _tb
    # Load the pyiqa CUDA model in the MAIN thread (see UniQAHead._timed_create):
    # creating it in a background thread and using it from the main thread faults
    # with 0xC0000005 in a fresh process.
    os.environ["IQA_MAIN_THREAD_LOAD"] = "1"
    try:
        in_npz, in_json, out_npz, out_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

        _arr = np.load(in_npz, allow_pickle=False)
        with open(in_json, encoding="utf-8") as f:
            _j = json.load(f)

        image_paths      = list(_j.get("image_paths") or [])
        _lum             = _j.get("lum_stats") or []
        lum_stats        = [tuple(t) for t in _lum] if _lum else None
        _ce              = _j.get("comp_eligible_paths") or []
        comp_eligible    = set(_ce) if _ce else None
        vlm_breakdowns   = _j.get("vlm_breakdowns") or None

        image_embeddings = _arr["image_embeddings"]
        clip_scores      = _arr["clip_scores"]
        prompt_embedding = _arr["prompt_embedding"] if "prompt_embedding" in _arr else None
        genre_ref_embs   = _arr["genre_ref_embs"]   if "genre_ref_embs"   in _arr else None

        # Pre-warm — in THIS main thread — every heavy module run_vision_heads
        # imports lazily deep in its call stack, several of them from spawned worker
        # threads (fast_ingestion in _load_images_parallel; pyiqa in _timed_create's
        # timeout thread; ultralytics/YOLO). Cold-importing from a fresh subprocess —
        # worse from a non-main thread — intermittently hits a Windows import glitch,
        # WinError 6714 (transactional-NTFS / Defender racing the src/ directory
        # scan), which would abort the whole IQA pass. Importing them once here, with
        # a short retry, puts them in sys.modules so the later (threaded) imports are
        # cache hits that never touch the filesystem.
        import time as _time
        for _mod in ("torch", "cv2", "pyiqa", "ultralytics", "fast_ingestion",
                     "vision_composition_heads", "yolo_auditor"):
            for _attempt in range(6):
                try:
                    __import__(_mod)
                    break
                except OSError as _oe:
                    print(f"[iqa_worker] import {_mod} retry {_attempt} ({_oe})", flush=True)
                    _time.sleep(0.25)
                except Exception:
                    break   # optional/absent dep → let run_vision_heads handle it

        from vision_grading_heads import run_vision_heads
        print(f"[iqa_worker] run_vision_heads on {len(image_paths)} images", flush=True)
        out = run_vision_heads(
            image_paths         = image_paths,
            image_embeddings    = image_embeddings,
            prompt_embedding    = prompt_embedding,
            clip_scores         = clip_scores,
            genre_ref_embs      = genre_ref_embs,
            lum_stats           = lum_stats,
            comp_eligible_paths = comp_eligible,
            vlm_breakdowns      = vlm_breakdowns,
        )

        np.savez(out_npz, quality=np.asarray(out.get("quality"), dtype=np.float32))
        payload = {
            "breakdowns":            out.get("breakdowns", []),
            "composition_overrides": out.get("composition_overrides", {}),
            "chiaroscuro_flags":     out.get("chiaroscuro_flags", {}),
            "person_detected":       out.get("person_detected", {}),
            "subject_bboxes":        out.get("subject_bboxes", {}),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=_json_default)
        print(f"[iqa_worker] done: quality={np.asarray(out.get('quality')).shape}", flush=True)
        # os._exit bypasses torch's CUDA atexit (see encode_worker.py) — outputs are
        # already flushed to disk, so there is no data loss.
        os._exit(0)
    except Exception:
        print(f"[iqa_worker] FATAL:\n{_tb.format_exc()}", flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
