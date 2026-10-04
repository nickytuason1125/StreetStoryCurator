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
# transformers imports TensorFlow whenever it is installed (it is — for the
# offline NIMA export script only). Nothing in a cull uses it, and loading it
# cost ~3.5 s per worker start plus its RAM (measured 2026-10-04). setdefault:
# an explicit environment still wins; child processes inherit it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Offline, like encode_worker (2026-10-04): TOPIQ's timm backbone asked
# huggingface.co to revalidate an already-cached resnet50; on a flaky network
# the SSL retries crashed the worker twice and the grade degraded to CLIP
# technical scores — 42% of a 600-photo test cull changed bucket. Verified
# topiq_nr loads from the local cache with this set. CLAUDE.md rule 5.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
    try:
        import work_counters as _wc
        _wc.install_exit_flush("iqa")
        import proc_qos as _pq
        _pq.opt_out_power_throttling()
        _wc.bump("worker.iqa")
    except Exception:
        pass
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
        # Phase-A results from detect_worker (only sent on an exact path match).
        _pre             = _j.get("precomputed") or None

        image_embeddings = _arr["image_embeddings"]
        clip_scores      = _arr["clip_scores"]
        prompt_embedding = _arr["prompt_embedding"] if "prompt_embedding" in _arr else None
        genre_ref_embs   = _arr["genre_ref_embs"]   if "genre_ref_embs"   in _arr else None

        # Pre-warm — in THIS main thread — every heavy module run_vision_heads
        # imports lazily deep in its call stack, several of them from spawned worker
        # threads (fast_ingestion in _load_images_parallel; pyiqa in _timed_create's
        # timeout thread). Cold-importing from a fresh subprocess —
        # worse from a non-main thread — intermittently hits a Windows import glitch,
        # WinError 6714 (transactional-NTFS / Defender racing the src/ directory
        # scan), which would abort the whole IQA pass. Importing them once here, with
        # a short retry, puts them in sys.modules so the later (threaded) imports are
        # cache hits that never touch the filesystem.
        import time as _time
        for _mod in ("torch", "cv2", "pyiqa", "fast_ingestion",
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

        # Subject sharpness OVERLAPS the quality model (2026-10-04). It only
        # needs the person pass's boxes, which exist before TOPIQ even loads;
        # run serially it added ~27 s per 600 RAWs of CPU work with the GPU
        # idle. The side thread is CPU-only (score_paths_cached never calls the
        # detector) — all CUDA stays on this main thread. Photos without cached
        # boxes are finished on the main thread below.
        import threading as _thr
        _sharp_on = (os.environ.get("FIRSTCUT_SUBJECT_SHARP", "1").strip() != "0"
                     and not (_pre and "subject_sharpness" in _pre))
        _vision_done = _thr.Event()
        _side = {"out": {}, "misses": list(image_paths), "err": None, "t": 0.0}

        def _sharp_side():
            import time as _ts
            try:
                import dfine_detector as _dd
                while not (_dd.PERSON_PASS_DONE.wait(0.5) or _vision_done.is_set()):
                    pass
                _t0 = _ts.monotonic()
                from subject_sharpness import score_paths_cached
                _side["out"], _side["misses"] = score_paths_cached(image_paths)
                _side["t"] = _ts.monotonic() - _t0
            except Exception as _e_side:
                _side["err"] = _e_side

        _side_thread = None
        if _sharp_on:
            _side_thread = _thr.Thread(target=_sharp_side, daemon=True, name="sharp-overlap")
            _side_thread.start()

        print(f"[iqa_worker] run_vision_heads on {len(image_paths)} images", flush=True)
        try:
            out = run_vision_heads(
                image_paths         = image_paths,
                image_embeddings    = image_embeddings,
                prompt_embedding    = prompt_embedding,
                clip_scores         = clip_scores,
                genre_ref_embs      = genre_ref_embs,
                lum_stats           = lum_stats,
                comp_eligible_paths = comp_eligible,
                vlm_breakdowns      = vlm_breakdowns,
                precomputed_detections = ((_pre["person_detected"], _pre["subject_bboxes"])
                                          if _pre else None),
            )
        finally:
            _vision_done.set()

        # Subject sharpness (2026-10-03): smeared subjects graded Mid because
        # nothing measured whether the SUBJECT is sharp. Runs here because it
        # needs D-FINE (GPU) — the grade worker itself must never touch CUDA.
        # Strictly additive: any failure leaves the map empty and no photo is
        # capped. FIRSTCUT_SUBJECT_SHARP=0 disables it.
        subject_sharp = {}
        if _pre and "subject_sharpness" in _pre:
            subject_sharp = dict(_pre["subject_sharpness"])
            print(f"[iqa_worker] subject sharpness reused from detect_worker "
                  f"({len(subject_sharp)} photos)", flush=True)
        elif _sharp_on:
            try:
                import time as _tss
                _t0 = _tss.monotonic()
                _side_thread.join()
                if _side["err"] is not None:
                    print(f"[iqa_worker] sharpness overlap failed ({_side['err']}) — "
                          f"measuring on the main thread", flush=True)
                subject_sharp = dict(_side["out"])
                _misses = _side["misses"] if _side["err"] is None else list(image_paths)
                if _misses:
                    from subject_sharpness import score_paths
                    subject_sharp.update(score_paths(_misses))
                print(f"[iqa_worker] subject sharpness {_side['t']:.1f}s overlapped "
                      f"+ {_tss.monotonic() - _t0:.1f}s after ({len(_misses)} on main thread)",
                      flush=True)
            except Exception as _sse:
                print(f"[iqa_worker] subject sharpness skipped: {_sse}", flush=True)
                subject_sharp = {}

        np.savez(out_npz, quality=np.asarray(out.get("quality"), dtype=np.float32))
        payload = {
            "subject_sharpness":     subject_sharp,
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
