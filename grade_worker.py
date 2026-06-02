"""
grade_worker.py — subprocess worker for GPU-intensive grading.

Runs in a separate process so a C-level crash (segfault, OOM kill) cannot
take down the FastAPI server.  Must be a top-level importable module because
Windows multiprocessing uses "spawn" (not fork).
"""
import os, sys, json, traceback as _tb
from pathlib import Path

# Suppress cmd-window flashes from child processes (same patch as server.py).
try:
    import suppress_console  # noqa: F401
except Exception:
    pass


def _setup_src_path() -> None:
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


def grade_worker_main(
    q,
    all_folders:   list,
    preset:        str,
    force_rescan:  bool,
    scan_mode:     bool,
    catalog_path_str: str,
    data_dir_str:  str,
) -> None:
    """
    Entry point called in the subprocess.  Progress and results are sent
    back via multiprocessing.Queue q.

    Progress messages:   {"progress": float, "desc": str}
    Gallery result:      {"done": True, "total": int, "strong": int, "mid": int,
                          "weak": int, "data": [...], "mogco_sequence": [...],
                          "mogco_error": str, "pipeline": "v2"}
    Error:               {"error": str, "traceback": str}
    """
    _setup_src_path()

    _CATALOG_PATH = Path(catalog_path_str)
    _DATA_DIR     = Path(data_dir_str)

    def _progress(frac: float, desc: str = "") -> None:
        q.put({"progress": round(frac, 3), "desc": desc})

    try:
        from grade_pipeline_v2 import run_v2
        n = len(all_folders)
        combined_gallery: list = []

        if _CATALOG_PATH.exists():
            try:
                _CATALOG_PATH.unlink()
                print("[grade_worker] Cleared stale catalog.json")
            except OSError as _e_del:
                print(f"[grade_worker] catalog delete warning: {_e_del}")

        for i, fp in enumerate(all_folders):
            p_start = i / n
            p_end   = (i + 1) / n

            def _fp(frac: float, desc: str = "", _s=p_start, _e=p_end) -> None:
                _progress(_s + frac * (_e - _s), desc)

            if n > 1:
                _fp(0.0, f"Grading folder {i+1}/{n}: {Path(fp).name}")

            result = run_v2(
                fp,
                preset       = preset,
                force_rescan = force_rescan,
                progress     = _fp,
                mogco_target = 0,
                scan_mode    = scan_mode,
            )
            combined_gallery.extend(result.get("gallery", []))

        gallery_slim = [
            {k: v for k, v in photo.items() if k != "embedding"}
            for photo in combined_gallery
        ]

        # NSGA-III across all photos
        _progress(0.97, "Running NSGA-III (strict literal constraints)…")
        mogco_sequence: list = []
        mogco_error_msg: str = ""
        try:
            import numpy as _np
            from nsga3_sequencer import run_nsga3_sequence_with_vlm, SequencerConstraintError
            try:
                from specvlm_pipeline import _CD_BRIEF as _brief
            except Exception:
                _brief = ""
            seq_candidates = [
                {
                    "path":          g["path"],
                    "score":         g.get("score", 0.5),
                    "embedding":     _np.array(
                        combined_gallery[idx].get("embedding", []),
                        dtype=_np.float32,
                    ),
                    "reasoning_log": g.get("reasoning_log", ""),
                    "breakdown":     g.get("breakdown", {}),
                }
                for idx, g in enumerate(gallery_slim)
                if "Strong" in g.get("grade", "") or "Mid" in g.get("grade", "")
            ]
            selected = run_nsga3_sequence_with_vlm(
                seq_candidates, target=5, brief=_brief
            )
            info_by_path = {g["path"]: g for g in gallery_slim}
            for rank, frame in enumerate(selected):
                base = dict(info_by_path.get(frame["path"], {"path": frame["path"]}))
                base.update({
                    "slot":             frame.get("slot", f"Slot {rank+1}"),
                    "slot_role":        frame.get("slot_role", ""),
                    "slot_score":       frame.get("slot_score", 0.0),
                    "mogco_objectives": frame.get("nsga3_objectives", {}),
                    "engine":           "nsga3",
                })
                mogco_sequence.append(base)
        except SequencerConstraintError as e:
            mogco_error_msg = str(e)
            print(f"[grade_worker] NSGA-III constraint error: {e}")
        except Exception as e:
            print(f"[grade_worker] NSGA-III failed: {e}")

        # Curation rationales
        _slim_by_path = {g["path"]: g for g in gallery_slim}
        if mogco_sequence:
            try:
                from creative_director_agent import generate_curation_rationales as _gen_rat
                _rationale_map = _gen_rat(mogco_sequence, _brief)
                for _r_path, _rat in _rationale_map.items():
                    if _r_path in _slim_by_path:
                        _slim_by_path[_r_path]["curation_rationale"] = _rat
                for _entry in mogco_sequence:
                    _rat = _rationale_map.get(_entry.get("path", ""), "")
                    if _rat:
                        _entry["curation_rationale"] = _rat
                if _rationale_map:
                    _progress(0.99, f"Rationales ready for {len(_rationale_map)} images…")
            except Exception as _e_rat:
                print(f"[grade_worker] Rationale generation failed: {_e_rat}")

        strong = sum(1 for g in combined_gallery if "Strong" in g.get("grade", ""))
        mid    = sum(1 for g in combined_gallery if "Mid"    in g.get("grade", ""))
        weak   = sum(1 for g in combined_gallery if "Weak"   in g.get("grade", ""))

        # Write final catalog
        try:
            import time as _cat_time
            _cat_folders = list(dict.fromkeys(
                str(Path(g["path"]).parent) for g in gallery_slim
            ))
            _CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _cat_tmp = _CATALOG_PATH.with_suffix(".json.tmp")
            _cat_tmp.write_text(
                json.dumps(
                    {"photos": gallery_slim, "folders": _cat_folders,
                     "saved_at": _cat_time.strftime("%Y-%m-%dT%H:%M:%S")},
                    ensure_ascii=False, indent=2,
                    default=lambda o: o.item() if hasattr(o, "item") else str(o),
                ),
                encoding="utf-8",
            )
            _cat_tmp.replace(_CATALOG_PATH)
            print(f"[grade_worker] catalog.json → {len(gallery_slim)} photos")
        except Exception as _e_cat:
            print(f"[grade_worker] catalog.json write failed: {_e_cat}")

        q.put({
            "done":           True,
            "total":          len(combined_gallery),
            "strong":         strong,
            "mid":            mid,
            "weak":           weak,
            "data":           gallery_slim,
            "mogco_sequence": mogco_sequence,
            "mogco_error":    mogco_error_msg,
            "pipeline":       "v2",
        })

    except Exception as exc:
        _full_tb = _tb.format_exc()
        print(f"[grade_worker] CRASH:\n{_full_tb}", file=sys.stderr, flush=True)
        try:
            _crash_path = _DATA_DIR / "crash.log"
            with open(_crash_path, "a", encoding="utf-8") as _cf:
                import datetime as _dt
                _cf.write(
                    f"\n{'='*60}\n{_dt.datetime.now().isoformat()} grade_worker crash:\n{_full_tb}\n"
                )
        except Exception:
            pass
        q.put({"error": str(exc), "traceback": _full_tb})


def grade_worker_loop(req_q, resp_q) -> None:
    """
    Persistent worker loop — stays alive between grading runs.

    Waits for request dicts on req_q, calls grade_worker_main() for each,
    then loops.  Exits after 30 minutes of inactivity or on a stop signal.
    Keeping the process alive preserves the SigLIP-2 singleton and all text
    embedding caches, eliminating the 15-30 s reload on repeat grades.
    """
    _setup_src_path()
    try:
        import suppress_console  # noqa: F401
    except Exception:
        pass

    import queue as _std_q
    _IDLE_TIMEOUT = 1800  # 30 min — exit if no request arrives
    print("[grade_worker] Persistent loop started — waiting for requests")

    while True:
        try:
            req = req_q.get(timeout=_IDLE_TIMEOUT)
        except _std_q.Empty:
            print("[grade_worker] Idle timeout (30 min) — exiting")
            return
        except Exception as _e_q:
            print(f"[grade_worker] Request queue error: {_e_q}")
            return

        if req is None or req.get("_stop"):
            print("[grade_worker] Stop signal — exiting")
            return

        grade_worker_main(
            resp_q,
            req["folders"],
            req["preset"],
            req["force_rescan"],
            req["scan_mode"],
            req["catalog_path"],
            req["data_dir"],
        )


if __name__ == "__main__":
    # Safety guard — this module should never be run directly.
    print("grade_worker.py is a subprocess module; import and call grade_worker_loop() instead.")
