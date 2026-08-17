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
    all_folders:      list,
    preset:           str,
    force_rescan:     bool,
    scan_mode:        bool,
    catalog_path_str: str,
    data_dir_str:     str,
    mogco_target:     int = 5,
    sample_limit:     int = 0,
    detect_only:      bool = False,
    deep_grade:       bool = False,
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

    # Force unbuffered stdout/stderr — block buffering loses the last prints before
    # a C-level crash (OOM/segfault), making the crash location impossible to read.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    _CATALOG_PATH = Path(catalog_path_str)
    _DATA_DIR     = Path(data_dir_str)

    # Cross-process "grade in progress" marker. local_launcher checks this after
    # its window closes and stays alive until the grade finishes, so a window
    # OOM-kill (or accidental close) can't abort the cull. detect_only sampling
    # runs are quick and not marked. Removed in the finally below.
    _LOCK_PATH = _DATA_DIR / "cache" / "grading.lock"
    if not detect_only:
        try:
            _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            _LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass

    def _progress(frac: float, desc: str = "") -> None:
        q.put({"progress": round(frac, 3), "desc": desc})

    try:
        # Pick the encoder quality tier BEFORE importing the pipeline. This must
        # happen first: grade_pipeline_v2, lance_store and siglip2_encoder each
        # read SIGLIP_TIER at module scope to fix their embedding dimension and
        # table name, so selecting afterwards would have no effect.
        try:
            import tier_select
            _tier, _tier_label, _tier_reason = tier_select.apply()
            print(f"[grade_worker] tier={_tier} ({_tier_label}) — {_tier_reason}")
            # Say up-front when RAM is short. select() deliberately falls through
            # to the smallest installed tier and lets the encoder's own floor
            # raise the one clear error (see tier_select.select docstring), but
            # that error arrives ~20s in, after the folder scan. The shortfall is
            # known here, so warn immediately instead of discarding the reason.
            # A warning only — the selector's estimate is conservative and runs
            # below it often still succeed, so this must never block a grade.
            _free = tier_select.free_ram_gb()
            if _free < tier_select.ram_need_gb(_tier):
                q.put({"progress": 0.0,
                       "desc": f"Quality: {_tier_label} — only {_free:.1f} GB RAM free; "
                               f"close a couple of apps if this fails"})
            else:
                q.put({"progress": 0.0, "desc": f"Quality: {_tier_label}"})
        except Exception as _e_tier:
            _tier_label = ""
            print(f"[grade_worker] tier selection skipped ({_e_tier}) — using default")

        from grade_pipeline_v2 import run_v2
        n = len(all_folders)
        combined_gallery: list = []

        # NOTE: catalog.json is intentionally NOT deleted here. It used to be
        # unlinked upfront ("clear stale state"), but _write_catalog() below
        # already does an atomic tmp-file + replace, which overwrites the old
        # catalog just as well once real results exist. Deleting it first only
        # created a window where a crash between the delete and the first
        # write (e.g. an early SigLIP-2/RAM failure) permanently lost the
        # user's whole graded catalog instead of just leaving the previous
        # (still-valid) one in place until fresh data replaces it.

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
                sample_limit = sample_limit,
                deep_grade   = deep_grade,
            )
            combined_gallery.extend(result.get("gallery", []))

        gallery_slim = [
            {k: v for k, v in photo.items() if k != "embedding"}
            for photo in combined_gallery
        ]

        # Drop the embeddings now that gallery_slim exists.
        #
        # Each entry's "embedding" is a 1536-element Python-float list (~49 KB);
        # across a few thousand photos combined_gallery holds hundreds of MB of
        # them. In cull mode (mogco_target<=0, which is what server.py always
        # sends) nothing below this line reads them again — the counts use
        # .get("grade"), the catalog write and the result message both use
        # gallery_slim — yet they stayed resident through the catalog write and
        # the whole JSON serialisation of the result, exactly when the process is
        # already at its peak. Only the NSGA-III branch needs them, so free them
        # whenever that branch is off.
        if mogco_target <= 0:
            import gc as _gc_emb
            for _photo in combined_gallery:
                _photo.pop("embedding", None)
            _gc_emb.collect()

        def _write_catalog(tag: str = "") -> None:
            """Persist gallery_slim to catalog.json (atomic). Safe to call repeatedly.
            No-op under detect_only so a sampled detection run never overwrites the
            user's real gallery catalog."""
            if detect_only:
                return
            try:
                # MERGE by path (see catalog_store): this used to write ONLY the
                # freshly graded folder, wiping every other folder out of the
                # user's gallery on each run.
                import catalog_store as _cat_store
                _cat_store.merge_write(gallery_slim, path=_CATALOG_PATH,
                                       tag=f"(worker {tag})".rstrip())
            except Exception as _e_cat:
                print(f"[grade_worker] catalog.json write failed: {_e_cat}")

        # Safety persist — grades are COMPLETE here. If NSGA-III or rationale
        # generation crashes the process downstream, the grades survive on disk
        # and the frontend recovers them on the next load.
        _write_catalog("(safety)")

        # Story sequencing (NSGA-III + the Judge's Verdict / rationale LLM) is NOT
        # part of culling. It loads additional models AFTER grading finishes, which
        # OOM-crashed the grade worker on the 6 GB GPU ("Grade worker died" with no
        # Python traceback). The Story tab builds sequences via its own endpoint
        # (/api/creative-direction/stream). The cull path sends mogco_target<=0, so
        # the entire block is skipped here.
        mogco_sequence: list = []
        mogco_error_msg: str = ""
        if mogco_target > 0:
            # NSGA-III across all photos
            _progress(0.97, "Running NSGA-III (strict literal constraints)…")
            try:
                import numpy as _np
                # run_nsga3_sequence_with_vlm / SequencerConstraintError no
                # longer exist — stale import crashed every mogco>0 run.
                from nsga3_sequencer import run_creative_story_sequencer
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
                selected = run_creative_story_sequencer(
                    seq_candidates, target=mogco_target, brief=_brief
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
            except Exception as e:
                mogco_error_msg = str(e)
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
        else:
            print("[grade_worker] Story sequencing skipped (cull mode, mogco_target<=0)")

        strong = sum(1 for g in combined_gallery if "Strong" in g.get("grade", ""))
        mid    = sum(1 for g in combined_gallery if "Mid"    in g.get("grade", ""))
        weak   = sum(1 for g in combined_gallery if "Weak"   in g.get("grade", ""))

        # Final catalog write — now enriched with curation rationales.
        _write_catalog("(final)")

        q.put({
            "done":           True,
            "quality":        _tier_label,   # "Fast" | "Balanced" | "Pro"
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
    finally:
        # Clear the in-progress marker so local_launcher knows the grade is done
        # (completed OR failed) and is free to exit.
        try:
            _LOCK_PATH.unlink(missing_ok=True)
        except Exception:
            pass


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

        # Wrap each run so a failure (e.g. insufficient RAM raised by the SigLIP
        # preflight) NEVER kills the persistent worker. A dead worker triggers a
        # multiprocessing respawn that fails on Windows with PermissionError
        # [WinError 5], wedging all future culls. Surviving = the user gets a
        # clean error, frees RAM, and retries against the SAME live worker.
        try:
            grade_worker_main(
                resp_q,
                req["folders"],
                req["preset"],
                req["force_rescan"],
                req["scan_mode"],
                req["catalog_path"],
                req["data_dir"],
                mogco_target=req.get("mogco_target", 5),
                sample_limit=req.get("sample_limit", 0),
                detect_only=req.get("detect_only", False),
                deep_grade=req.get("deep_grade", False),
            )
        except BaseException as _e_run:   # incl. MemoryError — keep the loop alive
            import traceback as _tb_run
            _msg = str(_e_run) or type(_e_run).__name__
            print(f"[grade_worker] run failed, worker stays alive: {_msg}", flush=True)
            try:
                resp_q.put({"error": _msg, "traceback": _tb_run.format_exc()})
            except Exception:
                pass


if __name__ == "__main__":
    # Safety guard — this module should never be run directly.
    print("grade_worker.py is a subprocess module; import and call grade_worker_loop() instead.")
