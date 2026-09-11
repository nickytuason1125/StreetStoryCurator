"""Grading routes — moved verbatim from server_impl.py (Milestone 4 split).

Decorators retargeted app -> router; every bare name that used to live in
server_impl resolves lazily through the module __getattr__ below (PEP 562),
so request-time access always sees the fully-initialised app without
circular imports. FastAPI names are imported eagerly because decorator-time
evaluation (parameter defaults like Query(...)) runs at import.
"""
from fastapi import (
    APIRouter, Body, Depends, File, Form, HTTPException, Query, Request,
    Response, UploadFile,
)
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, PlainTextResponse,
    StreamingResponse,
)
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, validator, model_validator

from server_impl import (  # shared state & helpers
    Path, _CATALOG_PATH, _DATA_DIR, _grading_active, _release_annotation_model, _trim_crash_log, analyzer, annotation_queue, asyncio, gpu_lock, os, sys,
)
import json

# The unit root — the directory that holds server_impl.py, grade_runner.py and
# crash.log. NOT this module's directory.
#
# This code lived in server_impl.py at the root, where dirname(__file__) was
# the root. The server split moved it into routers/, and the expression came
# along unchanged, so it silently began resolving to routers/. The runner path
# became routers/grade_runner.py, which does not exist, and every grade started
# from the UI failed to launch its subprocess. Route parity checks did not catch
# it (the route still registers) and neither did the harness, which invokes
# grade_runner.py directly and never goes through this handler.
_UNIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from routers.library import GradeRequest

router = APIRouter()


def _prewarm_folder_paths(folders: list) -> None:
    """Queue thumbnail generation for a graded folder's first screens.

    Runs in a daemon thread AFTER a grade's stream ends. Reuses library's
    prewarm pool and generator (same cache names as on-demand), honouring
    its RAM gate and in-flight dedup. A failure here is invisible by design —
    it is a warmth optimisation, never a correctness path.
    """
    try:
        from routers.library import _THUMB_PREWARM, _gen_one_thumb, _is_slow_volume
        cap = int(os.environ.get("FIRSTCUT_POSTGRADE_PREWARM", "300") or 300)
        for folder in folders:
            try:
                fdir = Path(folder)
                if not fdir.is_dir():
                    continue
                paths = sorted(str(p) for p in fdir.iterdir()
                               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp",
                                                        ".arw", ".cr2", ".cr3", ".nef",
                                                        ".raf", ".dng", ".heic", ".heif"))
            except Exception:
                continue
            for p in paths[:max(cap, 0)]:
                try:
                    _THUMB_PREWARM.submit(_gen_one_thumb, p, True)
                except Exception:
                    break
    except Exception:
        pass


def _precull_ram_sweep() -> None:
    """Free every parent-process cache that a cull does not need BEFORE it starts.

    The cull peak is ~2.5 GB (measured); on a machine already near its floor the
    cheapest headroom is releasing what the idle parent hoards: the preview/thumbnail
    decode caches and any CUDA blocks held by released models. Runs synchronously —
    it is fast (disk unlink + gc) and must complete before the encode subprocess
    spawns, or the memory it frees comes back too late.
    """
    import gc as _gc
    try:
        from server_impl import _evict_preview_cache
        _evict_preview_cache()
    except Exception:
        pass
    try:
        from vram_manager import VRAMManager
        VRAMManager.purge_vram()
    except Exception:
        pass
    _gc.collect()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)



@router.get("/api/ollama/status")
async def ollama_status():
    """
    Local text/vision model availability. Response shape is unchanged:
    {alive: bool, models: [{name, size_vram, size_total, until}]}

    The path keeps its name so the frontend's 15-second poll keeps working, but
    it no longer speaks to an Ollama daemon — there isn't one. `alive` now means
    "a local model is installed", and `models` lists those files with their
    on-disk size. size_vram is 0 because llama_cpp's residency is decided per
    call by the offload ladder, not held as a server-side fact.
    """
    def _sync() -> dict:
        try:
            import sys as _s, os as _o
            _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
            import local_llm as _llm
            import critique_engine as _ce

            models = []
            # Registry for BOTH. This hardcoded a "2b" filename while the
            # registry shipped the 3B checkpoint, so the vision model always
            # read as absent -- the fifth module in this repo to name a weight
            # file instead of asking.
            try:
                import model_registry as _mr_st
                _paths = (_llm.model_path(), _mr_st.gguf("vision").dest)
            except Exception:
                _paths = (_llm.model_path(),)
            for _p in _paths:
                if _p.exists():
                    models.append({"name": _p.name, "size_vram": 0,
                                   "size_total": _p.stat().st_size, "until": ""})
            return {"alive": bool(_llm.available() or _ce.vision_available()),
                    "models": models}
        except Exception as _e:
            return {"alive": False, "models": [], "error": str(_e)}

    result = await asyncio.get_running_loop().run_in_executor(None, _sync)
    return JSONResponse(result)


@router.get("/api/grading/status")
async def grading_status():
    """Is a grade runner alive right now? The UI polls this after a dropped
    stream: the runner is durable (detaches and finishes in the background),
    so 'stream ended' must not read as 'grade died'."""
    return {"grading": _grading_active.is_set()}


@router.post("/api/grade/v2/stream")
async def grade_photos_v2_stream(req: GradeRequest):
    """
    V2 pipeline: SigLIP → Q-Align → PersonalHead → MOGCO-II.
    Same SSE format as /api/grade/stream for drop-in frontend compatibility.
    Supports multi-folder: grades each folder, then runs MOGCO-II once across all.
    """
    import json as _json
    from fastapi.responses import StreamingResponse

    # ── Single-flight guard ────────────────────────────────────────────────
    # The check runs here and _grading_active is SET just before the
    # StreamingResponse is returned. Everything between the two is
    # synchronous (listdir, psutil, json — no awaits), so on the event loop
    # no second request can slip between check and set. The flag used to be
    # set only inside _stream_with_lock(), AFTER `await gpu_lock.acquire()` —
    # a genuine yield point — leaving a window where a reload mid-grade, a
    # second window, or a double-fired modal each spawned their own
    # grade_runner subprocess, competing for gpu_lock, VRAM, and the RAM
    # floor. This generator's finally remains the single CLEAR point.
    if _grading_active.is_set():
        raise HTTPException(
            409,
            "A grade is already running. Wait for it to finish before starting another.",
        )

    # ── Cross-process arm of the same guard ─────────────────────────────────
    # grade_worker marks cache/grading.lock for the life of a grade. The
    # in-process flag above cannot see a runner started by a DIFFERENT server
    # stack — the 2026-09-07 duplicate-stack incident had two grade_runners on
    # the same request file, each server's flag clear because the runner
    # belonged to the other process. A lock naming a dead process is removed
    # on read (see grade_lock.grade_in_progress), so a crashed grade cannot
    # wedge this gate; any other failure fails open and never blocks a grade.
    try:
        from src.grade_lock import grade_in_progress
        if grade_in_progress(_DATA_DIR):
            raise HTTPException(
                409,
                "A grade is already running (active grading.lock from another "
                "process). Wait for it to finish before starting another.",
            )
    except HTTPException:
        raise
    except Exception:
        pass  # the lock is a marker, not a source of truth — never block on it

    # NOTE: there was an Ollama health gate here that 503'd every non-scan grade
    # when http://localhost:11434 did not answer. Nothing installed Ollama — not
    # Setup.ps1, not requirements.txt — and no user-facing doc mentioned it, while
    # CLAUDE.md rule 5 promised a fully offline app. So a correct, complete install
    # could not grade a single photo. Grading never needed it either: the default
    # path is SigLIP zero-shot plus TOPIQ, and the modules that did call Ollama now
    # run the same models locally through llama_cpp. Removed, not made optional.

    # Resolve all valid folders — folder_paths (multi) takes priority over folder_path.
    # Same wake-up retry as GradeRequest.validate_folder_path: removable readers
    # behind USB selective suspend vanish for a second exactly at submit time.
    import time as _fp_retry_time
    def _dir_ok(fp: str) -> bool:
        for _probe in range(3):
            if os.path.isdir(fp):
                return True
            if _probe < 2:
                _fp_retry_time.sleep(2.0)
        return False
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if _dir_ok(fp)]
    if not all_folders:
        if req.folder_path and _dir_ok(req.folder_path):
            all_folders = [str(Path(req.folder_path).resolve())]
        else:
            raise HTTPException(400, "No valid folder path provided")

    # ── System RAM admission (single decision — see src/memory_plan.py) ─────
    # Deliberately AFTER folder resolution: what a cull costs depends on the
    # decode path and on how big the job is, so admission needs the job in
    # front of it. This used to be one constant checked before anything was
    # known — 1.8 GB, which was Balanced's ENCODER floor rather than a
    # whole-cull budget, so it admitted runs that then drove the machine to
    # 0.10 GB free and into the pagefile (111 s versus 25 s for the same
    # folder with room). It then refused outright when RAM was tight, even
    # though the codebase contained every ingredient of a graceful downgrade.
    #
    # Admission now picks the RICHEST plan that fits: full quality if it
    # fits, otherwise an AUTOMATIC downgrade to Scan (measured ~2.0 GB tree
    # vs 3.8–4.2 GB for full) — refusal only when even Scan cannot fit. The
    # chosen plan is reported to the UI via an SSE `notice` at stream start.
    _plan = None
    _plan_note = None
    _eff_scan = bool(req.scan_mode)
    _n_photos = 0
    _scope_note = None
    try:
        from src import memory_plan as _mp
        # Counting is a listdir, not a decode.
        _exts = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
                 ".rw2", ".raf", ".arw", ".cr2", ".cr3", ".nef", ".dng", ".orf")
        for _fp in all_folders:
            try:
                _n_photos += sum(1 for _f in os.listdir(_fp)
                                 if _f.lower().endswith(_exts))
            except OSError:
                pass
        _plan = _mp.plan_for(_n_photos, requested_scan=req.scan_mode)
        # Scope honesty (2026-09-07 SD-upload): a force_rescan run over a huge
        # library must never start silently. The user pressed "Re-grade
        # everything" on a 6k SD folder and the run tried to redo ~100k photos —
        # say exactly what is about to happen, and how to avoid it.
        if req.force_rescan and _n_photos > 20000:
            _scope_note = (f"This re-grade covers every photo in the selected folder — about "
                           f"{_n_photos:,} photos, which can take hours. If you only added new "
                           f"photos, cancel and use 'New photos only' to grade just those.")
        if _plan is None:
            _free_gb = _mp.free_ram_gb()
            _free_txt = f"only {_free_gb:.1f} GB free" if _free_gb is not None else "free RAM could not be measured"
            _hogs_txt = _mp.top_ram_hogs()
            _hogs_note = f" Biggest right now: {_hogs_txt} — closing one of those is usually enough." if _hogs_txt else ""
            _pagefile_hint = ""
            try:
                _commit = _mp.commit_headroom_gb()
                if _commit is not None and _free_gb is not None and _commit < _free_gb - 0.5:
                    _pagefile_hint = (" Your pagefile is the bottleneck, not RAM (commit headroom "
                                      f"{_commit:.1f} GB vs {_free_gb:.1f} GB physical) — set a fixed "
                                      "pagefile (initial 24576 MB / max 32768 MB) in System → Advanced → "
                                      "Virtual Memory, then reboot.")
            except Exception:
                pass
            return JSONResponse(
                status_code=503,
                content={"error": f"Not enough RAM to grade safely — {_free_txt}, and even a "
                         f"reduced-batch Scan (the lightest pass, ~{_mp.min_admission_gb():.1f} GB) does not "
                         f"fit. Close a couple of apps and retry.{_hogs_note}{_pagefile_hint}",
                         "alternatives": {"close_apps": True, "smaller_selection": True}},
            )
        _eff_scan = bool(req.scan_mode or _plan["scan_mode"])
        _plan_note = _plan.get("note") or None
    except Exception as _gate_exc:
        # Fail-OPEN (availability first — the encoder's own floors still
        # protect the machine) but LOUD: the old `except Exception: pass`
        # let a dead gate vanish silently.
        _plan_note = f"RAM admission check failed ({_gate_exc}) — relying on the encoder's own floors."
        print(f"[server] RAM admission check FAILED (fail-open): {_gate_exc}", flush=True)

    async def _stream_with_lock():
        # Hold gpu_lock for the full grading run so the annotation daemon
        # cannot load its GGUF concurrently and bust the 5.5 GB VRAM ceiling.
        # EVERYTHING from the acquire onward sits inside the try: the prelude
        # used to run outside it, so a failure there leaked gpu_lock — and,
        # now that the handler owns the single-flight SET, would have wedged
        # the guard until restart. This generator owns only the CLEAR.
        _gl = gpu_lock
        _acquired = False
        _proc = None
        _req_path = None
        _prog_path = None
        try:
            if _gl is not None:
                await _gl.acquire()
                _acquired = True
            if _plan_note:
                # The admission decision's message (downgrade explanation,
                # fail-open warning) reaches the UI before any heavy work.
                import json as _sj
                yield f"data: {_json.dumps({'notice': _plan_note})}\n\n"
                print(f"[server] Plan note: {_plan_note}", flush=True)
            if _scope_note:
                # Full-library re-grade warning — emitted before any heavy work
                # so the user can cancel while it is still cheap to do so.
                yield f"data: {_json.dumps({'notice': _scope_note})}\n\n"
                print(f"[server] Scope note: {_scope_note}", flush=True)
            # Free RAM held by background work before the grade's heavy model loads:
            #   - the niche detector's CLIP (~0.7 GB), reloads on the next folder-select
            #   - pause the background thumbnail prewarm (RAW decodes spike RAM)
            try:
                import fast_niche_detector as _fnd_rel
                _fnd_rel.release()
            except Exception:
                pass
            _release_annotation_model()   # free the ~1.5-4 GB annotation model
            _precull_ram_sweep()          # parent-cache sweep BEFORE the encode subprocess spawns
            import tempfile as _tf, subprocess as _sp

            # ── Run the grade as a CLEAN subprocess (grade_runner.py) ───────────
            # NOT a multiprocessing spawn child. A plain process running the pipeline
            # completes reliably where the multiprocessing worker died 0xC0000005 at
            # GPU-process boundaries (a GPU child exiting faulted the mp-spawn parent).
            # If this process crashes it is a separate OS process — the server and
            # window are unaffected; the catalog checkpoint + Resume recover the work.
            # SigLIP and IQA each still run in their OWN sub-subprocesses; this runner
            # does no direct GPU work.
            _fd, _req_path = _tf.mkstemp(suffix=".gradereq.json"); os.close(_fd)
            _prog_path = _req_path + ".progress.jsonl"
            open(_prog_path, "w", encoding="utf-8").close()
            with open(_req_path, "w", encoding="utf-8") as _rf:
                _json.dump({
                    "folders":      all_folders,
                    "preset":       req.preset,
                    "force_rescan": req.force_rescan,
                    "scan_mode":    _eff_scan,                 # honour a memory-plan downgrade
                    "deep_grade":   req.deep_grade and not _eff_scan,
                    "catalog_path": str(_CATALOG_PATH),
                    "data_dir":     str(_DATA_DIR),
                    "mogco_target": 0,   # cull only; Story sequencing is its own endpoint
                }, _rf)

            _runner = os.path.join(_UNIT_ROOT, "grade_runner.py")
            _flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
            _renv = dict(os.environ); _renv["PYTHONIOENCODING"] = "utf-8"
            # Route the runner's [v2] progress prints to crash.log (utf-8) for
            # debuggability — its result stream goes via the progress file above.
            _crash_path = os.path.join(_UNIT_ROOT, "crash.log")
            _trim_crash_log(_crash_path)
            _rlog = open(_crash_path, "a", encoding="utf-8", errors="replace")
            # ── Spawn-time RAM re-check: the admission reading can be minutes
            # old by now. SAME decision module as admission (src/memory_plan.py)
            # so the two gates cannot disagree: refuse if even Scan no longer
            # fits, honour a downgrade that only became necessary now (rewrite
            # the request file — the runner has not been spawned yet), and warn
            # in the tight-but-viable band.
            try:
                from src import memory_plan as _mp_spawn
                _spawn_plan = _mp_spawn.plan_for(_n_photos, requested_scan=req.scan_mode)
                if _spawn_plan is None:
                    _free_gb = _mp_spawn.free_ram_gb()
                    _free_txt = f"{_free_gb:.1f} GB" if _free_gb is not None else "unmeasurable"
                    import json as _sj
                    yield f"data: {_json.dumps({'error': f'Refused: {_free_txt} RAM free and even a Scan (the lightest pass) does not fit any more — at this level the machine page-thrashes and the app freezes (measured). Close a few apps — a browser tab or two is usually enough — and retry.', 'alternatives': {'close_apps': True, 'smaller_selection': True}})}\n\n"
                    print(f"[server] Grade REFUSED pre-spawn: RAM shrank below the Scan floor", flush=True)
                    return
                if _spawn_plan["degraded"] and not _eff_scan:
                    with open(_req_path, "w", encoding="utf-8") as _rf2:
                        _json.dump({
                            "folders":      all_folders,
                            "preset":       req.preset,
                            "force_rescan": req.force_rescan,
                            "scan_mode":    True,
                            "deep_grade":   False,
                            "catalog_path": str(_CATALOG_PATH),
                            "data_dir":     str(_DATA_DIR),
                            "mogco_target": 0,
                        }, _rf2)
                    import json as _sj
                    yield f"data: {_json.dumps({'notice': _spawn_plan['note']})}\n\n"
                    print(f"[server] Late downgrade to Scan at spawn: {_spawn_plan['free_gb']:.2f} GB free", flush=True)
                _need = _spawn_plan["need_gb"]
                _free_now = _spawn_plan["free_gb"]
                if _need is not None and _free_now is not None and _free_now < _need + _mp_spawn.TIGHT_BAND_GB:
                    import json as _sj
                    # Tight-but-viable band: warn once, do not block. The cull
                    # runs slower here; the checkpoint + Resume recover it if
                    # the machine still runs out mid-way.
                    yield f"data: {_json.dumps({'notice': f'Tight RAM: {_free_now:.1f} GB free — the cull will run slower. Closing a few apps keeps it fast.'})}\n\n"
                    print(f"[server] Grade advisory: tight RAM {_free_now:.2f} GB free — proceeding", flush=True)
            except Exception as _spawn_gate_exc:
                print(f"[server] pre-spawn RAM re-check failed (fail-open): {_spawn_gate_exc}", flush=True)

            # ── Hard DISK gate: a full drive fails persistence SILENTLY. ─────
            # Observed 2026-08-30 with 0 bytes free: the grade itself ran and
            # returned done, then catalog merge_write and Lance compaction
            # died with OSError Errno 28 — the work completed but could not
            # be saved. Refusing up front is the honest, crash-proof
            # behaviour (mirror of the RAM gate above).
            try:
                import shutil as _sh_gate
                _disk_gb = _sh_gate.disk_usage(str(_DATA_DIR)).free / 1e9
                if _disk_gb < 1.0:
                    yield f"data: {_json.dumps({'error': f'Refused: only {_disk_gb:.2f} GB disk free on the app drive — grades would complete but their results cannot be saved (catalog/Lance writes fail on a full disk). Free up space and retry.'})}\n\n"
                    print(f"[server] Grade REFUSED pre-spawn: {_disk_gb:.2f} GB disk free", flush=True)
                    return
            except Exception:
                pass

            import win_job as _wj
            _proc = _wj.popen(
                [sys.executable, _runner, _req_path, _prog_path],
                cwd=_UNIT_ROOT,
                creationflags=_flags, close_fds=True,
                stdin=_sp.DEVNULL, stdout=_rlog, stderr=_rlog, env=_renv,
            )
            try: _rlog.close()   # child keeps its inherited fd
            except Exception: pass
            print(f"[server] Grade runner subprocess pid={_proc.pid}", flush=True)

            _loop = asyncio.get_running_loop()
            _pos = 0
            _done = False
            # AUTO-RESUME budget: a crashed runner is re-spawned in place up to
            # 2 extra times (3 attempts total). Each respawn continues from the
            # encode checkpoint + incremental Lance cache, so no work is lost.
            _auto_resume_total = 2
            _auto_resume_left = _auto_resume_total
            def _read_new():
                # Binary read from the last byte offset; only consume up to the last
                # complete newline so a half-written progress line is never mis-parsed.
                nonlocal _pos
                try:
                    with open(_prog_path, "rb") as _pf:
                        _pf.seek(_pos)
                        _data = _pf.read()
                    _cut = _data.rfind(b"\n")
                    if _cut < 0:
                        return ""
                    _pos += _cut + 1
                    return _data[:_cut + 1].decode("utf-8", "replace")
                except Exception:
                    return ""

            def _emit(_line):
                nonlocal _done
                _line = _line.strip()
                if not _line:
                    return None
                try:
                    _msg = _json.loads(_line)
                except Exception:
                    return None
                if _msg.get("done") and annotation_queue is not None:
                    for _g in _msg.get("data", []):
                        _gpath = _g.get("path", "")
                        if _gpath and float(_g.get("score", 0.0)) > 0.0 and not _g.get("has_annotations"):
                            annotation_queue.put_nowait(_gpath)
                if _msg.get("done") or _msg.get("error"):
                    _done = True
                return _line

            while True:
                _chunk = await _loop.run_in_executor(None, _read_new)
                for _ln in _chunk.splitlines():
                    _out = _emit(_ln)
                    if _out is not None:
                        yield f"data: {_out}\n\n"
                if _done:
                    break
                if _proc.poll() is not None:
                    # Runner exited — drain any final lines, then if we never saw a
                    # done/error result it crashed. AUTO-RESUME (2026-09-11): the
                    # encode stage now checkpoints every chunk and the LanceDB
                    # incremental cache skips already-graded photos, so a re-spawn
                    # of the SAME request continues where the crash left off.
                    # Re-spawn up to 2 times automatically (RAM-gated), and only
                    # surface the recoverable-checkpoint error after the last try.
                    _tailchunk = await _loop.run_in_executor(None, _read_new)
                    for _ln in _tailchunk.splitlines():
                        _out = _emit(_ln)
                        if _out is not None:
                            yield f"data: {_out}\n\n"
                    if not _done and _auto_resume_left > 0:
                        _auto_resume_left -= 1
                        _cause = ""
                        # Wait for RAM recovery before re-spawning — respawning
                        # into the same low-memory state just burns the attempt.
                        _waited = 0
                        try:
                            from src import memory_plan as _mp_resume
                            _floor = _mp_resume.min_admission_gb()
                            while _waited < 120:
                                _free_r = _mp_resume.free_ram_gb()
                                if _free_r is None or _free_r >= _floor:
                                    break
                                await asyncio.sleep(5)
                                _waited += 5
                            _free_r = _mp_resume.free_ram_gb()
                            if _free_r is not None and _free_r < 2.0:
                                _cause = (f" The machine is still low on memory "
                                          f"({_free_r:.1f} GB free) — closing a few apps "
                                          "will make the restart stick. ")
                        except Exception:
                            pass
                        _attempt_no = 3 - _auto_resume_left
                        print(f"[server] Grade runner crashed (code={_proc.returncode}) — "
                              f"auto-restarting from checkpoint (attempt {_attempt_no}/3)",
                              flush=True)
                        yield f"data: {_json.dumps({'notice': f'The grade crashed but auto-restarts from its checkpoint (restart {_attempt_no} of 3).{_cause}'})}\n\n"
                        try:
                            _rlog = open(_crash_path, "a", encoding="utf-8", errors="replace")
                            _proc = _wj.popen(
                                [sys.executable, _runner, _req_path, _prog_path],
                                cwd=_UNIT_ROOT,
                                creationflags=_flags, close_fds=True,
                                stdin=_sp.DEVNULL, stdout=_rlog, stderr=_rlog, env=_renv,
                            )
                            try: _rlog.close()
                            except Exception: pass
                            print(f"[server] Grade runner RESPAWNED pid={_proc.pid} "
                                  f"(attempt {_attempt_no}/3)", flush=True)
                            continue   # keep streaming the new runner's progress
                        except Exception as _e_respawn:
                            print(f"[server] auto-resume spawn failed: {_e_respawn}",
                                  flush=True)
                    if not _done:
                        print(f"[server] Grade runner exited without result: code={_proc.returncode}", flush=True)
                        _recovered = 0
                        try:
                            if _CATALOG_PATH.exists():
                                _recovered = len(_json.loads(_CATALOG_PATH.read_text(encoding="utf-8")).get("photos", []))
                        except Exception:
                            pass
                        _cause = ""
                        try:
                            import psutil as _ps_death
                            _free_death = _ps_death.virtual_memory().available / 1e9
                            if _free_death < 2.0:
                                _cause = (f" The machine was low on memory when it died ({_free_death:.1f} GB free now) — "
                                          "close a few apps before resuming. ")
                        except Exception:
                            pass
                        yield f"data: {_json.dumps({'error': f'Grade process exited unexpectedly (code {_proc.returncode}) after {_auto_resume_total} auto-restart attempt(s). What IS kept: every committed chunk — the encode stage checkpoints partial embeddings, and a re-run skips both encoded chunks and already-graded photos. {_cause}Check crash.log for the technical detail', 'recovered': _recovered})}\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        finally:
            # Disconnect semantics (2026-09-10) — the runner is DURABLE.
            # This finally runs on normal completion AND on client disconnect
            # (GeneratorExit at a yield). It used to TERMINATE the runner on
            # disconnect: every window close / reload / UI reconnect killed an
            # in-flight grade mid-run ("Grade runner pid=… terminated on stream
            # close", three culls lost on 2026-09-10 alone) — contradicting the
            # app's own decoupled-backend architecture ("closing the window
            # must never abort an in-flight grade"). The runner is now left
            # ALIVE on disconnect: it holds the machine-wide grading.lock for
            # its whole life (so the 409 gates still refuse a second grade),
            # writes its durable checkpoints and the end-of-run catalog commit
            # itself, and cleans its own lock up in ITS finally. A monitor
            # thread owns the cleanup that used to live here: temp files,
            # the pre-spawn lock claim, _grading_active, gpu_lock, and the
            # post-grade prewarm — all after the runner actually exits.
            if _proc is not None and _proc.poll() is None:
                _proc_pid = _proc.pid
                print(f"[server] Grade runner pid={_proc_pid} DETACHED — client "
                      f"stream closed; the grade continues in the background "
                      f"and the catalog commits when it finishes", flush=True)

                def _reap_detached(proc=_proc, req=_req_path, prog=_prog_path,
                                   claim=_claim_path, loop=asyncio.get_running_loop()):
                    try:
                        proc.wait()
                        print(f"[server] Detached grade runner pid={_proc_pid} "
                              f"finished (code={proc.returncode})", flush=True)
                    except Exception:
                        pass
                    for _t in (req, prog):
                        try: os.unlink(_t)
                        except Exception: pass
                    try:
                        if claim is not None and \
                                claim.read_text(encoding="utf-8").strip() == str(os.getpid()):
                            claim.unlink(missing_ok=True)
                    except Exception:
                        pass
                    _grading_active.clear()
                    try:
                        if _gl is not None and _acquired and loop.is_running():
                            loop.call_soon_threadsafe(_gl.release)
                    except Exception:
                        pass
                    try:
                        if all_folders:
                            threading.Thread(
                                target=_prewarm_folder_paths,
                                args=(list(all_folders),),
                                daemon=True, name="post-grade-prewarm",
                            ).start()
                    except Exception:
                        pass

                threading.Thread(target=_reap_detached, daemon=True,
                                 name="detached-grade-reaper").start()
            else:
                # Normal completion or a prelude failure (no runner was ever
                # spawned — _proc/_req_path/_claim_path are all None then).
                for _tmp in (_req_path, _prog_path):
                    if _tmp:
                        try: os.unlink(_tmp)
                        except Exception: pass
                # Release OUR pre-spawn claim on cache/grading.lock — but only if
                # the runner never took ownership (it overwrites the file with its
                # own pid, and its own finally removes that). Never destroy a lock
                # whose contents we cannot prove are ours.
                try:
                    if _claim_path is not None and \
                            _claim_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                        _claim_path.unlink(missing_ok=True)
                except Exception:
                    pass
                _grading_active.clear()   # resume background thumbnail prewarm
                if _gl is not None and _acquired:
                    _gl.release()

                # Post-grade prewarm (2026-09-10): the cull was the reason the
                # grid's thumbnails were refused; when it ends the first browse of
                # the folder pays full decode latency for every tile. Kick the
                # prewarm over the just-graded folders NOW — fire-and-forget, in
                # the existing low-priority pool, so it never competes with the
                # user's first on-demand requests (in-flight dedup collapses any
                # overlap).
                try:
                    if all_folders:
                        threading.Thread(
                            target=_prewarm_folder_paths,
                            args=(list(all_folders),),
                            daemon=True, name="post-grade-prewarm",
                        ).start()
                except Exception:
                    pass

    # SET here, not inside the generator — see the single-flight comment above.
    # Every early-return above (400, 503) has already been passed, so the flag
    # cannot outlive a refused request.
    _grading_active.set()

    # ── Cross-process claim BEFORE the runner exists ─────────────────────────
    # The grade_runner subprocess only writes grading.lock once its Python is
    # up (well after Popen returns). That gap is a TOCTOU window in the
    # cross-process 409 gate: a second server stack clicking grade inside it
    # saw no lock and spawned a second runner on the same request (the
    # 2026-09-07 duplicate-runner incident). Claiming the lock HERE — with
    # this server's pid — closes it: another stack sees a live foreign python
    # pid and 409s; this stack matches pid == os.getpid() in
    # grade_lock.grade_in_progress, same as its own _grading_active. The
    # runner then overwrites the file with its own pid (ownership transfer);
    # the generator's finally removes the claim only if it still names us.
    _claim_path = None
    try:
        from src.grade_lock import lock_path as _claim_lock_path
        _claim_path = _claim_lock_path(_DATA_DIR)
        _claim_path.parent.mkdir(parents=True, exist_ok=True)
        _claim_path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        _claim_path = None   # marker, not truth — never block a grade on it

    return StreamingResponse(_stream_with_lock(), media_type="text/event-stream")


@router.post("/api/regrade")
async def regrade_photos(req: GradeRequest):
    """
    Force a full re-grade: clears catalog.json, runs the full IQA pipeline
    (force_rescan=True), and rebuilds the catalog. SSE streaming, same format
    as /api/grade/v2/stream.
    """
    # Move the previous catalog aside instead of deleting it. If the re-grade
    # fails (RAM refusal, crash, power loss) /api/catalog falls back to this
    # backup, so hours of grades are never destroyed by one failed run.
    import catalog_store
    catalog_store.back_up("regrade", path=_CATALOG_PATH)
    return await grade_photos_v2_stream(req.model_copy(update={"force_rescan": True, "scan_mode": False}))


@router.post("/api/scan")
async def scan_photos(req: GradeRequest):
    """
    Low-latency scan: clears catalog.json, runs embedding + IQA without full
    SpecVLM verification (scan_mode=True), and rebuilds the catalog. SSE streaming,
    same format as /api/grade/v2/stream.

    Backs the catalog up first, exactly as /api/regrade does. This clears the
    catalog by the same force_rescan=True route, so a scan that fails — and a
    RAM refusal is the ordinary failure here, not an exotic one — used to
    destroy the whole library with no .pre-regrade.bak for /api/catalog to fall
    back to. Two endpoints, one destructive act, one backup helper.
    """
    import catalog_store
    catalog_store.back_up("scan", path=_CATALOG_PATH)
    return await grade_photos_v2_stream(req.model_copy(update={"force_rescan": True, "scan_mode": True}))


@router.post("/api/personal/update")
async def personal_update(payload: dict):
    """
    Update the PersonalHead MLP when the user moves a photo between grade buckets.

    Body: { path1, grade1, path2, grade2 }
    Fetches embeddings from LanceDB and runs a Margin Ranking Loss update step.
    """
    try:
        import numpy as np
        import personal_head as ph
        import lance_store   as ls

        path1, grade1 = payload["path1"], payload["grade1"]
        path2, grade2 = payload["path2"], payload["grade2"]

        rows = ls.query_by_paths([path1, path2])
        by_path = {r["path"]: r for r in rows}

        if path1 not in by_path or path2 not in by_path:
            return JSONResponse({"ok": False, "error": "paths not found in LanceDB"})

        emb1 = by_path[path1]["embedding"]
        emb2 = by_path[path2]["embedding"]
        loss = await run_in_threadpool(ph.update, emb1, grade1, emb2, grade2)

        # Refresh personal scores for all stored photos
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = await run_in_threadpool(ph.score, all_embs)
            ls.update_personal_scores({r["path"]: float(s) for r, s in zip(all_rows, new_pers)})

        # Queue DPO preference events for background soul-alignment training
        try:
            import background_dpo_trainer as _dpo
            # path1 moved from grade2 → grade1 means path1 now has grade1
            # Queue: what changed grade, old → new
            _dpo.get_trainer().queue_event(path1, grade2, grade1)
        except Exception:
            pass  # DPO is best-effort; never block the main update

        return JSONResponse({"ok": True, "loss": round(loss, 5)})
    except Exception as e:
        print(f"[personal/update] failed: {e}", flush=True)
        raise HTTPException(500, "Taste update failed — see the server log for details.")


@router.post("/api/personal/star")
async def personal_star(payload: dict):
    """
    Train PersonalHead from a star rating on a single photo.

    Stars map to grade labels:
        4-5 → Strong ✅   3 → Mid ⚠️   1-2 → Weak ❌   0 → skip

    A contrastive photo with a different grade is pulled from LanceDB to form
    a preference pair for MarginRankingLoss.  The DPO queue is also updated so
    BackgroundDPOTrainer can fire once 20 events accumulate.
    """
    import random as _random
    try:
        path  = str(payload.get("path", "")).strip()
        stars = int(payload.get("stars", 0))
        if not path or stars == 0:
            return JSONResponse({"ok": True, "skipped": True})

        # Persist to the durable ratings store FIRST — this is the taste
        # baseline and must survive re-culls/catalog rebuilds even if the
        # PersonalHead training below fails.
        try:
            import ratings_store as _rs
            _rs.set_rating(path, stars)
        except Exception as _e_rs:
            print(f"[star] durable ratings_store write failed: {_e_rs}")

        star_grade = "Strong ✅" if stars >= 4 else ("Mid ⚠️" if stars == 3 else "Weak ❌")
        _RANK = {"Strong ✅": 2, "Mid ⚠️": 1, "Weak ❌": 0}
        star_rank = _RANK[star_grade]

        import numpy as np
        import personal_head as ph
        import lance_store   as ls

        rows = ls.query_by_paths([path])
        if not rows:
            return JSONResponse({"ok": False, "error": "path not found in LanceDB"})

        this_row   = rows[0]
        this_emb   = this_row["embedding"]
        this_grade = this_row.get("grade") or "Mid ⚠️"

        # Snapshot the machine score onto the rating now that we have it, so
        # accuracy measurement survives a later re-grade/migration wiping this
        # exact path out of LanceDB/catalog. The bare stars write above must
        # stay first (and stay independent) — this is a best-effort enrichment.
        try:
            _rs.set_rating(path, stars, score=this_row.get("score"),
                            personal_score=this_row.get("personal_score"))
        except Exception as _e_snap:
            print(f"[star] score snapshot skipped: {_e_snap}")

        # MasterJudge auto-refit: fires in a daemon thread once enough NEW
        # ratings have banked since the last champion/challenger fit (see
        # src/master_judge.maybe_autofit). A lost challenge only rewrites the
        # record — grades are untouched — so this is always safe to fire.
        # Never blocks this request.
        try:
            import master_judge as _mj
            _mj.maybe_autofit()
        except Exception as _e_mj:
            print(f"[star] MasterJudge autofit check skipped: {_e_mj}")

        # Queue DPO event: auto grade → user star grade
        try:
            import background_dpo_trainer as _dpo
            _dpo.get_trainer().queue_event(path, this_grade, star_grade)
        except Exception as _e_dpo:
            print(f"[star] DPO queue skipped: {_e_dpo}")

        # PersonalHead pair update: find a contrastive photo.
        # Memory fix (2026-08-30): this used to run ls.query_all() — loading
        # all 64k rows WITH embeddings (~400 MB of boxed floats) on every
        # star — and then rewrote personal scores for the whole store, which
        # OOM'd Lance's external sort under memory pressure. Now: a light scan
        # picks a contrastive path, only that row's embedding is fetched, and
        # the personal-score refresh covers just the two affected photos. Full
        # propagation happens on the periodic retrain, which already rewrites
        # all scores.
        light_rows   = ls.query_light_all()
        contrastive  = [r for r in light_rows
                        if r.get("path") != path
                        and _RANK.get(r.get("grade") or "Mid ⚠️", 1) != star_rank]

        loss = 0.0
        if contrastive:
            other      = _random.choice(contrastive[:40])
            other_rows = ls.query_by_paths([other["path"]])
            other_emb  = other_rows[0]["embedding"] if other_rows else None

            if other_emb is not None:
                other_grade = other_rows[0].get("grade") or "Mid ⚠️"
                loss = await run_in_threadpool(
                    ph.update, this_emb, star_grade, other_emb, other_grade)

                # Refresh personal scores for ONLY the two affected photos —
                # the periodic retrain propagates the new head store-wide.
                new_scores = {}
                for emb_obj, pth in ((this_emb, path), (other_emb, other["path"])):
                    try:
                        new_scores[pth] = float(ph.score(
                            np.asarray([emb_obj], dtype=np.float32))[0])
                    except Exception:
                        pass
                if new_scores:
                    ls.update_personal_scores(new_scores)

        # Auto-retrain the whole baseline every _RETRAIN_EVERY new ratings —
        # incremental pair-updates drift toward recent ratings; a periodic full
        # fit on the durable store keeps the head representative as the baseline
        # grows toward hundreds. Fire-and-forget so it never blocks the rating.
        retrained = None
        try:
            global _ratings_since_retrain
            _ratings_since_retrain += 1
            if _ratings_since_retrain >= _RETRAIN_EVERY:
                # L5: the old code discarded the task reference (failures
                # surfaced only as "exception was never retrieved") and reset
                # the counter BEFORE the retrain ran, so a failed retrain
                # waited another 25 ratings to retry. The done-callback now
                # logs the outcome and restores the counter on failure.
                fired_at = _ratings_since_retrain
                _ratings_since_retrain = 0
                import asyncio as _aio

                def _on_retrain_done(task: "_aio.Future") -> None:
                    global _ratings_since_retrain
                    try:
                        if task.cancelled():
                            _ratings_since_retrain = fired_at
                            print("[star] auto-retrain cancelled — counter restored", flush=True)
                            return
                        exc = task.exception()
                        if exc is not None:
                            _ratings_since_retrain = fired_at
                            print(f"[star] auto-retrain FAILED — counter restored to "
                                  f"{fired_at}: {exc}", flush=True)
                        else:
                            print("[star] auto-retrain completed", flush=True)
                    except Exception as _e_cb:
                        print(f"[star] auto-retrain callback error: {_e_cb}", flush=True)

                _task = _aio.create_task(run_in_threadpool(_retrain_personal_baseline))
                _task.add_done_callback(_on_retrain_done)
                _AUTO_RETRAIN_TASKS.add(_task)
                _task.add_done_callback(_AUTO_RETRAIN_TASKS.discard)
                retrained = "scheduled"
        except Exception as _e_rt:
            print(f"[star] auto-retrain schedule skipped: {_e_rt}")

        return JSONResponse({"ok": True, "star_grade": star_grade,
                             "loss": round(loss, 5), "retrain": retrained})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/api/taste/summary")
async def taste_summary():
    """The user's taste-authority standing, for the loupe's taste meter.

    Mirrors the tier logic in grade_pipeline_v2.py Step 5 EXACTLY — the UI must
    never claim an authority level the blend doesn't actually use. The ceiling
    grows with the durable star-rating count:
        <25 ratings → 0.35   ≥25 → 0.45   ≥50 → 0.55   ≥100 → 0.70
    FIRSTCUT_PH_WEIGHT_MAX, when set, is the same hard cap the pipeline
    applies, so the reported weight can never overstate the blend.
    """
    try:
        import ratings_store as _rs
        n = len(_rs.load())
    except Exception:
        n = 0
    if   n >= 100: weight, next_at, next_weight = 0.70, None,  None
    elif n >= 50:  weight, next_at, next_weight = 0.55, 100, 0.70
    elif n >= 25:  weight, next_at, next_weight = 0.45, 50,  0.55
    else:          weight, next_at, next_weight = 0.35, 25,  0.45
    try:
        import os as _os
        cap = _os.environ.get("FIRSTCUT_PH_WEIGHT_MAX", "").strip()
        if cap:
            weight = min(max(float(cap), 0.20), 0.80)
            next_at = next_weight = None      # cap overrides the ladder
    except Exception:
        pass
    return JSONResponse({"ratings": n, "weight": weight,
                         "next_at": next_at, "next_weight": next_weight})


# Full-baseline retrain plumbing — gathers every durable rating + its embedding
# and fits the PersonalHead from scratch (stable for 100s of ratings).
_RETRAIN_EVERY = 25
_ratings_since_retrain = 0
# L5: strong references to in-flight auto-retrain tasks, so a running retrain
# can never be garbage-collected mid-flight.
_AUTO_RETRAIN_TASKS: set = set()


def _gather_rating_samples() -> list:
    import numpy as np
    import ratings_store as _rs, lance_store as _ls
    ratings = _rs.load()
    if not ratings:
        return []
    rows = {r["path"]: r for r in _ls.query_all(min_score=0.0)}
    if not rows:
        # The embedding store can be wiped by an encoder-tier switch or a
        # re-grade. The catalog keeps the CURRENT grades on identical path
        # keys, so taste learning degrades to score-derived samples instead of
        # going dark: a photographer whose store rotated still trains the head.
        import json as _json
        from server_impl import _DATA_DIR as _dd
        cat = _dd / "cache" / "catalog.json"
        if cat.exists():
            try:
                data = _json.loads(cat.read_text(encoding="utf-8"))
                rows = {p["path"]: {"score": p.get("score")}
                        for p in data.get("photos", [])
                        if isinstance(p.get("score"), (int, float))}
            except Exception:
                pass
    _g = lambda s: "Strong ✅" if s >= 4 else ("Mid ⚠️" if s == 3 else "Weak ❌")
    out = []
    for p, s in ratings.items():
        r = rows.get(p)
        if r is None:
            continue
        src = _rs.get_source(p)
        emb = r.get("embedding")
        if emb is not None:
            out.append((np.asarray(emb, dtype=np.float32), _g(int(s)), src))
        elif isinstance(r.get("score"), (int, float)):
            # No embedding available — synthesize a degenerate 1-D sample so
            # PersonalHead at least sees the score↔star relationship.
            out.append((np.asarray([float(r["score"])], dtype=np.float32), _g(int(s)), src))
    return out


def _retrain_personal_baseline() -> dict:
    import numpy as np
    import personal_head as ph
    samples = _gather_rating_samples()
    if not samples:
        return {"n": 0}
    stats = ph.fit(samples)
    try:
        import lance_store as _ls
        rows = _ls.query_all()
        if rows:
            embs = np.stack([r["embedding"] for r in rows])
            pers = ph.score(embs)
            _ls.update_personal_scores({r["path"]: float(s) for r, s in zip(rows, pers)})
    except Exception:
        pass
    print(f"[personal] baseline retrained: {stats}")
    return stats


@router.post("/api/personal/retrain")
async def personal_retrain(payload: dict = None):
    """Manually retrain the PersonalHead on the full durable rating baseline."""
    stats = await run_in_threadpool(_retrain_personal_baseline)
    return JSONResponse({"ok": True, **stats})


@router.post("/api/master/retrain")
async def master_retrain(payload: dict = None):
    """Refit the MasterJudge head on the master star baseline and run the
    champion/challenger challenge. CPU-only (~10-20 s); returns the verdict:
    promoted true/false, held-out rho vs the incumbent's, and the fit history.
    A lost challenge rewrites the record only — grade output is untouched."""
    import master_judge as mj
    stats = await run_in_threadpool(mj.fit)
    return JSONResponse({"ok": True, **stats})


@router.post("/api/update_preference")
async def update_preference(payload: dict):
    """
    Update preference by providing a winner and a loser image path.
    Body: { "winner_path": str, "loser_path": str }
    Runs a MarginRankingLoss update on the PersonalHead and refreshes stored scores.
    """
    try:
        import numpy as np
        import personal_head as ph
        import lance_store as ls
        winner = payload.get("winner_path")
        loser = payload.get("loser_path")
        if not winner or not loser:
            return JSONResponse({"ok": False, "error": "winner_path and loser_path required"})

        rows = ls.query_by_paths([winner, loser])
        by_path = {r["path"]: r for r in rows}
        if winner not in by_path or loser not in by_path:
            return JSONResponse({"ok": False, "error": "paths not found in LanceDB"})

        emb_w = by_path[winner]["embedding"]
        emb_l = by_path[loser]["embedding"]

        loss = await run_in_threadpool(ph.update, emb_w, 1, emb_l, 0)

        # Refresh personal scores for all stored photos (LanceDB)
        all_rows = ls.query_all()
        if all_rows:
            all_embs = np.stack([r["embedding"] for r in all_rows])
            new_pers = await run_in_threadpool(ph.score, all_embs)
            ls.update_personal_scores({r["path"]: float(s) for r, s in zip(all_rows, new_pers)})

        return JSONResponse({"ok": True, "loss": round(loss, 5)})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/api/manage/sort-files")
async def sort_files(payload: dict):
    """
    Move graded photos into Strong / Mid / Weak subdirectories.
    Body: { folder_path, gallery: [...], copy: bool }
    """
    try:
        from grade_pipeline_v2 import sort_files as _sort
        from server_impl import _safe_dir_path
        import os as _os
        folder = _safe_dir_path(payload["folder_path"])

        # Only sort files that actually live inside the requested folder.
        # Stale UI state can carry paths from other folders or files deleted
        # since the grade; moving those is unrecoverable.
        norm_folder = _os.path.normcase(str(folder))

        def _in_folder(p: str) -> bool:
            try:
                return _os.path.commonpath([norm_folder, _os.path.normcase(p)]) == norm_folder
            except ValueError:
                return False

        raw_gallery = payload.get("gallery") or []
        gallery = [
            g for g in raw_gallery
            if isinstance(g, dict) and _in_folder(str(g.get("path", "")))
        ]
        skipped = len(raw_gallery) - len(gallery)

        result = _sort(
            str(folder),
            gallery,
            copy=bool(payload.get("copy", False)),
        )
        if isinstance(result, dict) and skipped > 0:
            result["skipped_outside_folder"] = skipped
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        # Full detail goes to the log; the client gets an actionable message
        # instead of raw Python exception text.
        print(f"[sort-files] failed: {e}", flush=True)
        raise HTTPException(500, "Could not sort the files — see the server log for details.")



