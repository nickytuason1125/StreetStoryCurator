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
    Path, _BG_EXECUTOR, _DATA_DIR, _GRADE_MIN_RAM_GB, _grading_active, _release_annotation_model, _trim_crash_log, analyzer, annotation_queue, asyncio, gpu_lock, os, sys,
)
import json
from routers.library import GradeRequest, _precompute_clusters, _run_vlm_deep_review

router = APIRouter()


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


@router.post("/api/grade")
async def grade_photos(req: GradeRequest):
    import asyncio
    global GLOBAL_CLUSTER_CACHE
    if not os.path.isdir(req.folder_path):
        raise HTTPException(400, "Invalid folder path")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: analyzer.analyze_folder(req.folder_path, preset=req.preset, force_rescan=True),
        )

        # Load any VLM critique results already on disk from a prior deep_review run.
        from pathlib import Path as _Path
        import json as _json
        _grade_cache_path = _DATA_DIR / "cache" / "vlm_rationale_cache.json"
        _vlm_grades: dict = (
            _json.loads(_grade_cache_path.read_text(encoding="utf-8"))
            if _grade_cache_path.exists() else {}
        )

        gallery = [{
            "path":        r[0],
            "grade":       r[1]["grade"],
            "score":       r[1]["score"],
            "critique":    r[1]["critique"],
            "breakdown":   r[1]["breakdown"],
            "nima_score":  r[1].get("nima_score"),
            "sim_flag":    r[1].get("sim_flag",   ""),
            "cluster_id":  r[1].get("cluster_id", -1),
            "faces":       r[1].get("faces", 0),
            "rationale": _vlm_grades.get(r[0]),   # None until background task completes
        } for r in results]
        strong = sum(1 for g in gallery if "Strong" in g["grade"])
        mid    = sum(1 for g in gallery if "Mid"    in g["grade"])
        weak   = sum(1 for g in gallery if "Weak"   in g["grade"])

        # Invalidate stale cache for a new folder, then kick off background tasks.
        GLOBAL_CLUSTER_CACHE = {}
        _BG_EXECUTOR.submit(_precompute_clusters, req.folder_path, results)
        if req.deep_review:
            _BG_EXECUTOR.submit(_run_vlm_deep_review, results)

        return JSONResponse({"status": "success", "total": len(gallery),
                             "strong": strong, "mid": mid, "weak": weak,
                             "data": gallery})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    # _grading_active is only SET inside _stream_with_lock() once the SSE body
    # starts iterating, so without this check a reload mid-grade, a second
    # window, or a double-fired modal each spawn their own grade_runner
    # subprocess — competing for gpu_lock, VRAM, and the RAM floor.
    if _grading_active.is_set():
        raise HTTPException(
            409,
            "A grade is already running. Wait for it to finish before starting another.",
        )

    # ── System RAM gate ─────────────────────────────────────────────────────
    # Lowered 3.0 → 1.8 GB: the HF SigLIP loader commits only ~1 GB (was ~9.5 GB
    # under open_clip), so culls run at far lower free RAM now. Qwen grading has
    # its own preflight that falls back to CLIP scoring if RAM is tight, so a low
    # but non-zero RAM cull still completes (in degraded mode) rather than 503.
    try:
        import psutil as _psutil
        _free_gb = _psutil.virtual_memory().available / 1e9
        if _free_gb < _GRADE_MIN_RAM_GB:
            return JSONResponse(
                status_code=503,
                content={"error": f"Not enough RAM to grade safely — only {_free_gb:.1f} GB free, need at least ~2 GB. "
                         "Close a couple of apps and retry."},
            )
    except Exception:
        pass

    # NOTE: there was an Ollama health gate here that 503'd every non-scan grade
    # when http://localhost:11434 did not answer. Nothing installed Ollama — not
    # Setup.ps1, not requirements.txt — and no user-facing doc mentioned it, while
    # CLAUDE.md rule 5 promised a fully offline app. So a correct, complete install
    # could not grade a single photo. Grading never needed it either: the default
    # path is SigLIP zero-shot plus TOPIQ, and the modules that did call Ollama now
    # run the same models locally through llama_cpp. Removed, not made optional.

    # Resolve all valid folders — folder_paths (multi) takes priority over folder_path
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders:
        if req.folder_path and os.path.isdir(req.folder_path):
            all_folders = [str(Path(req.folder_path).resolve())]
        else:
            raise HTTPException(400, "No valid folder path provided")

    async def _stream_with_lock():
        # Hold gpu_lock for the full grading run so the annotation daemon
        # cannot load its GGUF concurrently and bust the 5.5 GB VRAM ceiling.
        _gl = gpu_lock
        if _gl is not None:
            await _gl.acquire()
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
        _grading_active.set()
        try:
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
                    "scan_mode":    req.scan_mode,
                    "deep_grade":   req.deep_grade,
                    "catalog_path": str(_CATALOG_PATH),
                    "data_dir":     str(_DATA_DIR),
                    "mogco_target": 0,   # cull only; Story sequencing is its own endpoint
                }, _rf)

            _runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grade_runner.py")
            _flags = 0x08000000 if os.name == "nt" else 0   # CREATE_NO_WINDOW
            _renv = dict(os.environ); _renv["PYTHONIOENCODING"] = "utf-8"
            # Route the runner's [v2] progress prints to crash.log (utf-8) for
            # debuggability — its result stream goes via the progress file above.
            _crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
            _trim_crash_log(_crash_path)
            _rlog = open(_crash_path, "a", encoding="utf-8", errors="replace")
            # ── Hard RAM gate: fail fast BEFORE spawning into certain paging death.
            # The pipeline warns at 2.5 GB; below ~0.75 GB free a cull measurably
            # drove the machine to 0.01 GB free + 1.2 GB pagefile growth (see
            # grade_pipeline_v2 notes). At that point refusing is the honest,
            # crash-proof behaviour — the UI already advertises "the cull may
            # be refused" below its floor, and Resume recovers nothing lost.
            try:
                import psutil as _ps_gate
                _free_gb = _ps_gate.virtual_memory().available / 1e9
                if _free_gb < 0.75:
                    import json as _sj
                    yield f"data: {_json.dumps({'error': f'Refused: only {_free_gb:.1f} GB RAM free and a cull needs ~2.5 GB — running it would freeze this machine (measured). Close a few apps and retry.'})}\n\n"
                    print(f"[server] Grade REFUSED pre-spawn: {_free_gb:.2f} GB free", flush=True)
                    return
            except Exception:
                pass

            import win_job as _wj
            _proc = _wj.popen(
                [sys.executable, _runner, _req_path, _prog_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                creationflags=_flags, close_fds=True,
                stdin=_sp.DEVNULL, stdout=_rlog, stderr=_rlog, env=_renv,
            )
            try: _rlog.close()   # child keeps its inherited fd
            except Exception: pass
            print(f"[server] Grade runner subprocess pid={_proc.pid}", flush=True)

            _loop = asyncio.get_running_loop()
            _pos = 0
            _done = False

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
                    # done/error result it crashed: surface the recoverable checkpoint.
                    _tailchunk = await _loop.run_in_executor(None, _read_new)
                    for _ln in _tailchunk.splitlines():
                        _out = _emit(_ln)
                        if _out is not None:
                            yield f"data: {_out}\n\n"
                    if not _done:
                        print(f"[server] Grade runner exited without result: code={_proc.returncode}", flush=True)
                        _recovered = 0
                        try:
                            if _CATALOG_PATH.exists():
                                _recovered = len(_json.loads(_CATALOG_PATH.read_text(encoding="utf-8")).get("photos", []))
                        except Exception:
                            pass
                        yield f"data: {_json.dumps({'error': f'Grade process exited unexpectedly (code {_proc.returncode}) — {_recovered} grades were checkpointed and can be recovered. Check crash.log', 'recovered': _recovered})}\n\n"
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        finally:
            # This finally runs on normal completion AND on client disconnect
            # (GeneratorExit at a yield). Previously the subprocess kept running
            # after a disconnect while gpu_lock was released here — so a second
            # grade could start concurrently and bust the VRAM ceiling. Terminate
            # the runner if it's still alive, and always clean the temp files.
            try:
                if _proc.poll() is None:
                    _proc.terminate()
                    try:
                        _proc.wait(timeout=5)
                    except Exception:
                        _proc.kill()
                    print(f"[server] Grade runner pid={_proc.pid} terminated on stream close", flush=True)
            except Exception:
                pass
            for _tmp in (_req_path, _prog_path):
                try: os.unlink(_tmp)
                except Exception: pass
            _grading_active.clear()   # resume background thumbnail prewarm
            if _gl is not None:
                _gl.release()

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
    # backup below, so hours of grades are never destroyed by one failed run.
    _CATALOG_BAK = _CATALOG_PATH.with_name("catalog.json.pre-regrade.bak")
    try:
        if _CATALOG_PATH.exists():
            os.replace(str(_CATALOG_PATH), str(_CATALOG_BAK))
            print("[regrade] Previous catalog moved to catalog.json.pre-regrade.bak before re-grade")
    except Exception as _e:
        print(f"[regrade] catalog backup warning: {_e}")
    return await grade_photos_v2_stream(req.model_copy(update={"force_rescan": True, "scan_mode": False}))


@router.post("/api/scan")
async def scan_photos(req: GradeRequest):
    """
    Low-latency scan: clears catalog.json, runs embedding + IQA without full
    SpecVLM verification (scan_mode=True), and rebuilds the catalog. SSE streaming,
    same format as /api/grade/v2/stream.
    """
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

        # Queue DPO event: auto grade → user star grade
        try:
            import background_dpo_trainer as _dpo
            _dpo.get_trainer().queue_event(path, this_grade, star_grade)
        except Exception as _e_dpo:
            print(f"[star] DPO queue skipped: {_e_dpo}")

        # PersonalHead pair update: find a contrastive photo
        all_rows    = ls.query_all()
        contrastive = [r for r in all_rows
                       if r["path"] != path
                       and _RANK.get(r.get("grade") or "Mid ⚠️", 1) != star_rank]

        loss = 0.0
        if contrastive:
            other      = _random.choice(contrastive[:40])
            other_emb  = other["embedding"]
            other_grade = other.get("grade") or "Mid ⚠️"
            loss = await run_in_threadpool(ph.update, this_emb, star_grade, other_emb, other_grade)

            # Refresh personal scores across all stored photos
            if all_rows:
                all_embs = np.stack([r["embedding"] for r in all_rows])
                new_pers = await run_in_threadpool(ph.score, all_embs)
                ls.update_personal_scores(
                    {r["path"]: float(s) for r, s in zip(all_rows, new_pers)}
                )

        # Auto-retrain the whole baseline every _RETRAIN_EVERY new ratings —
        # incremental pair-updates drift toward recent ratings; a periodic full
        # fit on the durable store keeps the head representative as the baseline
        # grows toward hundreds. Fire-and-forget so it never blocks the rating.
        retrained = None
        try:
            global _ratings_since_retrain
            _ratings_since_retrain += 1
            if _ratings_since_retrain >= _RETRAIN_EVERY:
                _ratings_since_retrain = 0
                import asyncio as _aio
                _aio.create_task(run_in_threadpool(_retrain_personal_baseline))
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
    FRAMEGRADE_PH_WEIGHT_MAX, when set, is the same hard cap the pipeline
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
        cap = _os.environ.get("FRAMEGRADE_PH_WEIGHT_MAX", "").strip()
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
        emb = r.get("embedding")
        if emb is not None:
            out.append((np.asarray(emb, dtype=np.float32), _g(int(s))))
        elif isinstance(r.get("score"), (int, float)):
            # No embedding available — synthesize a degenerate 1-D sample so
            # PersonalHead at least sees the score↔star relationship.
            out.append((np.asarray([float(r["score"])], dtype=np.float32), _g(int(s))))
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


@router.post("/api/grade/stream")
async def grade_photos_stream(req: GradeRequest):
    """Streams grading progress as SSE, then emits the full result as the final event."""
    import asyncio, json
    from fastapi.responses import StreamingResponse

    global GLOBAL_CLUSTER_CACHE

    # Resolve which folders to grade — multi-folder takes priority
    all_folders = [str(Path(fp).resolve()) for fp in req.folder_paths if os.path.isdir(fp)]
    if not all_folders:
        if req.folder_path and os.path.isdir(req.folder_path):
            all_folders = [req.folder_path]
        else:
            raise HTTPException(400, "No valid folder path provided")

    loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()

    def _progress(frac: float, desc: str = "") -> None:
        loop.call_soon_threadsafe(
            aqueue.put_nowait, {"progress": round(frac, 3), "desc": desc}
        )

    async def _run() -> None:
        global GLOBAL_CLUSTER_CACHE
        try:
            # Grade each folder; combine all results
            n = len(all_folders)
            combined: list = []
            for i, fp in enumerate(all_folders):
                def _folder_progress(frac: float, desc: str = "", _i=i, _n=n) -> None:
                    _progress((_i + frac) / _n, desc)
                folder_results = await loop.run_in_executor(
                    None,
                    lambda _fp=fp: analyzer.analyze_folder(
                        _fp, preset=req.preset,
                        force_rescan=True, progress=_folder_progress,
                    ),
                )
                combined.extend(folder_results)
            results = combined

            from pathlib import Path as _Path
            import json as _json
            _grade_cache_path = _DATA_DIR / "cache" / "vlm_rationale_cache.json"
            _vlm_grades: dict = (
                _json.loads(_grade_cache_path.read_text(encoding="utf-8"))
                if _grade_cache_path.exists() else {}
            )
            gallery = [{
                "path":        r[0],
                "grade":       r[1]["grade"],
                "score":       r[1]["score"],
                "critique":    r[1]["critique"],
                "breakdown":   r[1]["breakdown"],
                "nima_score":  r[1].get("nima_score"),
                "sim_flag":    r[1].get("sim_flag", ""),
                "cluster_id":  r[1].get("cluster_id", -1),
                "faces":       r[1].get("faces", 0),
                "rationale": _vlm_grades.get(r[0]),
            } for r in results]
            strong = sum(1 for g in gallery if "Strong" in g["grade"])
            mid    = sum(1 for g in gallery if "Mid"    in g["grade"])
            weak   = sum(1 for g in gallery if "Weak"   in g["grade"])
            GLOBAL_CLUSTER_CACHE = {}
            _BG_EXECUTOR.submit(_precompute_clusters, all_folders[0], results)
            if req.deep_review:
                _BG_EXECUTOR.submit(_run_vlm_deep_review, results)

            # Run MOGCO beam search now that DuckDB is populated.
            # Enrich paths with gallery metadata so the frontend can render directly.
            mogco_sequence: list = []
            try:
                from mogco_sequencer import run_mogco_sequence
                beam = await loop.run_in_executor(None, run_mogco_sequence)
                if beam.get("paths"):
                    info_by_path = {g["path"]: g for g in gallery}
                    for path, slot, obj in zip(
                        beam["paths"], beam["slots"], beam["beam_objectives"]
                    ):
                        frame = dict(info_by_path.get(path, {"path": path}))
                        frame["slot"]             = slot
                        frame["mogco_objectives"] = obj
                        frame["engine"]           = "mogco-beam"
                        mogco_sequence.append(frame)
            except Exception:
                pass  # MOGCO failure never blocks grading result

            await aqueue.put({
                "done": True, "total": len(gallery),
                "strong": strong, "mid": mid, "weak": weak, "data": gallery,
                "mogco_sequence": mogco_sequence,
            })
        except Exception as exc:
            await aqueue.put({"error": str(exc)})

    asyncio.create_task(_run())

    async def _generate():
        while True:
            try:
                msg = await asyncio.wait_for(aqueue.get(), timeout=300)
            except asyncio.TimeoutError:
                yield "data: {\"ping\":true}\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("done") or msg.get("error"):
                break

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


