"""Extras routes — moved verbatim from server_impl.py (Milestone 4 split).

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
    Path, _DATA_DIR, _EXE_DIR, _HEIC_EXTS, _IMAGE_EXTS, _gen_preview, _safe_image_path, analyzer, asyncio, get_analyzer, os, sys,
)
from routers.export import _folder_watcher, _watched_folder
from routers.library import _RAW_EXTS

router = APIRouter()

# The unit root — the directory holding src/, scripts/ and static/. NOT this
# module's directory. See the identical note in routers/grading.py: the server
# split moved this code from the repo root into routers/, and every
# Path(__file__).parent silently began resolving one level too deep.
_UNIT_ROOT = Path(__file__).resolve().parent.parent


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.post("/api/watch/start")
async def watch_start(payload: dict):
    """
    Start watching a folder for new/modified images.
    New arrivals are auto-added to the vector DB (non-blocking background task).
    payload: { folder: str }
    """
    from engine_utils import FolderWatcher, LocalVectorDB

    global _folder_watcher, _watched_folder
    folder = payload.get("folder", "").strip()
    preset = payload.get("preset", "Classic Street")
    if not folder or not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")

    if _folder_watcher is not None:
        _folder_watcher.stop()

    vec_db = LocalVectorDB()
    _folder_watcher = FolderWatcher(analyzer=get_analyzer(), db=vec_db, preset=preset)
    _folder_watcher.watch(folder)
    _watched_folder = folder
    return {"status": "watching", "folder": folder}


@router.post("/api/watch/stop")
async def watch_stop():
    global _folder_watcher, _watched_folder
    if _folder_watcher:
        _folder_watcher.stop()
        _folder_watcher = None
    _watched_folder = ""
    return {"status": "stopped"}


@router.get("/api/watch/status")
async def watch_status():
    return {"watching": _folder_watcher is not None, "folder": _watched_folder}


# ---------------------------------------------------------------------------
# Vector DB search
# ---------------------------------------------------------------------------

@router.post("/api/search/similar")
async def search_similar(payload: dict):
    """
    Find visually similar images in the vector DB for a given source path.
    payload: { path: str, limit: int = 20 }
    """
    from engine_utils import LocalVectorDB
    path  = payload.get("path", "")
    limit = int(payload.get("limit", 20))
    emb   = analyzer.cache.get(path, {}).get("embedding")
    if emb is None:
        raise HTTPException(404, f"No embedding cached for {path!r} — grade it first")
    vec_db = LocalVectorDB()
    results = vec_db.search(emb, limit=limit)
    return JSONResponse({"query": path, "results": results})


# ---------------------------------------------------------------------------
# Reference bank (exemplar indexing)
# ---------------------------------------------------------------------------

@router.get("/api/exemplar-count")
def exemplar_count():
    return {"count": analyzer._ref_bank.count}


@router.get("/api/nima-status")
def nima_status():
    """Returns whether the NIMA ONNX has been generated and is loaded."""
    from pathlib import Path as _P
    onnx_exists = _P("models/onnx/nima.onnx").exists()
    session_loaded = (
        analyzer._ort_sessions is not None and
        "nima" in analyzer._ort_sessions
    )
    return {"available": session_loaded, "onnx_exists": onnx_exists}


async def _warm_and_run(fn):
    """Ensure ONNX is loaded, then run fn() in a thread executor."""
    import asyncio
    if analyzer._ort_sessions is None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, analyzer._ensure_sessions)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


@router.post("/api/index-exemplars")
async def index_exemplars(payload: dict):
    """Replace the entire bank with embeddings from a folder."""
    folder = payload.get("folder_path", "")
    if not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")
    count = await _warm_and_run(
        lambda: analyzer._ref_bank.build(
            folder,
            analyzer._ort_sessions["composition"],
            analyzer._comp_input,
        )
    )
    return {"status": "indexed", "count": count}


@router.post("/api/add-exemplars")
async def add_exemplars(payload: dict):
    """Append new exemplars from a folder to the existing bank (deduped)."""
    folder = payload.get("folder_path", "")
    if not os.path.isdir(folder):
        raise HTTPException(400, "Invalid folder path")
    added, skipped = await _warm_and_run(
        lambda: analyzer._ref_bank.add(
            folder,
            analyzer._ort_sessions["composition"],
            analyzer._comp_input,
        )
    )
    return {"status": "added", "added": added, "skipped": skipped,
            "total": analyzer._ref_bank.count}


@router.post("/api/clear-exemplars")
def clear_exemplars():
    """Remove all exemplars from the bank."""
    analyzer._ref_bank.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


# Mounted here so migrated routes register BEFORE the SPA catch-all below.



# ---------------------------------------------------------------------------
# Serve React frontend (catch-all — must be last)
# ---------------------------------------------------------------------------

DIST = _EXE_DIR / "frontend" / "dist"

import json




@router.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Ingest a local image file for use as a Creative Director peg.

    Saves the file to data/ingestion_queue/, hashes it with MD5,
    and inserts a stub LanceDB record so vector_search can locate it.
    Returns {"status": "success", "hash": file_hash, "path": saved_path}.
    """
    import hashlib as _hl

    # Bounded read — cap memory so an oversized upload can't OOM the process on a
    # RAM-tight machine (was an unbounded await file.read()).
    _MAX_IMG_BYTES = 64 * 1024 * 1024   # 64 MB
    _chunks, _size = [], 0
    while True:
        _c = await file.read(1024 * 1024)
        if not _c:
            break
        _size += len(_c)
        if _size > _MAX_IMG_BYTES:
            raise HTTPException(413, "Image too large (max 64 MB).")
        _chunks.append(_c)
    data      = b"".join(_chunks)
    file_hash = _hl.md5(data).hexdigest()

    queue_dir = _DATA_DIR / "data" / "ingestion_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in _IMAGE_EXTS:
        raise HTTPException(400, "Unsupported image format")
    dest = queue_dir / f"{file_hash}{suffix}"
    dest.write_bytes(data)

    # Phase 1 — VLM Pixel Inspector (Ollama qwen2.5vl:3b)
    # Runs before LanceDB insertion so the semantic profile is stored in breakdown.
    semantic_profile = ""
    try:
        import asyncio as _aio
        from fast_ingestion import run_pixel_inspector
        loop = _aio.get_running_loop()
        semantic_profile = await loop.run_in_executor(
            None, run_pixel_inspector, str(dest), 0.0
        )
    except Exception as _e_pi:
        print(f"[upload] pixel_inspector skipped: {_e_pi}")

    # Insert stub LanceDB record — zero-vector placeholder so peg lookup resolves.
    # semantic_profile is embedded in breakdown JSON so downstream Art Director can use it.
    import json as _json_mod
    breakdown_stub = _json_mod.dumps({"semantic_profile": semantic_profile})
    try:
        import lance_store as _ls
        _ls.upsert_batch([{
            "path":           str(dest),
            "embedding":      [0.0] * 1536,
            "score":          0.0,
            "personal_score": 0.0,
            "grade":          "Pending",
            "reasoning_log":  semantic_profile,
            "breakdown":      breakdown_stub,
            "exif_ts":        0.0,
        }])
        print(f"[upload] stub record inserted: {dest.name}  hash={file_hash}")
    except Exception as _e_ldb:
        print(f"[upload] LanceDB stub insert failed: {_e_ldb}")

    # TODO: Trigger full grading pipeline here

    return JSONResponse({"status": "success", "hash": file_hash, "path": str(dest)})


@router.get("/api/heatmap/technical/{image_hash:path}")
async def heatmap_technical(image_hash: str):
    """Return a Base64-encoded RGBA PNG blur heatmap for the given image path."""
    import asyncio
    from pipeline.heatmaps import generate_technical_heatmap

    p = _safe_image_path(image_hash)
    readable_path = str(p)

    if p.suffix.lower() in (_RAW_EXTS | _HEIC_EXTS):
        preview = _gen_preview(str(p))
        if preview:
            readable_path = str(preview)
        else:
            raise HTTPException(422, "Could not decode RAW/HEIC for heatmap")

    try:
        loop = asyncio.get_running_loop()
        b64 = await loop.run_in_executor(None, generate_technical_heatmap, readable_path)
        return JSONResponse({"b64": b64})
    except Exception as exc:
        raise HTTPException(500, f"Heatmap generation failed: {exc}")


@router.get("/api/search/semantic")
async def search_semantic(q: str = Query(default="", description="Free-text photo description")):
    """
    Natural-language semantic search over the LanceDB embedding index.

    Uses SigLIP-2's text tower to encode the query and returns the top-20
    most visually similar photos.  Reuses the grading singleton if loaded
    (no extra VRAM); falls back to a CPU encoder otherwise.
    """
    import asyncio as _aio

    if not q.strip():
        raise HTTPException(400, "Query parameter 'q' is required and must be non-empty")

    try:
        from creative_director import semantic_search as _ss
        loop    = _aio.get_running_loop()
        results = await loop.run_in_executor(None, lambda: _ss(query=q, limit=20))
        return JSONResponse({"results": results, "query": q})
    except Exception as exc:
        raise HTTPException(500, f"Semantic search failed: {exc}")


@router.post("/api/critique/details")
async def critique_details(req: Request):
    """
    On-demand deep text critique for a single selected photo.

    Sends the image to Qwen2.5-VL via Ollama using GENERATE_DEEP_TEXT_PROMPT —
    text only, no scores, no bounding boxes.  RAG context from
    cache/rag_concepts.json is injected automatically.

    Body: { "image_path": str, "mode": "story" | "competition" }
    Returns: { "narrative_arc": str, "geometry_composition": str }
    """
    data = await req.json()
    raw_path = data.get("image_path", "").strip()
    mode     = data.get("mode", "story")

    if not raw_path:
        raise HTTPException(400, "image_path is required")

    try:
        img_path = str(_safe_image_path(raw_path))
    except HTTPException:
        raise
    except Exception as _e:
        raise HTTPException(400, f"Invalid image path: {_e}")

    from qwen_vlm_grader import execute_vlm_text_deep_dive
    result = await run_in_threadpool(execute_vlm_text_deep_dive, img_path, mode)

    if result is None:
        raise HTTPException(503, "The critique model is not installed. Run the model "
                                 "downloader to fetch it — grading is unaffected.")

    return JSONResponse(result)


@router.get("/api/critique/jury/{image_hash:path}")
async def jury_critique(image_hash: str):
    """
    Generate a 3-paragraph DeepSeek-R1:8b jury critique for a single image.

    Runs the GGUF inference in an isolated subprocess so a C-level crash or
    OOM in llama-cpp-python kills only the child, not this server process.

    image_hash : MD5 stem used as the LanceDB path identifier.
    Returns {"critique": str, "think": str}.
    """
    import sys as _sys
    import asyncio as _aio
    import json as _json
    import subprocess as _sp

    if not image_hash.strip():
        raise HTTPException(400, "image_hash is required")

    # Inline script executed in the child process.
    # Uses sys.path.insert so it can import from src/ without package install.
    _critique_script = r"""
import sys, json
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else 'src')
image_hash = sys.argv[1]
try:
    from critique_engine import run_jury_critique
    result = run_jury_critique(image_hash=image_hash)
    print(json.dumps(result), flush=True)
except Exception as _e:
    import traceback
    print(json.dumps({
        "error":   f"{type(_e).__name__}: {_e}",
        "critique": "",
        "think":   traceback.format_exc(),
    }), flush=True)
"""

    try:
        # Use pythonw.exe on Windows — windowless Python never flashes a console.
        _py = _sys.executable
        if os.name == "nt" and _py.lower().endswith("python.exe"):
            _pyw = _py[:-10] + "pythonw.exe"
            if os.path.exists(_pyw):
                _py = _pyw
        _cflags = _sp.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = await _aio.create_subprocess_exec(
            _py, "-c", _critique_script, image_hash,
            stdout=_aio.subprocess.PIPE,
            stderr=_aio.subprocess.PIPE,
            cwd=str(_UNIT_ROOT),
            creationflags=_cflags,
        )

        try:
            # 90 s was not enough for the FIRST call: the text GGUF is 1.9 GB and
            # a cold load off disk can eat most of that budget before a single
            # token is generated, so the very first critique a user ever asked
            # for reliably reported a timeout on a model that was working. The
            # ceiling only needs to cover a cold load once — every later call
            # hits a warm model and returns in seconds.
            stdout_b, stderr_b = await _aio.wait_for(proc.communicate(), timeout=300)
        except _aio.TimeoutError:
            # Kill and reap the child so it doesn't linger as a zombie
            try:
                proc.kill()
                await _aio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            print("[jury] subprocess timed out after 300 s — killed and reaped")
            return JSONResponse(
                {"error": "Critique timed out (5 min). The writing model may still be loading on first use — try again.", "critique": "", "think": ""},
                status_code=504,
            )

        stderr_txt = stderr_b.decode(errors="replace").strip()
        if stderr_txt:
            # Forward child stderr to server log for debugging (truncated)
            print(f"[jury subprocess] {stderr_txt[:800]}")

        stdout_txt = stdout_b.decode(errors="replace").strip()
        if not stdout_txt:
            code = proc.returncode
            print(f"[jury] subprocess exited {code} with no stdout")
            return JSONResponse(
                {"error": f"Critique engine exited ({code}) with no output.", "critique": "", "think": stderr_txt[:400]},
                status_code=500,
            )

        # The child may emit debug prints before the final JSON line — find the last one.
        result: dict = {}
        for _line in reversed(stdout_txt.splitlines()):
            _line = _line.strip()
            if _line.startswith("{"):
                try:
                    result = _json.loads(_line)
                    break
                except _json.JSONDecodeError:
                    continue

        if not result:
            return JSONResponse(
                {"error": f"Could not parse critique output: {stdout_txt[:200]}", "critique": "", "think": ""},
                status_code=500,
            )

        if result.get("error") and not result.get("critique"):
            print(f"[jury] critique error from child: {result['error'][:200]}")
            return JSONResponse({"error": result["error"], "critique": "", "think": result.get("think", "")})

        return JSONResponse(result)

    except Exception as exc:
        import traceback as _tb
        _tb.print_exc()
        return JSONResponse(
            {"error": f"Server error: {type(exc).__name__}: {exc}", "critique": "", "think": ""},
            status_code=500,
        )


class ModelPullRequest(BaseModel):
    model_name: str

@router.post("/api/models/pull")
async def pull_model_stream(req: ModelPullRequest):
    """Stream the model downloader's progress as ndjson.

    This used to proxy Ollama's /api/pull. Ollama is gone, so it proxied a port
    nothing was listening on. It now runs scripts/fetch_models.py, which asks
    tier_select which encoder this machine will actually use and fetches only
    that — the difference between ~0.8 GB on a CPU laptop and the 20+ GB the old
    unconditional prefetch would have pulled.

    A subprocess, not an in-process call: downloads take minutes, and the event
    loop must stay free to serve the UI that is displaying this progress.
    """
    def _stream():
        # Imported here, not at module scope: suppress_console patches subprocess
        # at import line 1, and a local import picks up the patched module.
        import subprocess
        proc = None
        try:
            cmd = [sys.executable, str(_UNIT_ROOT / "scripts" / "fetch_models.py"),
                   "--json"]
            if req.model_name in ("optional", "all"):
                cmd.append("--with-optional" if req.model_name == "optional" else "--all")
            proc = subprocess.Popen(
                cmd, cwd=str(_UNIT_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:                      # type: ignore[union-attr]
                line = line.strip()
                if line.startswith("{"):
                    yield (line + "\n").encode()
            proc.wait(timeout=30)
        except Exception as exc:
            import json as _j
            yield (_j.dumps({"name": "plan", "status": "fail",
                             "message": str(exc)[:200]}) + "\n").encode()
        finally:
            # A disconnected client must not leave a multi-GB download orphaned.
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    return StreamingResponse(_stream(), media_type="application/x-ndjson")


@router.get("/api/health/engine")
async def health_engine():
    """Report which optional AI model files are present on disk.

      - status "online"  : at least one model file is installed
      - status "offline" : none are
      - missing_models   : the file names that are absent

    This used to be a hybrid check that first asked Ollama over HTTP which model
    tags it had pulled. That made a network round-trip on a 10-second UI poll to
    answer a question about the local filesystem, and every entry it could return
    named a third-party service the installer never set up. Every name here is now
    a .gguf file, which the frontend already distinguishes from a service tag, so
    the "install Ollama" branch of the banner is unreachable by construction.

    Grading does not depend on any of these. They power critique, annotations and
    Story Mode; the response is advisory, never a gate.
    """
    def _check_sync() -> dict:
        # From model_registry, not a fourth copy of the literals. The list here
        # named the 2B GGUFs, which do not exist on the Hub, so this endpoint
        # reported two permanently-missing files and the UI offered to download
        # something unobtainable. That is precisely why the registry exists.
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "src"))
        import model_registry as _mr
        missing = [m.dest.name for m in _mr.missing_gguf()]
        present = len(_mr.GGUF_MODELS) - len(missing)
        return {"status": "online" if present else "offline",
                "missing_models": missing}

    result = await asyncio.get_running_loop().run_in_executor(None, _check_sync)
    return JSONResponse(result)


@router.get("/api/annotations/{image_hash:path}")
async def get_annotations(image_hash: str):
    """Return has_annotations, score_factors, and eye_overlay_url for a single image."""
    # Resolve eye overlay URLs for canvas_renderer.py outputs
    _overlay_base = _UNIT_ROOT / "static" / "eye_feature_overlays"
    _verified     = _overlay_base / f"verified_{image_hash}.png"
    _critique     = _overlay_base / f"critique_{image_hash}.png"
    eye_overlay_url: str = ""
    if _verified.exists():
        eye_overlay_url = f"/static/eye_feature_overlays/verified_{image_hash}.png"
    elif _critique.exists():
        eye_overlay_url = f"/static/eye_feature_overlays/critique_{image_hash}.png"

    try:
        import lance_store as _ls
        # Indexed, embedding-free lookup — query_all() copied every row's
        # 1536-dim vector into RAM just to read two fields on each photo click.
        rows = _ls.query_by_path_fragment(
            image_hash, ["path", "has_annotations", "score_factors"]
        )
        record = next(
            (r for r in rows
             if Path(r["path"]).stem == image_hash or image_hash in Path(r["path"]).stem),
            None,
        )
        if record is None:
            return JSONResponse({
                "has_annotations": "",
                "score_factors":   [],
                "eye_overlay_url": eye_overlay_url,
            })
        import json as _json
        _sf_raw = record.get("score_factors", "")
        try:
            _sf = _json.loads(_sf_raw) if _sf_raw else []
        except Exception:
            _sf = []
        return JSONResponse({
            "has_annotations": record.get("has_annotations", ""),
            "score_factors":   _sf,
            "eye_overlay_url": eye_overlay_url,
        })
    except Exception as _e:
        return JSONResponse({
            "has_annotations": "",
            "score_factors":   [],
            "eye_overlay_url": eye_overlay_url,
            "error": str(_e),
        })


@router.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    """
    Upload a PDF reference document. Extracts text, runs LLM concept extraction,
    appends phrases to cache/rag_concepts.json.
    """
    import tempfile as _tmp, shutil as _sh
    try:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are supported")

        # Save upload to a temp file (bounded) then hand off to pdf_rag.ingest_pdf.
        # Cap at 200 MB so an oversized upload can't fill the disk / OOM the parser.
        suffix = ".pdf"
        _MAX_PDF_BYTES = 200 * 1024 * 1024
        with _tmp.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            _written = 0
            while True:
                _c = await file.read(1024 * 1024)
                if not _c:
                    break
                _written += len(_c)
                if _written > _MAX_PDF_BYTES:
                    tf.close(); Path(tf.name).unlink(missing_ok=True)
                    raise HTTPException(413, "PDF too large (max 200 MB).")
                tf.write(_c)
            tmp_path = tf.name

        from pdf_rag import ingest_pdf as _ingest
        phrases = await run_in_threadpool(_ingest, tmp_path, file.filename)

        Path(tmp_path).unlink(missing_ok=True)

        from pdf_rag import list_pdfs as _list

        # Bust the SigLIP-2 text embedding cache so next grade run re-encodes
        # with the new PDF phrases baked into the positive rubric.
        try:
            import sys as _sys
            _gp = _sys.modules.get("grade_pipeline_v2")
            if _gp is not None:
                _gp._text_emb_cache.clear()
                print("[server] RAG upload: cleared SigLIP text embedding cache")
        except Exception as _e_cache:
            print(f"[server] Cache clear skipped: {_e_cache}")

        return JSONResponse({"ok": True, "phrases": phrases, "pdfs": _list()})
    except HTTPException:
        raise
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        raise HTTPException(500, str(e))


@router.get("/api/rag/concepts")
async def rag_concepts():
    """Return all stored concept phrases and PDF metadata."""
    from pdf_rag import load_concepts as _lc, list_pdfs as _lp
    # for_display: show what is stored even when injection is off, so the
    # store is unused rather than invisible.
    return JSONResponse({"phrases": _lc(for_display=True), "pdfs": _lp()})


@router.delete("/api/rag/clear")
async def rag_clear():
    """Remove all stored concept phrases and uploaded PDF records."""
    from pdf_rag import clear_concepts as _cc
    await run_in_threadpool(_cc)
    try:
        import sys as _sys
        _gp = _sys.modules.get("grade_pipeline_v2")
        if _gp is not None:
            _gp._text_emb_cache.clear()
    except Exception:
        pass
    return JSONResponse({"ok": True})


@router.post("/api/reasoning_overlay")
async def reasoning_overlay(req: Request):
    """
    Render photographer-style reasoning annotations onto the image server-side.
    Body: { path: str }
    Returns: { overlay_url: str } or { error: str }
    """
    try:
        body      = await req.json()
        raw_path  = body.get("path", "")
        if not raw_path:
            return JSONResponse({"error": "path required"}, status_code=400)
        # sanitize_path was never defined here — every call to this endpoint
        # raised NameError -> 500. Use the same path-safety helper as the
        # other image endpoints.
        img_path  = str(_safe_image_path(raw_path))

        import lance_store as _ls
        # Indexed lookup — see get_annotations: query_all() here meant a full
        # library copy (embeddings included) per overlay render.
        rows = _ls.query_by_path_fragment(str(Path(img_path).stem))
        record = next(
            (r for r in rows if str(Path(r["path"])) == str(Path(img_path))),
            None,
        )
        if record is None:
            return JSONResponse({"error": "image not found in DB"}, status_code=404)

        bd = record.get("breakdown") or {}
        if isinstance(bd, str):
            import json as _json
            try:
                bd = _json.loads(bd)
            except Exception:
                bd = {}

        reasoning_log = record.get("reasoning_log", "")
        if not reasoning_log and bd:
            from src.specvlm_pipeline import _build_reasoning as _br, _detect_genre as _dg
            reasoning_log = _br(
                float(record.get("score", 0.5)),
                bd,
                bool(record.get("is_verified", False)),
                grade=record.get("grade", ""),
                genre=_dg(bd),
            )
        if not reasoning_log:
            return JSONResponse({"error": "no reasoning data for this image"}, status_code=404)

        from canvas_renderer import render_reasoning_overlay as _rro
        overlay_url = await run_in_threadpool(
            _rro,
            img_path,
            reasoning_log,
            bd,
            float(record.get("score", 0.5)),
            record.get("grade", ""),
        )
        return JSONResponse({"overlay_url": overlay_url})
    except Exception as _e:
        import traceback as _tb
        _tb.print_exc()
        return JSONResponse({"error": str(_e)}, status_code=500)


