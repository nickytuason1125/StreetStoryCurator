"""Creative routes — moved verbatim from server_impl.py (Milestone 4 split).

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
    Path, _BG_EXECUTOR, _DATA_DIR, _load_used_cd_paths, _save_used_cd_paths, asyncio,
)
import json

router = APIRouter()


def __getattr__(name):
    # Eager bindings above cover every static reference; this only serves
    # dynamic accesses (e.g. late-bound state added after the split).
    import server_impl as _si
    return getattr(_si, name)


@router.post("/api/creative-direction/stream")
async def creative_direction_stream(payload: dict):
    """
    SSE stream — Purist Creative Direction pipeline.

    Selects the best original captures for a 5-image Story Sequence.
    No pixel modification is performed. Output files are copies of originals.

    Payload:
        anchor_path  str  – reference image path (used for metadata only)
        folder_path  str  – base folder (locates Final_Portfolio output)
        style_prompt str  – style brief for the DeepSeek-R1 Agent Rule Set
        n_target     int  – target sequence length (5–10, default 7)
    """
    import asyncio, json, numpy as _np
    from fastapi.responses import StreamingResponse

    anchor_path     = (payload.get("anchor_path") or "").strip()
    folder_path     = (payload.get("folder_path") or "").strip()
    style_prompt    = (payload.get("style_prompt") or "").strip()
    n_target        = int(payload.get("n_target", 7))
    n_target        = max(3, min(10, n_target))
    peg_image_hash  = (payload.get("peg_image_hash") or "").strip() or None
    mode            = (payload.get("mode") or "story").strip().lower()
    if mode == "auto":
        try:
            from creative_director_agent import classify_mode
            mode = classify_mode(style_prompt)
        except Exception as _e_classify:
            print(f"[server] mode='auto' classification skipped ({_e_classify}) — defaulting to story")
            mode = "story"
    if mode not in ("story", "competition"):
        mode = "story"

    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    def _push(msg: dict):
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    def _progress(frac: float, desc: str):
        _push({"progress": round(frac, 3), "desc": desc})

    def _run():
        try:
            import numpy as np
            import json

            # ── Fetch all graded images (Strong + Mid + Weak) ─────────────────
            _progress(0.01, "Loading graded images…")
            strong_paths:  list[str] = []
            embeddings:    list      = []
            scores:        list      = []
            aspect_scores: list      = []
            IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp"}

            # Tier 0: catalog.json — always-available grade cache written by frontend
            try:
                if _CATALOG_PATH.exists():
                    _cat = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
                    _photos = _cat.get("photos", [])
                    if folder_path:
                        from pathlib import Path as _Path
                        _fp = str(_Path(folder_path).resolve())
                        _photos = [p for p in _photos if p.get("path", "").startswith(_fp)]
                    _photos = [p for p in _photos if float(p.get("score", 0)) > 0]
                    if _photos:
                        _photos.sort(key=lambda p: float(p.get("score", 0)), reverse=True)
                        strong_paths  = [p["path"]                          for p in _photos]
                        embeddings    = [np.zeros(1536, dtype=np.float32)   for _ in _photos]
                        scores        = [float(p.get("score", 0.5))         for p in _photos]
                        aspect_scores = [
                            p["breakdown"] if isinstance(p.get("breakdown"), dict)
                            else (json.loads(p["breakdown"]) if isinstance(p.get("breakdown"), str) else {})
                            for p in _photos
                        ]
                        _progress(0.02, f"Found {len(strong_paths)} graded images (catalog)")
            except Exception as _e:
                print(f"[cd] catalog.json read failed: {_e}")

            # Tier 1: LanceDB — primary source for embeddings; also fallback when catalog empty.
            # Always run to enrich embeddings even when catalog.json already provided paths —
            # catalog.json has no embeddings, so creative mode needs LanceDB for visual diversity.
            try:
                import lance_store as ls
                if strong_paths:
                    _lance_rows = ls.query_by_paths(strong_paths)
                else:
                    _lance_rows = ls.query_all(min_score=0.0)
                    if folder_path:
                        from pathlib import Path as _Path
                        fp = str(_Path(folder_path).resolve())
                        _lance_rows = [c for c in _lance_rows if c["path"].startswith(fp)]
                    _lance_rows = [c for c in _lance_rows if float(c.get("score", 0)) > 0]
                if _lance_rows:
                    if not strong_paths:
                        # Catalog was empty — use LanceDB as primary source
                        _lance_rows.sort(key=lambda c: float(c.get("score", 0)), reverse=True)
                        strong_paths  = [c["path"]                                  for c in _lance_rows]
                        scores        = [float(c.get("score", 0.5))                 for c in _lance_rows]
                        aspect_scores = [
                            c["breakdown"] if isinstance(c.get("breakdown"), dict)
                            else (json.loads(c["breakdown"]) if isinstance(c.get("breakdown"), str) else {})
                            for c in _lance_rows
                        ]
                        embeddings = [np.array(c["embedding"], dtype=np.float32) for c in _lance_rows]
                        _progress(0.02, f"Found {len(strong_paths)} graded images (LanceDB)")
                    else:
                        # Catalog provided paths — enrich embeddings from LanceDB
                        _emb_map = {c["path"]: np.array(c["embedding"], dtype=np.float32) for c in _lance_rows}
                        embeddings = [_emb_map.get(p, np.zeros(1536, dtype=np.float32)) for p in strong_paths]
                        n_real = sum(1 for e in embeddings if np.any(e != 0))
                        print(f"[cd] Embeddings enriched from LanceDB: {n_real}/{len(embeddings)} real")
            except Exception as e:
                print(f"[cd] LanceDB query failed: {e}")

            # Tier 2: Strong/ subfolder on disk (fallback when LanceDB is empty)
            if not strong_paths and folder_path:
                from pathlib import Path as _Path
                strong_dir = _Path(folder_path) / "Strong"
                if strong_dir.exists():
                    strong_paths = [
                        str(f) for f in sorted(strong_dir.iterdir())
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    ]
                    embeddings = [np.zeros(1536, dtype=np.float32) for _ in strong_paths]
                    scores     = [0.75] * len(strong_paths)

            # Tier 3: Scan folder directly for any images (cap at 50)
            # Extension alone isn't enough — a renamed/truncated garbage file with a
            # .jpg suffix passes an extension check trivially. The graded paths above
            # (catalog.json / LanceDB) already went through grade_pipeline_v2's
            # early-exit technical gate, which is why this is the one candidate
            # source that needs its own readability check before it can end up
            # copied into the user's Final_Portfolio output.
            if not strong_paths and folder_path:
                from pathlib import Path as _Path
                fp = _Path(folder_path)
                if fp.exists():
                    candidates = sorted(
                        f for f in fp.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )
                    all_imgs = []
                    for f in candidates:
                        if len(all_imgs) >= 50:
                            break
                        try:
                            from PIL import Image as _Image
                            with _Image.open(f) as _im:
                                _im.verify()
                        except Exception:
                            print(f"[cd] Tier 3 folder scan: skipping unreadable file {f.name}")
                            continue
                        all_imgs.append(str(f))
                    if all_imgs:
                        strong_paths = all_imgs
                        embeddings   = [np.zeros(1536, dtype=np.float32) for _ in strong_paths]
                        scores       = [0.5] * len(strong_paths)
                        _progress(0.02, f"Using {len(strong_paths)} folder images (grade folder for better selection)")

            if not strong_paths:
                _push({"error": "No images found. Grade your folder first."})
                return

            _progress(0.03, f"Found {len(strong_paths)} images for creative direction")

            # ── Release SigLIP-2 singleton before Creative Mode LLMs load ─────
            try:
                from grade_pipeline_v2 import release_grading_models
                release_grading_models()
            except Exception as _e_rel:
                print(f"[server] release_grading_models skipped: {_e_rel}")
                # Belt-and-suspenders: a release failure must never leave
                # stale GPU tensors resident before Story Mode's own GPU
                # work (contact-sheet critique GGUF) starts loading.
                try:
                    from vram_manager import VRAMManager
                    VRAMManager.purge_vram()
                except Exception:
                    pass

            # ── Run pipeline ──────────────────────────────────────────────────
            from creative_director import run_creative_direction

            avoid_paths = sorted(_load_used_cd_paths())

            result = run_creative_direction(
                strong_paths      = strong_paths,
                embeddings        = embeddings,
                scores            = scores or None,
                aspect_scores_list= aspect_scores or None,
                anchor_path       = anchor_path,
                output_dir        = folder_path or str(
                    Path(anchor_path).parent if anchor_path else _DATA_DIR / "cache"
                ),
                style_prompt      = style_prompt,
                n_target          = n_target,
                avoid_paths       = avoid_paths,
                progress          = _progress,
                peg_image_hash    = peg_image_hash,
                mode              = mode,
            )

            # Auto-mark generated images as used so next generation picks different ones.
            # Explicit save (save-sequence) persists to Story_<ts>/; this just rotates the pool.
            if result.get("outputs"):
                new_used = {
                    o["source_path"] for o in result["outputs"]
                    if o.get("success") and o.get("source_path")
                }
                if new_used:
                    updated = _load_used_cd_paths() | new_used
                    # Reset when the whole pool has cycled through
                    if len(updated) >= len(strong_paths):
                        updated = new_used
                    _save_used_cd_paths(updated)

            _push({"done": True, "data": result})

        except Exception as exc:
            _push({"error": str(exc)})
        finally:
            import gc as _gc
            _gc.collect()

    _BG_EXECUTOR.submit(_run)

    async def _event_stream():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if "done" in msg or "error" in msg:
                break

    return StreamingResponse(
        _event_stream(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/creative-direction/list-portfolio")
async def list_portfolio(payload: dict):
    """
    Return the list of stylized images in Final_Portfolio/ for a given folder.
    Body: { folder_path }
    """
    folder = (payload.get("folder_path") or "").strip()
    if not folder:
        return JSONResponse({"images": []})
    try:
        port_dir = Path(folder) / "Final_Portfolio"
        if not port_dir.is_dir():
            return JSONResponse({"images": []})
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted(
            str(f) for f in port_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        return JSONResponse({"images": images, "dir": str(port_dir)})
    except Exception as e:
        print(f"[cd/portfolio] list failed: {e}", flush=True)
        raise HTTPException(500, "Could not list the portfolio folder — see the server log for details.")


@router.post("/api/creative-direction/save-sequence")
async def save_cd_sequence(payload: dict):
    """
    Copy stylized outputs to a timestamped Story folder and write a manifest.
    Marks the source images as used so they are excluded from future sequences.

    Body: { outputs: [{source_path, output_path, params, success}], base_dir: str }
    """
    import shutil
    from datetime import datetime

    outputs  = payload.get("outputs", [])
    base_dir = (payload.get("base_dir") or "").strip()

    successes = [o for o in outputs if o.get("success") and o.get("output_path")]
    if not successes:
        return JSONResponse({"ok": False, "error": "No successful outputs to save"})

    # Resolve base dir
    if not base_dir:
        base_dir = str(Path(successes[0]["output_path"]).parent.parent)
    # Trust-boundary check: base_dir arrives from the client; route it through
    # the same validator every other filesystem write in this codebase uses.
    from server_impl import _safe_dir_path as _safe_dir
    base_dir_p = _safe_dir(base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    story_dir = base_dir_p / f"Story_{timestamp}"
    story_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    ROLE_ORDER = ["opener", "subject", "detail", "contrast", "closer"]
    sorted_out = sorted(
        successes,
        key=lambda o: ROLE_ORDER.index(o.get("params", {}).get("role", "")) if o.get("params", {}).get("role", "") in ROLE_ORDER else 99
    )

    for i, item in enumerate(sorted_out):
        src = Path(item["output_path"])
        if not src.exists():
            continue
        role = item.get("params", {}).get("role", "unknown")
        dest = story_dir / f"{i+1:02d}_{role}_{src.name}"
        shutil.copy2(str(src), str(dest))
        manifest.append({
            "seq": i + 1,
            "role":        role,
            "source_path": item.get("source_path", ""),
            "output_path": str(dest),
            "score":       item.get("params", {}).get("mogco_objectives", {}).get("set_cohesion", 0),
        })

    import json as _j
    (story_dir / "manifest.json").write_text(_j.dumps(manifest, indent=2))

    # Mark source paths as used
    source_paths = {o["source_path"] for o in successes if o.get("source_path")}
    used = _load_used_cd_paths() | source_paths
    _save_used_cd_paths(used)

    return JSONResponse({
        "ok":        True,
        "story_dir": str(story_dir),
        "count":     len(manifest),
        "used_total": len(used),
    })


@router.post("/api/creative-direction/clear-used")
async def clear_used_cd_paths():
    """Reset the used-image history so all photos are eligible again."""
    _save_used_cd_paths(set())
    return JSONResponse({"ok": True, "used_total": 0})


@router.get("/api/creative-direction/used-count")
async def get_used_cd_count():
    """Return how many source images are currently excluded from future sequences."""
    return JSONResponse({"count": len(_load_used_cd_paths())})


