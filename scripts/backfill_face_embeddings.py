r"""
Backfill face embeddings — builds the People Filter index.

What it does
------------
For every catalog photo that carries a face summary (24,635 rows as of
2026-09), run YuNet once, crop each face (25% pad), encode the crops with
the already-downloaded SigLIP-2 encoder, and upsert (path, face_idx,
embedding) into the faces table (lance_store.upsert_face_embeddings).

Why a script and not the grade pipeline
---------------------------------------
Grading never writes here, so the People index can be built or rebuilt
without re-grading 64k photos — and it can run on its own schedule, when
RAM allows, instead of competing with a cull.

Guards
------
* RAM gate: needs ~3.5 GB free (the encoder + decode buffers). Refuses
  below that rather than triggering the OOM dance the encode subprocess
  already knows how to lose.
* Resumable: paths already in the faces table are skipped, so an
  interrupted run continues where it stopped (delete the table to rebuild).
* Bounded batches: photos are processed in small groups so a crash loses
  at most one batch, and the encoder's own RAM floor applies.

Run:  venv\Scripts\python.exe scripts\backfill_face_embeddings.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import face_signals                               # noqa: E402
import lance_store                                # noqa: E402

CATALOG = ROOT / "cache" / "catalog.json"
# The encode subprocess reloads the ~4 GB model on EVERY batch, so small
# batches drown in model-load time (measured: 16/batch -> 2.4 s/photo;
# 256/batch amortizes the load to noise). Peak RAM is unchanged — the load
# happens in the isolated encode subprocess either way.
BATCH_PHOTOS = int(os.environ.get("FIRSTCUT_BACKFILL_BATCH", "256"))
# Same escape hatch as the encoder's SIGLIP_MIN_FREE_RAM_GB (0 = trust the
# pagefile and run anyway — slower, but the encode subprocess is isolated and
# the run is resumable, so a mid-run OOM just retries on the next launch).
MIN_FREE_GB = float(os.environ.get("FIRSTCUT_BACKFILL_MIN_RAM_GB", "3.5"))
# Decode parallelism: RAW/JPEG decode is I/O + C-extension bound and releases
# the GIL, so a small pool hides disk latency. Detection stays on the main
# thread (the shared YuNet session isn't guaranteed thread-safe).
DECODE_WORKERS = int(os.environ.get("FIRSTCUT_BACKFILL_DECODE_WORKERS", "4"))


def _decode_one(p: str):
    """Decode-only worker: -> (path, PIL.Image | None, error | None).

    Handles both _load_small return shapes: (im, src) for PIL-decodable
    formats, a bare RGB ndarray for RAW. Never raises.
    """
    try:
        res = face_signals._load_small(p)
        img = res[0] if isinstance(res, tuple) else res
        if img is None:
            return p, None, "unreadable/missing"
        rgb = img if isinstance(img, Image.Image) else Image.fromarray(img)
        return p, rgb, None
    except Exception as e:
        return p, None, f"{type(e).__name__}: {e}"


def free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1 << 30)
    except Exception:
        pass
    if sys.platform == "win32":     # psutil-less Windows fallback
        try:
            import ctypes
            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MEM(); m.dwLength = ctypes.sizeof(_MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullAvailPhys / (1 << 30)
        except Exception:
            pass
    return 99.0   # no reliable answer — let the encoder's own floor decide


def person_rows() -> list[str]:
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    out = []
    for p in data.get("photos", []):
        f = p.get("face") or {}
        if (f.get("faces_detected") or 0) > 0 and p.get("path"):
            out.append(p["path"])
    return out


def main(bench: int = 0) -> int:
    if free_ram_gb() < MIN_FREE_GB:
        print(f"Refusing: {free_ram_gb():.2f} GB free < {MIN_FREE_GB} GB needed "
              f"for the SigLIP-2 encoder. Close apps and re-run.")
        return 2
    if not CATALOG.exists():
        print(f"No catalog at {CATALOG}")
        return 2
    paths = person_rows()
    if bench:
        todo = paths[:bench]   # bench ignores index status and writes nothing
        print(f"BENCH: first {len(todo)} person-photos, no writes")
    else:
        todo = [p for p in paths if lance_store.faces_count_for_path(p) == 0]
        print(f"person photos: {len(paths)} | already indexed: {len(paths) - len(todo)} "
              f"| to embed: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    from siglip2_encoder import get_siglip2_encoder
    enc = get_siglip2_encoder()
    tmp = Path(tempfile.mkdtemp(prefix="fg_facecrop_"))
    t0 = time.time()
    done_faces = done_photos = skipped = 0
    try:
        # Double-buffered pipeline: batch N+1 decodes on the worker pool while
        # batch N is detected/cropped (main thread) and encoded (subprocess).
        pool = ThreadPoolExecutor(max_workers=DECODE_WORKERS)
        fut = pool.map(_decode_one, todo[:BATCH_PHOTOS])
        for start in range(0, len(todo), BATCH_PHOTOS):
            tb = time.time()
            decoded = list(fut)
            nxt = todo[start + BATCH_PHOTOS: start + 2 * BATCH_PHOTOS]
            fut = pool.map(_decode_one, nxt) if nxt else None
            t_decode = time.time() - tb

            crop_paths, meta = [], []
            for p, rgb, err in decoded:
                if rgb is None:
                    if err and not err.startswith("unreadable"):
                        print(f"  skip {Path(p).name}: {err}", flush=True)
                    skipped += 1
                    continue
                try:
                    bgr = np.asarray(rgb)[:, :, ::-1].copy()
                    faces = face_signals.detect_faces(bgr)
                except Exception as e:
                    print(f"  skip {Path(p).name}: {type(e).__name__}: {e}", flush=True)
                    skipped += 1
                    continue
                for fi, f in enumerate(faces[:12]):
                    x, y, fw, fh = f["box"]
                    W, H = rgb.size
                    pad = 0.25 * max(fw, fh)
                    c = rgb.crop((max(0, int(x - pad)), max(0, int(y - pad)),
                                  min(W, int(x + fw + pad)), min(H, int(y + fh + pad))))
                    cp = tmp / f"b{start}_p{len(meta)}_f{fi}.jpg"
                    c.save(cp, "JPEG", quality=88)
                    crop_paths.append(str(cp))
                    meta.append({"path": p, "face_idx": fi,
                                 "area_frac": round(f["area_frac"], 5)})
            t_prep = time.time() - tb - t_decode

            if not crop_paths:
                print(f"[{start + len(decoded)}/{len(todo)}] no faces — "
                      f"decode {t_decode:.1f}s prep {t_prep:.1f}s", flush=True)
                continue
            embs = enc.encode_images(crop_paths)
            t_encode = time.time() - tb - t_decode - t_prep
            if not bench:
                rows = [{**m, "embedding": embs[i].tolist()}
                        for i, m in enumerate(meta) if i < len(embs)]
                lance_store.upsert_face_embeddings(rows)
            done_faces += min(len(embs), len(meta))
            done_photos += len({m["path"] for m in meta})
            batch_s = time.time() - tb
            remaining = len(todo) - start - len(decoded)
            print(f"[{start + len(decoded)}/{len(todo)}] faces={done_faces} "
                  f"photos={done_photos} | dec {t_decode:.1f}s prep {t_prep:.1f}s "
                  f"enc {t_encode:.1f}s = {batch_s:.1f}s/batch "
                  f"({batch_s / max(1, len(decoded)):.2f}s/photo) "
                  f"eta={batch_s / max(1, len(decoded)) * remaining / 60:.0f}m",
                  flush=True)
        pool.shutdown(wait=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            enc.unload()
        except Exception:
            pass
    print(f"{'BENCH DONE' if bench else 'DONE'} — {done_faces} face embeddings "
          f"for {done_photos} photos ({skipped} skipped) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    _bench = 0
    if "--bench" in sys.argv:
        _bench = int(sys.argv[sys.argv.index("--bench") + 1])
    raise SystemExit(main(_bench))