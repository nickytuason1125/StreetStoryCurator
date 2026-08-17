"""
fetch_models.py — download exactly the weights THIS machine needs.

Run:
    venv\\Scripts\\python.exe scripts/fetch_models.py              # what grading needs
    venv\\Scripts\\python.exe scripts/fetch_models.py --dry-run    # plan + sizes, no network
    venv\\Scripts\\python.exe scripts/fetch_models.py --with-optional
    venv\\Scripts\\python.exe scripts/fetch_models.py --tier low   # override auto-selection
    venv\\Scripts\\python.exe scripts/fetch_models.py --json       # ndjson progress

Why this script exists
----------------------
Nothing delivered model weights. The one auto-downloader,
model_loader.ensure_all_models_downloaded(), was reachable only from
_bg_model_prefetch, which was commented out in server.py — and for good reason:
it fetched every model unconditionally. The giant encoder, a 6.8 GB Qwen, two
DeepSeek checkpoints, all of it, on every machine. That is over 20 GB, which does
not fit on a laptop and is absurd on one that has already been routed to the
768-d Fast encoder.

Worse was the failure mode when weights were absent. encode_worker fell back to
``open_clip.create_model_and_transforms(..., pretrained="webli")``, a ~7 GB fp32
download that peaks at 10.3 GB RAM — the heaviest possible path, triggered
automatically on the weakest possible machine.

So the rule here is: ask which encoder this machine will actually run, then fetch
that one. A 16 GB CPU laptop needs about 0.8 GB, not 20.

Everything is grouped by what breaks without it, because "optional" has to mean
something specific. REQUIRED is what grading cannot proceed without. OPTIONAL
degrades a named feature and never touches a grade. DEEP is the opt-in Deep Grade
path, off by default.

Process model
-------------
Never touches torch.cuda: this runs in the same process family as the grade
worker, and tier_select answers the GPU question from a cached subprocess probe.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SENTINEL = _ROOT / "models" / ".models_ready"

# Keep a margin beyond the download itself: snapshot_download stages files before
# moving them, and a disk that fills mid-fetch leaves a half-written checkpoint
# that looks installed. Refusing early is cheaper than diagnosing that later.
_DISK_MARGIN_GB = 3.0

REQUIRED, OPTIONAL, DEEP = "required", "optional", "deep"


def _emit(as_json: bool, **fields) -> None:
    """One progress record. ndjson when --json, human text otherwise."""
    if as_json:
        sys.stdout.write(json.dumps(fields) + "\n")
    else:
        status = fields.get("status", "")
        name = fields.get("name", "")
        msg = fields.get("message", "")
        prefix = {"start": "  →", "ok": "  ✓", "skip": "  ·", "fail": "  ✗"}.get(status, "   ")
        sys.stdout.write(f"{prefix} {name}{(': ' + msg) if msg else ''}\n")
    sys.stdout.flush()


def _free_disk_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1e9
    except Exception:
        return 999.0


# ── Individual fetchers ──────────────────────────────────────────────────────

def _fetch_encoder(tier: str, as_json: bool) -> bool:
    """Download + convert + validate + promote one tier's lean fp16 checkpoint.

    Delegates to build_lean_checkpoint.py, which already owns the per-tier repo
    table, the fp16 streaming conversion and the cosine validation against a
    reference encode. Reimplementing any of that here would create a second
    place for the repo IDs to drift.
    """
    import run_profile
    dest = _ROOT / run_profile.spec_for(tier).hf_dirname
    if (dest / "config.json").exists():
        _emit(as_json, name=f"encoder:{tier}", status="skip", message="already installed")
        return True

    _emit(as_json, name=f"encoder:{tier}", status="start",
          message="downloading and converting to fp16")
    r = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "build_lean_checkpoint.py"),
         "--tier", tier, "--promote"],
        cwd=str(_ROOT),
    )
    ok = r.returncode == 0 and (dest / "config.json").exists()
    _emit(as_json, name=f"encoder:{tier}", status="ok" if ok else "fail",
          message="" if ok else f"build_lean_checkpoint exited {r.returncode}")
    return ok


def _fetch_topiq(as_json: bool) -> bool:
    """TOPIQ NR runs on the DEFAULT grading path, so this is not optional.

    pyiqa fetches its own weights on first create_metric(); doing it here means it
    happens during setup with a progress line rather than silently, mid-cull, on a
    machine the user believed was offline-ready.
    """
    _emit(as_json, name="topiq_nr", status="start", message="caching quality-head weights")
    try:
        import model_loader
        ok = model_loader._download_topiq_nr_if_needed()
    except Exception as exc:
        _emit(as_json, name="topiq_nr", status="fail", message=str(exc)[:120])
        return False
    _emit(as_json, name="topiq_nr", status="ok" if ok else "fail")
    return ok


def _fetch_detectors(as_json: bool) -> bool:
    """D-FINE-nano (Apache-2.0) and the torchvision Mask R-CNN cache.

    download_detectors.py's own docstring says D-FINE exists to replace
    ultralytics YOLO, which is AGPL-3.0. Fetching it by default is what makes
    dropping YOLO an option rather than a project.
    """
    if (_ROOT / "models" / "dfine_nano" / "config.json").exists():
        _emit(as_json, name="detectors", status="skip", message="already installed")
        return True
    _emit(as_json, name="detectors", status="start", message="D-FINE-nano")
    try:
        import download_detectors
        download_detectors.main()
    except Exception as exc:
        _emit(as_json, name="detectors", status="fail", message=str(exc)[:120])
        return False
    _emit(as_json, name="detectors", status="ok")
    return True


def _fetch_gguf(key: str, as_json: bool) -> bool:
    import model_registry
    m = model_registry.gguf(key)
    if m.present():
        _emit(as_json, name=f"gguf:{key}", status="skip", message="already installed")
        return True
    _emit(as_json, name=f"gguf:{key}", status="start",
          message=f"{m.filename} (~{m.size_gb:.1f} GB) — {m.purpose}")
    try:
        from huggingface_hub import hf_hub_download
        got = hf_hub_download(repo_id=m.repo, filename=m.filename)
        m.dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy rather than symlink: the app must keep working if the HF cache is
        # cleared, and Windows symlinks need developer mode or elevation.
        shutil.copyfile(got, m.dest)
    except Exception as exc:
        _emit(as_json, name=f"gguf:{key}", status="fail", message=str(exc)[:120])
        return False
    _emit(as_json, name=f"gguf:{key}", status="ok")
    return True


def _fetch_vision_probe(as_json: bool) -> bool:
    """DINOv2-S/14 for ChiaroscuroHead.

    models/vision_probe/ has never existed on this install, so the head has been
    unavailable in every logged run — and its luminance fallback flagged 0 dark
    images every time, meaning intentionally dark work has had no protection at
    all. Small download, real grading consequence.
    """
    if (_ROOT / "models" / "vision_probe").exists() and \
            any((_ROOT / "models" / "vision_probe").rglob("*.safetensors")):
        _emit(as_json, name="vision_probe", status="skip", message="already installed")
        return True
    _emit(as_json, name="vision_probe", status="start", message="DINOv2-S/14 (~0.1 GB)")
    try:
        import model_loader
        ok = model_loader._download_vis_probe_if_needed()
    except Exception as exc:
        _emit(as_json, name="vision_probe", status="fail", message=str(exc)[:120])
        return False
    _emit(as_json, name="vision_probe", status="ok" if ok else "fail")
    return ok


def _fetch_deep_grade(as_json: bool) -> bool:
    """Qwen2.5-VL-3B for the opt-in Deep Grade toggle. 6.8 GB, off by default."""
    _emit(as_json, name="qwen_vlm", status="start", message="Qwen2.5-VL-3B (~6.8 GB)")
    try:
        import model_loader
        ok = model_loader._download_qwen_if_needed()
    except Exception as exc:
        _emit(as_json, name="qwen_vlm", status="fail", message=str(exc)[:120])
        return False
    _emit(as_json, name="qwen_vlm", status="ok" if ok else "fail")
    return ok


# ── Plan ─────────────────────────────────────────────────────────────────────

def build_plan(tier: str, groups: set) -> list:
    """[(group, name, size_gb, fetch_fn)] for the requested groups."""
    import model_registry
    enc_size = model_registry.ENCODER_SOURCES.get(tier, ("", 3.6, ""))[1]

    plan: list = []
    if REQUIRED in groups:
        plan += [
            (REQUIRED, f"encoder:{tier}", enc_size, lambda j: _fetch_encoder(tier, j)),
            (REQUIRED, "topiq_nr", 0.2, _fetch_topiq),
            (REQUIRED, "detectors", 0.1, _fetch_detectors),
        ]
    if OPTIONAL in groups:
        for key in ("vision", "vision_mmproj", "text"):
            m = model_registry.gguf(key)
            plan.append((OPTIONAL, f"gguf:{key}", m.size_gb,
                         (lambda k: (lambda j: _fetch_gguf(k, j)))(key)))
        plan.append((OPTIONAL, "vision_probe", 0.1, _fetch_vision_probe))
    if DEEP in groups:
        plan.append((DEEP, "qwen_vlm", 6.8, _fetch_deep_grade))
    return plan


def _installed(name: str) -> bool:
    """Is this plan item already on disk?

    topiq_nr and vision_probe are not listed: pyiqa owns its own cache location
    and both fetchers are cheap no-ops when the weights are already there, so
    they report 'skip' at run time rather than being predicted here.
    """
    import model_registry
    import run_profile
    if name.startswith("encoder:"):
        tier = name.split(":", 1)[1]
        return (_ROOT / run_profile.spec_for(tier).hf_dirname / "config.json").exists()
    if name.startswith("gguf:"):
        return model_registry.gguf(name.split(":", 1)[1]).present()
    if name == "detectors":
        return (_ROOT / "models" / "dfine_nano" / "config.json").exists()
    return False


def _pending_gb(plan: list) -> float:
    """Size of what is actually missing, so a re-run reports ~0 rather than the
    full catalogue total."""
    return sum(size for _g, name, size, _fn in plan if not _installed(name))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the weights this machine needs.")
    ap.add_argument("--tier", choices=("high", "mid", "low"),
                    help="override auto-selection")
    ap.add_argument("--with-optional", action="store_true",
                    help="also fetch critique/annotation/Story Mode models")
    ap.add_argument("--deep-grade", action="store_true",
                    help="also fetch the opt-in Deep Grade VLM (~6.8 GB)")
    ap.add_argument("--all", action="store_true", help="everything")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, download nothing")
    ap.add_argument("--json", action="store_true", help="ndjson progress on stdout")
    args = ap.parse_args()

    groups = {REQUIRED}
    if args.with_optional or args.all:
        groups.add(OPTIONAL)
    if args.deep_grade or args.all:
        groups.add(DEEP)

    # Tier first: it decides the largest single download in the plan.
    if args.tier:
        tier, label, reason = args.tier, args.tier, "chosen with --tier"
    else:
        import tier_select
        tier, label, reason = tier_select.select()

    plan = build_plan(tier, groups)
    pending = _pending_gb(plan)
    free = _free_disk_gb(_ROOT)

    if not args.json:
        print(f"\nFrameGrade — model download")
        print(f"  encoder tier : {label}  ({reason})")
        print(f"  to download  : ~{pending:.1f} GB")
        print(f"  free disk    : {free:.1f} GB\n")
    else:
        _emit(True, name="plan", status="start", tier=tier, label=label,
              reason=reason, pending_gb=round(pending, 2), free_disk_gb=round(free, 1),
              items=[n for _g, n, _s, _f in plan])

    if args.dry_run:
        for group, name, size, _fn in plan:
            mark = "already installed" if _installed(name) else f"~{size:.1f} GB"
            print(f"  [{group:8}] {name:22} {mark}")
        return 0

    if pending > 0 and free < pending + _DISK_MARGIN_GB:
        msg = (f"not enough disk: need ~{pending + _DISK_MARGIN_GB:.1f} GB "
               f"(download {pending:.1f} + {_DISK_MARGIN_GB:.0f} working margin), "
               f"have {free:.1f} GB")
        _emit(args.json, name="plan", status="fail", message=msg)
        return 1

    results: dict = {}
    for group, name, _size, fn in plan:
        try:
            results[name] = bool(fn(args.json))
        except Exception as exc:                      # a fetcher must never abort the run
            _emit(args.json, name=name, status="fail", message=str(exc)[:120])
            results[name] = False

    required_ok = all(ok for (g, n, _s, _f) in plan if g == REQUIRED
                      for ok in [results.get(n, False)])
    if required_ok:
        _SENTINEL.parent.mkdir(parents=True, exist_ok=True)
        _SENTINEL.write_text(tier, encoding="utf-8")

    failed = [n for n, ok in results.items() if not ok]
    _emit(args.json, name="plan", status="ok" if required_ok else "fail",
          message=("everything grading needs is installed"
                   + (f"; optional not fetched: {', '.join(failed)}" if failed else ""))
          if required_ok else f"failed: {', '.join(failed)}")
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
