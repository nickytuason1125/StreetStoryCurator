"""
tier_select.py — pick the best vision encoder that actually fits, right now.

The pipeline used to hard-demand the largest encoder and refuse to start when it
did not fit ("Not enough free RAM … need ~3.0 GB"). On a machine whose free RAM
swings between 1 and 6 GB depending on what else is open, that means a cull
either works or doesn't, with no middle ground.

This module picks a tier at grade start from two facts:
  * how much RAM is free at this instant
  * which encoders are actually installed

and exposes a user-facing quality label instead of model internals:

    Fast      smallest encoder  — runs on a busy machine
    Balanced  mid encoder
    Pro       largest encoder   — best quality, needs the most headroom

DELIBERATELY LIGHTWEIGHT: os + psutil only. It runs BEFORE grade_pipeline_v2 /
lance_store / siglip2_encoder are imported, because each of those reads
SIGLIP_TIER at module scope to fix its embedding dimension and table name.
Importing anything heavy here would defeat the point.

Switching tiers never destroys grades: each tier writes its own LanceDB table
(see lance_store._TBL_NAME), so a run at "Fast" does not purge the "Pro" work.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent

# tier -> (label, embed_dim, open_clip cache dir, ram need with the LEAN loader,
#          ram need with the heavy open_clip fp32 loader)
#
# "Pro" via the HF fp16 checkpoint is MEASURED at 2.71 GB peak (2026-07-31), so
# 3.0 is a true fit test. The open_clip figures are the heavy path: it reads a
# full fp32 checkpoint into RAM before casting, which for the largest model was
# measured at 10.3 GB. Mid/Low have no lean checkpoint yet, so they are costed
# on the heavy path — those two numbers are ESTIMATES scaled by parameter count,
# not measurements, and should be re-measured once their weights are installed.
_TIERS: dict = {}


def _build_tiers():
    """(label, dim, oc_cache, lean_need, heavy_need) derived from run_profile.

    The lean requirement IS the encoder's own hard floor, so the selector can
    never offer a tier the loader then refuses — that mismatch is what made
    Balanced get selected and then fail with "need ~3.0 GB".
    """
    import run_profile as _rp
    heavy = {"high": 10.3, "mid": 4.0, "low": 2.0}
    return {t: (_rp.spec_for(t).label, _rp.spec_for(t).embed_dim,
                _rp.spec_for(t).oc_cache, _rp.spec_for(t).ram_hard_gb,
                heavy[t]) for t in _rp.TIERS}


_TIERS = _build_tiers()
_ORDER = ["high", "mid", "low"]          # best quality first

# Checkpoint dirs come from run_profile — the single declaration of tier state.


def label(tier: str) -> str:
    return _TIERS.get(tier, _TIERS["high"])[0]


def embed_dim(tier: str) -> int:
    return _TIERS.get(tier, _TIERS["high"])[1]


def _hf_checkpoint_dir(tier: str = "high") -> str:
    import run_profile as _rp
    return os.environ.get("SIGLIP_HF_DIR") or str(_ROOT / _rp.spec_for(tier).hf_dirname)


def _has_hf(tier: str) -> bool:
    """True when the lean fp16 checkpoint for this tier is installed.

    Checked per tier rather than hardcoded, so building a lean checkpoint for a
    smaller tier immediately makes it the cheap option here too.
    """
    if os.environ.get("SIGLIP_ENC_USE_OC", "0").strip() == "1":
        return False
    return os.path.exists(os.path.join(_hf_checkpoint_dir(tier), "config.json"))


def available(tier: str) -> bool:
    """True when this tier's weights are on disk.

    The runtime is offline by contract, so a tier whose weights are absent can
    never load — selecting it would just relocate the failure instead of
    avoiding it.
    """
    if _has_hf(tier):
        return True
    cache = _ROOT / _TIERS.get(tier, _TIERS["high"])[2]
    if not cache.is_dir():
        return False
    for f in cache.rglob("*"):
        if (f.is_file() and f.suffix in {".pt", ".bin", ".safetensors"}
                and not f.name.endswith(".incomplete")):
            return True
    return False


def ram_need_gb(tier: str) -> float:
    """Free RAM required for this tier, given the loader it will actually use."""
    lean, heavy = _TIERS.get(tier, _TIERS["high"])[3:5]
    return lean if _has_hf(tier) else heavy


def free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 999.0        # unknown → don't let the probe itself block a grade


# Free RAM each tier needs when there is NO GPU and everything runs on the CPU.
# MEASURED on this machine with the GPU hidden (8 photos, batch 4):
#   Fast      2.83 GB peak, 2.16 s/img   -> a 514-photo cull in ~18 min
#   Balanced  4.52 GB peak, 2.94 s/img   -> ~25 min, and a big memory ask
#   Pro       6.5-7.6 GB,  11-14 s/img   -> ~2 hours; not a usable fallback
# CPU inference holds activations in system RAM instead of VRAM, so the
# requirement is several times the GPU figure — sizing the CPU path off the GPU
# numbers is what made the "fallback" fail with a bare 'bad allocation'.
_CPU_RAM_NEED = {"high": 8.0, "mid": 4.8, "low": 3.0}

# Without a GPU, quality above this tier costs more time than it is worth.
_CPU_MAX_TIER = "low"


_GPU_PROBE_CACHE = _ROOT / "cache" / "gpu_probe.json"


def has_gpu() -> bool:
    """True when some GPU actually exists to run on.

    CRITICAL: this must NOT call torch.cuda.* in this process. tier_select runs
    in the grade worker (the PARENT of the isolated encode subprocess), and
    touching torch.cuda there initialises CUDA in the parent; when the encode
    child later exits, the parent faults with 0xC0000005. That is a crash class
    this project already fixed once, and calling torch.cuda.is_available() here
    reintroduced it — verified: identical culls, exit 139 with the direct call
    and exit 0 with it bypassed.

    So the probe runs in a THROWAWAY subprocess and the answer is cached on
    disk. Hardware does not change between runs, so this costs one subprocess
    the first time and nothing afterwards.

    torch is still the honest signal: onnxruntime lists an execution provider
    when the WHEEL supports it, which is not the same as a device being present.
    FRAMEGRADE_ASSUME_GPU overrides; delete cache/gpu_probe.json to re-probe.
    """
    env = os.environ.get("FRAMEGRADE_ASSUME_GPU", "").strip()
    if env:
        return env not in ("0", "false", "no")
    try:
        if _GPU_PROBE_CACHE.exists():
            import json
            return bool(json.loads(_GPU_PROBE_CACHE.read_text(encoding="utf-8"))["gpu"])
    except Exception:
        pass
    gpu = _probe_gpu_subprocess()
    try:
        import json
        _GPU_PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _GPU_PROBE_CACHE.write_text(json.dumps({"gpu": gpu}), encoding="utf-8")
    except Exception:
        pass
    return gpu


def _probe_gpu_subprocess() -> bool:
    """Ask a separate process, so CUDA is never initialised in ours."""
    import subprocess
    import sys
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "import torch;print(int(torch.cuda.is_available() "
             "and torch.cuda.device_count()>0))"],
            capture_output=True, text=True, timeout=180)
        return r.stdout.strip().splitlines()[-1].strip() == "1"
    except Exception:
        return False        # unknown → assume CPU, which is the safe tier


def select(free_gb: Optional[float] = None,
           gpu: Optional[bool] = None) -> tuple:
    """Return (tier, label, reason). Never raises.

    Picks the highest-quality INSTALLED tier whose requirement fits in free RAM.
    If none fits, returns the smallest installed tier anyway — the encoder's own
    RAM floor then produces one clear, actionable error rather than this module
    inventing a second failure mode.

    With no GPU the choice changes on both axes: the RAM requirement rises
    several-fold AND the big encoders get slow enough to be unusable, so the
    ceiling drops to Fast regardless of how much memory is free.
    """
    free = free_ram_gb() if free_gb is None else free_gb
    have_gpu = has_gpu() if gpu is None else gpu
    installed = [t for t in _ORDER if available(t)]
    if not installed:
        return "high", label("high"), "no encoder weights installed"

    if not have_gpu:
        capped = [t for t in installed
                  if _ORDER.index(t) >= _ORDER.index(_CPU_MAX_TIER)] or installed[-1:]
        for tier in capped:
            if free >= _CPU_RAM_NEED.get(tier, 3.0):
                return (tier, label(tier),
                        f"no GPU detected — running '{label(tier)}' on the CPU "
                        f"({free:.1f} GB free)")
        smallest = capped[-1]
        return (smallest, label(smallest),
                f"no GPU detected and only {free:.1f} GB free — "
                f"'{label(smallest)}' needs {_CPU_RAM_NEED.get(smallest, 3.0):.1f} GB "
                f"on the CPU")

    for tier in installed:
        if free >= ram_need_gb(tier):
            best = installed[0]
            if tier == best:
                reason = f"{free:.1f} GB free — best available quality"
            else:
                reason = (f"{free:.1f} GB free — '{label(best)}' needs "
                          f"{ram_need_gb(best):.1f} GB, using a lighter encoder")
            return tier, label(tier), reason

    smallest = installed[-1]
    return (smallest, label(smallest),
            f"only {free:.1f} GB free — below every installed encoder's "
            f"requirement ({ram_need_gb(smallest):.1f} GB for '{label(smallest)}')")


def apply(force: bool = False) -> tuple:
    """Select a tier and publish it via SIGLIP_TIER. Returns (tier, label, reason).

    MUST run before grade_pipeline_v2 / lance_store / siglip2_encoder are
    imported — all three read SIGLIP_TIER at module scope.

    An explicit SIGLIP_TIER from the user/env always wins: auto-selection is a
    convenience, not a policy that overrides a deliberate choice.
    """
    explicit = os.environ.get("SIGLIP_TIER", "").strip().lower()
    if explicit in _TIERS and not force:
        return explicit, label(explicit), "set explicitly via SIGLIP_TIER"

    tier, lbl, reason = select()
    os.environ["SIGLIP_TIER"] = tier
    missing = [label(t) for t in _ORDER if not available(t)]
    if missing:
        reason += f" (not installed: {', '.join(missing)})"
    print(f"[tier] Quality: {lbl} — {reason}", flush=True)
    return tier, lbl, reason
