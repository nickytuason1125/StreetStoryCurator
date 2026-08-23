"""
run_profile.py — the single source of truth for "which encoder am I running?".

Why this module exists
----------------------
Every value that depends on the tier used to be re-derived independently in
five modules. In one afternoon that produced three separate bugs, all the same
shape:

  * encode_worker gated the ONNX *vision* graph on tier but not the *text*
    graph, so Balanced produced 1024-d images and 1536-d text and every
    ``embs @ probes.T`` was a shape error
  * siglip2_encoder's RAM floors were the giant's (3.0/4.0 GB), so Balanced
    refused to load despite a measured 2.05-2.23 GB peak
  * grade_pipeline_v2's encoder-source marker was global while the LanceDB
    tables were per tier, so every Pro<->Fast switch re-encoded the folder

siglip2_encoder even carried the comment "must mirror
encode_worker._HF_DIR_BY_TIER". A comment asking a human to keep two tables in
sync IS the bug, written down.

The fix is structural, not another patch: one frozen profile, built once per
process, that owns every tier-derived value. Adding a tier becomes one row here
instead of five places to remember.

Process model
-------------
This must stay a PURE function of environment + filesystem. The pipeline runs
work in isolated subprocesses (encode, IQA, detect), and each child derives its
own profile from ``SIGLIP_TIER``. Passing an object across the boundary would
not survive; deriving the same answer from the same inputs does.

Hard rule: NOTHING here may touch ``torch.cuda``. This module is imported by the
grade worker, which is the PARENT of the encode subprocess, and initialising
CUDA in the parent makes it fault with 0xC0000005 when the child exits. That
crash has been introduced twice. GPU presence comes from tier_select's cached
subprocess probe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

_ROOT = Path(__file__).resolve().parent.parent

TIERS = ("high", "mid", "low")


@dataclass(frozen=True)
class TierSpec:
    """Everything that differs between encoders, in one row."""
    tier: str
    label: str                 # user-facing; never a model name
    embed_dim: int
    model_tag: str             # open_clip / timm tag
    hf_dirname: str            # lean fp16 checkpoint
    oc_cache: str              # open_clip fallback cache dir
    ram_hard_gb: float         # refuse below this (measured peak, minus slack)
    ram_soft_gb: float         # "fits comfortably"
    cpu_ram_gb: float          # requirement with NO GPU (activations in RAM)


# MEASURED, not estimated. GPU floors come from real encode peaks; CPU figures
# from runs with the GPU hidden (8-16 photos, batch 4):
#   high  ONNX/GPU 1.20 GB;  CPU 6.5-7.6 GB and 11-14 s/img -> unusable on CPU
#   mid   GPU peak 2.05-2.23 GB;  CPU 4.52 GB, 2.94 s/img
#   low   GPU peak 1.46 GB;       CPU 2.83 GB, 1.05-1.24 s/img
_SPECS = {
    "high": TierSpec("high", "Pro", 1536, "ViT-gopt-16-SigLIP2-384",
                     "models/siglip2_hf_fp16", "models/siglip2",
                     3.0, 4.0, 8.0),
    "mid":  TierSpec("mid", "Balanced", 1024, "ViT-L-16-SigLIP2-384",
                     "models/siglip2_L_hf_fp16", "models/siglip2_L",
                     1.8, 2.4, 4.8),
    "low":  TierSpec("low", "Fast", 768, "ViT-B-16-SigLIP2-384",
                     "models/siglip2_B_hf_fp16", "models/siglip2_B",
                     1.2, 1.6, 3.0),
}


# ── settings registry ───────────────────────────────────────────────────────
# The 31 environment variables were undocumented tuning knobs scattered through
# the code, each parsed slightly differently and each a chance to crash on a
# typo. They are declared here once, with a type, a default and a reason.
# Garbage values fall back to the default rather than raising: a bad env var
# must never be able to stop a cull.
@dataclass(frozen=True)
class Setting:
    kind: type
    default: Any
    why: str


SETTINGS = {
    # encoder selection
    "SIGLIP_TIER":            Setting(str, "", "force a tier; empty = auto-select"),
    "SIGLIP_HF_DIR":          Setting(str, "", "override the lean checkpoint dir"),
    "SIGLIP_ENC_USE_OC":      Setting(bool, False, "force the heavy open_clip loader"),
    "FRAMEGRADE_ENCODER":     Setting(str, "", "'onnx' or 'torch'; empty = auto"),
    "FRAMEGRADE_ASSUME_GPU":  Setting(str, "", "skip the GPU probe (testing)"),
    "FRAMEGRADE_ORT_PROVIDERS": Setting(str, "", "comma-separated ONNX provider order"),
    # batching / chunking — all FIXED per device, never derived from free RAM,
    # which is what made 47 of 514 grades wobble between identical runs
    "SIGLIP_ENC_BATCH":       Setting(int, 0, "encode batch; 0 = per-device default"),
    "SIGLIP_ENC_CHUNK":       Setting(int, 0, "encode chunk; 0 = default"),
    "FRAMEGRADE_DEDUP_CHUNK": Setting(int, 0, "dedup similarity block size"),
    "FRAMEGRADE_DFINE_CHUNK": Setting(int, 0, "person-detector chunk"),
    "FRAMEGRADE_IQA_CHUNK":   Setting(int, 0, "IQA decode chunk (RAM-bounded)"),
    "FRAMEGRADE_LANCE_CHUNK": Setting(int, 500, "rows per LanceDB write"),
    "FRAMEGRADE_LANCE_RETENTION_DAYS": Setting(
        int, 7, "keep LanceDB versions this many days; history was unbounded"),
    "FRAMEGRADE_STREAM_CHUNK": Setting(int, 0, "streaming stage chunk"),
    # RAM floors
    "SIGLIP_MIN_FREE_RAM_GB": Setting(float, 0.0, "override soft floor; 0 = per-tier"),
    "SIGLIP_HARD_MIN_RAM_GB": Setting(float, 0.0, "override hard floor; 0 = per-tier"),
    "FRAMEGRADE_MIN_RAM_GB":  Setting(float, 0.0, "legacy alias of the soft floor"),
    "FRAMEGRADE_ANNOTATE_MIN_RAM_GB": Setting(float, 2.0, "skip annotation below this"),
    # quality gates
    "FRAMEGRADE_BLUR_MIN":    Setting(float, 0.0, "blur rejection threshold"),
    "FRAMEGRADE_FLAT_STD":    Setting(float, 0.0, "flat/void frame std threshold"),
    "FRAMEGRADE_FUNNEL":      Setting(bool, True, "percentile pre-cull funnel"),
    "FRAMEGRADE_FUNNEL_FRAC": Setting(float, 0.35, "fraction kept by the funnel"),
    "FRAMEGRADE_FUNNEL_CEIL": Setting(int, 0, "max photos through the funnel"),
    "FRAMEGRADE_STEP4E":      Setting(bool, True, "borderline re-judge pass"),
    "FRAMEGRADE_PH_WEIGHT_MAX": Setting(float, 0.35, "taste-model blend ceiling"),
    # decode
    "FRAMEGRADE_SHARED_DECODE": Setting(bool, True, "reuse decoded frames across stages"),
    "FRAMEGRADE_DFINE_DRAFT": Setting(bool, True, "DCT-domain draft decode for detection"),
    "FRAMEGRADE_LUM_DRAFT":   Setting(bool, True, "draft decode for luminance stats"),
    "FRAMEGRADE_GATE_WORKERS": Setting(int, 0, "early-gate worker threads; 0 = auto"),
    "FRAMEGRADE_IQA_SLICE":   Setting(int, 0, "IQA slice size"),
    # local LLM (replaces the Ollama HTTP dependency)
    "FRAMEGRADE_LOCAL_LLM_GGUF": Setting(str, "", "override the text GGUF path"),
    "FRAMEGRADE_LOCAL_LLM_MIN_RAM_GB": Setting(
        float, 0.0, "override the text-GGUF RAM floor; 0 = derive from weight size"),
    "FRAMEGRADE_DIRECTOR_POOL": Setting(
        int, 12, "candidates shown to the Art Director; prefill cost is superlinear"),
    "FRAMEGRADE_USE_RAG_CONCEPTS": Setting(
        bool, False, "inject book-derived RAG phrases into prompts; off by default"),
    "FRAMEGRADE_STORY_REVISION": Setting(
        bool, False, "contact-sheet critique pass; ~200s per iteration on CPU"),
    "FRAMEGRADE_STORY_VERDICT": Setting(
        bool, False, "Judge's Verdict narrative; ~92s on CPU"),
    # diagnostics
    "FRAMEGRADE_RAM_TRACE":   Setting(bool, False, "log RSS/peak per stage"),
    "FRAMEGRADE_FUSION_DUMP": Setting(bool, False, "dump per-photo fusion terms"),
}


def setting(name: str) -> Any:
    """Read a declared setting, tolerating garbage.

    A malformed value returns the default instead of raising — an env typo must
    not be able to stop a cull. Undeclared names raise, so a misspelled setting
    is caught at the call site rather than silently reading as unset forever.
    """
    if name not in SETTINGS:
        raise KeyError(f"undeclared setting {name!r} — add it to run_profile.SETTINGS")
    spec = SETTINGS[name]
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return spec.default
    raw = raw.strip()
    try:
        if spec.kind is bool:
            return raw.lower() not in ("0", "false", "no", "off")
        if spec.kind is int:
            v = int(float(raw))
            return v if v >= 0 else spec.default
        if spec.kind is float:
            v = float(raw)
            return v if v >= 0 else spec.default
        return raw
    except Exception:
        return spec.default


# ── the profile ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RunProfile:
    spec: TierSpec
    gpu: bool

    # -- identity -----------------------------------------------------------
    @property
    def tier(self) -> str:
        return self.spec.tier

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def embed_dim(self) -> int:
        return self.spec.embed_dim

    @property
    def model_tag(self) -> str:
        return self.spec.model_tag

    # -- paths --------------------------------------------------------------
    @property
    def hf_dir(self) -> str:
        """Lean fp16 checkpoint. SIGLIP_HF_DIR overrides (used by the builder's
        validation, which must run against a STAGING dir before promotion)."""
        return setting("SIGLIP_HF_DIR") or str(_ROOT / self.spec.hf_dirname)

    @property
    def oc_cache_dir(self) -> str:
        return str(_ROOT / self.spec.oc_cache)

    @property
    def has_lean_checkpoint(self) -> bool:
        if setting("SIGLIP_ENC_USE_OC"):
            return False
        return os.path.exists(os.path.join(self.hf_dir, "config.json"))

    # -- cache naming -------------------------------------------------------
    # 'high' keeps the historical unsuffixed names so existing caches built
    # before tiers existed stay valid. Anything else would silently invalidate
    # a user's whole library on upgrade.
    def cache_name(self, stem: str, ext: str) -> str:
        return f"{stem}{ext}" if self.tier == "high" else f"{stem}_{self.tier}{ext}"

    @property
    def lance_table(self) -> str:
        return "photos" if self.tier == "high" else f"photos_{self.tier}"

    @property
    def encoder_source(self) -> str:
        loader = "hf" if self.has_lean_checkpoint else "openclip"
        return f"{loader}-{self.tier}-{self.model_tag}"

    # -- resource decisions -------------------------------------------------
    @property
    def ram_hard_gb(self) -> float:
        override = setting("SIGLIP_HARD_MIN_RAM_GB")
        if override:
            return override
        if not self.has_lean_checkpoint:
            return 1.5          # heavy open_clip path; deliberately permissive
        return self.spec.ram_hard_gb

    @property
    def ram_soft_gb(self) -> float:
        override = (setting("SIGLIP_MIN_FREE_RAM_GB")
                    or setting("FRAMEGRADE_MIN_RAM_GB"))
        if override:
            return override
        if not self.has_lean_checkpoint:
            return 2.0
        return self.spec.ram_soft_gb

    @property
    def cpu_ram_gb(self) -> float:
        return self.spec.cpu_ram_gb

    @property
    def encode_batch(self) -> int:
        """Keyed on DEVICE, never on free RAM.

        Sizing from available memory changed batch COMPOSITION between runs;
        kernel selection then shifted embeddings in the last bits and 47 of 514
        borderline grades flipped. CPU measured 25% faster at 16 vs 4, with peak
        RSS unchanged.
        """
        return setting("SIGLIP_ENC_BATCH") or (8 if self.gpu else 16)

    # -- encoder path -------------------------------------------------------
    @property
    def onnx_vision(self) -> str:
        return str(_ROOT / "models" / "onnx" / "vision.onnx")

    @property
    def onnx_text(self) -> str:
        return str(_ROOT / "models" / "onnx" / "text.onnx")

    def onnx_enabled(self, *, text: bool = False) -> bool:
        """One rule for BOTH graphs. They must agree: the exported graph is the
        giant's, so using it for text on a 1024-d tier produced 1536-d probes
        against 1024-d images. Vision and text asking the same question is the
        point of putting this here.
        """
        sel = str(setting("FRAMEGRADE_ENCODER")).lower()
        if sel == "torch":
            return False
        graph = self.onnx_text if text else self.onnx_vision
        if not os.path.exists(graph):
            return False
        if text and not os.path.exists(os.path.join(self.hf_dir, "tokenizer.json")):
            return False
        if sel == "onnx":
            return True
        # Auto: only the 'high' tier has an exported graph, and only with a GPU
        # (on CPU it needs 6.5-7.6 GB and 11-14 s/image).
        return self.tier == "high" and self.gpu

    @property
    def ort_providers(self) -> list:
        pref = str(setting("FRAMEGRADE_ORT_PROVIDERS"))
        if pref:
            return [p.strip() for p in pref.split(",") if p.strip()]
        return ["CUDAExecutionProvider", "DmlExecutionProvider",
                "CoreMLExecutionProvider", "ROCMExecutionProvider",
                "CPUExecutionProvider"]

    def describe(self) -> str:
        return (f"tier={self.tier} ({self.label}) dim={self.embed_dim} "
                f"gpu={self.gpu} batch={self.encode_batch} "
                f"floor={self.ram_hard_gb}/{self.ram_soft_gb} GB "
                f"table={self.lance_table}")


_CURRENT: Optional[RunProfile] = None
_CURRENT_KEY: Optional[tuple] = None


def _env_key() -> tuple:
    """The inputs the profile is derived from."""
    return (str(setting("SIGLIP_TIER")).lower(),
            str(setting("SIGLIP_HF_DIR")),
            str(setting("FRAMEGRADE_ASSUME_GPU")),
            bool(setting("SIGLIP_ENC_USE_OC")))


def current(*, refresh: bool = False) -> RunProfile:
    """The profile for THIS process, derived from the environment.

    Cached, but keyed on the environment it was derived from — NOT cached
    blindly. tier_select.apply() publishes SIGLIP_TIER at grade start, and a
    plain cache would hand back a stale profile to anything that happened to
    import this module first. That is an import-order landmine of exactly the
    kind this module exists to remove, so the key is re-checked on every call
    and the profile rebuilt if the environment moved.

    The GPU flag is not part of the key: probing is the expensive part, hardware
    does not change mid-run, and tier_select caches the probe on disk anyway.
    """
    global _CURRENT, _CURRENT_KEY
    key = _env_key()
    if _CURRENT is not None and not refresh and key == _CURRENT_KEY:
        return _CURRENT
    tier = key[0] if key[0] in _SPECS else "high"
    gpu = _CURRENT.gpu if (_CURRENT is not None and not refresh
                           and key[2] == (_CURRENT_KEY or ("",) * 4)[2]) else _gpu_present()
    _CURRENT = RunProfile(spec=_SPECS[tier], gpu=gpu)
    _CURRENT_KEY = key
    return _CURRENT


def _gpu_present() -> bool:
    """Delegates to tier_select's CACHED SUBPROCESS probe.

    Never calls torch.cuda here: this module is imported by the parent of the
    encode subprocess, and initialising CUDA there faults the parent with
    0xC0000005 when the child exits.
    """
    env = str(setting("FRAMEGRADE_ASSUME_GPU"))
    if env:
        return env.lower() not in ("0", "false", "no")
    try:
        import tier_select
        return tier_select.has_gpu()
    except Exception:
        return False


def spec_for(tier: str) -> TierSpec:
    return _SPECS.get(tier, _SPECS["high"])
