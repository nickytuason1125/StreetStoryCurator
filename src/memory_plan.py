"""
MemoryPlan — the ONE module that owns cull admission and degradation.

Before 2026-09-07 there were four disconnected RAM checks that could disagree:
  1. the router's admission gate (required_ram_gb → 503),
  2. the in-stream pre-spawn gate (encoder floor + 0.2 → SSE refuse),
  3. a warn-only whole-cull check inside the pipeline (2.5 GB),
  4. the SigLIP encoder's own soft/hard floors (adaptive, batch reduction).
Each was individually defensible; together they produced the worst outcome —
a hard 503 refusal in a situation where the codebase already contained every
ingredient of a graceful downgrade (scan mode, batch reduction, checkpoint
resume). This module replaces 1–3 with a single decision:

  pick the RICHEST plan that fits the machine RIGHT NOW; refuse only when
  even the minimal plan (Scan, measured 1.01 GB process tree, 2.0 GB
  admitted) cannot fit.

Deliberately NOT in the ladder: tier downgrades. The active tier selects the
Lance table and the embedding space (photos_high vs photos_low, different
embed dims) — silently switching tiers per run would invalidate the
incremental cache and mix subtly different scores into one library. Tier is
a user choice (Lite mode); memory degradation happens within it.

Failpoint for tests: FIRSTCUT_OOM_FAILPOINT=<stage> raises MemoryError at
that stage ("admission", "encode", "judge", "save").
"""
from __future__ import annotations

import os

# Measured constants — see run_profile._RAM_NEED_GB header and
# reports/benchmark_report.md. Nothing here is invented.
SCAN_TREE_MEASURED_GB = 2.0   # scan process tree measured 1.01 GB peak; 2× headroom
TIGHT_BAND_GB = 0.8           # free between need and need+this → "tight but viable" advisory

_STAGES = ("admission", "encode", "judge", "save")


def _global_memory_status():
    """(avail_phys_gb, avail_commit_gb) via GlobalMemoryStatusEx, or None.

    avail_commit is the paging-file headroom — the resource that actually
    ran out on 2026-09-07: "The paging file is too small for this operation
    to complete" is a COMMIT failure, not a physical-RAM one, and it hits
    while Windows is still reporting physical pages available.
    """
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return (st.ullAvailPhys / 1e9, st.ullAvailPageFile / 1e9)
    except Exception:
        pass
    return None


def free_ram_gb() -> float | None:
    """Gigabytes available for new allocations right now, or None when it
    cannot be read.

    Returns the SCARCER of physical RAM and commit headroom: a machine can
    report physical pages free while the commit limit is exhausted (the
    pagefile cannot grow fast enough mid-burst) — gating on physical alone
    admitted runs that then died mid-import with "The paging file is too
    small". psutil (physical only) remains the fallback; the encoder's own
    floors still protect the machine when even this fails.
    """
    g = _global_memory_status()
    if g is not None:
        return min(g)
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        pass
    return None


def commit_headroom_gb() -> float | None:
    """Paging-file headroom (commit limit − commit charge), or None.

    When this is much smaller than physical free RAM, the pagefile is the
    bottleneck — the fix is a fixed-size pagefile (16–32 GB), not closing
    apps.
    """
    g = _global_memory_status()
    return g[1] if g is not None else None


def _need_full_gb(n_photos: int) -> float:
    import run_profile
    return run_profile.required_ram_gb(n_photos, scan_mode=False)


def _need_scan_gb() -> float:
    import run_profile
    return run_profile.required_ram_gb(0, scan_mode=True)


def _hard_floor_gb() -> float:
    """The active encoder's refuse-below floor (Lite HF ≈ 1.2–1.6 GB)."""
    try:
        from siglip2_encoder import _default_ram_floor_gb
        return _default_ram_floor_gb()
    except Exception:
        return 1.5


def min_admission_gb() -> float:
    """The LAST rung's cost — below this, genuinely nothing fits."""
    return _hard_floor_gb() + 0.3


def plan_for(n_photos: int, requested_scan: bool) -> dict | None:
    """The richest plan that fits the machine right now.

    Ladder, richest first:
      full quality            → measured whole-cull figures (3.8–7.0 GB)
      Scan                    → measured scan tree, 2.0 GB admitted
      Scan, reduced batch     → encoder hard floor + 0.3 (the old pre-spawn
                                gate's territory — the encoder degrades to
                                batch=2 instead of refusing)

    Returns {"plan", "scan_mode", "need_gb", "free_gb", "degraded", "note"}
    — "plan" names the rung; "degraded" True means the request was
    downgraded (callers must honour plan["scan_mode"]); None means even the
    reduced-batch Scan cannot fit and the request should be refused.
    """
    free = free_ram_gb()
    if free is None:
        # Cannot measure → admit, same fail-open semantics the old gate had,
        # but LOUD: the encoder's own floors still protect the machine.
        return {"plan": "scan" if requested_scan else "full",
                "scan_mode": requested_scan, "need_gb": None, "free_gb": None,
                "degraded": False,
                "note": "Free RAM could not be measured — relying on the encoder's own floors."}

    ladder = []
    if not requested_scan:
        ladder.append(("full", False, _need_full_gb(n_photos)))
    ladder.append(("scan", True, _need_scan_gb()))
    ladder.append(("scan+reduced-batch", True, min_admission_gb()))

    for i, (name, scan_mode, need) in enumerate(ladder):
        if free >= need:
            degraded = i > 0
            note = ""
            if degraded:
                if name == "scan+reduced-batch":
                    # Set the batch reduction HERE, in the server process, so
                    # the runner child inherits it from its very first spawn.
                    os.environ["SIGLIP_ENC_BATCH"] = "2"
                    note = (f"RAM is very tight ({free:.1f} GB free) — running a "
                            f"Scan with a reduced encode batch (needs ~{need:.1f} GB). "
                            f"Slower, but everything is still graded.")
                else:
                    note = (f"RAM is tight ({free:.1f} GB free) — downgraded to a Scan "
                            f"({need:.1f} GB needed instead of {ladder[0][2]:.1f} GB for full "
                            f"quality). Everything is still graded; Deep-Grade verification "
                            f"is skipped. Re-run when more RAM is free for full quality.")
            return {"plan": name, "scan_mode": scan_mode, "need_gb": need,
                    "free_gb": free, "degraded": degraded, "note": note}

    return None  # even the reduced-batch Scan does not fit → caller refuses


def memory_checkpoint(stage: str, progress=None) -> None:
    """Stage-boundary RAM check for the running pipeline.

    - failpoint (tests): FIRSTCUT_OOM_FAILPOINT == stage → injected MemoryError.
    - free < encoder hard floor → MemoryError with a recoverable message:
      everything graded so far is checkpointed (Lance rows + IQA caches), so
      the run aborts CLEANLY and a re-run resumes instead of restarting.
    - between hard and hard+0.8 → shrink the encode batch and say so.
    """
    if stage not in _STAGES:
        raise ValueError(f"memory_checkpoint: unknown stage {stage!r}")

    failpoint = os.environ.get("FIRSTCUT_OOM_FAILPOINT", "").strip()
    if failpoint == stage:
        raise MemoryError(
            f"[failpoint] injected OOM at the '{stage}' stage — progress is "
            f"saved; re-run to resume from the checkpoint.")

    free = free_ram_gb()
    if free is None:
        return

    try:
        from siglip2_encoder import _default_ram_floor_gb
        hard = _default_ram_floor_gb()
    except Exception:
        hard = 1.5

    if free < hard:
        raise MemoryError(
            f"Out of memory at the '{stage}' stage: {free:.1f} GB free, "
            f"need ~{hard:.1f} GB to keep going. Everything graded so far is "
            f"saved — close a few apps and re-run; the cull resumes from its "
            f"checkpoint instead of starting over.")

    if free < hard + TIGHT_BAND_GB:
        msg = (f"Tight RAM at '{stage}': {free:.1f} GB free — shrinking the "
               f"encode batch for the rest of this stage.")
        print(f"[memory_plan] {msg}", flush=True)
        if os.environ.get("SIGLIP_ENC_BATCH") != "2":
            os.environ["SIGLIP_ENC_BATCH"] = "2"
        if progress is not None:
            try:
                progress(0.0, f"Low memory ({free:.1f} GB free) — running slower to stay safe")
            except Exception:
                pass

