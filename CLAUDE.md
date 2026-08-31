# FirstCut — formerly FrameGrade (renamed 2026-08-30)

> **Deprecation notice:** the product formerly known as **FrameGrade** is now
> **FirstCut**. All user-facing strings, docs, env vars (`FRAMEGRADE_*` →
> `FIRSTCUT_*`), the Tauri identity (`com.firstcut.app`), the Rust crate, and
> the PyInstaller spec (`FirstCut.spec`) were migrated in one pass
> (`scripts/deprecate_framegrade.py`). If an old script still sets a
> `FRAMEGRADE_*` variable, rename it to `FIRSTCUT_*` — the old names are
> deprecated and will not be recognized going forward.

# FirstCut — Frontier 2026 Architectural Contract

## Model Stack (Sequential, VRAM-safe)

| Phase | Model | Size | VRAM | Status |
|---|---|---|---|---|
| Embedding + dedup | SigLIP-2 ViT-g/14 NaFlex | 1536-d | ~1.5 GB | always runs |
| **Primary grader** | **Qwen2.5-VL-3B-Instruct INT4** | vision scoring | **~2.2 GB** | **runs when cached** |
| Fallback grader | SpecVLMPipeline (CLIP cosine sim) | instant | 0 GB extra | when Qwen absent |
| IQA heads | TOPIQ NR + MANIQA | technical quality | ~0.5 GB | always runs |
| Sequencing | NSGA-III (pymoo) | CPU | 0 GB | always runs |
| Preference | PersonalHead MLP 1536→256→64→1 | CPU | 0 GB | when weights present |
| Annotations / Critique | Qwen2.5-VL-2B GGUF | UI overlays only | ~1.5 GB | when GGUF present |

**Hard constraint: MAX 5.5 GB VRAM peak. Models never run concurrently.**

> **Note (corrected 2026-08-23):** the claim that DeepSeek entries "have been
> removed" was WRONG when written — `model_registry` still listed
> `deepseek-r1-8b-q5.gguf` as the text model and the 5.73 GB file was on disk.
> It is true now: DeepSeek was deleted on 2026-08-22 and replaced by
> **Qwen3-4B** (`bartowski/Qwen_Qwen3-4B-GGUF`, 2.5 GB, Apache-2.0) for Story
> and Competition selection, the Judge's Verdict and RAG extraction. On the
> 16 GB target the 8B never loaded at all — it needs ~6.6 GB free and 2.3-4.0 GB
> was measured — so Story mode had been silently score-sorting.
>
> The primary grader is unchanged: Qwen2.5-VL-3B on the opt-in Deep Grade path,
> falling back to SpecVLM CLIP.
>
> **Never name a weight file in code.** Ask `model_registry`. Five modules
> hardcoded `deepseek-r1-8b-q5.gguf` and a sixth hardcoded a `2b` vision
> filename the registry never shipped; after the swap a correct install reported
> models missing and a stale one reported success.

## VRAM Sequential Protocol

```
SigLIP-2.encode_images()          # dedup + archetype embeddings
  → VRAMManager.purge_vram()
  → QwenVLMGrader.grade_images_scored()   # primary: direct vision scoring
      OR SpecVLMPipeline.grade_images()   # fallback: CLIP cosine similarity
  → VRAMManager.purge_vram()
  → IQA heads (TOPIQ NR + MANIQA)
  → VRAMManager.purge_vram()
  → PersonalHead.adjust_scores()  # CPU only
```

`purge_vram()` must always call all three: `torch.cuda.empty_cache()`,
`torch.cuda.ipc_collect()`, and `gc.collect()`.

## RAG Context Injection

PDF reference books can be uploaded via the UI (`POST /api/rag/upload`).
Concept phrases are extracted and stored in `cache/rag_concepts.json`.
At grade time, up to 8 phrases are injected into the Qwen2.5-VL scoring prompt
as a rubric block — providing style-aware context without embedding computation.
When no PDFs are uploaded the prompt runs without the rubric block.

## Grading Path Decision Tree

Default is **SigLIP zero-shot** (SpecVLM CLIP). Qwen is opt-in via the
`deep_grade` flag (frontend "Deep Grade" toggle, default OFF). This keeps the
common path off the GPU-heavy Qwen stage — no Qwen VRAM footprint and none of
the WebView2 GPU-contention crash surface (0xC0000005 at Qwen load).

```
scan_mode=True             → SpecVLM CLIP, IQA skipped (ultra-fast niche pass)
deep_grade=True  (opt-in)  → Qwen2.5-VL-3B (direct vision, RAG) + TOPIQ IQA
deep_grade=False (DEFAULT) → SpecVLM CLIP zero-shot + TOPIQ IQA
```

`deep_grade` is threaded server.py `GradeRequest` → grade_worker → `run_v2`.
When `deep_grade=True` but Qwen weights are missing, it still falls back to
SpecVLM CLIP. TOPIQ IQA runs for both grade modes; only `scan_mode` skips it.

## Vector Store

LanceDB with **1536-d** IVF-PQ schema. Schema includes `reasoning_log` (string).
Auto-migrates from legacy 1152-d (SigLIP-So400M) on first run.

## Grade Buckets

Absolute thresholds — the grade reflects the photo itself, not its rank in the batch:
- Strong ✅  score ≥ 0.60
- Mid ⚠️    0.41 ≤ score < 0.60
- Weak ❌   score < 0.41

Quantile/historical-anchor calibration was removed (2026-06): it forced a
~25% Strong / 20% Weak split on every run regardless of actual quality.
Also removed: the Step-4e batch score stretch (rescaled clustered batches onto
[0.18, 0.88]) and the 0.62–0.68 archetype Strong-floors (now 0.50–0.55 Mid-band
penalty protection only). Do not reintroduce per-batch relative grading or any
floor ≥ 0.60 — a photo reaches Strong only on actual fused quality.

## PersonalHead / DPO

Endpoint: `POST /api/personal/update` (path1/grade1/path2/grade2).
Score blend (grade_pipeline_v2 Step 5): **confidence-adaptive** —
`final = (1-w)*grader + w*head`, where `w = 0.20 + (ceil-0.20)*conf` and
`conf = |head-0.5|/0.5`. A neutral head (~0.5, i.e. a genre it hasn't learned)
collapses `w` to the 0.20 floor → identical to the legacy flat 0.80/0.20, so it
can never regress; a confident head rises toward `ceil` (env
`FIRSTCUT_PH_WEIGHT_MAX`, default 0.35, clamped ≤0.60) so taste becomes a
first-class vote only where it has coverage. Adding a few ratings in a new genre
raises the head's confidence there → grades shift toward the user's taste
automatically, with zero effect where it hasn't learned.
Weights persist to `models/personal_head.pt` via `PersonalHead.save()`.

## Deprecated Graders

Legacy models (Q-Align, NIMA ONNX, MobileViT, DINOv2-small) live in
`src/deprecated/`. Import from there raises `DeprecationWarning`.
Production code must NOT import from `qalign_grader`, `onealign_scorer`,
or `lightweight_analyzer` directly — use `grade_pipeline_v2.run_v2()`.

## Frontend Reasoning Display

The right panel has three tabs when graded: Breakdown · Analysis · EXIF.
- **Analysis tab**: merged tab showing score, verdict, per-aspect observation rows,
  best/weakest footer, and jury critique fallback. Displays `VERIFIED · 7B` badge
  when `photo.is_verified === true`. Contains a "Draw on image" / "Hide overlay"
  toggle button (Eye/EyeOff) that controls `isAuditModeActive` — when active,
  the `reasoningOverlayUrl` annotation PNG is overlaid on the photo in the viewer.

## --force-frontier Flag

Activated by `python main.py --force-frontier` (or `FORCE_FRONTIER=1` env var).

Pre-flight sequence (before server starts):
1. `check_model_integrity()` — aborts if SigLIP-2 or Vision-R1-7B weights absent.
2. `validate_vram_overhead(5.0)` — aborts if free VRAM < 5.0 GB.

Runtime enforcement (`src/frontier_config.py`):
- `grade_pipeline_v2`: raises `RuntimeError` instead of falling back to QAlign/NIMA/V1.
- `grade_pipeline_v2`: raises `RuntimeError` if encoder produces 1152-d (SigLIP So400M fallback).
- `lance_store`: drops 1152-d table with a FRONTIER ENFORCEMENT log message.
- Frontend: Breakdown tab displays full reasoning text + VERIFIED badge instead of percentage bars.

`GET /api/config` returns `{"force_frontier": bool}` for the frontend to read.

Tests: `tests/test_frontier_lock.py` covers all enforcement paths.

## Ratings Are Never Ground Truth (2026-08-30)

**Product rule: star ratings must never change how any image is graded —
for anyone.** The 794-photo rating baseline (LX3/TPE, tagged `tpe_master`
in `cache/user_ratings.json`) was explicitly PLACEHOLDER data. Ratings are
collected in `ratings_store` purely as *measurement and learning data*;
the grader stays objective and absolute by default.

Measured against the placeholder baseline (`scripts/master_backtest.py`):
incumbent grader ρ = +0.87, bands monotone, aspect signal =
Composition +0.64 · Narrative +0.51 · Technical +0.38 · Lighting +0.33 ·
**Human/Culture −0.15**. This informed the tooling below; it does not
steer grading.

The master algorithm follows a TWO-PHASE contract:

| Phase | What happens | Rating influence |
|---|---|---|
| **BUILD** (offline) | challengers train on a rating baseline (`scripts/master_backtest.py --fit`, `POST /api/master/retrain`, background auto-refit every 25 new ratings); each fit runs a champion/challenger exam on held-out photos and writes a record to `cache/master_judge.json` | records only — grades are untouched |
| **SHIP** (deliberate, operator-run) | a challenger that WON its exam is baked into `data/master_judge_defaults.json` via `scripts/promote_master_judge.py` / `promote_to_shipped()` — refused for anything unpromoted, stale, or incomplete | the curated training data becomes part of the algorithm |
| **RUN** (every install) | the SHIPPED master judge grades BY DEFAULT — no flags, no ratings, like the shipped encoder weights; precedence + guards in `master_judge.active()` | zero |

Opt-in flags (default OFF) for non-shipped influence:

| Flag | Enables | Guardrails |
|---|---|---|
| `FIRSTCUT_PERSONAL_TASTE=1` | PersonalHead taste blend (Step 5) | confidence-adaptive 0.20–0.70 weight, tier-mismatch guard |
| `FIRSTCUT_MASTER_JUDGE=1` | local cache challenger + background auto-refit | champion/challenger promotion required |
| `FIRSTCUT_MASTER_ANCHORS=1` | human-anchored calibration ruler (`cache/master_anchors.json`) | probe-fingerprint freshness + lo/hi span + 3★ monotonicity refusal |
| `FIRSTCUT_MASTER_JUDGE_OFF=1` | kill switch — disables the shipped master judge too | — |

Non-negotiables:
- Never default-on the opt-in flags; never hand-edit `promoted` to true;
  never ship an unpromoted record. An unpromoted `cache/master_judge.json`
  is a RECORD of a lost challenge, not a bug.
- Precedence in `active()`: kill switch → opted-in local challenger →
  shipped master → nothing. The shipped master needs no flags; a local
  challenger always needs its opt-in.
- The stacked design `[machine score, 5 aspects]` is load-bearing: the
  aspect-only challenger lost 0.78 vs 0.87. "Aesthetic" is NOT a judge
  feature — it IS the base grader score and doesn't exist in breakdowns
  at grade time.
- Every fit appends to the `history` list in cache/master_judge.json
  (last 20: n, rho_holdout, rho_baseline, promoted) — the ρ trend IS the
  improvement log.
- The placeholder-trained PersonalHead weights were parked as
  `cache/personal_head.placeholder.bak.pt/.npz` (recoverable by renaming
  back); grading never loads them unless the taste flag is on.

## Reliability & Benchmark Tooling (2026-08-30)

| Tool | What it measures |
|---|---|
| `scripts/benchmark.py [--scan|--deep|--skip-cull|--full-story]` | cull wall time, s/image, peak process-tree RSS, per-stage timeline; MOGCO sequencing determinism (run twice) → `reports/benchmark_report.md` |
| `scripts/reliability_check.py` | repeat-run score determinism (0 drift tolerated), process hygiene (no stray workers, `grading.lock` cleaned), corrupt-file resilience, empty-folder behaviour, RAM floor, **disk head-room** → `reports/reliability_report.md` |

Measured on this machine (RTX 3060 Laptop, 16.8 GB RAM, 100-photo dataset):
full cull **56 s / 0.56 s per photo / 2.75 GB peak**; scan cull **23 s /
0.23 s per photo / 1.01 GB peak**; sequencing deterministic, 0.22 s.

Reliability findings baked into code that day:
- **A full app-drive fails persistence silently** — with 0 bytes free, a
  grade completed and then `catalog_store.merge_write` + Lance compaction
  died with OSError Errno 28 (work lost). The server grade path now has a
  **hard DISK gate** (mirror of the RAM gate): refuses below 1 GB free on
  the app drive with a clear message. Do not remove it.
- High-tier culling needs ≥ ~1.8 GB *available* RAM just for SigLIP-2 — on
  a 16 GB desktop under normal load this is the binding constraint, not
  CPU/GPU. Scan mode (1 GB peak) is the fallback for loaded machines.
- A corrupt .jpg among good ones is dropped with a logged warning; the run
  completes and the good photos grade. Never "fix" this by widening the
  try/except to hide other errors.

## Story / Competition (2026-08-23)

Selection runs over the WHOLE graded pool via `src/story_selector.py`, not the
40 nearest neighbours of the top-scoring frame. Length is the user's choice,
4-10, via the existing `n_target`. Cohesion is REPORTED, never gated: measured
floors were a cliff (0.55 and 0.80 both returned 10 every time; 0.85 and 0.88
returned 1), and no floor survives without grading on a curve.

Two stages are OPT-IN because they were measured, not guessed:

| Setting | Default | Cost when on |
|---|---|---|
| `FIRSTCUT_STORY_REVISION` | off | ~200s per iteration (170s to encode one contact sheet on CPU) |
| `FIRSTCUT_STORY_VERDICT` | off | ~92s for a 200-token narrative |

With both off a Story run is **57.5s** end to end. With them on it did not
return in ten minutes.

Also measured, and load-bearing:
- Grammar-constrained decoding DEGRADES selection: 11/14 against 14/14
  unconstrained, biasing toward small ids. Do not add it back.
- Manifest size drives latency superlinearly: 25 candidates 36.1s, 12 candidates
  4.7s. `FIRSTCUT_DIRECTOR_POOL` defaults to 12.
- Shot type does NOT discriminate in a street library: largest face measured was
  0.88% of frame against an 8% "close" boundary. Do not build narrative roles on
  camera distance.

## Rules for New Code

1. Never load two GPU models simultaneously — always `purge_vram()` between.
2. Never import legacy graders outside of `src/deprecated/`.
3. All embeddings are 1536-d; reject 1152-d vectors at the API boundary.
4. `asyncio.get_running_loop()` in async route handlers, never `get_event_loop()`.
5. No external network calls at runtime — fully offline app.
6. Use `frontier_config.is_force_frontier()` (function call) — never `from frontier_config import FORCE_FRONTIER` (captures value at import time).
