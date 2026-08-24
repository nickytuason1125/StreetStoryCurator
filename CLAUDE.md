# FrameGrade — Frontier 2026 Architectural Contract

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
`FRAMEGRADE_PH_WEIGHT_MAX`, default 0.35, clamped ≤0.60) so taste becomes a
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

## Story / Competition (2026-08-23)

Selection runs over the WHOLE graded pool via `src/story_selector.py`, not the
40 nearest neighbours of the top-scoring frame. Length is the user's choice,
4-10, via the existing `n_target`. Cohesion is REPORTED, never gated: measured
floors were a cliff (0.55 and 0.80 both returned 10 every time; 0.85 and 0.88
returned 1), and no floor survives without grading on a curve.

Two stages are OPT-IN because they were measured, not guessed:

| Setting | Default | Cost when on |
|---|---|---|
| `FRAMEGRADE_STORY_REVISION` | off | ~200s per iteration (170s to encode one contact sheet on CPU) |
| `FRAMEGRADE_STORY_VERDICT` | off | ~92s for a 200-token narrative |

With both off a Story run is **57.5s** end to end. With them on it did not
return in ten minutes.

Also measured, and load-bearing:
- Grammar-constrained decoding DEGRADES selection: 11/14 against 14/14
  unconstrained, biasing toward small ids. Do not add it back.
- Manifest size drives latency superlinearly: 25 candidates 36.1s, 12 candidates
  4.7s. `FRAMEGRADE_DIRECTOR_POOL` defaults to 12.
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
