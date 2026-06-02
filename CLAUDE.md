# Street Story Curator — Frontier 2026 Architectural Contract

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

> **Note:** DeepSeek-R1-Distill GGUF entries have been removed — those weights were
> never downloaded and the code paths were dead. The primary grader is now
> Qwen2.5-VL-3B (transformers INT4, cached to `models/qwen_vlm/`). If that cache
> is absent the pipeline falls back to SpecVLM CLIP cosine similarity automatically.

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

```
scan_mode=True  → SpecVLM CLIP (always — speed over accuracy)
scan_mode=False → models/qwen_vlm/*.safetensors present?
                    yes → Qwen2.5-VL-3B  (direct vision, RAG context)
                    no  → SpecVLM CLIP   (cosine similarity fallback)
```

## Vector Store

LanceDB with **1536-d** IVF-PQ schema. Schema includes `reasoning_log` (string).
Auto-migrates from legacy 1152-d (SigLIP-So400M) on first run.

## Grade Buckets

Relative quantile per run (n ≥ 4): top 25% → Strong, bottom 20% → Weak, rest → Mid.
Absolute floor: Strong requires score ≥ 0.50 (uniformly bad batch gets no Strong).
Fallback for n < 4: Strong ≥ 0.60, Weak < 0.41.
- Weak ❌  ≤ 0.40

## PersonalHead / DPO

Endpoint: `POST /api/personal/update` (path1/grade1/path2/grade2).
Score blend: `0.80 * grader_score + 0.20 * head_score`.
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

## Rules for New Code

1. Never load two GPU models simultaneously — always `purge_vram()` between.
2. Never import legacy graders outside of `src/deprecated/`.
3. All embeddings are 1536-d; reject 1152-d vectors at the API boundary.
4. `asyncio.get_running_loop()` in async route handlers, never `get_event_loop()`.
5. No external network calls at runtime — fully offline app.
6. Use `frontier_config.is_force_frontier()` (function call) — never `from frontier_config import FORCE_FRONTIER` (captures value at import time).
