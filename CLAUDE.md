# Street Story Curator — Frontier 2026 Architectural Contract

## Model Stack (Sequential, VRAM-safe)

| Phase | Model | Size | VRAM |
|---|---|---|---|
| Embedding | SigLIP-2 ViT-g/14 NaFlex FP8 | 1536-d | ~1.5 GB |
| Draft grading | DeepSeek-R1-Distill-Qwen-1.5B INT4 | always runs | ~1.0 GB |
| Verify grading | DeepSeek-R1-Distill-Qwen-7B INT4 | threshold-gated | ~3.5 GB |
| Sequencing | NSGA-III (pymoo) | CPU | 0 GB |
| Preference | PersonalHead MLP 1536→256→64→1 | CPU | 0 GB |

**Hard constraint: MAX 5.5 GB VRAM peak. Models never run concurrently.**

## VRAM Sequential Protocol

```
SigLIP-2.encode_images()
  → VRAMManager.purge_vram()          # empty_cache + ipc_collect + gc.collect
  → SpecVLM.grade_all()
  → VRAMManager.purge_vram()
  → PersonalHead.adjust_scores()      # CPU only
```

`purge_vram()` must always call all three: `torch.cuda.empty_cache()`,
`torch.cuda.ipc_collect()`, and `gc.collect()`.

## Priority Gate Threshold

`DRAFT_CONFIDENCE_THRESHOLD = 0.85` in `src/specvlm_pipeline.py`.
The verifier (7B) fires only when draft confidence **≤ 0.85**.

## Vector Store

LanceDB with **1536-d** IVF-PQ schema. Schema includes `reasoning_log` (string).
Auto-migrates from legacy 1152-d (SigLIP-So400M) on first run.

## Grade Buckets

- Strong ✅ ≥ 0.60
- Mid ⚠️  0.41–0.59
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

The right panel has four tabs when graded: Breakdown · Analysis · Reasoning · EXIF.
- **Reasoning tab**: shows `photo.reasoning_log` (raw VLM chain-of-thought).
  Displays `VERIFIED · 7B` badge when `photo.is_verified === true`.
- **Analysis tab**: shows `reasoning_log` falling back to `critique`.

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
