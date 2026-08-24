# FrameGrade — System Architecture

> Lead Systems Architect Reference — generated 2026-05-26
>
> This document describes how the photo curation pipeline currently operates end-to-end:
> every model, every data hop, every feedback loop.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Model Stack](#2-model-stack)
3. [Life of a Single Photograph](#3-life-of-a-single-photograph)
4. [DPO Star Rating Feedback Loop](#4-dpo-star-rating-feedback-loop)
5. [Eye Feature Rendering Layer](#5-eye-feature-rendering-layer)
6. [PDF RAG Aesthetic Rubric](#6-pdf-rag-aesthetic-rubric)
7. [NSGA-III Story Sequencer](#7-nsga-iii-story-sequencer)
8. [Backend Failsafes](#8-backend-failsafes)
9. [Calibration Script](#9-calibration-script)
10. [Component Reference](#10-component-reference)

---

## 1. System Overview

```
+-----------------------------------------------------------------------------+
|  Browser (React + Vite)                                                     |
|                                                                             |
|   CreativeDirection panel --> PDF RAG upload  --> /api/rag/upload           |
|   Gallery grid -----------> star click        --> /api/personal/star        |
|   Detail panel -----------> Eye toggle        --> CSS opacity crossfade     |
|   Scan button ------------> /api/scan          --> triggers full pipeline   |
+------------------------+----------------------------------------------------+
                         | HTTP (axios)  port 8000
+------------------------v----------------------------------------------------+
|  FastAPI  (server.py)                                                       |
|                                                                             |
|  asyncio.Lock  gpu_lock  -- serialises ALL GPU work server-wide             |
|  AsyncQueueManager       -- deduplicates concurrent /api/scan requests      |
|                                                                             |
|  /api/scan ------------> grade_pipeline_v2.run_pipeline()                   |
|  /api/rag/upload -------> pdf_rag.ingest_pdf()  +  cache bust               |
|  /api/personal/star ----> PersonalHead.update()  +  DPO queue               |
|  /api/sequence ---------> nsga3_sequencer.sequence()                        |
|  /api/vision/story -----> vision_story_mode.run()                           |
+------------------------+----------------------------------------------------+
                         |
+------------------------v----------------------------------------------------+
|  GPU / CPU Workers  (all serialised through gpu_lock)                       |
|                                                                             |
|  SigLIP-2 896px         TOPIQ-NR           MUSIQ                            |
|  Qwen2-VL-7B            YOLOv8             Phi-4-mini-reasoning             |
|  LanceDB vector store   PersonalHead MLP   BackgroundDPOTrainer             |
+-----------------------------------------------------------------------------+
```

The server is a single-process FastAPI application. Every GPU-touching operation
acquires `gpu_lock` before proceeding; this prevents VRAM thrashing on a single
consumer GPU.

---

## 2. Model Stack

| Role | Model | File / Module | Runs on |
|---|---|---|---|
| Visual embedding | SigLIP-2 SO400M 896px | `src/siglip2_encoder.py` | GPU |
| Aesthetic regressor (no-ref IQR) | TOPIQ-NR | `src/grade_pipeline_v2.py` | GPU |
| Aesthetic regressor (multi-scale) | MUSIQ | `src/grade_pipeline_v2.py` | GPU |
| Object detection / saliency | YOLOv8 (threshold 0.35) | `src/grade_pipeline_v2.py` | GPU |
| VLM caption + aspect scoring | Qwen2-VL-7B-Instruct | `src/grade_pipeline_v2.py` | GPU |
| Purism / noise judge | Phi-4-mini-reasoning | `src/grade_pipeline_v2.py` | GPU |
| Final verdict | Llama-3-8B-Instruct (jury) | `src/grade_pipeline_v2.py` | GPU |
| Personal taste head | MLP 1536->256->64->1 | `src/personal_head.py` | CPU |
| DPO fine-tuning | QLoRA Rank-4 adapter | `src/background_dpo_trainer.py` | GPU (background) |
| RAG concept extraction | DeepSeek-R1-8B GGUF | `src/pdf_rag.py` | CPU/GPU |
| Story sequencing | NSGA-III + CLIP | `src/nsga3_sequencer.py` | CPU |
| Vision story analysis | CLIP + Qwen + TSP | `src/vision_story_mode.py` | GPU |

---

## 3. Life of a Single Photograph

Nine phases from raw file on disk to a graded, embedded, sequenced entry.

### Phase 1 — Ingestion

`server.py` receives `/api/scan`. `AsyncQueueManager` deduplicates concurrent
requests (second caller waits for the first scan to finish and reuses its result).
The scanner walks the configured `PHOTO_DIR` and collects all JPEG/RAW paths.
Previously-seen files whose `mtime` has not changed are skipped (LanceDB hit check).

### Phase 2 — YOLO Saliency

YOLOv8 runs at threshold 0.35. The top detection box is used to compute
`subject_area_pct` (bounding box area / frame area x 100) and centroid position.
`person_kill_switch` fires if a person occupies > 60% of the frame for street
photos set to strict-purist mode.

### Phase 3 — SigLIP-2 Embedding

`siglip2_encoder.SigLIP2Encoder` processes the image at 896 x 896 px and returns
a 1536-dim L2-normalised embedding. The text embedding cache (`_text_emb_cache`)
is built once per process lifetime (or after a cache bust) from `_POS_PROMPTS` +
`_NEG_PROMPTS` + `_ASPECT_PROMPTS`, all augmented at build time with any PDF RAG
phrases loaded from `cache/rag_concepts.json`.

```
image_emb (1536)  --> cos_sim --> pos_text_embs  --> aesthetic_score
                  --> cos_sim --> neg_text_embs  --> penalty
                  --> cos_sim --> aspect_embs    --> aspect_scores[]
```

RAM preflight check (`src/siglip2_encoder.py`) aborts gracefully if available RAM
< 4 GB to prevent the OOM crash observed in production (C-level segfault at 4%).

### Phase 4 — Vision Regression Stack

Three independent quality regressors run sequentially (all serialised through
`gpu_lock`):

| Regressor | Weight | What it measures |
|---|---|---|
| TOPIQ-NR | 0.35 | No-reference IQR quality (sharpness, noise, exposure) |
| MUSIQ | 0.35 | Multi-scale aesthetic quality |
| SigLIP-2 semantic | 0.30 | Aesthetic concept alignment |

Weighted sum -> `raw_score in [0, 1]`. `SIM_THRESH = 0.88` gates semantic
similarity; scores below it are zeroed for the YOLO-detected subject area.

### Phase 5 — PersonalHead Score

`personal_head.PersonalHead` (MLP 1536->256->64->1, MarginRankingLoss) loads
`cache/personal_head.pt` and scores the same 1536-dim embedding. The final
blended score is:

```
final = 0.80 x raw_score  +  0.20 x personal_score
```

On a fresh install with no star data, `personal_score ~= 0.5` so the blend
has negligible effect until the user starts rating photos.

### Phase 6 — Grade Bucketing

Grades are assigned relative to the current batch, not absolute thresholds:

| Grade | Criterion |
|---|---|
| Strong | `final_score` in top 25% of batch AND >= 0.50 floor |
| Mid | Middle 55% |
| Weak | Bottom 20% of batch |

This means grade labels drift with batch composition, which is intentional:
the user should see a distribution of grades in every scan, not a sea of
"Strong" on a technically easy batch.

### Phase 7 — LanceDB Storage

Results are upserted into a LanceDB table keyed by file path. Stored fields:

- `path`, `embedding` (1536-dim float32 vector)
- `score`, `personal_score`, `final_score`, `grade`
- `aspect_scores` (dict), `spatial_facts` (dict: `subject_area_pct`, `h_gap`)
- `yolo_score`, `caption`, `eye_overlay_url`
- `stars` (user rating, 0-5), `approved_by_vision` (bool)

### Phase 8 — NSGA-III Sequencing

`nsga3_sequencer.sequence()` runs a multi-objective evolutionary algorithm over
the graded photo set to assign narrative slot roles. See Section 7 for slot
scoring weights.

### Phase 9 — Annotation Queue

`vision_story_mode.run()` processes the NSGA-III sequence through a
CLIP + Qwen2-VL + TSP pipeline, generating story annotations, h_gap contrast
measurements, and `eye_overlay_url` paths for the frontend overlay toggle.

---

## 4. DPO Star Rating Feedback Loop

```
User clicks 4 stars on a photo
        |
        v
handleSetStars()  -->  POST /api/personal/star  { path, stars }
                                |
                        server.py resolves star -> grade label
                        (5 stars / 4 stars -> Strong, 3 stars -> Mid, 2/1 -> Weak)
                                |
                        +-------+------------------------------------------+
                        |  PersonalHead.update() (immediate)               |
                        |  MarginRankingLoss on                            |
                        |  (this_emb, star_grade) vs                      |
                        |  random contrastive from all_rows               |
                        |  Re-scores ALL photos in LanceDB                |
                        +--------------------------------------------------+
                                |
                        INSERT INTO dpo_prefs (path, old_grade, new_grade,
                                               embedding, created_at)
                                |
                        BackgroundDPOTrainer wakes when
                        untrained rows >= BATCH_THRESHOLD (20)
                                |
                        +-------v------------------------------------------+
                        |  Hard Negative Mining selection                   |
                        |  40% of batch: delta=2 rows only                  |
                        |  (Strong<->Weak flips, highest signal)            |
                        |  60% of batch: delta DESC, recency               |
                        +---------------------------------------------------+
                                |
                        QLoRA Rank-4 fine-tune of SpecVLM adapter
                        (loaded back into grade_pipeline_v2 on next scan)
```

### Hard Negative Mining SQL Detail

```sql
-- 8 hard-negative slots (delta = 2: Strong <-> Weak only)
SELECT * FROM dpo_prefs
WHERE trained = 0
  AND ABS(CASE new_grade WHEN 'Strong' THEN 2
                         WHEN 'Mid'    THEN 1
                         ELSE 0 END
        - CASE old_grade WHEN 'Strong' THEN 2
                         WHEN 'Mid'    THEN 1
                         ELSE 0 END) = 2
ORDER BY created_at DESC
LIMIT 8;

-- 12 soft slots (any remaining untrained, by signal strength)
SELECT * FROM dpo_prefs
WHERE trained = 0
  AND id NOT IN (<hard_negative_ids>)
ORDER BY delta DESC, created_at DESC
LIMIT 12;
```

Strong<->Weak flips carry the highest gradient signal. Filling 40% of every
training batch with these examples prevents the adapter from over-fitting to
easy Mid<->Weak distinctions.

---

## 5. Eye Feature Rendering Layer

Each photo processed by `vision_story_mode.py` may produce an `eye_overlay_url`
— a path to a PNG that annotates the judge's critique (saliency heatmap,
bounding boxes, text overlay) directly on the image.

The frontend renders two stacked `<img>` elements in the detail panel:

```
+-----------------------------------------+
|  Base photo  (always rendered, z=0)     |
|  +---------------------------------+    |
|  |  Eye overlay  (z=1)             |    |
|  |  opacity: 0 -> 1  (350ms ease) |    |
|  +---------------------------------+    |
|                          +-----------+  |
|                          |  Eye btn  |  |
|                          +-----------+  |
+-----------------------------------------+
```

Key implementation details:

- The overlay `<img>` is always in the DOM when `eye_overlay_url` is non-null;
  only its CSS `opacity` changes. This avoids layout reflow and the harsh snap
  of conditional rendering.
- The floating Eye button renders only when `eye_overlay_url` is non-null.
- `showEyeOverlay` state resets to `false` when the selected photo changes
  (useEffect on `selId`).
- The Eye icon swaps to an Eye-slash SVG when the overlay is active.

---

## 6. PDF RAG Aesthetic Rubric

Users can upload photography books or style guides (PDF) via the Creative
Direction panel. The pipeline extracts aesthetic concept phrases and bakes them
permanently into SigLIP-2's positive prompt rubric for the lifetime of the
process (until the next scan after a new upload or clear).

```
PDF file upload  -->  /api/rag/upload
                           |
                   pdf_rag.ingest_pdf()
                   PyMuPDF text extraction
                   DeepSeek-R1-8B GGUF concept extraction
                   -> cache/rag_concepts.json  { phrases: [...] }
                   -> cache/rag_pdfs/<hash>.pdf
                           |
                   server.py clears _text_emb_cache
                   (grade_pipeline_v2._text_emb_cache.clear())
                           |
                   Next /api/scan -> fresh encoder build
                   _pos_prompts_augmented = _POS_PROMPTS + rag_phrases
                   SigLIP-2 encodes augmented list -> new text embeddings
                   Cached for all subsequent scans
```

RAG phrases are indistinguishable from built-in aesthetic rubric prompts at
scoring time. They influence the semantic score component (0.30 weight) of the
vision regression stack.

Copyright note: Only concept phrases (ideas) are extracted, never verbatim
text. Ideas are not copyrightable; this extraction is legally equivalent to
reading a book and taking notes.

---

## 7. NSGA-III Story Sequencer

`src/nsga3_sequencer.py` assigns each graded photo to one of four narrative
slots using a multi-objective evolutionary algorithm (NSGA-III). Slot-specific
score weights control which photos are pulled toward which role:

| Slot | kw_sc | asp_sc | lum_sc | raw_sc | Notes |
|---|---|---|---|---|---|
| Opener | 0.25 | 0.50 | 0.20 | 0.05 | Establishes scene; aspect quality dominates |
| Subject/Interaction | 0.20 | 0.45 | — | 0.35 | DPO-personalised; raw score weighted highest |
| Detail/Accent | 0.30 | 0.60 | — | 0.10 | Aspect quality dominates |
| Contrast | 0.30 | 0.30 | 0.40 | — | Luminance contrast drives selection |
| Closer/Resolution | 0.30 | 0.60 | — | 0.10 | Same as Detail/Accent |

The Subject slot carries the highest raw_sc weight (0.35) so that PersonalHead
DPO preferences most strongly influence which photo becomes the story's
emotional centrepiece.

### Strict Slot Constraints

Each slot enforces hard constraints checked before NSGA-III assignment:

- Opener: `subject_area_pct < OPENER_MAX_AREA` (wide establishing shot)
- Subject: `subject_area_pct > DOMINANT_AREA_PCT` (strong presence)
- Contrast: `abs(h_gap) > H_CONTRAST_THRESH` (measurable tonal contrast)

Threshold values are read at runtime from `config.json`, written by
`calibrate_backend.py`.

---

## 8. Backend Failsafes

### GPU Lock

```python
gpu_lock = asyncio.Lock()

async with gpu_lock:
    result = await run_in_threadpool(grade_pipeline_v2.run_pipeline, ...)
```

All GPU-touching operations (scan, sequence, vision story) acquire this lock.
Concurrent requests queue behind it rather than racing for VRAM.

### VRAM Sequential Protocol

Within `grade_pipeline_v2.run_pipeline()`, models are loaded and unloaded in
strict sequence. No two heavyweight models are resident simultaneously:

```
load TOPIQ-NR  -> score all photos -> unload
load MUSIQ     -> score all photos -> unload
load SigLIP-2  -> embed all photos -> unload
load Qwen2-VL  -> caption batch    -> unload
load Phi-4     -> judge batch      -> unload
load Llama-8B  -> verdict batch    -> unload
```

### Async Queue Manager

`AsyncQueueManager` in `server.py` ensures that if two `/api/scan` requests
arrive simultaneously, the second waits for the first to complete and then
receives the same result, rather than spawning a duplicate pipeline run.

### Atomic File Writes

All JSON files (`config.json`, `calibration_telemetry.json`, `rag_concepts.json`,
`final_story_manifest.json`) are written atomically:

```python
tmp = target_path.with_suffix(".tmp")
tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
os.replace(str(tmp), str(target_path))
```

`os.replace()` is atomic on POSIX and Windows NTFS: readers never see a
partially-written file.

### RAM Preflight (SigLIP-2)

`siglip2_encoder.py` checks available RAM before loading the 896px model:

```python
import psutil
if psutil.virtual_memory().available < 4 * 1024**3:
    raise MemoryError("Insufficient RAM for SigLIP-2 (need >= 4 GB free)")
```

This surfaces as a clean HTTP 500 with a human-readable message instead of a
C-level OOM crash at the 4% progress mark.

---

## 9. Calibration Script

`calibrate_backend.py` derives pipeline thresholds from real-world approved data
stored in `final_story_manifest.json` (entries where `approved_by_vision=True`).

### IQR-Filtered Median

Raw values are cleaned before statistics are computed:

```python
arr = np.array(values, dtype=float)
lo, hi = np.percentile(arr, 10), np.percentile(arr, 90)
core = arr[(arr >= lo) & (arr <= hi)]
if len(core) == 0:
    core = arr          # nothing survived -- use all
m = float(np.median(core))
s = float(np.std(core, ddof=1)) if len(core) > 1 else 0.0
```

The 10th-90th percentile window removes outliers (a single unusually large
subject or unusually dark frame) before computing the median. The median is
more robust than the mean for small, skewed samples.

### Derived Thresholds

| Threshold | Formula | Floor |
|---|---|---|
| `DOMINANT_AREA_PCT` | Subject/Interaction median + 0.5 x std | 15.0 |
| `H_CONTRAST_THRESH` | median(abs h_gap values > 80) / 1.5 | 80.0 |
| `OPENER_MAX_AREA` | Opener median + 1.0 x std | 15.0 |

Fallback values (20.0, 161.0, 30.0) are written to `config.json` when
fewer than one approved sample exists for a slot.

Run after accumulating >= 10 approved sequences:

```bash
python calibrate_backend.py
python calibrate_backend.py --dry-run   # preview without writing
```

---

## 10. Component Reference

| File | Role |
|---|---|
| `server.py` | FastAPI app, all HTTP endpoints, gpu_lock, AsyncQueueManager |
| `src/grade_pipeline_v2.py` | Full grading pipeline: TOPIQ+MUSIQ+SigLIP-2+Qwen+Phi+Llama |
| `src/siglip2_encoder.py` | SigLIP-2 896px encoder, text embedding cache, RAM preflight |
| `src/personal_head.py` | PersonalHead MLP, MarginRankingLoss, score blending |
| `src/background_dpo_trainer.py` | SQLite DPO queue, hard negative mining, QLoRA fine-tune |
| `src/nsga3_sequencer.py` | NSGA-III slot assignment, slot-aware score weighting |
| `src/vision_story_mode.py` | CLIP+Qwen+TSP story annotation, eye overlay generation |
| `src/pdf_rag.py` | PDF ingestion, DeepSeek-R1-8B concept extraction, cache management |
| `calibrate_backend.py` | IQR-filtered threshold calibration from manifest telemetry |
| `canvas_renderer.py` | Final story canvas composition (reads calibration_telemetry.json) |
| `frontend/src/App.tsx` | React UI: gallery, detail panel, star rating, eye toggle, RAG upload |
| `cache/rag_concepts.json` | Live RAG phrase store (cleared on upload/clear) |
| `cache/personal_head.pt` | Trained PersonalHead weights |
| `cache/dpo_prefs.db` | SQLite DPO preference queue |
| `config.json` | Runtime thresholds written by calibrate_backend.py |
| `calibration_telemetry.json` | Per-slot stats for canvas_renderer.py |
| `final_story_manifest.json` | Approved sequences, source of truth for calibration |

---

*Last updated: 2026-05-26*