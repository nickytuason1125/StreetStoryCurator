# Re-Score Architecture — grading a library WITHOUT re-encoding it

Status: DESIGN (approved slice #1 spike, 2026-09-07). Not yet implemented.
Goal line: **"Re-grade everything" must cost minutes and ~flat RAM, not hours
and a multi-GB pipeline run.**

---

## 1. The insight

A grade has exactly one expensive, photo-derived artifact: the **SigLIP-2
image embedding** (tiered 1536/1024/768-dim). Everything else that determines
a photo's grade is either:

- **derived from that embedding** (probe/anchor similarities, personal head),
- **a cached per-photo measurement** (TOPIQ IQA `q`, luminance mean/std),
- **a pure function of the preset** (text prompts -> probe vectors), or
- **a threshold rule** (`assign_grades`: Strong >= 0.60 / Mid / Weak).

Re-grading with a different preset, niche, taste-head update, or judge blend
therefore does NOT require re-decoding or re-encoding a single photo. Today
`force_rescan=True` throws the embeddings' validity away and re-runs the whole
pipeline anyway — hours of encode for a re-judgement that is pure arithmetic
over vectors already in LanceDB.

## 2. What the score actually consumes (audited, grade_pipeline_v2.py)

| Input | Source | Re-score cost |
|---|---|---|
| `embs` (N x D image embeddings) | **Lance `photos[_tier]` table** (already persisted, ~1304-1337) | chunked read, O(batch) RAM |
| `arr_a / arr_fa / arr_t / arr_comp / arr_light / arr_hc / arr_narr` | cosine(image emb x text probe groups) — probes re-encoded per preset (`encode_text_groups`, ~400 ms each, cached) | cheap, re-run per preset |
| `arr_lum, arr_std` (luminance mean/std) | **computed during decode today — NOT persisted** (~2419-2420) | needs persistence (see §4) |
| TOPIQ IQA `q` | IQA pass, checkpointed per (folder, preset) (`_iqa_ckpt_*`) | reuse checkpoint / persist to Lance (§4) |
| personal head `pers` | `personal_head.pt` scored ON EMBEDDINGS (~3093-3097) | cheap re-run |
| MasterJudge blend | `master_judge.predict_many(breakdowns, final_scores)` (~3174) | cheap re-run |
| VLP / Anchor / chiaroscuro masks | thresholds on `arr_a/arr_lum/arr_std` (~2534-2549) | pure arithmetic |
| Soft-Focus Gate | threshold on fine-art sim + score (~3201-3207) | pure arithmetic |
| street-fit | `_is_human_centric(preset)` (~2523) | preset flag |
| grades | `assign_grades(final_scores)` (~3223) | pure arithmetic |

**Conclusion:** with embeddings + IQA + two floats of luminance stats
persisted, a full re-score needs **zero image decodes and zero encoder
loads**.

## 3. Target architecture

### 3.1 Extract the scoring core into a pure function

Steps 3-6 of `run_v2` (probe build -> sim matrices -> masks -> personal blend
-> MasterJudge -> soft-focus gate -> `assign_grades`) become:

```python
def score_from_embeddings(
    embs,                        # (N, D) float32 — from Lance
    preset: str,
    iqa_q=None,                  # (N,) cached TOPIQ, None -> placeholders
    lum_stats=None,              # (N, 2) cached lum mean/std, None -> skip masks
):
    """returns (final_scores, grades, per_photo_breakdowns)"""
```

`run_v2` refactors to CALL this (same numbers, same tests), so there is one
scoring implementation, not two that drift.

### 3.2 New entry point: `rescore_folder(folder_path, preset, ...)`

Lives beside `run_v2` (new `src/rescore.py` importing the shared core):

1. Chunked `query_all` on Lance for the folder -> (path, embedding, iqa, lum,
   exif_ts). Batch 500 — RAM is one batch, never the folder.
2. Build the preset's probe vectors (`encode_text_groups` — the ONLY model
   load; the SigLIP text tower is small and can reuse the warm encoder worker).
3. `score_from_embeddings(...)` per chunk.
4. Chunked `upsert_batch` back to Lance (new scores/grades/breakdowns).
5. `catalog_store.merge_write` per chunk (crash-safe, not end-loaded).
6. Return counts + a **slim gallery** (no embeddings — same contract as the
   cull path since the 2026-09-07 slim-by-construction change).

`grade_worker` routes `force_rescan=True` calls to `rescore_folder` when the
folder's rows are already encoded, and falls back to `run_v2` for anything
missing an embedding (new photos) — i.e. "re-grade everything" becomes
"re-score everything + grade the genuinely-new".

### 3.3 UX mapping (pre-grade modal)

- **New photos only** -> today's incremental `run_v2` (unchanged).
- **Re-grade everything** -> `rescore_folder` when Lance coverage is complete;
  honest SSE notice of which path was taken ("re-judged 6,000 photos from
  cached analysis — no re-encoding needed").

## 4. Prerequisite: persist the two non-embedding per-photo facts

Add nullable columns to the Lance table (and backfill lazily):

| Column | Producer | Backfill |
|---|---|---|
| `lum`, `lum_std` | the decode step's luminance stats (already computed at ~2418-2420's source) | decode-on-demand per photo, chunked, then upsert |
| `iqa_q` (TOPIQ NR 0-1) | IQA stage checkpoint (`_iqa_ckpt_*`) | promote from checkpoint at write time; re-run IQA only for rows missing it |

Both backfills are resumable and O(batch) RAM — they reuse the existing
chunked-upsert pattern (`FIRSTCUT_LANCE_CHUNK`).

## 5. RAM / time profile (16 GB machine, 100k photos)

| | Today (force_rescan) | Re-score |
|---|---|---|
| Encoder load | yes (1.2-3 GB tier-dependent) | **no** |
| Decodes | every photo | **zero** (after §4 backfill) |
| Peak RAM | pipeline 2-4.2 GB + gallery spikes | ~O(500-photo chunk), ~200-300 MB tree |
| Wall time | hours (encode-bound) | minutes (vector math + one text encode) |
| Failure surface | full pipeline | arithmetic + DB writes only |

## 6. Edge cases and rules

- **Preset / niche change:** probe vectors are the only preset-dependent model
  input -> re-score covers every preset/niche re-grade.
- **Qwen Deep-Grade rows (`is_verified`):** those grades came from a VLM
  reading pixels; a re-score must either keep them (default, flagged) or
  invalidate them explicitly — never silently overwrite with the SigLIP score.
- **Personal head retrain / MasterJudge update:** these change weights, not
  embeddings -> re-score re-applies them correctly by design.
- **Tiers:** embeddings live in per-tier tables (`photos_low`/`photos`); a
  tier switch is NOT a re-score — it is a re-encode (different vector space).
  The modal must never route a tier change through rescore.
- **Draft-decode luminance:** `lum` stats must be captured on the same decode
  path the score was calibrated on; the backfill reuses the production decode
  path verbatim.
- **Catalog:** entries stay embedding-free (slim-by-construction, 2026-09-07).

## 7. Rollout

1. §4 schema columns + backfill (decode-on-demand) — resumable, safe.
2. Extract `score_from_embeddings` from `run_v2`; golden-test: identical
   grades on a fixed folder before/after refactor.
3. `rescore_folder` + `grade_worker` routing + modal notice.
4. Tests: grade-parity golden test; rescore RAM profile test (batch-bound);
   tier-change-must-not-rescore guard.

---

*Line numbers referenced are from grade_pipeline_v2.py as of 2026-09-07 and
will drift; they anchor the audit, they are not an API.*
