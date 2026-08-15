# One Ruler — Absolute Calibration Across All Three Tiers

**Date:** 2026-08-16
**Status:** design, awaiting approval
**Scope:** Project 1 of 4 (see Decomposition at the end)

## Problem

Grading compares each photo against RAG-augmented positive probes and a fixed
negative set. The raw discriminant — `max(img·pos) - max(img·neg)` — spans about
±0.05, so it must be stretched onto `[0.10, 0.95]` before it becomes a score. The
stretch needs a reference. `specvlm_pipeline._calibrate` takes `(lo, hi)` anchors
(p1/p99 over a reference corpus) for that reference, and falls back to per-batch
min/max when they are absent.

Three defects make that fallback the live behaviour.

### 1. The stale-anchor alarm is not wired to anything

`cache/calibration_anchors.json` was derived 2026-08-04 with fingerprint
`8047158f17c20234`. The 2026-08-13 rubric clean (60 → 31 phrases, removing 29
Halsman-biography lines) changed the positive probes, so the fingerprint no
longer matches.

`load_anchors` detects this correctly and reports `calibration anchors are
STALE`. It then calls `_shipped_defaults`, which matches on **tier only and never
compares the fingerprint** (`specvlm_pipeline.py:378-402`), and returns the same
numbers that were just rejected. Every Pro grade since 2026-08-13 has been scored
against a ruler measured on the polluted rubric.

### 2. Balanced and Fast have no ruler at all

`data/calibration_defaults.json` contains only `by_tier.high`. For `mid` and
`low`, `_shipped_defaults` returns `None`, `load_anchors` returns `None`, and
`_calibrate` falls through to per-batch min/max (`specvlm_pipeline.py:522-532`).
It prints a warning and grades on a curve anyway.

This is the direct answer to "why don't the tiers agree": two of the three are
not grading absolutely. Pro answers "how good is this photo"; Fast answers "how
good is this photo relative to the batch you just handed me". `CLAUDE.md:79`
forbids per-batch relative grading; the fallback does it silently.

### 3. The reference corpora for the small tiers are not diverse

Deriving `mid`/`low` anchors from their existing stores would reproduce the bug:

| table | rows | folders | largest folder |
|---|---|---|---|
| `photos` (high) | 38,055 | 194 | 4,463 (12%) |
| `photos_mid` | 767 | 4 | 514 (67%) |
| `photos_low` | 909 | 4 | 514 (57%) |

`derive_calibration_anchors.py` warns about exactly this: anchors from one shoot
reproduce the original bug with extra steps. A diverse re-encode at `mid` and
`low` is unavoidable and is the bulk of this project's cost.

## Goal

All three tiers grade on a fixed, absolute, fingerprint-validated scale, and we
hold a measured number for how closely their bucket assignments agree.

**Non-goals.** Changing what counts as a good photo (the rubric is correct as of
2026-08-13). Changing fusion weights or thresholds. Improving grading quality.
Any cross-tier score mapping — that decision is deferred until change 5 reports
the residual disagreement.

## Design

### Change 1 — Validate shipped defaults; refuse rather than curve

Two parts, both in `specvlm_pipeline.py`.

*Fingerprint the shipped defaults.* Each `by_tier` entry already carries a
`fingerprint` field; `_shipped_defaults` must compare it against the live probe
fingerprint and return `None` on mismatch. A shipped default is a ruler like any
other and gets the same validation.

*Refuse when no valid ruler exists.* `_calibrate(anchors=None)` currently degrades
to per-batch min/max. It must instead raise, with a message naming the tier and
the remedy. This is a deliberate policy change and the core of the fix: the
codebase already prefers refusal over silent degradation (the derive script
refuses a degenerate span, the encoder floors refuse rather than OOM), and a
silent curve is both forbidden by contract and undetectable in the output.

An escape hatch, `FRAMEGRADE_ALLOW_UNCALIBRATED`, keeps the old behaviour for
development and for the bootstrap: change 2 needs a cull to run at `mid` and
`low` to regenerate their probe caches, and at that moment those tiers still have
no ruler. It is declared in `run_profile.SETTINGS` like every other knob,
defaults to off, and logs loudly when used.

Ordering note: change 4 ships valid rulers for all three tiers, so a normal user
never reaches the refusal. Change 1 must not merge ahead of change 4.

### Change 2 — Regenerate probe embeddings

No code. The probe cache key hashes `_pos_prompts_augmented`, so the rubric edit
already invalidated it; `cache/probe_embs_low.npz` and `_mid.npz` are dated
2026-08-12 (pre-clean) and the `high` cache is absent entirely. One run per tier
rebuilds all three against the clean 31 phrases. This is a prerequisite for
change 4 — the derive script reads the probe cache.

### Change 3 — Build a diverse reference corpus at `mid` and `low`

New `scripts/build_reference_corpus.py`.

Sample from the `photos` table, stratified across its 194 folders: proportional
allocation with a per-folder cap, so a 4,463-photo card cannot dominate the way
the LX3 folder dominates the current small-tier stores. Target ~2,000 photos;
emit the realised folder histogram so the spread is auditable rather than
asserted.

Encode the sample at `mid` and `low` through the existing isolated encode
subprocess — no new encode path, and no direct GPU work in the parent
(`feedback_never_touch_cuda_in_parent`). Measure throughput on a 100-photo
sample first and report the projected full-run time before committing to 2,000.

Resumable: a photo already present in the target tier's table is skipped, so an
interrupted run costs only its current chunk.

### Change 4 — Derive and ship anchors for all three tiers

Run `derive_calibration_anchors.py` once per tier. Pro is re-derived too — its
current anchors are the stale 2026-08-04 measurement.

New `scripts/promote_calibration_defaults.py` merges a derived
`cache/calibration_anchors.json` into `data/calibration_defaults.json` under
`by_tier.<tier>`, preserving the other tiers and carrying both fingerprints. The
existing hand-maintained shape is kept; only the write becomes scripted, because
hand-editing a fingerprinted file is how it drifts.

The derive script's existing guards stay: refuse below 200 usable photos, drop
rows whose embedding norm is not ~1.0, refuse a degenerate span, refuse when one
bucket takes >80% of the corpus.

### Change 5 — Measure cross-tier agreement

New `scripts/compare_tier_grades.py`.

Take the photos present in all three tables (709 today, more after change 3),
compute each tier's calibrated overall score under its own ruler, and report:

- a 3×3 bucket-agreement matrix (Strong/Mid/Weak) per tier pair
- Spearman rank correlation per tier pair
- the score-delta distribution (median, p90, max)
- the worst disagreements by path, so they can be eyeballed

This is the acceptance evidence for "almost the same", and the input to the
deferred decision on whether a cross-tier mapping is warranted.

Scope note: this compares the **CLIP term under each tier's ruler**, not the
fully fused grade, because fusion needs TOPIQ and the other per-photo arrays that
are not stored for these rows. The CLIP term is what the ruler governs, so it is
the right thing to measure for this project.

### Change 6 — Persist fusion inputs; lazy re-grade of the back catalogue

38,055 rows carry grades made against a wrong or absent ruler. The fusion
(`grade_pipeline_v2.py:2450-2558`) is ~25 chained clamps, floors and gates; the
clamps are lossy, so it cannot be inverted and TOPIQ cannot be recovered from a
stored score. Re-scoring existing rows therefore means re-running the stages that
feed fusion — not a cheap recompute.

Decision: **do not re-cull the back catalogue.** Instead:

*Persist the inputs going forward.* Extend the LanceDB schema with a
`fusion_inputs` JSON column holding the per-photo arrays fusion consumes
(`arr_a`, `arr_t`, `arr_fa`, `arr_lum` and the rest already assembled at the
`FRAMEGRADE_FUSION_DUMP` site, `grade_pipeline_v2.py:2573`), plus the anchor
fingerprint the row was graded under. Future ruler changes then become a pure
CPU recompute, which is what makes this class of problem cheap next time.

*Re-grade lazily.* A folder re-grades when next opened. Rows whose stored
fingerprint does not match the live ruler are marked stale rather than silently
mixed with current ones; surfacing that mark is Project 3.

The column is additive and nullable, so existing rows stay readable — the store
already auto-migrates schema on first run.

### Change 7 — Chore: compact the vector store

`photos.lance` is 15.5 GB across 780 versions. `FRAMEGRADE_LANCE_RETENTION_DAYS`
defaults to 7 but history is plainly not being reclaimed. Run
`optimize(cleanup_older_than=)` and confirm the retention path actually executes
on a normal run; a prior audit recovered 894 MB → 18 MB on this same table, so
the mechanism works when it runs.

## Testing

TDD applies to the code changes; changes 2–5 and 7 are scripts and runs whose
evidence is their output.

Unit tests, written before the fix:

1. A shipped default whose fingerprint does not match the live probes is
   rejected — this is defect 1, so it gets a failing test first.
2. A shipped default whose fingerprint matches is accepted.
3. `_calibrate` with no anchors raises, and does not return per-batch min/max.
4. `FRAMEGRADE_ALLOW_UNCALIBRATED=1` restores the fallback and logs.
5. `load_anchors` still prefers a valid user cache over a valid shipped default.
6. The corpus sampler respects the per-folder cap on a synthetic histogram.

Acceptance evidence:

- `derive_calibration_anchors.py` completes for all three tiers without hitting
  a refusal guard.
- `compare_tier_grades.py` output, reported in full.
- One real cull at each tier that grades without the uncalibrated warning.

## Risks

**The refusal could block a user mid-cull.** Mitigated by ordering — change 4
ships rulers for every tier before change 1 merges — and by the escape hatch.

**The small-tier encode may be slower than expected.** Mitigated by measuring a
100-photo sample and reporting the projection before the full run.

**Tier agreement may still be poor after honest rulers.** That is a finding, not
a failure; change 5 exists to produce the number, and the cross-tier mapping
option stays available.

**Schema change to a 15.5 GB table.** The column is additive and nullable, and
change 7 compacts the table first so the migration runs against 18 GB of history
rather than a store still carrying 780 versions.

## Decomposition context

This spec is Project 1 of 4, agreed 2026-08-15:

1. **One ruler (this spec)** — backend calibration.
2. **Break up `App.tsx`** — 282 KB, one function from line 836. Prerequisite for 3 and 4.
3. **Show the ruler** — surface tier and calibration state per grade. Needs 1 and 2.
4. **Visual design pass** — direction exists in `design_bundle4/.../Street Photo Curator v3.html` and `v3-mockup.png`. Needs 2.
