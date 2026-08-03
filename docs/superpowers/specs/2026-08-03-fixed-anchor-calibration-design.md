# Fixed-anchor calibration — design

**Date:** 2026-08-03
**Goal:** Make the default grade absolute and comparable across culls, by
anchoring CLIP-discriminant calibration to fixed reference values instead of to
each batch's own min and max.

## The bug

`specvlm_pipeline._calibrate` min-max stretches every batch: batch min → 0.10,
batch max → 0.95. Demonstrated with identical input:

```
photo raw discriminant = 0.020 in BOTH batches
  in an all-STRONG batch ->  0.10   (Weak)
  in an all-WEAK   batch ->  0.95   (Strong)
```

The same photograph is Weak or Strong depending only on what it was culled
alongside.

This is on the live default path: `cal_overall` (specvlm_pipeline.py:605) flows
into `overall_clip` (:658), which is the grade for every photo that does not go
through `deep_grade`.

It contradicts the project's own contract, `CLAUDE.md:79`:

> Do not reintroduce per-batch relative grading or any floor >= 0.60 — a photo
> reaches Strong only on actual fused quality.

**How it survived:** the 2026-06 absolute-grading work removed quantile
calibration from `grade_pipeline_v2` and recorded the job as done. This min-max
lived in a different file — the CLIP scorer — and was never looked at. The
invariant was enforced in one place and violated in another.

### Consequences

- Grades are not comparable across culls. Re-grading a photo in a different
  folder changes its grade.
- Every batch is *guaranteed* to contain a ~0.10 and a ~0.95, so a folder of
  uniformly excellent work always manufactures "Weak" rejects.
- The absolute thresholds (0.60 / 0.41) are applied to an already
  batch-normalised input, which makes them decorative.
- It explains why `Strong=62 Mid=324 Weak=128` is so stable run to run: min-max
  forces a spread regardless of content.

## Why not simply delete the calibration

The existing docstring is honest about the constraint, and it is real: the raw
discriminant (`max(img·pos) - max(img·neg)`) spans only about ±0.05 in 1536-d
space. Feeding that to a 0.60/0.41 threshold puts every photo in one bucket. An
earlier IQR-based attempt compressed everything into [0.33, 0.67] — every photo
Mid, and TOPIQ's contribution irrelevant.

So the scale expansion must stay. Only its *reference* changes: from "this
batch" to "a fixed corpus".

## Design

### Anchors

Replace batch min/max with two constants per calibration context:

```
lo  = p1  of the raw discriminant over a reference corpus   -> maps to 0.10
hi  = p99 of the raw discriminant over the same corpus      -> maps to 0.95
score = clip(0.10 + (raw - lo) * (0.95 - 0.10) / (hi - lo), 0.10, 0.95)
```

Percentiles, not min/max: a single outlier frame must not move the scale for
every photo graded afterwards. Values outside [lo, hi] clamp, so an
exceptionally good photo saturates at 0.95 rather than redefining the top.

### Anchors are per-context, and the context must be fingerprinted

This is the part most likely to be got wrong, and it is the same bug class the
project already fixed once with `RunProfile`: a value that depends on tier state
being stored in one place and used in another.

The discriminant distribution depends on **three** things:

1. **Encoder tier** — high/mid/low emit 1536/1024/768-d embeddings with
   different similarity scales. Anchors derived at one tier are meaningless at
   another.
2. **The probe set** — change `_STREET_POS_PROBES` and the distribution moves.
3. **The encoder checkpoint** — a different checkpoint at the same dim is a
   different space.

Anchors are therefore stored keyed on a fingerprint of all three, e.g.
`sha1(tier | encoder_source | sha1(sorted(pos_probes) + sorted(neg_probes)))`.
On mismatch the pipeline **must refuse to use stale anchors** and fall back to
the current batch-relative behaviour with a loud warning, rather than silently
grading against the wrong scale. Silent wrong-scale grading is worse than the
bug being fixed.

Anchors belong in `run_profile.RunProfile` alongside the other tier-derived
values, not in a module-level dict in `specvlm_pipeline`.

### Deriving them

`scripts/derive_calibration_anchors.py`:

1. Load all embeddings for the active tier from LanceDB.
2. Load the live probe embeddings for that tier.
3. Compute the raw discriminant for every photo.
4. Write `{fingerprint, tier, n, p1, p99, generated}` to
   `cache/calibration_anchors.json`.

The reference corpus must be **diverse** — the whole library, not one folder.
Anchors derived from a single shoot reproduce the bug with extra steps.

## Open question the implementer must resolve first

The probe caches are inconsistent and this must be understood before anchors are
derived, or they will be derived against the wrong vectors:

- `cache/probe_embs_low.npz` holds `pos (74, 768)`, `neg (5, 768)` — but
  `_STREET_POS_PROBES` has **307** entries and `_STREET_NEG_PROBES` **31**. The
  file also has separate `sp` / `sn` keys, which may be the street probes.
- `cache/probe_embs_mid.npz` exists; **no high-tier cache exists at all**, so
  the Pro tier appears to recompute probes every run.

Establish which array `_raw_discriminant` actually receives before deriving
anything.

## Migration

Every existing grade was produced under batch-relative calibration and is not
comparable to a fixed-anchor grade. LanceDB holds 1829 rows at the high tier.

Re-grade rather than migrate: the scores are recomputable, and a mixed catalog
where some grades are batch-relative and some are absolute is worse than either.
Embeddings are cached, so a re-grade is the fast path (~130 s for 514 photos),
not a re-encode.

**The `Strong=62 Mid=324 Weak=128` reference dies with this change.** Every
verification in this project measures against it, so a new reference must be
captured immediately after the fix and the memory updated. Until then there is
no regression baseline.

## Verification

The defining test, which the current code fails:

```python
def test_same_photo_scores_the_same_in_any_batch():
    """A photo's grade must not depend on what it was culled alongside."""
    strong_batch = np.array([0.020, 0.022, 0.025, 0.028, 0.030])
    weak_batch   = np.array([0.020, 0.010, 0.008, 0.005, 0.002])
    assert calibrate(strong_batch)[0] == pytest.approx(calibrate(weak_batch)[0])
```

Plus:

- A uniformly strong batch produces **no** 0.10 scores.
- A uniformly weak batch produces **no** 0.95 scores.
- Stale/absent anchors fall back loudly, and never silently grade.
- Scores stay within [0.10, 0.95] for inputs far outside the anchor range.
- A real cull produces a plausible distribution — if the new reference is
  ~0 Strong or ~0 Weak, the anchors are wrong, not the photos.

## Risks

| Risk | Mitigation |
|---|---|
| Anchors derived from a non-representative corpus | Use the whole library; sanity-check the resulting distribution before accepting |
| Stale anchors after a probe or encoder change | Fingerprint covering tier + encoder + probe set; refuse rather than guess |
| New distribution looks alarming | Expected — grades become absolute. A uniformly good shoot *should* now be mostly Strong. Judge by whether individual grades are defensible, not by whether the histogram matches the old one |
| Anchors re-derived per tier get out of sync | Store on `RunProfile`, the single declaration of tier state |
