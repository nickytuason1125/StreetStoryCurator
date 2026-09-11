# AADB + Unsplash — External Human-Accuracy Signals for the Master Judge

**Date:** 2026-09-11
**Status:** design, awaiting approval
**Scope:** two sub-projects sharing one evaluation harness — AADB-trained
aesthetic feature (primary), Unsplash calibration anchors + exemplar feature
(secondary, built on the same gate). Both are strictly additive to the
existing Master Judge; neither is allowed to touch live grading unless it
wins an exam.

## Problem

Grading today combines three things: zero-shot SigLIP/CLIP probe cosine
similarity (squashed into a narrow band), NIMA (a single MobileNetV2 model
trained on AVA, wired in as the aesthetic component — see
`project_nima_aesthetic_undeprecated`), and the Master Judge
(`src/master_judge.py`), a ridge regression stacked on
`[machine score, Technical, Composition, Lighting, Narrative, Human/Culture,
arch:geo, arch:night, arch:layer, arch:messy, arch:maxdoc]` (the `DESIGN`
constant, `src/master_judge.py:73`). The Master Judge is shipped
(`data/master_judge_defaults.json`, ρ_holdout=0.8738 on 791 photos,
`promoted: true`) and runs by default per `CLAUDE.md`'s BUILD/SHIP contract.

Three concrete gaps:

1. **No externally-validated human-aesthetic signal exists anywhere in the
   stack.** Every number the pipeline currently reports — the 0.874
   ρ_holdout, the `+0.231` rank agreement vs `+0.149` chance
   (`project_grading_measurement_bounds`) — is measured against this
   project's own rating data (the `tpe_master` placeholder baseline, or the
   user's own growing `cache/user_ratings.json`, currently 3 entries).
   Nothing checks whether the grader agrees with human aesthetic judgment
   *in general*, independent of this project's own labels.
2. **`Human/Culture` is a live, negatively-correlated feature.** Per
   `CLAUDE.md`'s 2026-08-30 backtest: Composition +0.64, Narrative +0.51,
   Technical +0.38, Lighting +0.33, **Human/Culture −0.15**. It's still in
   `FEATURES` and still gets fit a coefficient. Out of scope to fix here
   (flagged so it isn't conflated with this work) — but any new feature
   added alongside it should be evaluated with this in mind: a second
   badly-correlated feature would compound the problem, not offset it.
3. **`cache/master_anchors.json` (the `FIRSTCUT_MASTER_ANCHORS` human-anchored
   calibration ruler) has never been populated.** Absolute Strong/Mid/Weak
   boundaries currently run on generic per-batch percentile stretch, not on
   curated human-judged reference photos.

## Goal

Add genuinely new signal — trained/anchored on human aesthetic judgment this
project has never seen — to the Master Judge's feature set, and prove via a
held-out exam whether each addition makes the fused grade more accurate to
**both** benchmarks that count as "human": AADB's own held-out ratings
(general human judgment, external to this project) and this project's live
rating baseline via the existing champion/challenger harness (personal
judgment — the user considers `PersonalHead`'s signal legitimate and it is
not to be treated as a lesser check). A challenger ships only if it does not
regress either.

**Non-goals:** replacing NIMA or the CLIP probe banks (both stay as-is);
inventing a new evaluation methodology (reuse `scripts/master_backtest.py`'s
champion/challenger exam and `src/master_judge.py`'s fit/promote pipeline
wholesale); retraining `PersonalHead` itself; fixing the `Human/Culture`
feature (separate, already-identified, out of scope).

## Design

### Sub-project 1 — AADB-trained feature (primary)

**Data.** AADB (Aesthetics and Attributes Database, Kong et al. ECCV 2016):
~10,000 photos, individually selected for Creative Commons licensing
specifically so the dataset could be redistributed for training — the
cleanest license story of the datasets considered (see prior discussion:
AVA's images aren't officially redistributable, PARA is academic-gated,
Flickr-AES is link-based and rots). Ships an overall aesthetic score plus
attributes (balancing, color harmony, depth of field, content, etc.).

**Setup script — `aadb_setup.py` (repo root, mirrors `nima_setup.py`'s
one-time pattern).**
1. Download AADB (images + score CSV) into a scratch location outside the
   repo (never committed — same `models/` gitignore precedent as NIMA's
   `.hdf5`/`.onnx`).
2. Fixed 80/20 train/held-out split, seeded, saved alongside the download so
   the held-out set is stable across re-runs (re-shuffling on every run
   would make the AADB ρ non-reproducible).
3. Encode every AADB image through the **existing** SigLIP-2 encoder
   (`src/encode_worker.py`'s isolated-subprocess pattern — never touch
   `torch.cuda` in the parent, per `feedback_never_touch_cuda_in_parent`) —
   same embedding space grading already uses, so the resulting head is a
   probe on real production embeddings, not a foreign feature space.
4. Fit a small head (`1536→256→64→1`, matching `PersonalHead`'s shape and
   `personal_head_np.py`'s numpy-mirror pattern for the CUDA-free grade
   worker) via ridge regression against AADB's aesthetic score, on the TRAIN
   split only.
5. **Report ρ against the AADB held-out split.** This is the external,
   general-human-judgment number. Print it; do not silently proceed if it's
   weak — a head that doesn't predict AADB's own held-out labels has no
   business being added to a live feature set.
6. Save `models/aadb_head.pt` + numpy mirror (gitignored, same as
   `personal_head.pt`).

**Grading integration.** A new Step (alongside Step 4a-NIMA in
`grade_pipeline_v2.run_v2()`) computes the AADB head's score per photo from
the SigLIP-2 embeddings already resident in memory (near-zero marginal
cost — no extra encode pass, unlike NIMA which needs its own decode/forward
pass) and writes it into `per_photo_breakdowns[idx]["AADB"]`. **Not** named
or stored as `"aesthetic"` — `master_judge.py:51` is explicit that
`"Aesthetic"` is deliberately excluded from `FEATURES` because it IS the base
grader score; conflating the two would corrupt the design matrix.

**Master Judge integration.** Add `"AADB"` to `FEATURES` in
`src/master_judge.py:56-57`. Per the file's own comment, reordering/adding
to this list invalidates the saved fingerprint (`_feature_fingerprint`,
`master_judge.py:261`) — this is intentional and exactly the mechanism that
forces a re-fit/re-exam rather than silently reinterpreting old weights.

**Evaluation gate.** `scripts/master_backtest.py --fit` already fits a
challenger via `master_judge.fit_from_rows()`, computes ρ_holdout on this
project's own baseline, and refuses promotion below the existing champion
(`_MIN_PROMOTE_N=60`, `_MIN_HOLDOUT_N=12`, both already satisfied by current
baseline size). Two additions:
- A new `--aadb-check` (or folded into `--fit`) step reports the AADB-only ρ
  from step 5 above, surfaced in the same report `master_backtest.py`
  already prints, so both numbers are visible together at fit time.
- `scripts/promote_master_judge.py`'s gate becomes: promote only if the new
  challenger's ρ_holdout (this project's baseline) does not regress vs. the
  current champion **and** the AADB head's own external ρ clears a minimum
  bar (e.g. must exceed the current baseline's chance-level rank agreement,
  `+0.149`, from `project_grading_measurement_bounds` — a head that can't
  beat chance on its own labeled data is not "more humanly accurate," it's
  noise). Exact threshold is a detail for the implementation plan, not the
  design.

**Degrade behavior.** Matches NIMA's precedent exactly: if
`models/aadb_head.pt` is absent, the new grading step is skipped and
`"AADB"` is simply missing from the breakdown — `feature_vector()`
(`master_judge.py:127-137`) already emits `NaN` for missing inputs rather
than imputing zero, so a fresh clone without the model file grades exactly
as it does today.

### Sub-project 2 — Unsplash anchors + exemplar feature (secondary)

Built after sub-project 1 ships (or at minimum after it's validated), using
the same evaluation harness — no new eval code for this piece either.

**Data.** Unsplash's dataset/API, filtered to curated Street Photography /
Documentary collections. Unsplash's license explicitly permits training ML
models (an actual, current grant — unlike the scraped-web ambiguity behind
LAION/AVA). Only derived artifacts (embeddings, anchor statistics, exemplar
vectors) are ever stored or shipped — raw Unsplash images are never
committed or redistributed, same boundary already drawn for the RAG PDFs
(`project_rag_pdfs`) and for AADB above.

**Setup script — `unsplash_setup.py`.**
1. Pull a curated Strong/Weak split from Street Photography / Documentary
   collections: Strong = staff-picked / top-quartile-by-engagement photos
   within those collections; Weak = bottom-quartile-by-engagement photos
   from the *same* collections (not a random off-topic pool — a weak
   exemplar has to still be a street/documentary photo, just a worse one,
   or the contrast is trivial and teaches nothing about this project's
   actual grading boundary).
2. Encode through the same SigLIP-2 encoder.
3. **Anchors:** populate `cache/master_anchors.json` via
   `master_judge.human_anchor_lo_hi()` (`master_judge.py:442`), which already
   defines the lo/hi-span contract this file needs to satisfy — this is
   filling an existing, currently-empty feature, not building new anchor
   machinery.
4. **Exemplar bank:** save the Strong-pool embeddings as a reference set in
   a new small module, `src/exemplar_scorer.py` — computes per-photo cosine
   similarity to the k nearest Strong exemplars at grade time, written to
   `per_photo_breakdowns[idx]["Exemplar"]`, added to `FEATURES` the same way
   as `"AADB"` above.

**Evaluation gate.** Identical mechanism to sub-project 1: `"Exemplar"`
becomes a new `FEATURES` entry, `master_backtest.py --fit` re-fits and
reports ρ_holdout, `promote_master_judge.py` refuses regression. No
AADB-style "external held-out" check applies here (Unsplash isn't a rated
dataset) — the exemplar feature is validated purely against this project's
existing rating baseline.

## Testing

- **Sub-project 1:** `aadb_setup.py` is deterministic given a fixed seed —
  test that a re-run reproduces the same train/held-out split and the same
  reported AADB ρ (within floating-point tolerance). Integration test:
  grade a small fixture set with `models/aadb_head.pt` present vs. absent,
  confirm `"AADB"` appears/is absent from breakdowns accordingly and nothing
  else changes (degrade path). `master_backtest.py --fit` run against a
  synthetic/small baseline confirms the promotion gate correctly refuses a
  challenger whose AADB ρ is below the chance-level floor.
- **Sub-project 2:** `unsplash_setup.py` populates `cache/master_anchors.json`
  in the exact shape `human_anchor_lo_hi()` expects (existing
  fingerprint-freshness + monotonicity checks already validate this — reuse
  them, don't reimplement). `exemplar_scorer.py` gets a unit test on a small
  fixed embedding set with known nearest-neighbor answers.
- **Shared:** neither sub-project may change a single live grade unless its
  respective `promote_master_judge.py` run reports a win. This is the
  existing BUILD/SHIP contract; the test is simply "run the exam, confirm it
  refuses/accepts correctly," not new infrastructure.

## Licensing note (carried into the plan, not re-litigated here)

Both datasets are used strictly as training/calibration data whose *images*
are never committed to the repo or shipped in the built app — only derived
numeric artifacts (head weights, anchor statistics, exemplar embedding
vectors) leave the setup scripts. This mirrors the precedent already set for
the RAG PDFs (source material removed from the repo, only extracted generic
phrases kept) and for NIMA (idealo's Apache-2.0 weights shipped, not AVA's
raw images).

## Decomposition

This spec covers both sub-projects because they share one evaluation
mechanism and were approved together. If implementation reveals hidden
complexity in either half (per the brainstorming skill's ratchet rule), that
half should be re-scoped independently rather than blocking the other.
Sub-project 1 (AADB) is the dependency-free starting point; sub-project 2
(Unsplash) can proceed in parallel or after, but its exemplar feature's
promotion exam is meaningless without sub-project 1's `FEATURES`-extension
mechanism already wired up, so implementation order should be 1 then 2.
