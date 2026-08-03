# Lean disk footprint — design

**Date:** 2026-08-03
**Goal:** Cut FrameGrade's on-disk footprint from ~24 GB to between ~10 and
~17 GB depending on one user decision, and make what remains a *fixed* cost
rather than one that grows with use.

## Why

Measured on the development machine, same day, same hardware:

| | FrameGrade | Lightroom Classic 15.4.1 |
|---|---|---|
| Install / model weights | 22.87 GB | 4.20 GB |
| Working data | 1.03 GB (1,745 photos) | 22.71 GB previews + 0.33 GB catalogs |
| **Total** | **23.90 GB** | **27.24 GB** |

Near parity, but the two numbers behave differently. Lightroom's previews are a
*marginal* cost that grows with every import — already 22.71 GB across 18
catalogs on this machine. FrameGrade's model weights are a *fixed* cost.

That framing was only partly true, which is what prompted this work. Two leaks:

1. **Roughly 6 GB of model weights appear unreferenced, plus a 7.1 GB
   maybe.** `ViT-L-14.pt` (702 MB) has zero references in live code.
   `deepseek-r1-8b-q5.gguf` (5.4 GB) backs a grader whose weights the project
   notes record as never wired up. Separately, `models/siglip2` (7.1 GB) is the
   source for the 3.5 GB ONNX actually loaded at runtime — removable in
   principle, but see "Handled separately" below. These are candidates the
   audit must confirm, not conclusions.

2. **`cache/lance.db/photos.lance` is 859 MB holding ~10 MB of vectors.**
   350 data fragments, 409 versions, for 1,745 photos at 1536 float32 dims
   (10 MB of actual vector data). `compact_after_write()` calls
   `compact_files()`, which merges fragments but **does not delete old
   versions**. Every re-cull leaves its history behind, unbounded.

Leak 2 is the more important find: it is the same category of marginal growth
this project claims Lightroom suffers from and it does not.

## Non-goals

- RAM ceiling and cull speed. Both were considered and deferred; disk is the
  larger measurable win and leak 2 is an actual bug.
- A general disk-budget / eviction subsystem. Two specific leaks with known
  causes do not justify that machinery (YAGNI).
- Removing `cache/previews` (92 MB), `cache/thumbs` (14 MB), or
  `cache/rag_pdfs` (50 MB). All are live, earn their size, and are small.

## Part 1 — LanceDB version retention

### Change

`lance_store.compact_after_write()` currently calls `tbl.compact_files()`.
Replace with:

```python
tbl.optimize(cleanup_older_than=timedelta(days=RETENTION_DAYS),
             delete_unverified=False)
```

Verified available in the installed lancedb 0.30.2:
`optimize(*, cleanup_older_than, delete_unverified, retrain)` performs
compaction *and* version reaping in one call.

### Decisions

- **Retention window, default 7 days.** Declared in the existing `SETTINGS`
  table as `FRAMEGRADE_LANCE_RETENTION_DAYS`, not a literal — per the project's
  one-declaration rule. A window rather than "delete all but current" preserves
  a rollback path if a cull writes bad grades.
- **`delete_unverified=False`.** Refuses to remove fragments it cannot prove are
  orphaned. This matters here specifically: three tier tables (`photos.lance`,
  `photos_mid.lance`, `photos_low.lance`) live in one database and a reader on
  another tier may hold references.
- **Failure stays non-fatal.** The existing `try/except` and "Compaction
  skipped" log are retained. Cleanup is housekeeping running *after* grades are
  durably committed; it must never fail a completed cull.

### Expected effect

`photos.lance` 859 MB → tens of MB. Growth becomes bounded by the retention
window instead of unbounded in the number of culls.

## Part 1b — Delete `src/lance_migration.py`

Found while reviewing Part 1. The file is dead three times over:

- **Line 1 is the literal text `but ar`**, before the module docstring. The file
  raises `SyntaxError` and cannot be imported.
- **Nothing imports it.** No reference in any `.py` in the project.
- Its `cache/lancedb_v2` directory does not exist, confirming it has never run —
  the module calls `DB_DIR.mkdir()` at import time.

It is not merely dead, it is a trap. It defines a *second* LanceDB
(`cache/lancedb_v2`, table `photos_v2`) with a schema that disagrees with the
live store — `confidence`, and `breakdown` as a JSON string rather than a struct.
Anyone who "fixes" line 1 without reading on would create a parallel vector
store beside the real one, with a CWD-relative path, as an import side effect.

It also migrates from SQLite and FAISS, neither of which the project still uses.

**Action:** delete the file. It is tracked in git, so this is reversible in a way
the model weights are not — no quarantine step needed. Verify by confirming the
suite still passes and no import breaks.

## Part 2 — Model weight audit and quarantine

### Why not just grep

A static scan cannot see paths built at runtime, and the project's own guidance
names model weights as the case where caution beats cleverness. Two passes, and
a file must be untouched by *both* to become a candidate:

- **Static:** reference scan across `src/`, root scripts, and `scripts/`,
  excluding `src/deprecated/`.
- **Dynamic:** trace every file actually opened under `models/` during a real
  cull, plus the other weight-loading modes — story, competition, annotation.

### Quarantine, not deletion

Candidates move to `models/_quarantine/` preserving relative paths. Then the
full suite and a real Pro-tier cull run against the quarantined tree. Only after
that passes does the user delete.

This is reversible at every step and costs one extra cull. The audit can only
prove a weight unused *for the paths it exercises*; a mode never triggered will
not appear in the trace. Quarantine is what makes that acceptable.

### Handled separately

`models/siglip2` (7.1 GB) is the source for the ONNX export actually loaded at
runtime. Very likely removable, but removing it means re-downloading to re-export
ONNX. It gets its own decision rather than riding along with the obvious dead
weights.

## Verification

The project's existing bar, unchanged:

- 169 tests pass.
- A real Pro-tier LX3 cull gives `Strong=62 Mid=324 Weak=128` with **zero**
  per-photo score or grade drift against a Pro baseline. Bucket counts alone are
  not sufficient — a Balanced-tier run reproduces the counts while 489/514
  individual scores differ.
- Pin the tier: `SIGLIP_TIER=high SIGLIP_MIN_FREE_RAM_GB=1.2
  SIGLIP_HARD_MIN_RAM_GB=1.0`. Below 3 GB free the ladder silently drops to
  Balanced.
- Leave `cache/encoder_source.txt` at its live value so embeddings are reused —
  isolates the change under test.
- `cache/catalog.json` still holds all 1,745 photos afterwards.

New tests:

- Retention keeps no more than the window's versions and never drops the
  current one.
- A table remains readable and row-complete after cleanup.
- Quarantine leaves the live model set intact — the app loads and grades.

## Actual result (2026-08-03, measured)

**The projection above was wrong. Recorded here rather than edited away, because
the error is the useful part.**

| | Projected | Actual |
|---|---|---|
| `cache/` | ~0.2 GB | **0.15 GB** (894 MB → 18 MB) ✅ |
| `models/` | ~9.7–16.8 GB | **22.1 GB** (only 0.74 GB removable) |
| **Total reclaimed** | ~7–11 GB | **~1.6 GB** |

The LanceDB half landed as designed: 894 MB → 18 MB across three tier tables,
row counts unchanged (1829 / 707 / 514), and growth is now bounded by a
retention window instead of unbounded in the number of culls.

The model-weight half was almost entirely a mirage. Two of the three candidates
I named from `du` output and a stale project note are live:

- **`deepseek-r1-8b-q5.gguf` (5.4 GB) is live.** It backs `jury_engine.py`,
  `creative_director_agent.py`, `pdf_rag.py` and two paths in `server.py`. The
  project note recording DeepSeek's removal referred to the *grader*; the GGUF
  was later repurposed as the Jury engine.
- **`models/siglip2` (7.1 GB) is a live fallback** — the high tier's open_clip
  checkpoint (`siglip2_encoder.py:38`), reachable via `SIGLIP_ENC_USE_OC`.
- **`ViT-L-14.pt` (0.74 GB) is genuinely dead.** `human_grader.py:9` documents
  its only consumer, "a redundant manual ViT-L/14 CLIP load", as removed.

**Lesson: audit before quantifying.** Every number in the original projection
came from directory sizes plus recollection. The two-pass audit contradicted it
within minutes of first being run against the real tree.

The audit also caught three of its own false-positive bugs, each of which would
have quarantined a live weight — immediate-parent-only directory matching,
counting `.md` as source (which let this very spec keep a weight alive), and
whitespace-splitting the traced command. All three now have tests.

## Where the disk actually is

`models/` is 22.1 GB and essentially all of it is in use. The largest single
live weight is `deepseek-r1-8b-q5.gguf` at 5.4 GB, loaded through llama.cpp at
`jury_engine.py:108` with `n_ctx=1024`. Replacing it with a 3–4B at Q5
(~2–2.5 GB) would reclaim more than this entire spec did.

Two open questions, deliberately not answered here:

1. Does R1's `<think>` output overflow `n_ctx=1024` and truncate verdicts? If
   so that is a live bug, and it argues for a non-reasoning model.
2. Can a smaller model match it on jury critique? Unlike the VLM grader
   tournament there is no numeric ground truth, so this needs blind pairwise
   preference judged by the user.

Against Lightroom Classic's 4.2 GB install plus 22.7 GB of *growing* previews,
FrameGrade's ~22 GB remains a fixed cost. That comparison still holds — it was
never the model weights that made it hold.

## Risks

| Risk | Mitigation |
|---|---|
| A weight is loaded on a path neither audit pass exercises | Quarantine, not deletion; full cull + suite before the user deletes |
| Version cleanup removes a version another tier's reader needs | `delete_unverified=False`; 7-day window; all three tier tables exercised |
| Cleanup fails and breaks a cull | Non-fatal `try/except`, runs after grades are committed |
| Reclaimed space is smaller than projected | Numbers are measured, not estimated; `siglip2` (7.1 GB) is the one uncertain line and is decided separately |
