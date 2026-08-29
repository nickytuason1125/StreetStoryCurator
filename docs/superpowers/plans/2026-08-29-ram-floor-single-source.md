# RAM Floor Single Source — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_profile.required_ram_gb()` the only place the cull RAM floor is stated, then commit the uncommitted RAM/draft-decode batch and regenerate the stale accuracy audit.

**Architecture:** The working tree already moved the effective floor from a hardcoded `1.8` to a computed `~3.8` GB, but four literal `1.8`s survived in display paths — three in the frontend and one in the Rust telemetry shim. All four fail in the under-warning direction: they promise the user 1.8 GB is enough, then the gate 503s at 3.8. The fix removes the literals rather than updating them, so the next floor move cannot desync anything: the frontend renders nothing until the server tells it the floor, and the Rust shim computes the same table from the same env vars, locked by a test.

**Tech Stack:** Python 3 / FastAPI (backend), React + TypeScript + Vite (frontend), Rust + axum (telemetry shim), pytest, hand-rolled `scripts/test-*.mjs` node tests, `cargo test`.

**Spec:** No separate spec document. The authority for every number here is the measurement block in `src/run_profile.py` above `_RAM_NEED_GB` (measured 2026-08-28 with a 0.2 s process-tree sampler) and the rationale comments in the uncommitted diff to `routers/grading.py` and `server_impl.py`. Read those two comment blocks before starting — they explain why encoder-scoped `TierSpec.ram_hard_gb` and whole-cull `required_ram_gb()` are different questions, and why conflating them was the original bug.

## Global Constraints

- Repo root for every command in this plan: `street-story-curator/`. Paths are relative to it.
- Python is the venv interpreter: `./venv/Scripts/python.exe`. The system Python lacks uvicorn and will not run this project.
- Shell is Git Bash (POSIX sh) on Windows. Use forward slashes.
- The floor is **never** written as a literal in display code. Exactly one Python function (`run_profile.required_ram_gb`) and one Rust mirror (locked by a parity test) may contain the numbers.
- The env overrides `FRAMEGRADE_MIN_RAM_GB` (absolute override, wins when `> 0`) and `FRAMEGRADE_DRAFT_DECODE` (`"0"` disables draft decode, anything else enables) must behave identically in the Python and Rust implementations.
- Do not change `TierSpec.ram_hard_gb` or `_GRADE_MIN_RAM_GB`'s role. Raising the tier floors deletes the smaller encoder tiers — that regression has been made before and is documented in `run_profile.py`.
- Frontend tests are plain node scripts under `frontend/scripts/`, named `test-*.mjs`, discovered automatically by `scripts/run-tests.mjs`. Do not add vitest or any test framework.
- Commit messages follow the repo's existing voice: lowercase conventional prefix, then a sentence describing the user-visible consequence (`fix: the Grade button could not start a grade`).

---

### Task 1: The frontend stops restating the floor

**Files:**
- Modify: `frontend/src/lib/grading.ts:8-33` (the `ramReadiness` function)
- Modify: `frontend/src/App.tsx:2095` (critical-tone status row)
- Modify: `frontend/src/App.tsx:2370` (RAM chip popover)
- Test: `frontend/scripts/test-ram-floor.mjs` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ramReadiness(gs)` gains a `min: number | null` field in its return object. Full return type after this task:
  ```ts
  {
    level: 'clear' | 'tight' | 'critical' | 'unknown';
    free: number | null;
    total: number | null;
    percent: number | null;
    min: number | null;
    readout: string;
    tip: string;
  }
  ```
  Task 2 does not consume this. No other task does.

Background: `ramReadiness` currently does `const min = gs?.ram_min_gb ?? 1.8;` and then both `App.tsx` call sites *independently* re-read `sysRam?.ram_min_gb ?? graderStatus?.ram_min_gb ?? 1.8`. Three copies of one number. The fix is for `ramReadiness` to be the single reader and to return what it read, so the call sites have nothing left to guess. When the server has not answered yet, the honest answer is "unknown" — and `App.tsx` already renders nothing for `level === 'unknown'`, at both sites.

- [ ] **Step 1: Write the failing test**

Create `frontend/scripts/test-ram-floor.mjs`:

```js
// The cull RAM floor is served by the backend as ram_min_gb (computed by
// run_profile.required_ram_gb). It has moved once already — 1.8 -> 3.8 — and
// the move left three stale literal 1.8s in display code, each of which told
// the photographer that 1.8 GB was enough while the gate refused the cull at
// 3.8. Every one of those failures under-warns, which is the dangerous
// direction.
//
// These lock the rule: ramReadiness is the ONLY reader of ram_min_gb, it
// reports what it read as `min`, and when the server has not said, the answer
// is 'unknown' rather than an invented number.
//
// Run:  node scripts/test-ram-floor.mjs
import { ramReadiness } from '../src/lib/grading.ts';

let failed = 0;
const check = (name, actual, expected) => {
  const ok = Object.is(actual, expected);
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}` + (ok ? '' : `  (got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)})`));
};

// It reports the floor the server sent, verbatim.
check('min is the served floor',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 3.8 }).min, 3.8);
check('min follows the server when the floor moves',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 6.6 }).min, 6.6);

// No floor from the server = unknown. Never a guess.
check('absent floor is unknown',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44 }).level, 'unknown');
check('absent floor reports no min',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44 }).min, null);
check('null payload is unknown',
      ramReadiness(null).level, 'unknown');
check('absent free RAM is still unknown',
      ramReadiness({ ram_min_gb: 3.8 }).level, 'unknown');

// The readiness bands are driven by the SERVED floor, not by a baked-in one.
// At a 3.8 floor, 3.0 GB free is below it: critical.
check('3.0 free under a 3.8 floor is critical',
      ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 3.8 }).level, 'critical');
// The same 3.0 GB under the old 1.8 floor was merely tight — proving the band
// actually follows the server rather than a constant.
check('3.0 free under a 1.8 floor is tight',
      ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 1.8 }).level, 'tight');
check('9.0 free under a 3.8 floor is clear',
      ramReadiness({ ram_free_gb: 9.0, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 3.8 }).level, 'clear');

// The tip must quote the served floor, so the popover and the chip cannot
// disagree with each other or with the gate.
const crit = ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 3.8 });
check('tip quotes the served floor', crit.tip.includes('3.8'), true);
check('tip does not quote the dead 1.8 floor', crit.tip.includes('1.8'), false);

console.log(failed ? `\n${failed} FAILED` : '\nall green');
process.exit(failed ? 1 : 0);
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && node --experimental-strip-types scripts/test-ram-floor.mjs`

Expected: FAIL. Several lines fail — `min is the served floor` gets `undefined` (the field does not exist yet), and `absent floor is unknown` gets `'clear'` (the current code substitutes `1.8` and carries on).

- [ ] **Step 3: Make `ramReadiness` the single reader**

In `frontend/src/lib/grading.ts`, replace the whole `ramReadiness` function (currently lines 8-33) with:

```ts
export function ramReadiness(gs: any): {
  level: 'clear' | 'tight' | 'critical' | 'unknown';
  free: number | null; total: number | null; percent: number | null;
  min: number | null; readout: string; tip: string;
} {
  const free    = gs?.ram_free_gb ?? null;
  const total   = gs?.ram_total_gb ?? null;
  const percent = gs?.ram_percent ?? null;
  // The floor is the SERVER'S, always. It is computed per-machine by
  // run_profile.required_ram_gb() from a measured table, and it has already
  // moved once (1.8 -> 3.8). A literal default here would be a second source
  // of truth that goes stale silently and under-warns — which is how the
  // photographer gets told 1.8 GB is fine and then gets a 503.
  const min: number | null = typeof gs?.ram_min_gb === 'number' ? gs.ram_min_gb : null;
  const unknown = { level: 'unknown' as const, free, total, percent, min,
                    readout: '', tip: 'System memory unknown' };
  if (free == null || min == null) return unknown;
  // Headroom is the only number that changes a decision here, and it is the
  // only one the grade floor is expressed in, so it is the only one the chip
  // carries. Printing "% in use" beside it made this the widest element in a
  // toolbar that already needs 2008px of a 1500px window; the percentage is
  // still in the tooltip and in the popover, where there is room for context.
  const readout = percent != null || total != null
    ? `${free.toFixed(1)} GB free`
    : `${free.toFixed(1)} GB`;
  const usedTip = percent != null ? ` (${percent.toFixed(0)}% in use — matches Task Manager)` : '';
  // "clear" needs headroom above the floor, not merely clearance of it: a cull
  // that starts at exactly the floor has nothing left for the browser beside
  // it. 1.2 GB of margin, with an absolute 5 GB backstop for the case where a
  // low floor would otherwise call a genuinely tight machine clear.
  const clearThresh = Math.max(min + 1.2, 5.0);
  if (free < min)           return { level: 'critical', free, total, percent, min, readout, tip: `Only ${free.toFixed(1)} GB free${usedTip} — grading needs ~${min} GB. Close some apps before grading.` };
  if (free < clearThresh)   return { level: 'tight',    free, total, percent, min, readout, tip: `${free.toFixed(1)} GB free${usedTip} — enough to grade, but close Chrome or other heavy apps first for a stable cull.` };
  return { level: 'clear', free, total, percent, min, readout, tip: `${free.toFixed(1)} GB free${usedTip} — clear to grade.` };
}
```

Note the `clearThresh` comment changed too: the old one justified the 5 GB backstop with SigLIP encoder load figures (~2 GB + ~1 GB baseline) that the 2026-08-28 whole-tree measurement superseded. The threshold value is unchanged; only the reasoning is corrected.

- [ ] **Step 4: Delete the two literals in App.tsx**

In `frontend/src/App.tsx` line 2095, inside the `critical:` row, replace:

```tsx
                  critical: { col:T.alarmCrit, text:`Low system memory — only ${r.free?.toFixed(1)} GB free, below the ~${(sysRam?.ram_min_gb ?? graderStatus?.ram_min_gb ?? 1.8)} GB needed. Close some apps before grading.` },
```

with:

```tsx
                  critical: { col:T.alarmCrit, text:`Low system memory — only ${r.free?.toFixed(1)} GB free, below the ~${r.min} GB needed. Close some apps before grading.` },
```

Then at line 2367-2370, replace the comment and the `ramFloorGb` binding:

```tsx
          // Served by /api/system/ram as ram_min_gb (_GRADE_MIN_RAM_GB). Never
          // restate it as a literal here: two copies of the floor were baked into
          // display strings and would have gone stale the moment the gate moved.
          const ramFloorGb = sysRam?.ram_min_gb ?? graderStatus?.ram_min_gb ?? 1.8;
```

with:

```tsx
          // Served by /api/system/ram as ram_min_gb (_GRADE_MIN_RAM_GB), read
          // ONCE by ramReadiness and handed back as r.min. Never restate it as
          // a literal here: three copies of the floor were baked into display
          // strings and every one went stale the moment the gate moved from
          // 1.8 to 3.8, each of them under-warning the photographer.
          const ramFloorGb = r.min;
```

`r.min` cannot be null at this point — the enclosing block returns early on `r.level === 'unknown'`, and `unknown` is exactly the state where `min` is null. Both `App.tsx` sites sit inside that guard.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && node --experimental-strip-types scripts/test-ram-floor.mjs`

Expected: PASS on every line, ending `all green`.

- [ ] **Step 6: Confirm no literal floor survives in the frontend**

Run: `cd frontend && grep -rn "1\.8" src/ | grep -iv "0\.18\|1\.85\|z-\|leading"`

Expected: no output. If any line comes back, it is a fourth copy of the floor — remove it the same way.

- [ ] **Step 7: Run the full frontend gate**

Run: `cd frontend && npm run test && npx tsc --noEmit`

Expected: `Frontend tests OK — 2 file(s).` then tsc exits silently with code 0. The new file is picked up automatically by `scripts/run-tests.mjs`, which discovers every `scripts/test-*.mjs`.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/lib/grading.ts frontend/src/App.tsx frontend/scripts/test-ram-floor.mjs
git commit -m "fix(ui): the RAM floor is the server's number, not three stale copies of it"
```

---

### Task 2: The Rust telemetry shim serves the real floor

**Files:**
- Modify: `native/framegrade-rs/src/main.rs:63-75` (the `system_ram` handler), plus a new function and a new test module
- Test: `native/framegrade-rs/src/main.rs` (`#[cfg(test)] mod tests` at the end of the file)

**Interfaces:**
- Consumes: nothing from Task 1. This task is independent and can be done in either order.
- Produces: two Rust functions, used only within this file and its tests:
  ```rust
  fn ram_need_gb(draft: bool, n_photos: u32, override_gb: Option<f64>) -> f64
  fn required_ram_gb(n_photos: u32) -> f64
  ```

Background: `native/framegrade-rs` is "slice 1" of a native orchestrator whose stated contract is *byte-compatible response shapes, so the React frontend can point at this process without knowing the difference*. Its `/api/system/ram` handler hardcodes `"ram_min_gb": 1.8`. After the Python change that is a live disagreement between two servers claiming to be interchangeable — and it is the shim that is wrong.

The shim cannot import `run_profile`, so this is a deliberate second implementation. Contain the risk by keeping the arithmetic in a pure function with no env or I/O, and locking the table with tests. The env-reading wrapper stays trivial enough to eyeball.

- [ ] **Step 1: Write the failing test**

Append to the end of `native/framegrade-rs/src/main.rs`:

```rust
// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ram_need_gb;

    // This crate claims byte-compatible responses with the Python server, so a
    // number it invents is a bug even when it is plausible. These lock the
    // table to src/run_profile.py::_RAM_NEED_GB. If that table is re-measured,
    // both sides move together or this fails — which is the point.
    //
    // Pure function, no env reads: env is process-global and `cargo test` runs
    // threads in parallel, so testing the wrapper directly would be flaky.

    #[test]
    fn draft_on_small_job_matches_python() {
        assert_eq!(ram_need_gb(true, 0, None), 3.8);
        assert_eq!(ram_need_gb(true, 58, None), 3.8);
        assert_eq!(ram_need_gb(true, 300, None), 3.8);
    }

    #[test]
    fn draft_on_large_job_uses_the_extrapolated_figure() {
        assert_eq!(ram_need_gb(true, 301, None), 4.2);
        assert_eq!(ram_need_gb(true, 5000, None), 4.2);
    }

    #[test]
    fn draft_off_roughly_doubles_the_requirement() {
        // Full-resolution decode was measured at 5.85-6.35 GB against
        // 2.89-3.60 GB drafting. The gate is deliberately above the measured
        // peak, not at it.
        assert_eq!(ram_need_gb(false, 0, None), 6.6);
        assert_eq!(ram_need_gb(false, 301, None), 7.0);
    }

    #[test]
    fn an_explicit_override_wins_over_every_branch() {
        assert_eq!(ram_need_gb(true, 0, Some(2.5)), 2.5);
        assert_eq!(ram_need_gb(false, 9000, Some(2.5)), 2.5);
    }

    #[test]
    fn a_zero_or_negative_override_is_ignored() {
        // Python treats "" and 0 as absent (`override if override > 0`). A
        // gate of 0 GB would admit every cull, so this must not be a way to
        // switch the gate off by accident.
        assert_eq!(ram_need_gb(true, 0, Some(0.0)), 3.8);
        assert_eq!(ram_need_gb(true, 0, Some(-1.0)), 3.8);
    }

    #[test]
    fn the_dead_floor_is_gone() {
        // 1.8 was Balanced's ENCODER floor masquerading as a whole-cull
        // budget. No branch may return it.
        for &draft in &[true, false] {
            for &n in &[0u32, 58, 300, 301, 5000] {
                assert_ne!(ram_need_gb(draft, n, None), 1.8);
            }
        }
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd native/framegrade-rs && cargo test`

Expected: FAIL at compile time — `cannot find function 'ram_need_gb' in this scope`. That is the correct failure; the function does not exist yet.

- [ ] **Step 3: Write the implementation**

In `native/framegrade-rs/src/main.rs`, immediately above the `// ── handlers: telemetry ──` comment (currently around line 61), insert:

```rust
// ── cull RAM requirement ────────────────────────────────────────────────────
// MIRRORS src/run_profile.py::required_ram_gb. Two implementations of one
// number is exactly the drift that put a stale `1.8` in this file's
// system_ram handler while the Python gate had moved to 3.8 — the shim told
// the photographer a cull would fit and the server then refused it.
//
// This crate cannot import Python, so the table is copied verbatim and locked
// by the tests at the bottom of this file. If run_profile's measurements are
// redone, change both or the tests fail.
//
//   photos   draft ON   draft OFF
//    <=300     3.8 GB     6.6 GB
//     >300     4.2 GB     7.0 GB

/// Pure arithmetic — no env, no I/O — so it is testable without races.
fn ram_need_gb(draft: bool, n_photos: u32, override_gb: Option<f64>) -> f64 {
    // Python: `return override if override > 0 else need`. A zero or negative
    // override means "unset", never "no floor".
    if let Some(v) = override_gb {
        if v > 0.0 {
            return v;
        }
    }
    match (draft, n_photos <= 300) {
        (true, true) => 3.8,
        (true, false) => 4.2,
        (false, true) => 6.6,
        (false, false) => 7.0,
    }
}

/// Env-reading wrapper. Same two variables the Python side honours:
/// FRAMEGRADE_MIN_RAM_GB (absolute override) and FRAMEGRADE_DRAFT_DECODE
/// ("0" disables scaled decode, which roughly doubles the requirement).
fn required_ram_gb(n_photos: u32) -> f64 {
    let override_gb = std::env::var("FRAMEGRADE_MIN_RAM_GB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());
    let draft = std::env::var("FRAMEGRADE_DRAFT_DECODE")
        .map(|s| s.trim() != "0")
        .unwrap_or(true);
    ram_need_gb(draft, n_photos, override_gb)
}
```

Then in the `system_ram` handler, replace:

```rust
        "ram_min_gb":   1.8,
```

with:

```rust
        // n_photos = 0: this endpoint is polled with no job in front of it, so
        // it reports the floor for a small cull. The real per-cull gate lives
        // in the Python grading router, which knows the folder size.
        "ram_min_gb":   required_ram_gb(0),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd native/framegrade-rs && cargo test`

Expected: `test result: ok. 6 passed; 0 failed`.

- [ ] **Step 5: Confirm the crate still builds clean**

Run: `cd native/framegrade-rs && cargo build 2>&1 | tail -20`

Expected: `Finished` with no warnings about `required_ram_gb` being unused (it is used by the handler).

- [ ] **Step 6: Commit**

```bash
git add native/framegrade-rs/src/main.rs
git commit -m "fix: the native shim reported the dead 1.8 GB floor as if it were live"
```

---

### Task 3: Stop tracking the scratch audit artefacts

**Files:**
- Modify: `.gitignore`
- Delete from the working tree: `accuracy_report.txt`, `tsc_audit.txt`
- Restore: `frontend/src/lib/grading.ts` — **only if** Task 1 was skipped (see Step 1)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks. Task 4 relies on `git status` being clean of these two files.

Background: `accuracy_report.txt` (2026-08-25) and `tsc_audit.txt` (13 bytes, containing `EXITCODE=0`) are generated scratch output sitting untracked in the repo root. `.gitignore` already ignores the closed VLM tournament's `/_ab_*` set and `*.log` by the same logic; these two were simply missed. `accuracy_report.txt` is regenerated in Task 5.

- [ ] **Step 1: Check for a content-free diff on grading.ts**

Run: `git diff --stat frontend/src/lib/grading.ts`

If Task 1 has already run, this shows real changes — leave them alone and skip to Step 2.

If Task 1 has *not* run and this shows a changed file with **zero** added/removed lines, the diff is line-endings only (LF vs CRLF) and must not enter a commit. Restore it:

```bash
git checkout -- frontend/src/lib/grading.ts
```

- [ ] **Step 2: Write the failing check**

Run: `git status --short | grep -E "accuracy_report\.txt|tsc_audit\.txt"`

Expected (the failure being fixed): two lines, both prefixed `??`.

- [ ] **Step 3: Add the ignore rules**

In `.gitignore`, find the existing block:

```
# Throwaway experiment + debug artefacts at the repo root. The _ab_* set is the
# closed 2026-06 VLM tournament; nothing outside that set imports them.
/_ab_*
/_cull_*.out
/_dl_*.out
/_tmp_*.py
/_train_baseline.py
/srv_out*.txt
/srv_err*.txt
```

and append two lines to it, so the whole block reads:

```
# Throwaway experiment + debug artefacts at the repo root. The _ab_* set is the
# closed 2026-06 VLM tournament; nothing outside that set imports them.
/_ab_*
/_cull_*.out
/_dl_*.out
/_tmp_*.py
/_train_baseline.py
/srv_out*.txt
/srv_err*.txt
# Generated audit output. Both are REPORTS, regenerated on demand, and both
# went stale in the tree: accuracy_report.txt sat three days behind the fix to
# its own nan bug and was read as if current.
/accuracy_report.txt
/tsc_audit.txt
```

- [ ] **Step 4: Remove the stale copies from the working tree**

```bash
rm -f accuracy_report.txt tsc_audit.txt
```

These are regenerable: `accuracy_report.txt` comes from Task 5, and `tsc_audit.txt` is just captured output of `npx tsc --noEmit`.

- [ ] **Step 5: Run the check to verify it passes**

Run: `git status --short | grep -E "accuracy_report\.txt|tsc_audit\.txt" ; echo "exit=$?"`

Expected: no matching lines, and `exit=1` (grep found nothing). The only change `git status` should now show for this task is `M .gitignore`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore the generated audit reports; a stale one was read as current"
```

---

### Task 4: Verify and commit the RAM / draft-decode batch

**Files:**
- Commit (no edits): `src/fast_ingestion.py`, `src/run_profile.py`, `src/grade_pipeline_v2.py`, `src/vision_grading_heads.py`, `routers/grading.py`, `server_impl.py`, `tests/test_ram_sensitivity.py`, `tests/test_draft_decode.py`

**Interfaces:**
- Consumes: Tasks 1-3 complete, so the tree holds no line-ending-only diffs and no untracked scratch.
- Produces: nothing. This is the shipping gate.

Background: this is roughly 285 lines of measured work — draft decode (JPEG DCT-domain downscaling, 279 ms/img decode against 3 ms of model time), the encoder-floor / whole-cull-budget split, the D-FINE 640-not-512 finding, and the `FRAMEGRADE_LUM_DRAFT` default flip after a verified zero-bucket-change diff. It is complete and tested; it has simply never been committed. Do not restructure it — commit it.

- [ ] **Step 1: Confirm what is about to be committed**

Run: `git status --short && git diff --stat`

Expected: exactly eight modified files (the list above, minus the two `tests/` files, which are untracked) plus untracked `tests/test_draft_decode.py`. If `models/qwen3_vl/` still appears as untracked, leave it — it is handled by its own plan (`2026-08-29-qwen3-vl-disposition.md`) and must not enter this commit.

- [ ] **Step 2: Run the tests that cover this change**

Run: `./venv/Scripts/python.exe -m pytest tests/test_ram_sensitivity.py tests/test_draft_decode.py -q`

Expected: `28 passed`.

- [ ] **Step 3: Run the wider backend suite for regressions**

Run: `./venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -20`

Expected: green, or the same failures the tree had before this batch. If something fails, check it against a stash first:

```bash
git stash push -- src/ routers/ server_impl.py
./venv/Scripts/python.exe -m pytest tests/ -q 2>&1 | tail -20
git stash pop
```

A failure present in both runs is pre-existing and not this batch's to fix. A failure only in the unstashed run is a real regression — stop and fix it before committing.

- [ ] **Step 4: Confirm the gate and the display agree**

Run:

```bash
./venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import run_profile as rp; print('display floor (n=0):', rp.required_ram_gb(0)); print('small cull  (n=58):', rp.required_ram_gb(58)); print('large cull (n=5000):', rp.required_ram_gb(5000)); print('draft decode on:', rp.draft_decode_enabled())"
```

Expected exactly:

```
display floor (n=0): 3.8
small cull  (n=58): 3.8
large cull (n=5000): 4.2
draft decode on: True
```

If `draft decode on` prints `False`, something has `FRAMEGRADE_DRAFT_DECODE=0` set in the environment and the floors will read 6.6/7.0. Unset it and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/fast_ingestion.py src/run_profile.py src/grade_pipeline_v2.py \
        src/vision_grading_heads.py routers/grading.py server_impl.py \
        tests/test_ram_sensitivity.py tests/test_draft_decode.py
git commit -m "perf: decode JPEGs at the size the cull actually uses, and gate on what a cull really costs

Decode was 99% of the quality stage — 279 ms/img against 3 ms of model — because
every frame was decoded at full resolution and then shrunk to <=512 px. Drafting
in the DCT domain takes a 7.7 MB JPEG from 325 to 83 ms and a 50 MB one from 772
to 374 ms, with 36-64x less RAM per image, and moved no grade bucket on either
of two verified folders.

The RAM gate moves with it. It was 1.8 GB — Balanced's ENCODER floor, which
answers 'which encoder fits', not 'will this cull fit'. It admitted culls that
then drove the machine to 0.10 GB free and into the pagefile (111 s against 25 s
for the same folder with room). required_ram_gb() now answers the second
question from a measured whole-process-tree table, and the gate runs after
folder resolution so it can see the job size.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQsEXLPg6yw7GbjZC5B64r"
```

- [ ] **Step 6: Verify the tree is clean**

Run: `git status --short`

Expected: only `?? models/qwen3_vl/` remains (handled by its own plan). Nothing else.

---

### Task 5: Regenerate the accuracy audit

**Files:**
- Run (no edits): `scripts_accuracy_report.py`
- Create: `docs/superpowers/specs/2026-08-29-accuracy-baseline.md`

**Interfaces:**
- Consumes: Task 4 committed, so the report describes a known tree state.
- Produces: a recorded baseline that `2026-08-29-rank-agreement-investigation.md` Task 1 reads as its "before" figure.

Background: the report on disk was generated 2026-08-25 and printed `Spearman rho = +nan` for section 3. That nan was **already fixed** two days later in `95998de` — `personal_shift_lines()` now prints `Spearman rho = undefined` plus a named cause. The file was simply never regenerated, and was subsequently read as if current. Regenerating it costs one command and settles what the real numbers are.

- [ ] **Step 1: Confirm the ratings store is present**

Run: `./venv/Scripts/python.exe -c "import json; d=json.load(open('cache/user_ratings.json')); r=d.get('ratings',d); print(len(r),'ratings')"`

Expected: `124 ratings` (or more, if photos have been rated since).

- [ ] **Step 2: Run the audit**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py | tee accuracy_report.txt`

Expected: four sections printed — the header counts, `1. RANK AGREEMENT`, `2. BAND MONOTONICITY`, and `3. PERSONAL SHIFT`. Section 3 must **not** contain the string `nan`; if it has no usable data it now says `Spearman rho = undefined` followed by a sentence naming the cause.

- [ ] **Step 3: Verify the nan regression cannot come back**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: green, including `test_personal_shift_names_the_cause_instead_of_printing_nan`.

- [ ] **Step 4: Record the baseline**

Create `docs/superpowers/specs/2026-08-29-accuracy-baseline.md`, pasting the real output from Step 2 into the fenced block:

```markdown
# Accuracy Baseline — 2026-08-29

Recorded immediately after the RAM/draft-decode batch was committed, so the
rank-agreement investigation has a dated "before" to argue against. Supersedes
the 2026-08-25 `accuracy_report.txt`, which predated the fix in `95998de` and
printed `+nan` for section 3.

Generated by: `./venv/Scripts/python.exe scripts_accuracy_report.py`

## Output

```
<paste the full Step 2 output here, verbatim>
```

## Reading

- **Rank agreement** is the legitimacy metric. Record the rho and the n.
- **Band monotonicity** can PASS on a trivial margin. Record the three means,
  not just the verdict — Strong 4.02 / Mid 3.95 on the 2026-08-25 run was a
  0.07-star gap, which passes the ordering check while telling you the bands
  barely separate.
- **Personal shift** reports `undefined` rather than a number when
  `personal_score` has no variance. If it says undefined, the named cause is
  the finding — the metric is not broken, the input is degenerate.
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-29-accuracy-baseline.md
git commit -m "docs: record the accuracy baseline the stale report was hiding"
```

Note `accuracy_report.txt` itself is not added — Task 3 made it ignored. The durable record is the dated spec.

---

## Self-Review

**Spec coverage.** No formal spec exists; the authority is `run_profile.py`'s measurement block and the diff's own rationale comments, both named in the header. Every literal `1.8` found in the audit is assigned to a task: `grading.ts:15` and `App.tsx:2095,2370` to Task 1, `main.rs:73` to Task 2. The two untracked scratch reports go to Task 3, the uncommitted batch to Task 4, the stale audit to Task 5.

**Placeholder scan.** One deliberate placeholder remains: `<paste the full Step 2 output here, verbatim>` in Task 5 Step 4. It cannot be filled in advance — the numbers come from running the script — and the step says exactly which command produces it.

**Type consistency.** `ramReadiness` returns `min: number | null` (Task 1 Step 3); both `App.tsx` call sites read `r.min` (Step 4) and sit inside the `level === 'unknown'` early return, which is the only state where `min` is null. The Rust `ram_need_gb(draft: bool, n_photos: u32, override_gb: Option<f64>) -> f64` signature is identical in the test module (Task 2 Step 1) and the implementation (Step 3); `required_ram_gb(n_photos: u32) -> f64` is called as `required_ram_gb(0)` in the handler, matching. The Rust table matches Python's `_RAM_NEED_GB` on all four branches.
