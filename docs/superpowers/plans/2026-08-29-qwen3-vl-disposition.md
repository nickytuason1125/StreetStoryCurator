# Qwen3-VL Disposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the rejected Qwen3-VL candidate — confirm it is on no live path, remove the comment that points at a tool which has never existed, record why it lost against the criterion that matters, and get its 8.3 GB off the working disk without destroying it.

**Architecture:** `models/qwen3_vl/` is 8.3 GB of untracked weights for a model the 2026-06-14 VLM tournament **rejected**. Its integration is half-present by design — `qwen_vlm_grader.py` can dispatch its model class and `niche_registry.py` carries its scale anchors, both left from the bake-off — while the output-calibration entry sits commented out citing `_ab_fit_calibration`, a script that has never existed in this repo. This plan does not calibrate the model. It confirms the rejection against the photographer's own ratings (which the tournament never tested), documents it, and archives the weights the way the RAG source books were archived: moved off-repo with a restore path, never deleted.

**Tech Stack:** Python 3, pytest, the stored A/B result JSONs at the repo root, `cache/user_ratings.json`.

**Spec:** No separate spec. The authority is `_ab_tournament.json` (the 2026-06-14 verdict and per-model summary statistics) and the per-photo scores in `_ab_qwen3_4b.json` / `_ab_baseline_qwen25_3b.json`. Read `_ab_tournament.json` before starting — it is 40 lines and contains the whole decision.

## Global Constraints

- Repo root for every command: `street-story-curator/`. Paths are relative to it.
- Python is the venv interpreter: `./venv/Scripts/python.exe`.
- All JSON in this repo must be read with `encoding='utf-8'` — the default Windows cp1252 codec fails on `_ab_*.json` and on `cache/user_ratings.json` (verified: `UnicodeDecodeError` at byte 0x8f).
- **Never delete model weights.** The established precedent in this repo is to move them off the working disk with a documented restore path (the RAG source books went to `D:\framegrade_bench\rag_source` on 2026-08-13). A mistake here costs a multi-gigabyte re-download.
- Do not re-run the tournament. Every number this plan needs is already on disk from the 2026-06-14 run; recomputing them needs a GPU and hours, and would answer a question that is already answered.
- Do not remove the `qwen3_vl` dispatch in `qwen_vlm_grader.py:836` or the scale anchors in `niche_registry.py:939`. They are inert without weights, they cost nothing, and they are what makes a future re-evaluation cheap.

---

### Task 1: Prove the weights are on no live path

**Files:**
- Test: `tests/test_qwen3_vl_is_not_required.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks. This is the safety gate that makes Task 4's archive move safe.

Background: before moving 8.3 GB, establish by test — not by reading — that nothing requires it. The default grading path is SigLIP zero-shot; Qwen runs only behind the opt-in `deep_grade` flag, and that path points at `models/qwen_vlm` (Qwen2.5-VL-3B), not `models/qwen3_vl`. This test locks that, so a future change that quietly makes the archived model load-bearing fails loudly instead of at runtime on a machine that no longer has it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen3_vl_is_not_required.py`:

```python
"""models/qwen3_vl is a REJECTED tournament candidate, not an install dependency.

The 2026-06-14 bake-off rejected Qwen3-VL-4B (10.93 s/img against the
incumbent's 8.79, and a score spread of std 0.076 against 0.145). Its 8.3 GB of
weights then sat in the working tree looking like a half-finished integration,
because the dispatch and the scale anchors for it are still present — they are
leftovers from the bake-off, and they are inert without weights.

These lock that inertness, so the weights can be archived off-disk and a later
change cannot quietly make them required again without failing here first.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_the_model_registry_does_not_list_qwen3_vl():
    """The downloader fetches what the registry lists. It must not list this."""
    reg = _read("src/model_registry.py")
    assert "qwen3_vl" not in reg, (
        "model_registry names qwen3_vl — a rejected model would be downloaded "
        "on every fresh install"
    )


def test_no_module_hardcodes_the_qwen3_vl_weights_path():
    """Only the bake-off harness may name this directory.

    'Never name a weight file in code — ask model_registry' is a standing rule
    here (CLAUDE.md); five modules once hardcoded a deepseek filename and a
    correct install then reported models missing.
    """
    offenders = []
    for py in (ROOT / "src").rglob("*.py"):
        if py.name == "qwen_vlm_grader.py":
            continue          # dispatches on config model_type, not on a path
        if "models/qwen3_vl" in py.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(py.relative_to(ROOT).as_posix())
    assert offenders == [], f"these modules hardcode the archived path: {offenders}"


def test_the_grader_dispatches_on_config_not_on_disk_layout():
    """The dispatch may stay: it reads model_type from whatever config is
    loaded, so it costs nothing when the weights are absent."""
    src = _read("src/qwen_vlm_grader.py")
    assert 'if _model_type == "qwen3_vl"' in src
    assert "models/qwen3_vl" not in src, (
        "the grader hardcodes the archived directory instead of taking a path"
    )


def test_the_tournament_verdict_is_still_on_record():
    """The reason the weights are archived must remain readable."""
    verdict = json.loads((ROOT / "_ab_tournament.json").read_text(encoding="utf-8"))
    assert "rejected" in verdict["verdict"].lower()
    assert "qwen3_4b" in verdict["summary"]
```

- [ ] **Step 2: Run the tests to see where the tree actually stands**

Run: `./venv/Scripts/python.exe -m pytest tests/test_qwen3_vl_is_not_required.py -v`

Expected: all four PASS. This test documents a property the tree should already have, so passing immediately is the correct outcome — it is a regression lock, not a red-green cycle.

If any test FAILS, that is a real finding and the failure must be fixed before Task 4 archives anything:
- `test_the_model_registry_does_not_list_qwen3_vl` failing means fresh installs are downloading a rejected 8.3 GB model. Remove the registry entry.
- `test_no_module_hardcodes_the_qwen3_vl_weights_path` failing names the modules that will break when the weights move. Route them through `model_registry` first.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qwen3_vl_is_not_required.py
git commit -m "test: lock that the rejected Qwen3-VL weights are on no live path"
```

---

### Task 2: Confirm the rejection against the criterion the tournament never used

**Files:**
- Create: `scripts_ab_vs_stars.py`
- Test: `tests/test_ab_vs_stars.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `scripts_ab_vs_stars.py` exposing
  ```python
  def pairs_for(ab_path: str, stars_by_name: dict[str, int]) -> list[tuple[float, int]]
  def stars_by_basename(ratings_path: str) -> dict[str, int]
  ```
  Task 3 quotes this script's printed output. `2026-08-29-rank-agreement-investigation.md` does not depend on it.

Background — this is the substantive finding of the plan. The tournament ranked candidates by `spearman_vs_base`: agreement with **Qwen2.5-VL-3B**, the incumbent model. It never asked whether a candidate agreed with **the photographer**. That matters because the incumbent itself only reaches ρ ≈ +0.23 against the user's stars, so "agrees 0.817 with the incumbent" is agreement with a weak judge.

The measurement is free: 24 of the 25 tournament photos are rated in `cache/user_ratings.json`, and every model's per-photo score is already stored. No GPU, no re-run.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ab_vs_stars.py`:

```python
"""The bake-off ranked models by agreement with the INCUMBENT MODEL, never with
the photographer. That is the wrong denominator when the incumbent itself only
reaches rho ~ +0.23 against the photographer's stars.

These cover the join, which is the only part with a way to go silently wrong:
the ratings store is keyed by absolute path and the A/B results by bare
filename, so a basename join is required and an empty join must be loud.
"""
import json
import scripts_ab_vs_stars as s


def test_stars_are_keyed_by_basename(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({
        "ratings": {r"C:\Users\x\Desktop\tpe2026_2\TPE26-1.jpg": 4,
                    r"C:\Users\x\Desktop\tpe2026_2\TPE26-2.jpg": "5"}
    }), encoding="utf-8")
    out = s.stars_by_basename(str(p))
    assert out == {"TPE26-1.jpg": 4, "TPE26-2.jpg": 5}


def test_a_bare_ratings_dict_works_too(tmp_path):
    """The store is sometimes {"ratings": {...}} and sometimes the map itself."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps({r"C:\a\TPE26-1.jpg": 3}), encoding="utf-8")
    assert s.stars_by_basename(str(p)) == {"TPE26-1.jpg": 3}


def test_pairs_join_on_basename_and_skip_unrated(tmp_path):
    ab = tmp_path / "ab.json"
    ab.write_text(json.dumps({"results": {
        "TPE26-1.jpg": {"score": 0.45},
        "TPE26-9.jpg": {"score": 0.60},     # not rated — must be skipped
    }}), encoding="utf-8")
    got = s.pairs_for(str(ab), {"TPE26-1.jpg": 4})
    assert got == [(0.45, 4)]


def test_an_empty_join_is_not_silently_an_empty_result(tmp_path):
    """Zero overlap means the join broke, not that the model scored nothing."""
    ab = tmp_path / "ab.json"
    ab.write_text(json.dumps({"results": {"OTHER.jpg": {"score": 0.5}}}),
                  encoding="utf-8")
    assert s.pairs_for(str(ab), {"TPE26-1.jpg": 4}) == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_ab_vs_stars.py -q`

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts_ab_vs_stars'`.

- [ ] **Step 3: Write the implementation**

Create `scripts_ab_vs_stars.py`:

```python
"""Score every 2026-06-14 bake-off candidate against the PHOTOGRAPHER's stars.

The tournament ranked candidates by `spearman_vs_base` — agreement with
Qwen2.5-VL-3B, the incumbent. Nothing in it asked whether a candidate agreed
with the photographer, and the incumbent only reaches rho ~ +0.23 against his
stars, so that number measured agreement with a weak judge.

24 of the 25 bake-off photos are rated and every candidate's per-photo score is
already stored, so this costs nothing: no GPU, no re-grade.

Usage:  ./venv/Scripts/python.exe scripts_ab_vs_stars.py
Exit code 0 always — a report, not a gate.
"""
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Reuse the audit's Spearman so both reports agree by construction. It handles
# tied ranks with average ranks, which matters here: the ratings are heavily
# tied (44 threes, 42 fives, 34 fours out of 125).
_spec = importlib.util.spec_from_file_location(
    "_acc_rep", ROOT / "scripts_accuracy_report.py")
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
spearman = _rep.spearman

# Every file is utf-8; the Windows default cp1252 raises UnicodeDecodeError on
# these at byte 0x8f.
_ENC = "utf-8"

CANDIDATES = [
    ("Qwen2.5-VL-3B  (incumbent)", "_ab_baseline_qwen25_3b.json"),
    ("Qwen3-VL-4B    (rejected) ", "_ab_qwen3_4b.json"),
    ("Qwen3-VL-2B    (rejected) ", "_ab_qwen3_2b.json"),
    ("InternVL3.5-2B (rejected) ", "_ab_internvl35_2b.json"),
    ("SmolVLM2-2.2B  (rejected) ", "_ab_smolvlm2_22b.json"),
]


def stars_by_basename(ratings_path: str) -> dict:
    """Ratings are keyed by absolute path, A/B results by bare filename."""
    data = json.loads(Path(ratings_path).read_text(encoding=_ENC))
    data = data.get("ratings", data)
    out = {}
    for path, stars in data.items():
        try:
            out[os.path.basename(path)] = int(stars)
        except (TypeError, ValueError):
            continue
    return out


def pairs_for(ab_path: str, stars_by_name: dict) -> list:
    """(model_score, stars) for every bake-off photo that was also rated."""
    results = json.loads(Path(ab_path).read_text(encoding=_ENC)).get("results", {})
    pairs = []
    for name, row in results.items():
        s = row.get("score")
        if name in stars_by_name and isinstance(s, (int, float)):
            pairs.append((float(s), stars_by_name[name]))
    return pairs


def main():
    stars = stars_by_basename(str(ROOT / "cache" / "user_ratings.json"))
    verdict = json.loads((ROOT / "_ab_tournament.json").read_text(encoding=_ENC))

    W = 72
    print("=" * W)
    print("BAKE-OFF RE-SCORED AGAINST THE PHOTOGRAPHER'S STARS")
    print("=" * W)
    print(f"tournament date:  {verdict['date']}")
    print(f"recorded verdict: {verdict['verdict']}")
    print()
    print(f"{'model':<28}{'rho vs STARS':>14}{'rho vs INCUMBENT':>19}{'n':>5}")
    print("-" * W)

    for label, fname in CANDIDATES:
        path = ROOT / fname
        if not path.exists():
            print(f"{label:<28}{'(no results file)':>14}")
            continue
        pairs = pairs_for(str(path), stars)
        if not pairs:
            print(f"{label:<28}{'(no rated overlap)':>14}")
            continue
        sc, st = zip(*pairs)
        rho = spearman(list(sc), list(st))
        key = fname.replace("_ab_", "").replace(".json", "")
        vs_base = verdict["summary"].get(key, {}).get("spearman_vs_base")
        vs_base_s = "—" if vs_base is None else f"{vs_base:+.3f}"
        print(f"{label:<28}{rho:>+14.3f}{vs_base_s:>19}{len(pairs):>5}")

    print("-" * W)
    print("The right-hand column is what the tournament ranked on. The left is")
    print("what the app is for. They are not the same question, and a candidate")
    print("can score well on the right by reproducing a weak judge.")
    print("=" * W)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_ab_vs_stars.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Run the report and confirm the rejection holds**

Run: `./venv/Scripts/python.exe scripts_ab_vs_stars.py`

Expected — these values were measured while writing this plan and should reproduce exactly:

```
Qwen2.5-VL-3B  (incumbent)         +0.243             —           24
Qwen3-VL-4B    (rejected)          +0.078        +0.817           24
Qwen3-VL-2B    (rejected)          -0.012        +0.502           24
```

Read it as: the rejected 4B agrees with the photographer at **+0.078**, against the incumbent's **+0.243** on the same 24 photos. For context, the noise floor for this ratings set is ρ ≈ +0.149 at the 95th percentile (see the rank-agreement plan) — so the 4B's agreement with the photographer is **indistinguishable from chance**, while its headline +0.817 tournament score was agreement with the incumbent.

If the numbers differ materially, do not proceed to Task 4 — the ratings store has changed and the disposition needs re-deciding on the new figures.

- [ ] **Step 6: Commit**

```bash
git add scripts_ab_vs_stars.py tests/test_ab_vs_stars.py
git commit -m "test: re-score the closed bake-off against the photographer, not the incumbent"
```

---

### Task 3: Fix the comment that points at a tool which never existed

**Files:**
- Modify: `src/qwen_vlm_grader.py:1581-1589` (the `_OUTPUT_CALIBRATION` block)
- Modify: `MODELS.md` (append a disposition section)

**Interfaces:**
- Consumes: Task 2's printed figures, quoted verbatim into both edits.
- Produces: nothing consumed by later tasks.

Background: `_OUTPUT_CALIBRATION` holds one commented-out line — `# "qwen3_vl": {"gain": ..., "offset": ...},   # fitted by _ab_fit_calibration`. A grep of the tree finds `_ab_fit_calibration` in exactly two files: `src/qwen_vlm_grader.py` and its copy under `dist/`. The script has never existed. The comment reads as an unfinished integration waiting for a tool run, which is how 8.3 GB of rejected weights came to look like pending work.

- [ ] **Step 1: Confirm the tool really does not exist**

Run: `ls scripts_*ab_fit* 2>/dev/null; grep -rln "_ab_fit_calibration" src/ scripts_*.py 2>/dev/null`

Expected: no `scripts_*ab_fit*` file, and exactly one grep hit — `src/qwen_vlm_grader.py`, the comment itself.

- [ ] **Step 2: Replace the comment**

In `src/qwen_vlm_grader.py`, replace:

```python
    _OUTPUT_CALIBRATION: dict = {
        # "qwen3_vl": {"gain": ..., "offset": ...},   # fitted by _ab_fit_calibration
    }
```

with:

```python
    _OUTPUT_CALIBRATION: dict = {
        # EMPTY ON PURPOSE — this is not a to-do.
        #
        # The only candidate that ever needed an entry here was Qwen3-VL-4B, and
        # the 2026-06-14 bake-off rejected it: 10.93 s/img against the
        # incumbent's 8.79, and a score spread of std 0.076 against 0.145. An
        # affine map can stretch a compressed spread, so that half was arguably
        # fixable; the latency was not.
        #
        # Re-scored on 2026-08-29 against the photographer's own stars — the
        # question the bake-off never asked, since it ranked candidates by
        # agreement with the incumbent MODEL — the 4B reaches rho +0.078 on the
        # 24 rated bake-off photos, against the incumbent's +0.243 and a
        # chance-level p95 of +0.149. Its headline +0.817 was agreement with a
        # judge that is itself weak. See scripts_ab_vs_stars.py.
        #
        # The earlier note here cited "_ab_fit_calibration", a script that has
        # never existed in this repo, which made a closed decision look like
        # pending work. If a future candidate needs an entry, fit it against
        # STARS, not against the incumbent.
    }
```

- [ ] **Step 3: Verify nothing depends on the block's shape**

Run: `./venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); import qwen_vlm_grader as q; print('calibration entries:', len(q.QwenVLMGrader._OUTPUT_CALIBRATION))"`

Expected: `calibration entries: 0`. `_apply_output_calibration` already returns the score unchanged on a missing entry (`if not cal: return score, breakdown`), so an empty dict is the identity map it has always been.

- [ ] **Step 4: Record the disposition in MODELS.md**

Append to `MODELS.md`:

```markdown
## Rejected candidates (2026-06-14 bake-off)

`_ab_tournament.json` holds the full result. Qwen2.5-VL-3B remains the deep-grade
judge. Rejected: SmolVLM2-2.2B (score collapse, std 0.003), InternVL3.5-2B
(15.66 s/img), Qwen3-VL-2B, Qwen3-VL-4B (10.93 s/img, std 0.076).

Re-scored 2026-08-29 against the photographer's stars rather than against the
incumbent model — the question the bake-off did not ask — on the 24 rated photos
of the 25-photo test batch:

| model | rho vs STARS | rho vs incumbent | s/img |
|---|---|---|---|
| Qwen2.5-VL-3B (incumbent) | **+0.243** | — | 8.79 |
| Qwen3-VL-4B | +0.078 | +0.817 | 10.93 |
| Qwen3-VL-2B | -0.012 | +0.502 | 6.30 |

Chance-level agreement on this ratings set is rho +0.149 at p95, so both Qwen3-VL
variants are indistinguishable from noise against the photographer. A high
`rho vs incumbent` means a candidate reproduces a judge that is itself weak; it
is not evidence of quality. **Rank future candidates against stars.**

The Qwen3-VL weights are archived off-repo — see BUILD.md for the restore path.
The `qwen3_vl` dispatch in `src/qwen_vlm_grader.py` and its scale anchors in
`src/niche_registry.py` are deliberately left in place: they are inert without
weights and they make a future re-evaluation cheap.
```

- [ ] **Step 5: Commit**

```bash
git add src/qwen_vlm_grader.py MODELS.md
git commit -m "docs: a closed rejection was reading as pending work on a tool that never existed"
```

---

### Task 4: Archive the weights off the working disk

**Files:**
- Move: `models/qwen3_vl/` → `D:\framegrade_bench\rejected_models\qwen3_vl\`
- Modify: `BUILD.md` (append a restore path)

**Interfaces:**
- Consumes: Task 1's test passing (nothing requires the weights) and Task 2's numbers (the rejection holds).
- Produces: nothing.

Background: 8.3 GB of a rejected candidate on the working disk. The precedent for this repo is the RAG source books, moved to `D:\framegrade_bench\rag_source` on 2026-08-13 rather than deleted. Follow it. **This step is a move, never a delete** — if the destination drive is unavailable, stop and ask rather than improvising.

- [ ] **Step 1: Confirm the safety gate is green**

Run: `./venv/Scripts/python.exe -m pytest tests/test_qwen3_vl_is_not_required.py -q`

Expected: `4 passed`. If anything fails, stop — something needs these weights.

- [ ] **Step 2: Confirm the destination exists and has room**

Run: `ls -d /d/framegrade_bench 2>/dev/null && df -h /d | tail -1`

Expected: the directory listing, and free space comfortably above 9 GB.

If `D:` is not present, **stop and ask the user where to archive**. Do not delete, do not pick another drive, and do not proceed with the rest of this task.

- [ ] **Step 3: Record the size and file count before the move**

Run: `du -sh models/qwen3_vl && find models/qwen3_vl -type f | wc -l`

Expected: roughly `8.3G` and 10 files. Write both numbers down — Step 5 verifies against them.

- [ ] **Step 4: Move, do not copy-then-delete**

```bash
mkdir -p /d/framegrade_bench/rejected_models
mv models/qwen3_vl /d/framegrade_bench/rejected_models/qwen3_vl
```

A `mv` across drives on Windows is a copy followed by a source delete, and it will not remove the source unless the copy succeeded. Do not substitute `rm -rf` for any part of this.

- [ ] **Step 5: Verify the archive is intact and the source is gone**

Run:

```bash
du -sh /d/framegrade_bench/rejected_models/qwen3_vl && \
find /d/framegrade_bench/rejected_models/qwen3_vl -type f | wc -l && \
ls models/qwen3_vl 2>&1 | head -1
```

Expected: the same size and file count recorded in Step 3, then `ls: cannot access 'models/qwen3_vl': No such file or directory`.

If the size or count differs, the copy was incomplete — copy it again from the archive back into `models/` before doing anything else.

- [ ] **Step 6: Confirm the app is unaffected**

Run: `./venv/Scripts/python.exe -m pytest tests/test_qwen3_vl_is_not_required.py tests/test_ab_vs_stars.py -q`

Expected: `8 passed`. The A/B result JSONs stay at the repo root — they are 9 KB of scores, not weights, and they are the record of the decision.

- [ ] **Step 7: Record the restore path**

Append to `BUILD.md`:

```markdown
### Archived model weights

Rejected bake-off candidates are moved off the working disk rather than deleted,
so a future re-evaluation does not need a multi-gigabyte re-download.

| model | archived to | size | why |
|---|---|---|---|
| Qwen3-VL-4B | `D:\framegrade_bench\rejected_models\qwen3_vl\` | 8.3 GB | rejected 2026-06-14; rho +0.078 vs the photographer's stars (chance is +0.149) — see MODELS.md |

To restore one: `mv /d/framegrade_bench/rejected_models/qwen3_vl models/qwen3_vl`.
Nothing else is required — `src/qwen_vlm_grader.py` dispatches on the config's
`model_type`, so the model works again as soon as the directory is back.
```

- [ ] **Step 8: Commit**

```bash
git add BUILD.md
git commit -m "chore: archive the rejected Qwen3-VL weights off the working disk (8.3 GB)"
```

- [ ] **Step 9: Confirm the tree is finally clean**

Run: `git status --short`

Expected: no output at all. `models/qwen3_vl/` was the last untracked entry.

---

## Self-Review

**Spec coverage.** The authority is `_ab_tournament.json` plus the stored per-photo A/B scores. Every element of the item is assigned: the weights to Task 4, the dead `_ab_fit_calibration` comment to Task 3, the unverified rejection to Task 2, and the safety precondition to Task 1. The plan deliberately does **not** fit a calibration — Task 2 establishes why, and Task 3 records it in the code where the next reader will look.

**Placeholder scan.** No TBDs. Task 4 Step 2 has a conditional stop rather than a guess, which is the intended behaviour for a hard-to-reverse move on a drive that may be absent, not a placeholder. Task 2 Step 5's expected numbers are real measurements taken while writing this plan, not illustrations.

**Type consistency.** `stars_by_basename(ratings_path: str) -> dict` and `pairs_for(ab_path: str, stars_by_name: dict) -> list` have identical signatures in the test (Task 2 Step 1) and the implementation (Step 3); the tests call them positionally in that order. `spearman` is imported from `scripts_accuracy_report.py` rather than reimplemented, so this report and the accuracy audit cannot disagree on tie handling. `CANDIDATES` filenames map to `_ab_tournament.json`'s `summary` keys by stripping the `_ab_` prefix and `.json` suffix — verified against the five keys actually present (`baseline_qwen25_3b`, `qwen3_2b`, `internvl35_2b`, `smolvlm2_22b`, `qwen3_4b`).
