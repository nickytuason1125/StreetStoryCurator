# Rank Agreement Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out why the grader agrees with the photographer at ρ ≈ +0.23 when a merely competent grader would reach ≈ +0.95, by adding four diagnostics to the accuracy audit — then choose the one lever the evidence supports.

**Architecture:** This is a diagnosis, not a fix. The grader's legitimacy metric sits close to chance and nobody knows which stage is responsible: the aspect scores, their fusion into one number, the band thresholds, or the taste head that is supposed to personalise all of it. Guessing which to change is how a grader gets tuned into a curve. So each task adds one section to `scripts_accuracy_report.py`, built from pure functions with unit tests, and each section answers one falsifiable question. The last task reads all four answers and writes the spec for the actual fix. **No task in this plan changes a grade, a threshold, or a weight.**

**Tech Stack:** Python 3, pytest, `scripts_accuracy_report.py`, LanceDB via `src/lance_store.py`, `cache/user_ratings.json`.

**Spec:** No separate spec — this plan produces one (Task 5). The baseline it argues against is `docs/superpowers/specs/2026-08-29-accuracy-baseline.md`, written by Task 5 of `2026-08-29-ram-floor-single-source.md`. Run that first if it does not exist.

## Global Constraints

- Repo root for every command: `street-story-curator/`. Paths are relative to it.
- Python is the venv interpreter: `./venv/Scripts/python.exe`.
- Read every JSON with `encoding='utf-8'`. Windows cp1252 raises `UnicodeDecodeError` on `cache/user_ratings.json` at byte 0x8f.
- **Report only.** No task here may write a score, move a threshold, or retrain a head. `scripts_accuracy_report.py` exits 0 always — it is a report, not a gate — and stays that way.
- Reuse the existing `spearman` and `_rankdata` in `scripts_accuracy_report.py`. Do not add scipy or numpy for statistics that are twenty lines of Python; the module deliberately has no third-party dependency.
- Grade bands are **absolute** (Strong ≥ 0.60, Mid ≥ 0.41). Task 4 measures alternative cut points and prints them. It must not apply them. Per-batch or quantile grading has been removed twice and must never come back.
- Aspect keys are not fixed: Qwen emits niche-specific axis names, and matching is case-insensitive throughout this repo. Any code that reads a breakdown must normalise keys, never assume the canonical five.

## The numbers this plan is measured against

Measured 2026-08-29 on the live `cache/user_ratings.json` (n = 125):

| quantity | value | meaning |
|---|---|---|
| current agreement | **+0.231** | the grader, as shipped |
| chance, p95 | **+0.149** | 2000 random-score trials; below this is noise |
| achievable ceiling | **+0.948** | 2000 trials of a grader that orders the star groups correctly but is blind within a group |

Star histogram: `{1: 2, 2: 3, 3: 44, 4: 34, 5: 42}`.

The ceiling matters most. The ratings are heavily tied — 120 of 125 sit on three values — and the obvious excuse is that a grader cannot be measured against so coarse a signal. **That excuse is false.** A grader that merely sorted the 3s below the 4s below the 5s would score +0.948. The tie structure is not the constraint; the grader is at +0.231, roughly 1.5× the noise floor, against a reachable +0.948.

**Success gate for the fix this plan specifies: ρ ≥ 0.45.** Below +0.35, the lever chosen was wrong and Task 5's spec should be rewritten rather than implemented.

---

### Task 1: Section 0 — every rho gets its own bounds

**Files:**
- Modify: `scripts_accuracy_report.py` (add two functions and a call in `main()`)
- Test: `tests/test_accuracy_report.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  def star_histogram(ratings: dict) -> dict[int, int]
  def measurement_bounds(stars: list[int], trials: int = 2000, seed: int = 0) -> dict
      # -> {"noise_p95": float, "ceiling": float, "n": int}
  def variance_lines(ratings: dict) -> list[str]
  ```
  Tasks 2-4 print their own rho values and compare them against `measurement_bounds`.

Background: `+0.231` means nothing on its own. Printed beside a +0.149 noise floor and a +0.948 ceiling it means "barely distinguishable from chance, against a target that is reachable". Every later section needs the same frame, so it goes in first.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_accuracy_report.py`:

```python
def test_star_histogram_counts_every_level():
    ratings = {"a": 5, "b": "5", "c": 3, "d": 3, "e": 1}
    assert rep.star_histogram(ratings) == {1: 1, 3: 2, 5: 2}


def test_star_histogram_skips_unparseable_ratings():
    assert rep.star_histogram({"a": 4, "b": None, "c": "x"}) == {4: 1}


def test_the_ceiling_is_high_even_when_the_ratings_are_heavily_tied():
    """The tie structure is not an excuse.

    120 of the photographer's 125 ratings sit on three values, which looks like
    it should cap what any grader can score. It does not: a grader that merely
    orders the star GROUPS correctly, while being blind within each group,
    reaches ~0.95. This test exists so nobody re-derives the excuse.
    """
    stars = [3] * 44 + [4] * 34 + [5] * 42 + [1] * 2 + [2] * 3
    b = rep.measurement_bounds(stars, trials=200, seed=0)
    assert b["ceiling"] > 0.90, f"ceiling collapsed to {b['ceiling']}"
    assert b["n"] == 125


def test_the_noise_floor_is_well_below_the_ceiling():
    stars = [3] * 44 + [4] * 34 + [5] * 42 + [1] * 2 + [2] * 3
    b = rep.measurement_bounds(stars, trials=200, seed=0)
    assert 0.0 < b["noise_p95"] < 0.35
    assert b["noise_p95"] < b["ceiling"]


def test_bounds_are_deterministic_for_a_given_seed():
    """A report whose numbers move between runs cannot be quoted."""
    stars = [3] * 20 + [5] * 20
    a = rep.measurement_bounds(stars, trials=100, seed=7)
    b = rep.measurement_bounds(stars, trials=100, seed=7)
    assert a == b


def test_variance_lines_name_the_verdict_not_just_the_numbers():
    ratings = {f"p{i}": (3 if i < 44 else 5) for i in range(80)}
    text = "\n".join(rep.variance_lines(ratings))
    assert "0. RATING VARIANCE" in text
    assert "ceiling" in text.lower()
    assert "chance" in text.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: FAIL — `AttributeError: module 'rep' has no attribute 'star_histogram'`.

- [ ] **Step 3: Write the implementation**

In `scripts_accuracy_report.py`, add above `def main()`:

```python
def star_histogram(ratings: dict) -> dict:
    """{stars: count}, skipping anything that will not parse as an int."""
    hist: dict = {}
    for raw in ratings.values():
        try:
            s = int(raw)
        except (TypeError, ValueError):
            continue
        hist[s] = hist.get(s, 0) + 1
    return dict(sorted(hist.items()))


def measurement_bounds(stars: list, trials: int = 2000, seed: int = 0) -> dict:
    """What rho values are reachable against THIS set of ratings.

    noise_p95 — 95th percentile of rho for a grader scoring at random. A result
                below this is indistinguishable from chance.
    ceiling   — rho for a grader that orders the star GROUPS perfectly but is
                blind within a group. This is the realistic best, because a
                continuous score inevitably spreads photos the photographer
                rated identically, and that spread is pure noise.

    Seeded, so the report can be quoted: an unseeded bound would drift between
    runs and every comparison against it would be unfalsifiable.
    """
    import random as _random
    rng = _random.Random(seed)

    def _pct95(xs: list) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(0.95 * len(xs)))]

    noise = [spearman([rng.random() for _ in stars], stars) for _ in range(trials)]
    # `s + random()` keeps every group strictly ordered against its neighbours
    # while randomising inside it — exactly "gets the groups right, guesses
    # within them".
    ceil = [spearman([s + rng.random() for s in stars], stars) for _ in range(trials)]
    return {
        "noise_p95": round(_pct95(noise), 3),
        "ceiling":   round(sum(ceil) / len(ceil), 3),
        "n":         len(stars),
    }


def variance_lines(ratings: dict) -> list:
    """Section 0, as lines, so it can be tested without file I/O."""
    hist = star_histogram(ratings)
    stars = [s for s, c in hist.items() for _ in range(c)]
    if not stars:
        return ["", "0. RATING VARIANCE", "   no parseable ratings"]
    b = measurement_bounds(stars)
    total = len(stars)
    lines = ["", "0. RATING VARIANCE AND MEASUREMENT BOUNDS"]
    for s, c in hist.items():
        bar = "#" * max(1, round(40 * c / total))
        lines.append(f"   {s}star  n={c:<4} {bar}")
    lines.append(f"   chance (p95):      {b['noise_p95']:+.3f}   "
                 f"a rho at or below this is not distinguishable from guessing")
    lines.append(f"   reachable ceiling: {b['ceiling']:+.3f}   "
                 f"a grader that only orders the star groups correctly")
    lines.append("   Read section 1's rho between these two. Heavily tied")
    lines.append("   ratings do NOT cap the ceiling — ordering the groups is enough.")
    return lines
```

Then in `main()`, immediately after the `print()` that follows the `store miss:` line, add:

```python
    for line in variance_lines(ratings):
        print(line)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: all green, including the pre-existing nan tests.

- [ ] **Step 5: Run the report and read section 0**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py`

Expected: a new section 0 showing the histogram, `chance (p95): +0.149`, `reachable ceiling: +0.948`. If the ceiling comes back below +0.90, stop — the ratings changed shape and this plan's whole framing needs rechecking.

- [ ] **Step 6: Commit**

```bash
git add scripts_accuracy_report.py tests/test_accuracy_report.py
git commit -m "feat: the audit reports what rho was reachable, not just what it was"
```

---

### Task 2: Section 4 — which aspect actually tracks the photographer

**Files:**
- Modify: `scripts_accuracy_report.py`
- Test: `tests/test_accuracy_report.py` (extend)

**Interfaces:**
- Consumes: `measurement_bounds` from Task 1.
- Produces:
  ```python
  def aspect_pairs(rows: dict, ratings: dict) -> dict[str, list[tuple[float, int]]]
  def aspect_lines(rows: dict, ratings: dict, fused_rho: float | None) -> list[str]
  ```

Background: the fused score reaches +0.231. If a single aspect — composition, say — reaches +0.40 on its own, then the aspects are informative and **the fusion is destroying the signal**, which is a completely different fix from "the grader cannot see". PersonalHead training already hinted at this: its taste signature came out with composition as the dominant driver. This section tests that directly.

Aspect scores live in the `breakdown` column, a JSON blob that `lance_store.query_all` returns already parsed. Keys are niche-specific and must be matched case-insensitively.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_accuracy_report.py`:

```python
def test_aspect_pairs_normalise_key_case_and_whitespace():
    """Qwen emits niche-specific axis names, and casing drifts between niches."""
    rows = {
        "a": {"score": 0.7, "breakdown": {"Composition": 0.8, "lighting ": 0.4}},
        "b": {"score": 0.3, "breakdown": {"COMPOSITION": 0.2, "Lighting": 0.9}},
    }
    got = rep.aspect_pairs(rows, {"a": 5, "b": 3})
    assert set(got) == {"composition", "lighting"}
    assert sorted(got["composition"]) == [(0.2, 3), (0.8, 5)]


def test_aspect_pairs_skip_non_numeric_and_private_fields():
    """_tech_audit is a nested dict of technical fields, not an aspect."""
    rows = {"a": {"score": 0.7, "breakdown": {
        "Composition": 0.8, "_tech_audit": {"blur": 1}, "verdict": "good"}}}
    got = rep.aspect_pairs(rows, {"a": 4})
    assert set(got) == {"composition"}


def test_aspect_pairs_ignore_photos_with_no_breakdown():
    rows = {"a": {"score": 0.7, "breakdown": {}}, "b": {"score": 0.4}}
    assert rep.aspect_pairs(rows, {"a": 4, "b": 5}) == {}


def test_aspect_lines_flag_an_aspect_that_beats_the_fused_score():
    """The finding this section exists to surface."""
    rows = {}
    for i in range(20):
        # composition tracks stars exactly; the fused score is inverted noise
        stars = 1 + (i % 5)
        rows[f"p{i}"] = {"score": 1.0 - stars / 5,
                         "breakdown": {"Composition": stars / 5, "Lighting": 0.5}}
    ratings = {f"p{i}": 1 + (i % 5) for i in range(20)}
    text = "\n".join(rep.aspect_lines(rows, ratings, fused_rho=0.231))
    assert "composition" in text.lower()
    assert "beats the fused score" in text.lower()


def test_aspect_lines_are_quiet_when_no_aspect_beats_fusion():
    rows = {f"p{i}": {"score": 0.5, "breakdown": {"Composition": 0.5}}
            for i in range(10)}
    ratings = {f"p{i}": 3 for i in range(10)}
    text = "\n".join(rep.aspect_lines(rows, ratings, fused_rho=0.9))
    assert "beats the fused score" not in text.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: FAIL — `AttributeError: module 'rep' has no attribute 'aspect_pairs'`.

- [ ] **Step 3: Write the implementation**

Add to `scripts_accuracy_report.py` above `def main()`:

```python
def aspect_pairs(rows: dict, ratings: dict) -> dict:
    """{aspect_key: [(aspect_score, stars), ...]} over every rated photo.

    Keys are lowercased and stripped: axis names are niche-specific here and
    their casing drifts, so a literal match silently drops half the data.
    Private fields (leading underscore) and non-numeric values are skipped —
    `_tech_audit` is a nested dict, not an axis.
    """
    out: dict = {}
    for path, raw_stars in ratings.items():
        row = rows.get(path)
        if not row:
            continue
        try:
            stars = int(raw_stars)
        except (TypeError, ValueError):
            continue
        bd = row.get("breakdown") or {}
        if not isinstance(bd, dict):
            continue
        for k, v in bd.items():
            if not isinstance(k, str) or k.startswith("_"):
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            out.setdefault(k.strip().lower(), []).append((float(v), stars))
    return out


def aspect_lines(rows: dict, ratings: dict, fused_rho) -> list:
    """Section 4, as lines.

    The question: does any single aspect agree with the photographer BETTER
    than the fused score does? If one does, the aspects carry signal and the
    fusion is throwing it away — a different repair from "the grader is blind".
    """
    per = aspect_pairs(rows, ratings)
    lines = ["", "4. PER-ASPECT AGREEMENT (each aspect vs your stars)"]
    if not per:
        lines.append("   no per-aspect breakdowns stored for your rated photos")
        return lines

    scored = []
    for key, pairs in sorted(per.items()):
        if len(pairs) < 8:
            continue
        vals, stars = zip(*pairs)
        scored.append((spearman(list(vals), list(stars)), key, len(pairs)))
    if not scored:
        lines.append("   too few rated photos carry a breakdown to measure")
        return lines

    scored.sort(reverse=True)
    for rho, key, n in scored:
        lines.append(f"   {key:<18} rho = {rho:+.3f}   (n={n})")

    best_rho, best_key, _ = scored[0]
    if fused_rho is not None and not math.isnan(fused_rho) and best_rho > fused_rho:
        lines.append("")
        lines.append(f"   '{best_key}' ({best_rho:+.3f}) BEATS THE FUSED SCORE "
                     f"({fused_rho:+.3f}).")
        lines.append("   The aspects carry signal the fusion is discarding. The")
        lines.append("   repair is in how they are combined, not in the grader's eye.")
    return lines
```

Then in `main()`, after the band-monotonicity block and before the `if ppairs:` block, add:

```python
    for line in aspect_lines(rows, ratings, rho):
        print(line)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: all green.

- [ ] **Step 5: Run the report and record the finding**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py`

Expected: section 4 lists every aspect with its own rho, best first. Write down the top aspect and its rho — Task 5 needs both.

Three readings, all useful:
- **An aspect beats the fused score** → the fusion weights are the bug.
- **Every aspect is near or below +0.149** → the grader genuinely cannot see what the photographer values; the repair is upstream, in the probes or the prompt.
- **Every aspect is roughly equal** → the aspects are not independent; the grader is emitting one opinion five times, which is the "Aesthetic-monopoly" pattern already recorded for this pipeline.

- [ ] **Step 6: Commit**

```bash
git add scripts_accuracy_report.py tests/test_accuracy_report.py
git commit -m "feat: the audit says which aspect actually tracks the photographer"
```

---

### Task 3: Section 3 — say why the taste head is silent

**Files:**
- Modify: `scripts_accuracy_report.py` (extend `personal_shift_lines`)
- Test: `tests/test_accuracy_report.py` (extend)

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `personal_shift_lines(ppairs, rho)` keeps its signature; its output gains a diagnosis line when `personal_score` is degenerate.

Background — with a strong hypothesis already available. `lance_store._make_schema` gives `personal_score` a **default of 0.5**, and the migration backfill uses `0.5` too. So a store where PersonalHead never wrote is not empty — it is a column of identical 0.5s. Zero variance, Spearman undefined, which is exactly what the audit reported. The existing fix made that honest (`undefined` plus a named cause) but stopped short of saying which of the two causes it is: never written, or written and genuinely flat.

This matters because the taste head should be a *large* lever here — the blend gives taste a rising share of the vote as the rating count grows, and the photographer has 125 ratings. If it is sitting at its default, a substantial correction is simply not being applied.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_accuracy_report.py`:

```python
def test_an_all_default_personal_score_is_named_as_never_written():
    """0.5 is the schema default. A column of them means PersonalHead never
    wrote, which is a different problem from a head that learned nothing."""
    ppairs = [(0.5, s) for s in (3, 4, 5, 3, 4, 5, 5, 3)]
    text = "\n".join(rep.personal_shift_lines(ppairs, rho=0.231))
    assert "undefined" in text.lower()
    assert "0.5" in text
    assert "never written" in text.lower()


def test_a_flat_non_default_personal_score_is_not_called_unwritten():
    """Genuinely flat at some other value is a trained-but-useless head."""
    ppairs = [(0.62, s) for s in (3, 4, 5, 3, 4, 5, 5, 3)]
    text = "\n".join(rep.personal_shift_lines(ppairs, rho=0.231))
    assert "never written" not in text.lower()


def test_a_varying_personal_score_still_reports_the_shift():
    ppairs = [(0.2, 3), (0.4, 4), (0.6, 5), (0.8, 5)]
    text = "\n".join(rep.personal_shift_lines(ppairs, rho=0.231))
    assert "never written" not in text.lower()
    assert "taste learning moved agreement" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: FAIL on `test_an_all_default_personal_score_is_named_as_never_written` — the current output names a generic no-variance reason without identifying the 0.5 default.

- [ ] **Step 3: Write the implementation**

In `scripts_accuracy_report.py`, inside `personal_shift_lines`, replace the degenerate branch:

```python
    if math.isnan(rho_p):
        lines.append(f"   Spearman rho = undefined   (n={n})")
        lines.append(f"   {_no_variance_reason(ps, pst, n, 'personal_score')}")
        return lines
```

with:

```python
    if math.isnan(rho_p):
        lines.append(f"   Spearman rho = undefined   (n={n})")
        lines.append(f"   {_no_variance_reason(ps, pst, n, 'personal_score')}")
        # 0.5 is lance_store's schema default for this column (and its
        # migration backfill). A column of them is not a head that learned
        # nothing — it is a head whose output was never written back, which is
        # a far bigger miss: the taste blend gives the head a rising share of
        # the vote as ratings accumulate, so at this rating count a real
        # correction is simply not being applied.
        if ps and all(abs(p - 0.5) < 1e-9 for p in ps):
            lines.append("   every personal_score is exactly 0.5 — the schema")
            lines.append("   default. PersonalHead was NEVER WRITTEN to the store")
            lines.append("   for these photos; the blend is inert, not neutral.")
            lines.append("   Check that update_personal_scores() runs after a grade.")
        return lines
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: all green.

- [ ] **Step 5: Run the report and confirm which case is live**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py`

Expected: section 3 now either reports a real shift, or names the 0.5 default explicitly.

If it says **never written**, verify directly before treating it as fact:

```bash
./venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import lance_store as ls
vals = [r.get('personal_score') for r in ls.query_all(min_score=0.0)]
uniq = sorted(set(vals))
print('rows:', len(vals), '| distinct personal_score values:', uniq[:10])
"
```

A single distinct value of `0.5` confirms it. Record the result for Task 5.

- [ ] **Step 6: Commit**

```bash
git add scripts_accuracy_report.py tests/test_accuracy_report.py
git commit -m "fix: the audit says whether the taste head is untrained or simply unwritten"
```

---

### Task 4: Section 5 — where the band cuts should have been

**Files:**
- Modify: `scripts_accuracy_report.py`
- Test: `tests/test_accuracy_report.py` (extend)

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces:
  ```python
  def best_cuts(pairs: list[tuple[float, int]]) -> tuple[float, float, float]
      # -> (strong_cut, mid_cut, separation) — separation is mean(Strong stars) - mean(Weak stars)
  def threshold_lines(pairs: list[tuple[float, int]]) -> list[str]
  ```

Background: the bands pass their ordering check on a 0.07-star margin (Strong 4.02, Mid 3.95, Weak 3.50). Ordering is the wrong test — it passes on noise. The useful question is how much separation the *current* cuts (0.60 / 0.41) achieve and how much was available at any cut pair. A large gap means the thresholds are misplaced for this library; a small one means no threshold choice rescues the underlying score, and the repair is upstream.

**This section prints. It does not apply.** Absolute thresholds are load-bearing here — relative and quantile grading have been removed twice — and a cut fitted to maximise separation on the photographer's own ratings is exactly a curve. Task 5 decides what, if anything, to do with the finding.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_accuracy_report.py`:

```python
def test_best_cuts_finds_an_obvious_split():
    """Scores below 0.5 are 1-star, above are 5-star: any cut in the gap works
    and separation is the full 4 stars."""
    pairs = [(0.1, 1), (0.2, 1), (0.3, 1), (0.8, 5), (0.9, 5), (0.95, 5)]
    strong, mid, sep = rep.best_cuts(pairs)
    assert sep == 4.0
    assert 0.3 < mid <= strong <= 0.9


def test_best_cuts_report_low_separation_when_the_score_is_uninformative():
    """Scores unrelated to stars: no cut pair can separate them."""
    pairs = [(0.5 + (i % 7) / 100, 3 + (i % 3)) for i in range(30)]
    _, _, sep = rep.best_cuts(pairs)
    assert sep < 1.5


def test_best_cuts_never_returns_mid_above_strong():
    pairs = [(i / 20, 1 + i % 5) for i in range(20)]
    strong, mid, _ = rep.best_cuts(pairs)
    assert mid <= strong


def test_threshold_lines_always_show_the_shipped_cuts_for_comparison():
    pairs = [(0.1, 1), (0.9, 5)] * 6
    text = "\n".join(rep.threshold_lines(pairs))
    assert "0.60" in text and "0.41" in text
    assert "report only" in text.lower()


def test_threshold_lines_survive_too_little_data():
    assert "not enough" in "\n".join(rep.threshold_lines([(0.5, 3)])).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: FAIL — `AttributeError: module 'rep' has no attribute 'best_cuts'`.

- [ ] **Step 3: Write the implementation**

Add to `scripts_accuracy_report.py` above `def main()`:

```python
# The shipped absolute cuts. Named here ONLY so the report can print them
# beside the fitted ones; this module never applies a threshold.
_SHIPPED_STRONG, _SHIPPED_MID = 0.60, 0.41


def _separation(pairs: list, strong: float, mid: float):
    """mean(stars in Strong) - mean(stars in Weak). None if a band is empty."""
    hi = [s for sc, s in pairs if sc >= strong]
    lo = [s for sc, s in pairs if sc < mid]
    if not hi or not lo:
        return None
    return sum(hi) / len(hi) - sum(lo) / len(lo)


def best_cuts(pairs: list) -> tuple:
    """Scan every (mid, strong) cut pair; return the best-separating one.

    Separation, not accuracy: the question is whether the score can be cut
    anywhere that puts the photographer's better photos on one side. A grid of
    the observed scores is enough — cuts between two adjacent scores are
    equivalent.
    """
    if len(pairs) < 4:
        return (_SHIPPED_STRONG, _SHIPPED_MID, 0.0)
    grid = sorted({round(sc, 3) for sc, _ in pairs})
    best = (_SHIPPED_STRONG, _SHIPPED_MID, 0.0)
    for mid in grid:
        for strong in grid:
            if strong < mid:
                continue
            sep = _separation(pairs, strong, mid)
            if sep is not None and sep > best[2]:
                best = (strong, mid, sep)
    return best


def threshold_lines(pairs: list) -> list:
    """Section 5, as lines. REPORT ONLY — see the banner in the output."""
    lines = ["", "5. BAND THRESHOLDS (report only — nothing is applied)"]
    if len(pairs) < 4:
        lines.append("   not enough rated photos with scores to fit cuts")
        return lines

    shipped = _separation(pairs, _SHIPPED_STRONG, _SHIPPED_MID)
    strong, mid, sep = best_cuts(pairs)
    shipped_s = "undefined (a band is empty)" if shipped is None else f"{shipped:+.2f} stars"
    lines.append(f"   shipped  Strong>={_SHIPPED_STRONG:.2f}  Mid>={_SHIPPED_MID:.2f}"
                 f"   separation = {shipped_s}")
    lines.append(f"   best fit Strong>={strong:.2f}  Mid>={mid:.2f}"
                 f"   separation = {sep:+.2f} stars")
    gain = None if shipped is None else sep - shipped
    if gain is not None:
        lines.append(f"   available by re-cutting alone: {gain:+.2f} stars")
        if gain < 0.5:
            lines.append("   Small. No threshold choice rescues this score — the")
            lines.append("   repair is upstream of the cut.")
        else:
            lines.append("   Large. The cuts are misplaced for this library.")
    lines.append("   These are FITTED ON YOUR OWN RATINGS. Applying them would be")
    lines.append("   grading on a curve, which was removed twice. Report only.")
    return lines
```

Then in `main()`, after the aspect section added in Task 2, add:

```python
    for line in threshold_lines(pairs):
        print(line)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/test_accuracy_report.py -q`

Expected: all green.

- [ ] **Step 5: Run the report and record the gain**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py`

Expected: section 5 prints the shipped separation, the best-fit separation, and the difference. Record the difference — Task 5 uses it to decide whether thresholds are even a candidate lever.

- [ ] **Step 6: Commit**

```bash
git add scripts_accuracy_report.py tests/test_accuracy_report.py
git commit -m "feat: the audit measures how much separation the band cuts leave on the table"
```

---

### Task 5: Read the four diagnostics and specify the fix

**Files:**
- Create: `docs/superpowers/specs/2026-08-29-rank-agreement-findings.md`

**Interfaces:**
- Consumes: the printed output of Tasks 1-4, and the baseline at `docs/superpowers/specs/2026-08-29-accuracy-baseline.md`.
- Produces: a spec that a later plan implements. **This task writes no code.**

- [ ] **Step 1: Generate the complete report**

Run: `./venv/Scripts/python.exe scripts_accuracy_report.py > /tmp/report.txt 2>&1; cat /tmp/report.txt`

Expected: six sections — 0 variance/bounds, 1 rank agreement, 2 band monotonicity, 3 personal shift, 4 per-aspect, 5 thresholds.

- [ ] **Step 2: Apply the decision rule**

Read the four diagnostics in this order and stop at the first that fires. They are ordered by how much of the gap each can close, largest first:

1. **Section 3 says `personal_score` is all 0.5** → the lever is *wiring*, not modelling. A trained head's output is never reaching the store, so the taste blend is inert at a rating count where it should carry real weight. Cheapest fix available and the only one that is a plain bug.
2. **Section 4 shows an aspect beating the fused score** → the lever is *fusion*. The aspects see what the photographer values and the combination step discards it. Specify a reweighting fitted on held-out ratings, never on all 125.
3. **Section 5 shows ≥ 0.5 stars available from re-cutting** → the lever is *thresholds*, with a hard constraint: any new cut must be justified absolutely, not fitted to this library. Fitting on the photographer's ratings is the curve that was removed twice.
4. **None of the above; every aspect near the +0.149 noise floor** → the lever is *upstream*. The grader cannot see what this photographer values, and the repair is in the probes and prompt, not in any downstream arithmetic. This is the expensive branch — say so plainly rather than dressing it up.

- [ ] **Step 3: Write the findings spec**

Create `docs/superpowers/specs/2026-08-29-rank-agreement-findings.md`:

```markdown
# Rank Agreement — Findings and Proposed Fix

**Baseline:** rho +0.231 (n=125), chance p95 +0.149, reachable ceiling +0.948.
The grader sits at roughly 1.5x the noise floor against a target that a merely
competent grader would reach.

## Diagnostics

<paste sections 0 and 3-5 of the report verbatim>

## Which lever fired

<name the first rule from the decision rule that fired, and quote the numbers
that fired it>

## Proposed change

<one paragraph: the single change, the files it touches, and why this lever and
not the others>

## Success gate

Re-run `scripts_accuracy_report.py` after the change.

- **rho >= 0.45** — the fix worked. Record the new baseline.
- **0.35 <= rho < 0.45** — partial. Keep it only if section 4 or 5 also improved;
  a rho that moved while every diagnostic stayed flat is a fit to these 125
  ratings, not a better grader.
- **rho < 0.35** — the lever was wrong. Revert and re-read the diagnostics.

## What must not change

- Bands stay absolute. No quantile, per-batch, or historical-anchor calibration.
- No threshold fitted to maximise agreement on the photographer's own ratings.
- The audit stays report-only and keeps exiting 0.
```

- [ ] **Step 4: Verify the spec is decidable**

Re-read what was written and confirm three things: the named lever is one of the four, the quoted numbers appear in the report output, and the proposed change names actual files. A spec that says "improve the grading" is a plan failure — send it back.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-08-29-rank-agreement-findings.md
git commit -m "docs: name the lever behind the weak rank agreement"
```

---

## Self-Review

**Spec coverage.** This plan produces a spec rather than implementing one. The item it covers — "rank agreement is weak" — decomposes into the four candidate causes any repair would have to choose between (taste wiring, fusion, thresholds, upstream perception), and each gets a task that answers it with a number. Task 5's decision rule consumes all four and is ordered, so it terminates on one lever rather than licensing a rewrite.

**Placeholder scan.** Task 5 Step 3's angle-bracketed slots are the deliverable's content, which cannot exist before Tasks 1-4 run; each is accompanied by the command that produces it and Step 4 gates on them being filled. Everywhere else, every function is given in full with its tests.

**Type consistency.** `spearman(list, list) -> float` and `_rankdata` are reused from the existing module, never reimplemented. `measurement_bounds(stars, trials, seed) -> {"noise_p95", "ceiling", "n"}` has matching keys in the test (Task 1 Step 1) and the implementation (Step 3). `aspect_pairs(rows, ratings) -> dict[str, list[tuple[float, int]]]` feeds `aspect_lines(rows, ratings, fused_rho)`, which is called in `main()` with `rho` — the same variable section 1 computes, and which is `None` when section 1 had no data, a case `aspect_lines` guards with `fused_rho is not None and not math.isnan(...)`. `best_cuts(pairs) -> (strong, mid, sep)` is unpacked in that order in `threshold_lines` and in all three of its tests. `pairs` and `ppairs` keep the `(score, stars)` shape they already have in `main()`.
