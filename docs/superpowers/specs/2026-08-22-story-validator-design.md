# Story/Competition prompt accuracy and reliability — validator design

## Context

Story and Competition mode ask a language model to obey editorial rules —
"slot 1 is a wide establishing shot", "close on something quiet", "every frame
must stand alone" — and then trust that it did. Measured today, across four
models, it mostly does not.

| Model | Correct | Per answer |
|---|---|---|
| LFM2.5-VL-1.6B | 5/6 | 1.7s |
| LFM2.5-VL-3B | 4/6 | 4.6s |
| Qwen3-4B | 6/6 | 1.7s (thinking off) |
| DeepSeek-R1-8B | 2/6 completed | 127s |

Those numbers are for the easy case: pick ONE item matching ONE attribute.
Asked to fill seven slots against pacing rules from a 25-item manifest, every
model tested drifted to the low ids and ignored the opener rule. The first time
any model placed a genuine wide establishing shot in slot 1 was after measured
facts were added to the payload — and even then only at a reduced manifest size.

Three levers were measured and are now fixed points rather than open questions:

- **Grammar constraining degrades the CHOICE**, not just the format:
  `from_json_schema` scored 11/14 against 14/14 unconstrained, biasing toward
  small ids. Do not constrain the selection call.
- **Payload size drives latency superlinearly**: 25 candidates 36.1s, 12
  candidates 4.7s. Attention is quadratic in sequence length. Cap the manifest.
- **Thinking must be suppressed** on hybrid reasoning models: 31.9s to 1.7s at
  identical accuracy.

None of those make the model obey the rules. They make it fast and unbiased
enough to have a chance.

## The idea

Every constraint in these prompts is machine-checkable, because `story_facts`
measures the things the prompts ask for. If the prompt says "slot 1 is wide"
and the model returns id 7, we can check whether id 7 is wide.

So: stop relying on prompt wording, and verify the answer instead. Reliability
becomes a property of the pipeline rather than a hope about the model.

## Goals

- Every stated rule has a matching checker.
- A broken rule is repaired deterministically, and the repair is reported.
- Story and Competition get genuinely different rule sets — today Competition is
  Story with a stricter dedup threshold and a different prompt string.
- Nothing derived from copyrighted source material enters the prompt path.

## Non-goals

- Rewriting the four prompt strings. A rule that is checked does not need to be
  phrased perfectly, so prompt wording matters much less once this exists.
- RAG concepts. Dropped from the prompt path by decision: the store holds
  phrases extracted from copyrighted books, and roughly 25 of 62 rubric phrases
  were previously found to be biography text rather than photographic criteria.
  The concept vocabulary is `niche_registry`, which is hand-authored in source.
- The vision-critic call sites. Same validator, wired in a later change.

## Architecture

`src/story_validator.py`, one module, three parts:

```
Rule      name, scope ("story" | "competition"), checker
validate  run the rule set -> list[Violation]
repair    fix each violation -> (picks, list[Repair])
```

A checker returns one of three things, and the third is the important one:

- `None` — the rule holds.
- `Violation(slot, rule, detail)` — the rule is broken.
- `Unverifiable(rule, why)` — the fact needed is not known.

**Unverifiable is not a violation.** 68% of Strong photos in this library carry
no focal length — they are Lightroom exports with the lens data stripped. A
"slot 0 must be wide" rule that treated unknown as wrong would fail two thirds
of frames for their export settings rather than for being wrong. Unverifiable
rules are reported and never repaired against.

## Rule sets

Story:

| Rule | Fact used |
|---|---|
| exactly N picks, unique, in range | the picks themselves |
| slot 0 is `wide` | `framing` |
| final slot is quiet (low subject scale) | subject scale |
| adjacent luminance delta < 25% | `luminance`, existing `_LUM_SMOOTH_THRESH` |
| no adjacent near-duplicates | embeddings, existing `_DUP_SIM_THRESH` (0.88) |

Competition:

| Rule | Fact used |
|---|---|
| exactly N picks, unique, in range | the picks themselves |
| pairwise similarity <= 0.85 | embeddings |
| framing variety across the set | `framing` |
| no narrative dependence — each frame standalone | subject scale, `session` |

Thresholds reuse the constants already in `creative_director.py` rather than
introducing parallel ones.

**Fact availability at time of writing.** `framing`, `luminance`, `session`,
`score` and `personal_score` exist in `story_facts` today. **Subject scale does
not** — box area is the intended second framing source and is unbuilt. Every
rule depending on it therefore returns `Unverifiable` on every photo until that
lands. That is correct behaviour, not a bug: the rule is declared, reported as
uncheckable, and starts working the day its fact arrives. It does mean the
"quiet closer" and "no narrative dependence" rules are inert on delivery, and
the validator should not be described as enforcing them until they are not.

## Repair

Deterministic, no second model call. For a violated slot, take the
highest-scoring unused candidate whose facts satisfy the rule. If none exists,
leave the slot as the model chose it and record why. A repair that cannot be
made is reported, not forced.

## Data flow

```
facts_for_pool(rows)
  -> prompt (unchanged, manifest capped at ~12)
  -> model picks
  -> validate(picks, facts, rules_for(mode))
  -> repair(picks, facts, violations)
  -> sequence + repair_log
```

`repair_log` rides alongside `director_fallback` into the result dict, the
per-photo rationale, and the results toolbar — the same path built today for
reporting a degraded run.

## Error handling

Every failure degrades to a note. A checker that raises, a fact that is absent,
an empty pool, a rule set that does not apply — all produce an entry in the log
and let the run continue. The pipeline's job is to return photographs.

## Testing

TDD, table-driven over the rule set. The tests that matter are the ones that
fail against `main` today:

- a pool of wide frames plus one close-up must not open on the close-up
- two near-identical frames placed adjacently must get one swapped
- a pool with no focal data must produce Unverifiable notes, not seven repairs
- a repair that cannot be satisfied must be reported, not forced
- Competition and Story must return materially different sets from one pool

## Risks

**Repair could flatten the edit.** Deterministic substitution takes the
highest-scoring candidate that fits, which is a score sort wearing a rule's
clothes. If most slots end up repaired, the model is contributing nothing and we
should say so rather than hide it behind a rule-compliant sequence. The repair
log makes that visible; a run where most slots were repaired should be reported
as prominently as a fallback.

**Framing coverage is currently 32%.** Until subject-box area lands as a second
source, the opener rule is unverifiable for most photos. This design is correct
under that limit but delivers less than it will once coverage improves.
