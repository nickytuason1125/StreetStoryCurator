# Cohesive story selection — design

Supersedes the framing rules in `2026-08-22-story-validator-design.md`. That
document's opener rule, detail role and Competition framing-variety rule rest on
shot type, which measurement has since shown does not vary in this library. The
validator's other ideas — machine-checkable constraints, Unverifiable as a
first-class result — survive and are folded in here.

## Context

Story mode is handed a library of graded photographs and must return a set that
hangs together. Today it does this:

```
5,634 graded Strong photos
   -> top-40 NEAREST NEIGHBOURS of the single highest-scoring frame   (:1123)
   -> dedup
   -> manifest of 12                                                  (:1299)
   -> model picks 7
```

Two things are wrong with the funnel. The model sees a tiny fraction of the
library, and the fraction is chosen by **resemblance to one photograph**. If that
top scorer is an outlier, the whole story is built inside a cul-de-sac. Worse,
the step selects for SIMILARITY immediately before the sequencer is asked to find
contrast.

It was not pure damage: taking 40 look-alikes bought visual coherence, which a
photo story genuinely needs. A set drawn evenly from 5,634 frames would be a
jumble. So coherence must become an explicit criterion, not something lost by
deleting the step that accidentally supplied it.

## What measurement ruled out

**Shot type is not a discriminating axis for this photographer.** Face sizes
across 80 real Strong photos: median 0.22% of frame, largest 0.88%. The "close"
boundary is 8%. There are no close-ups in this library — it is street work, where
people are small in the frame. A "wide opener" rule would pass on nearly every
frame and a "detail" role could never be filled.

**Zero-shot SigLIP cannot judge framing.** Validated against 488 Strong photos
and rejected: the "close" probe returned a waterfall, then a wide-angle sun-flare
landscape; "portrait" won zero frames of 488, every margin negative. Rewriting
with explicit scale words, a missing "medium" class and shared negatives did not
fix it. CLIP-family models are strong on CONTENT and weak on FRAMING.

**Automatic length has no working operating range.** Growing a set while cohesion
stays above a floor was measured across four briefs:

```
floor 0.55  ->  10, 10, 10, 10
floor 0.80  ->  10, 10, 10, 10
floor 0.85  ->  10,  1,  2,  1
floor 0.88  ->   1,  1,  1,  1
```

It is a cliff, not a curve. Cohesion is measured against a centroid that moves as
frames are added, so with one photo it is trivially 1.0 and the second drops it
off the edge. No floor produces lengths in the 4-10 range. Length is therefore
the USER'S choice, not the system's.

## Design

**The user sets k with a slider, 4-10, default 7.** This maps onto the `n_target`
field the API already accepts, so the backend contract is unchanged.

**Selection maximises value subject to two constraints**, over the whole graded
library rather than 40 neighbours:

```
taste_w(i) = 0.10 + 0.40 * conf(i),   conf(i) = |personal_score(i) - 0.5| / 0.5

merit(i)   = (1 - taste_w) * score(i) + taste_w * personal_score(i)

value(i)   = 0.40 * brief_match(i)
           + 0.40 * merit(i)
           + 0.20 * cohesion_to_selected(i)    <- preference, not a gate

constraint
  duplicate   no pair above _DUP_SIM_THRESH (0.88) -- the existing constant
```

**Taste is the baseline, scoped to where it has evidence.** A flat weight was
wrong in both directions. Measured over 5,634 Strong photos, `personal_score`
correlates with the aesthetic score at **0.010** — near zero. It is not a tinted
copy of quality; it is the only term carrying information the others lack, which
argues for weighting it well above a 0.10 afterthought.

But it is trained on **124 ratings** into a 1536-256-64-1 network, and the
project's own notes record its alignment at 0.52 against a 0.50 coin flip. Its
spread says the same: mean 0.568, std 0.075, so for most of the library the head
holds no strong opinion. A handful at 0.249 and 0.756 are where it speaks.

So the weight follows the head's own confidence, the mechanism already proven in
`grade_pipeline_v2` Step 5: near 0.5 it collapses to a 0.10 floor and quality
leads; far from 0.5 it rises toward 0.50 and taste leads. It can never degrade a
photo the head knows nothing about, and it genuinely governs the ones it does.

The real lever is not this weight — it is the 124. Every genre the user rates
widens where the head has an opinion, and therefore where it leads.

**Cohesion is a term in the objective, not a threshold.** This is the resolution
of a contradiction found in spec self-review: the document argued cohesion could
not be given a defensible floor, then listed it as a hard constraint anyway. But
deleting it outright would throw away the one thing the old anchor step got
right.

As a weighted term it needs no magic number and has no cliff. A frame that fits
the set is preferred; a frame that fits badly can still be chosen if it is
strongly relevant and high quality — which is what "the striking outlier that
earns its place" looks like. The measured failure of the floor approach was
caused precisely by treating a smooth quantity as a boundary.

Only the duplicate ceiling stays hard, because 0.88 is not a taste judgement:
above it the two frames are the same photograph.

`brief_match` is cosine to the embedded brief, min-max normalised across the
library so it is comparable with the other terms. `score` and `personal_score`
are already computed and stored per photo; `personal_score` is the PersonalHead
trained on the user's own ratings, and the director currently discards it.

**Cohesion is also reported.** Beyond steering selection as a weighted term, the
achieved cohesion is returned with the set — "these 6 hang together at 0.88" —
so the user can judge whether the story holds rather than trusting that it does.
A number the user can act on beats a threshold nobody can justify.

**Bundled demo images are excluded by path.** 56 files under `dataset_images/`
are the app's own assets and were being selected into user stories.

## Architecture

`src/story_selector.py`, pure functions over arrays, no model loaded:

```
library_matrix(rows)          -> (rows, M) normalised embeddings
select(brief_vec, rows, M, k) -> (indices, diagnostics)
cohesion(M, indices)          -> mean, min, max_pair
```

The LLM is not involved in selection. It receives the chosen set and writes the
story — language, which is what it is good at. This is the inversion the whole
investigation arrived at: the computer decides, the model narrates.

## Data flow

```
brief -> embed_text_query()            (existing, creative_director.py:83)
library -> lance_store.query_all()     (existing)
        -> story_selector.select(k)    (new; replaces the :1123 anchor block)
        -> story_facts.facts_for_pool  (existing, for the rationale)
        -> prompt -> model -> narrative
```

## Error handling

An empty library, a brief that embeds to nothing, or fewer than k candidates
surviving the duplicate constraint all return what was found plus a stated
reason, alongside `director_fallback`. Returning 4 photographs and saying why
beats returning 7 by relaxing a constraint silently.

## Testing

TDD. The tests that fail against `main` today:

- a library dominated by one cluster must not return k frames from that cluster
  alone (the anchor cul-de-sac)
- two frames above 0.88 similarity must never both appear
- `dataset_images/` paths must never be selected
- k is honoured exactly for every k in 4..10, or a reason is given
- a brief with no good matches must report low cohesion rather than silently
  returning the k highest-scoring photos regardless of the brief

## Risks

**The weights are guesses.** 0.40 brief / 0.40 merit / 0.20 cohesion are not
measured — they are a starting point. They should be tuned against sets the user
judges, and until then the split is arbitrary and should be described that way in
the UI, not presented as a considered balance.

**The taste model may be measuring noise.** Its 0.010 correlation with the
aesthetic score is the argument for including it — and is equally consistent with
it having learned nothing, since noise is uncorrelated with everything too. 124
ratings against that many parameters is the classic setup for memorising rather
than generalising. The confidence weighting limits the damage (a head with no
opinion cannot steer anything) but does not resolve the question. Resolving it
needs held-out ratings, which do not exist yet.

The cohesion weight is the most arbitrary of the four, and the most consequential:
it is the single dial between "a repetitive set that scores well" and "a jumble
of individually strong frames". It is the first thing to tune against real
output.

**Cohesion may not mean what I think.** It is cosine to a set centroid in SigLIP
space, which correlates with "looks similar", not with "belongs in the same
story". Those come apart: five photographs of one doorway score perfectly and
make a bad edit. The duplicate ceiling catches identical frames but not
near-repetition of subject.

**No validation against user judgement yet.** Every measurement so far tests
whether the machinery does what it claims, not whether the resulting sets are
good. That requires the user looking at output, and nothing in this design
substitutes for it.
