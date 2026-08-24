# Frontend Architecture

FrameGrade's UI follows a strict design system (`src/theme/tokens.css`, enforced
by `npm run build` via `scripts/lint-tokens.mjs`). This document maps the
**code structure**: what lives where, and the agreed path for shrinking
`App.tsx`.

## Current layout

```
src/
  theme/tokens.css      THE source of truth for colour/type/space/motion.
  theme/tokens.ts       Typed accessors (T.*) that emit var(--x).
  components/ui/*       The primitive kit: Button, Chip, Modal, Field, Panel,
                        Score, Segmented, StarRating, StatusBar, Toolbar…
  components/ExifPanel.tsx
  lib/api.ts            Backend base URL, thumb/photo URL builders,
                        path sanitisation, offline fetch-guard.   ← extracted
  lib/grading.ts        ramReadiness(), gc() grade→token colour.      ← extracted
  hooks/useGuardedInterval.ts  visibility-aware polling.             ← extracted
  App.tsx               Application shell + feature screens.
```

## Rules

1. **No raw hex or px outside `theme/tokens.css`.** The linter fails the build.
2. **Chrome is dead neutral; warm colour (`--mark`) is reserved for the user's
   own marks** — see the manifesto comment atop tokens.css before adding any.
3. **New UI must use the `components/ui` kit**, not bespoke inline controls.
4. **Never reintroduce emoji as status** (✅⚠️❌) — the grade rule under each
   frame carries verdicts.

## App.tsx decomposition plan (in progress)

`App.tsx` is still one large file. It is being split **along its existing
seams** — every component defined before the `App()` function is already
props-driven and can move without behaviour change:

| Target module | Contents (former App.tsx lines) | Status |
|---|---|---|
| `lib/api.ts` | fetch-guard, API, thumbUrl/photoUrl, sanitizePath | ✅ done |
| `hooks/useGuardedInterval.ts` | visibility-aware interval hook | ✅ done |
| `lib/grading.ts` | ramReadiness, gc | ✅ done |
| `components/gallery/FilmThumb.tsx` | memo'd filmstrip cell | next |
| `components/gallery/GridView.tsx` | contact-sheet grid + select bar | next |
| `components/dnd/SortableItem.tsx` | dnd-kit wrapper | next |
| `components/panels/ExportModal.tsx` | export dialog | next |
| `features/annotations/regions.tsx` | REGION_GUIDE, tier* helpers, REGION_BOX | next |
| `features/annotations/CritiqueOverlays.tsx` | parseCritique, FactorAnnotations, AnalysisHUD, ASPECT_DIM | next |
| `lib/slogans.ts` | progress slogans (_SLOGANS/toSlogan) | next |
| `features/niche/NICHE_GROUPS.ts` | niche registry mirror | next |

### Extraction protocol (each step)

1. Move the block verbatim; add `export` to its top-level names.
2. Generate imports from the identifier table in App's original import block.
3. `npm run build` must pass (tsc + token lint) before moving on.
4. One feature per commit.

Never attempt a big-bang split: the App body wires ~60 state hooks through
these components, and prop-threading mistakes are runtime errors the compiler
cannot catch.
