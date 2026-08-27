/* Token references for the places TypeScript still has to name a colour:
 * SVG `fill`/`stroke`, canvas, and values computed at runtime from a score.
 *
 * These are `var(--x)` strings, NOT literals. tokens.css stays the only file
 * where a value is actually decided, so `npm run lint:tokens` can forbid hex
 * everywhere else without carving out an exception for this file. SVG paint
 * attributes accept custom properties, so `fill={T.mark}` resolves correctly.
 */

export const T = {
  well: 'var(--well)',
  ground: 'var(--ground)',
  surface: 'var(--surface)',
  raised: 'var(--raised)',
  raisedHover: 'var(--raised-hover)',
  line: 'var(--line)',
  lineStrong: 'var(--line-strong)',

  ink: 'var(--ink)',
  ink2: 'var(--ink-2)',
  ink3: 'var(--ink-3)',
  ink4: 'var(--ink-4)',

  gradeStrong: 'var(--grade-strong)',
  gradeWeak: 'var(--grade-weak)',
  gradePending: 'var(--grade-pending)',

  /** The photographer's own marks ONLY — stars, selects. Never chrome. */
  mark: 'var(--mark)',
  markInk: 'var(--mark-ink)',
  markDim: 'var(--mark-dim)',

  alarmWarn: 'var(--alarm-warn)',
  alarmCrit: 'var(--alarm-crit)',

  /** Machine voice — AI verdicts, score bars, command palette. Never user actions. */
  ai: 'var(--ai)',
  aiDim: 'var(--ai-dim)',
  aiInk: 'var(--ai-ink)',

  focus: 'var(--focus)',
  focusInset: 'var(--focus-inset)',

  /** Dialog/overlay wash. Already carries its own alpha — do not add another. */
  scrim: 'var(--scrim)',

  /** Chrome floating above artwork — pair with the `.glass` blur utility. */
  glass: 'var(--glass)',
} as const;

/* ── Grade vocabulary ────────────────────────────────────────────────────────
 * The four functions below are the shared grade language; everything
 * downstream inherits from them, which is why they are the first thing the
 * restyle converts. Absolute thresholds, matching the backend contract:
 * Strong >= 0.60, Mid 0.41-0.59, Weak < 0.41.
 */

export type GradeKey = 'strong' | 'mid' | 'weak' | 'pending';

export function gradeKey(g: string | null | undefined): GradeKey {
  if (g?.includes('Strong')) return 'strong';
  if (g?.includes('Mid')) return 'mid';
  if (g?.includes('Weak')) return 'weak';
  return 'pending';
}

/** Plain-language label. */
export function gradeLabel(g: string | null | undefined): string {
  return { strong: 'Strong', mid: 'Mid', weak: 'Weak', pending: 'Pending' }[gradeKey(g)];
}

/* The machine's verdict is a 2px rule under the frame, not a badge.
 *
 * Shape, not colour, was the original plan (a grease-pencil ring for a select).
 * It was cut because glyph shape is not preattentive: telling a ring from a
 * slash needs a fixation, so scanning 500 cells would cost 500 of them. A rule
 * in a fixed position aligns into visible bands down a column, so runs of
 * Strong are legible without reading anything, and it occludes no image.
 *
 * Mid returns null on purpose. Silence is the correct rendering of "no
 * opinion", and Mid is the majority bucket, so the grid stays quiet. */
export function gradeRule(g: string | null | undefined): string | null {
  const k = gradeKey(g);
  if (k === 'strong') return T.gradeStrong;
  if (k === 'weak') return T.gradeWeak;
  return null; // mid and pending carry no rule
}

/* The cell's OTHER verdict channel: the glass chip over the frame.
 *
 * It answers the same question as gradeRule - does this grade get a mark? - so
 * it has to give the same answer, and the only way to guarantee that is to ask
 * here rather than re-deciding in JSX. GridView previously did re-decide, with
 * a guard of `!isPending`, so Mid fell through and rendered a label in --ink-2
 * while the rule beneath the same cell stayed silent. On a real folder that is
 * most of the grid, which is precisely the noise Mid-silence exists to prevent.
 *
 * Returns the chip's text colour, or null when the grade does not speak. */
export function gradeBadge(g: string | null | undefined): string | null {
  return gradeRule(g);
}

/** Weak frames physically sink. Cheapest high-value scanning affordance here. */
export function gradeOpacity(g: string | null | undefined): number {
  return gradeKey(g) === 'weak' ? 0.55 : 1;
}

/* Photographers read `f/1.4` and `1/250` fluently, so the leading zero on a
 * 0-1 score is noise. `.612` is faster to scan in a column and reads as a
 * measurement rather than as a progress percentage. */
export function formatScore(s: number | null | undefined): string {
  if (typeof s !== 'number' || Number.isNaN(s)) return '—';
  if (s >= 1) return '1.00';
  return s.toFixed(3).replace(/^0/, '');
}
