import { formatScore, gradeKey } from '../../theme/tokens';
import { cn } from '../../lib/cn';

/* A score as a typographic object, not a progress bar.
 *
 * Five progress bars per photo was the previous treatment. A bar communicates
 * "loading", not "measured", and per-photo bars cannot be compared across
 * frames — which is the only comparison that matters when culling. Set as mono
 * numerals with tabular figures, the same digit column lines up on every frame
 * in the filmstrip, so the numbers become scannable down the strip.
 *
 * The leading zero is dropped: photographers read f/1.4 and 1/250 fluently, so
 * `.612` is faster to parse and reads as a measurement rather than a percentage.
 */

export function Score({
  value,
  size = 'sm',
  grade,
  className,
}: {
  value: number | null | undefined;
  size?: 'sm' | 'xl';
  /** Only used to grey out a Weak score; the hue still comes from the rule. */
  grade?: string | null;
  className?: string;
}) {
  const weak = gradeKey(grade) === 'weak';
  return (
    <span
      className={cn(
        't-num tabular-nums',
        size === 'xl' ? 'text-xl' : 'text-xs',
        weak ? 'text-ink-3' : 'text-ink-2',
        className,
      )}
    >
      {formatScore(value)}
    </span>
  );
}

/* The five aspects as a compact segmented strip rather than five bars.
 *
 * Fixed column order across every photo, so a weak Composition sits in the same
 * position on frame 1 and frame 400 and the eye can compare down a column. */
export function AspectStrip({
  breakdown,
  className,
}: {
  breakdown: Record<string, number> | null | undefined;
  className?: string;
}) {
  const entries = Object.entries(breakdown ?? {}).slice(0, 5);
  if (!entries.length) return null;
  return (
    <div className={cn('flex w-full gap-px', className)} aria-hidden>
      {entries.map(([key, v]) => (
        <span
          key={key}
          title={`${key} ${formatScore(v)}`}
          className="h-1 flex-1 bg-raised"
        >
          <span
            className="block h-full bg-ink-4"
            style={{ width: `${Math.max(0, Math.min(1, Number(v) || 0)) * 100}%` }}
          />
        </span>
      ))}
    </div>
  );
}
