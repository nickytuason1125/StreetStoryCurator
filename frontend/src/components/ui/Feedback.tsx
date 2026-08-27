import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Progress, Skeleton, EmptyState — the three "the app is thinking" renderings.
 *
 * They share one rule: motion is honest. A determinate bar shows real progress;
 * an indeterminate sweep says "working, duration unknown"; a skeleton marks
 * exactly what will appear where; an empty state explains itself instead of
 * showing blank space. The old UI had eleven hand-rolled spinners and silent
 * blank panels — the user couldn't tell loading from broken.
 *
 * All three are pure CSS (no JS timers), so they cost nothing while idle.
 */

/* ── ProgressBar ────────────────────────────────────────────────────────────
 * value: null → indeterminate sweep. 0..1 → determinate fill.
 * The fill is neutral ink on raised — progress is chrome, not a judgement,
 * so it never wears the grade teal or the mark orange. */
export function ProgressBar({
  value,
  label,
  className,
}: {
  /** 0..1, or null for indeterminate. */
  value: number | null;
  label?: string;
  className?: string;
}) {
  const indeterminate = value == null;
  return (
    <div className={cn('flex flex-col gap-1', className)}>
      {(label || !indeterminate) && (
        <div className="flex items-baseline justify-between gap-2">
          {label && <span className="t-label">{label}</span>}
          {!indeterminate && (
            <span className="t-num text-xs text-ink-3">
              {Math.round(value * 100)}%
            </span>
          )}
        </div>
      )}
      <div
        role={indeterminate ? undefined : 'progressbar'}
        aria-valuenow={indeterminate ? undefined : Math.round(value * 100)}
        aria-valuemin={indeterminate ? undefined : 0}
        aria-valuemax={indeterminate ? undefined : 100}
        className="h-1 w-full overflow-hidden rounded-sm bg-raised"
      >
        <div
          className={cn(
            'h-full rounded-sm bg-ink-2',
            indeterminate ? 'animate-sweep w-1/3' : 'transition-[width] duration-slow ease',
          )}
          style={indeterminate ? undefined : { width: `${Math.min(100, Math.max(0, value * 100))}%` }}
        />
      </div>
    </div>
  );
}

/* ── Skeleton ───────────────────────────────────────────────────────────────
 * Shimmer block standing in for content that is loading. Size it like the
 * thing it replaces — a skeleton the wrong shape causes layout shift when the
 * real content lands, which reads as jank even when nothing moved. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn('animate-shimmer rounded-sm bg-raised', className)}
    />
  );
}

/* ── EmptyState ─────────────────────────────────────────────────────────────
 * Icon + one-line title + optional hint + optional action. Used by the gallery
 * before import, empty filters, and cleared panels. Centered in its container;
 * the parent decides vertical placement. */
export function EmptyState({
  icon,
  title,
  hint,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 px-6 py-8 text-center',
        className,
      )}
    >
      {icon && <div className="text-ink-4">{icon}</div>}
      <div>
        <p className="text-md font-medium text-ink-2">{title}</p>
        {hint && <p className="mx-auto mt-1 max-w-[36ch] text-sm leading-prose text-ink-3">{hint}</p>}
      </div>
      {action}
    </div>
  );
}