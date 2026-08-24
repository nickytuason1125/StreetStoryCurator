import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Segmented control — one primitive for what the header hand-rolled three
 * separate times (main tabs, loupe/grid, and the grade filters), each with
 * slightly different heights, radii and hover behaviour.
 *
 * The selected segment is marked by a luminance step, never by the accent:
 * warm colour belongs to the photographer's marks alone. Square-ish corners
 * (--r-sm) because a contact sheet has square corners; pill-shaped tabs are
 * generic dashboard furniture.
 */

export interface SegmentedOption<T extends string> {
  value: T;
  label?: ReactNode;
  icon?: ReactNode;
  title?: string;
  /** Trailing count, set in tabular figures so the control doesn't jitter. */
  count?: number | string;
  /** Semantic dot (grade filters). The ONLY place a segment carries colour. */
  dot?: string;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  iconOnly,
  className,
}: {
  options: SegmentedOption<T>[];
  /** null means "nothing selected" — used by the grade filters. */
  value: T | null;
  onChange: (v: T) => void;
  iconOnly?: boolean;
  className?: string;
}) {
  if (!options.length) return null;
  return (
    <div
      role="tablist"
      className={cn(
        'inline-flex shrink-0 overflow-hidden rounded-sm border border-line-strong bg-well',
        className,
      )}
    >
      {options.map((o, i) => {
        const active = value === o.value;
        return (
          <button
            key={o.value}
            role="tab"
            aria-selected={active}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={cn(
              'inline-flex h-6 items-center justify-center gap-1 border-0 text-sm font-medium',
              'cursor-pointer whitespace-nowrap transition-colors duration-fast ease',
              iconOnly ? 'w-8' : 'px-3',
              i > 0 && 'border-l border-line-strong',
              active
                ? 'bg-raised-hover text-ink'
                : 'bg-transparent text-ink-3 hover:bg-raised hover:text-ink',
            )}
          >
            {o.dot && (
              <span
                className="h-1 w-1 shrink-0 rounded-full"
                style={{ background: o.dot }}
              />
            )}
            {o.icon}
            {!iconOnly && o.label}
            {o.count != null && <span className="t-num text-ink-4">({o.count})</span>}
          </button>
        );
      })}
    </div>
  );
}
