import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Status chip — RAM, GPU, grader mode, model download.
 *
 * The rule this primitive exists to enforce: a chip is NEUTRAL by default and
 * takes colour ONLY when the user needs to act. "Everything is fine" is
 * rendered as silence, not as a green badge.
 *
 * That is not a style preference in a photo tool. The old header carried up to
 * six saturated pills next to the photographs, and saturated chrome next to an
 * image both competes for attention and biases how you read the image's own
 * colour. Reserving colour for exceptions means anything coloured on screen is
 * genuinely worth looking at.
 */

export type ChipTone = 'neutral' | 'warn' | 'crit';

export interface ChipProps {
  /** Short uppercase name — RAM, GPU. Rendered in the label role. */
  label: string;
  /** Optional readout: "8.2 GB", "Deep Edit". Numeric values get tabular figures. */
  value?: ReactNode;
  tone?: ChipTone;
  /** Numeric readouts must set this so digits align and don't jitter as they tick. */
  numeric?: boolean;
  title?: string;
  className?: string;
}

const TONES: Record<ChipTone, string> = {
  /* Neutral chips are unboxed — dot + label + value floating directly on the
   * toolbar (index-readout style, à la fontsinuse.com). Boxing is reserved
   * for the alarm tones: a visible container is itself a signal, so "all
   * clear" must not have one. */
  neutral: 'border-transparent bg-transparent text-ink-3',
  warn: 'border-alarm-warn bg-surface text-alarm-warn',
  crit: 'border-alarm-crit bg-surface text-alarm-crit',
};

const DOT: Record<ChipTone, string> = {
  neutral: 'bg-ink-4',
  warn: 'bg-alarm-warn',
  crit: 'bg-alarm-crit',
};

export function Chip({ label, value, tone = 'neutral', numeric, title, className }: ChipProps) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex h-6 shrink-0 items-center gap-1 rounded-sm border px-2',
        'transition-colors duration-fast ease',
        TONES[tone],
        className,
      )}
    >
      <span className={cn('h-1 w-1 shrink-0 rounded-full', DOT[tone])} />
      <span className="t-label !text-current">{label}</span>
      {value != null && (
        <span className={cn('text-xs opacity-80', numeric && 't-num')}>{value}</span>
      )}
    </span>
  );
}
