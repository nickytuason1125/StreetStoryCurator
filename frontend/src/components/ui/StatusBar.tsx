import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* StatusBar — the fixed bottom rail for live telemetry (RAM, GPU, grader mode,
 * progress) and transient messages.
 *
 * Fixed height, hairline top edge, everything vertically centred: telemetry
 * that changes every second must never change the layout around it, or the
 * whole app shimmers. Numeric readouts use .t-num so ticking digits don't
 * jitter horizontally.
 *
 * Colour policy comes from Chip: neutral until action is needed. A calm system
 * is silent; amber/red means look here now.
 */

export function StatusBar({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <footer
      className={cn(
        'flex h-6 shrink-0 items-center gap-3 overflow-hidden border-t border-line bg-surface px-3',
        className,
      )}
    >
      {children}
    </footer>
  );
}

/* A labelled readout slot inside the status bar. */
export function StatusItem({
  label,
  value,
  numeric = true,
  tone = 'neutral',
  title,
}: {
  label: string;
  value?: ReactNode;
  numeric?: boolean;
  tone?: 'neutral' | 'warn' | 'crit';
  title?: string;
}) {
  const toneClass =
    tone === 'crit' ? 'text-alarm-crit'
    : tone === 'warn' ? 'text-alarm-warn'
    : 'text-ink-3';
  return (
    <span className={cn('inline-flex shrink-0 items-baseline gap-1', toneClass)} title={title}>
      <span className="t-label !text-current">{label}</span>
      {value != null && (
        <span className={cn('text-xs', numeric && 't-num', tone === 'neutral' && 'text-ink-2')}>
          {value}
        </span>
      )}
    </span>
  );
}

/* Separator dot between status items. */
export function StatusDot() {
  return <span className="h-1 w-1 shrink-0 rounded-full bg-ink-4" aria-hidden />;
}