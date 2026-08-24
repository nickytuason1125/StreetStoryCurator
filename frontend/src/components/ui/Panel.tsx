import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Panel — the sectioned container used by every side column and dialog body.
 *
 * One header treatment (label role + optional trailing action) instead of the
 * five slightly-different section headers the old panels had grown. The panel
 * itself is transparent: it sits on --surface chrome and draws only a hairline
 * between sections, so a column of panels reads as one surface with rules
 * rather than a stack of boxes. Boxes-in-boxes is most of what made the old
 * layout feel busy.
 */

export function Panel({
  title,
  action,
  children,
  className,
  bodyClassName,
}: {
  /** Uppercase micro-label — rendered in the .t-label role. */
  title?: string;
  /** Control aligned right in the header (e.g. "Clear", an IconButton). */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn('flex min-h-0 flex-col', className)}>
      {title && (
        <header className="flex h-8 shrink-0 items-center justify-between gap-2 border-b border-line px-3">
          <span className="t-label">{title}</span>
          {action}
        </header>
      )}
      <div className={cn('min-h-0 flex-1 overflow-auto p-3', bodyClassName)}>
        {children}
      </div>
    </section>
  );
}

/* Divider between stacked panels — a hairline, not a gap-plus-border pair.
 * Panels own their internal padding; the column owns only this rule. */
export function PanelDivider() {
  return <div className="h-px shrink-0 bg-line" aria-hidden />;
}