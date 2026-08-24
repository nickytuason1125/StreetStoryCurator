import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Toolbar — the horizontal control rail used by the header and panel headers.
 *
 * One job: guarantee alignment. Every child is vertically centred in a fixed
 * height, groups are separated by a hairline (not margins), and the toolbar
 * itself never wraps — a wrapping toolbar is how the old header ended up with
 * controls at three different baselines depending on window width.
 *
 * `gap` between siblings is the spacing rhythm's 2-step; use ToolbarGroup to
 * cluster related controls with a tighter internal gap.
 */

export function Toolbar({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      role="toolbar"
      className={cn(
        'flex h-8 shrink-0 items-center gap-2 overflow-x-auto border-b border-line px-3',
        className,
      )}
    >
      {children}
    </div>
  );
}

/* A visual cluster inside a toolbar. Separated from siblings by a vertical
 * hairline so grouping is structural, not a spacing guess. */
export function ToolbarGroup({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('flex h-full shrink-0 items-center gap-1 border-l border-line pl-2 first:border-l-0 first:pl-0', className)}>
      {children}
    </div>
  );
}

/* Spacer that pushes everything after it to the far end of the bar. */
export function ToolbarSpacer() {
  return <div className="min-w-2 flex-1" aria-hidden />;
}