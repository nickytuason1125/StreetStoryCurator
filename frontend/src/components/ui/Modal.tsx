import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { X } from 'lucide-react';
import { cn } from '../../lib/cn';

/* Dialog shell — one implementation for what was three hand-rolled overlays.
 *
 * Beyond styling, this fixes behaviour the originals didn't have: Escape closes,
 * focus moves into the dialog on open and returns to the trigger on close, and
 * the surface is announced as a dialog. Those were missing everywhere, which
 * meant a keyboard user could tab straight out of an open modal into the page
 * behind it.
 *
 * The scrim is a solid ground wash rather than a blur. Backdrop blur on a
 * screenful of decoded photographs is expensive, and it smears the very images
 * the dialog is usually describing.
 */

export function Modal({
  title,
  subtitle,
  onClose,
  children,
  footer,
  width = 560,
}: {
  title: string;
  subtitle?: ReactNode;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  width?: number;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const restoreTo = useRef<Element | null>(null);

  useEffect(() => {
    restoreTo.current = document.activeElement;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); onClose(); }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      (restoreTo.current as HTMLElement | null)?.focus?.();
    };
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[500] flex items-center justify-center bg-scrim"
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        className={cn(
          'animate-fade-in flex max-h-[80vh] flex-col overflow-hidden outline-none',
          'rounded-md border border-line-strong bg-surface shadow-lg',
        )}
        style={{ width, maxWidth: 'calc(100vw - 2rem)' }}
      >
        <div className="flex shrink-0 items-center gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0 flex-1">
            <p className="text-md font-semibold text-ink">{title}</p>
            {subtitle && <p className="mt-px text-xs text-ink-3">{subtitle}</p>}
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="flex cursor-pointer rounded-sm border-0 bg-transparent p-1 text-ink-3 transition-colors duration-fast ease hover:bg-raised hover:text-ink"
          >
            <X size={13}/>
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto px-4 py-2">{children}</div>

        {footer && (
          <div className="flex shrink-0 justify-end gap-2 border-t border-line px-4 py-3">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}
