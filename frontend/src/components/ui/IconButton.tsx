import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Square icon-only button — the toolbar workhorse.
 *
 * Split out of Button because icon buttons have different geometry (square,
 * fixed width, centred glyph) and different density needs than labelled ones.
 * Keeping them separate stops the "w-8 h-8 px-0" hacks that were appearing at
 * every call site.
 *
 * Same variant contract as Button: emphasis by luminance, never by hue — the
 * warm mark colour is reserved for the photographer's own judgements.
 */

type Variant = 'default' | 'quiet' | 'solid' | 'danger';
type Size = 'sm' | 'md';

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** The glyph. Pass a lucide component instance: <IconButton icon={<X size={14}/>}/> */
  icon: ReactNode;
  /** Required — an icon button with no accessible name is invisible to a screen reader. */
  label: string;
  variant?: Variant;
  size?: Size;
}

const VARIANTS: Record<Variant, string> = {
  default: 'bg-transparent border border-line-strong text-ink-2 hover:bg-raised hover:text-ink active:bg-raised-hover',
  quiet:   'bg-transparent border border-transparent text-ink-3 hover:bg-raised hover:text-ink active:bg-raised-hover',
  solid:   'bg-raised border border-line-strong text-ink hover:bg-raised-hover active:bg-raised',
  danger:  'bg-transparent border border-line-strong text-ink-2 hover:border-alarm-crit hover:text-alarm-crit active:bg-raised',
};

const SIZES: Record<Size, string> = {
  sm: 'h-6 w-6',
  md: 'h-8 w-8',
};

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { icon, label, variant = 'quiet', size = 'md', className, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      aria-label={label}
      title={label}
      className={cn(
        'inline-flex shrink-0 items-center justify-center rounded-sm',
        'cursor-pointer select-none',
        'transition-colors duration-fast ease',
        'disabled:opacity-reject disabled:cursor-not-allowed disabled:hover:bg-transparent',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon}
    </button>
  );
});