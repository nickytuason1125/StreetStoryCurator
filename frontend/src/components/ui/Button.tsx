import { forwardRef } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* The single button in the application.
 *
 * Note what is NOT here: an accent-coloured "primary" variant. The warm mark
 * colour is reserved for the photographer's own judgements, so a button may
 * never wear it — that reservation is the main thing separating this UI from a
 * generic dark dashboard, and it only holds if the primitives refuse to break it.
 * Emphasis is carried by luminance and border weight instead.
 *
 * Hover/focus/active live here once. The old code had 8 hover handlers across
 * 539 styled elements, because inline styles cannot express :hover at all —
 * ~98% of controls simply did not react to the cursor, which is most of what
 * read as "dated".
 */

type Variant = 'default' | 'quiet' | 'solid' | 'ink' | 'danger';
type Size = 'sm' | 'md';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
}

const VARIANTS: Record<Variant, string> = {
  // Bordered, transparent — the default for toolbar actions.
  default: 'bg-transparent border border-line-strong text-ink-2 hover:bg-raised hover:text-ink active:bg-raised-hover',
  // No border. For dense clusters where borders would create visual noise.
  quiet: 'bg-transparent border border-transparent text-ink-3 hover:bg-raised hover:text-ink active:bg-raised-hover',
  // Filled. Emphasis via luminance, not hue. The inset top-light hairline gives
  // the fill physical presence — the same edge --shadow-3 paints on dialogs.
  solid: 'bg-raised border border-line-strong text-ink hover:bg-raised-hover active:bg-raised [box-shadow:inset_0_1px_0_rgb(255_255_255/_.06)]',
  // Inverted ink — the strongest step the chrome may take. Full-luminance fill,
  // well-coloured label: a white button on the dark ground. Reserved for the
  // one primary action on a screen (the welcome hero's CTA); still strictly
  // neutral, so the mark-colour reservation holds. Hover dims the fill rather
  // than swapping tokens, so the inversion survives the cursor.
  ink: 'bg-ink border border-ink text-well hover:opacity-90 active:opacity-85',
  // The one place chrome may take a hue: a destructive action.
  danger: 'bg-transparent border border-line-strong text-ink-2 hover:border-alarm-crit hover:text-alarm-crit active:bg-raised',
};

const SIZES: Record<Size, string> = {
  sm: 'h-6 px-2 gap-1 text-xs',
  md: 'h-8 px-3 gap-2 text-sm',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'default', size = 'md', icon, className, children, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center rounded-sm font-sans font-medium',
        'cursor-pointer select-none whitespace-nowrap',
        'transition duration-fast ease active:scale-[.98]',
        'disabled:opacity-reject disabled:cursor-not-allowed disabled:hover:bg-transparent',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
});
