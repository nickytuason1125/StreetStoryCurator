import { forwardRef } from 'react';
import type {
  InputHTMLAttributes,
  SelectHTMLAttributes,
} from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '../../lib/cn';

/* Text input and select — one implementation each.
 *
 * The old code styled these per-instance with inline styles, which produced
 * four different input heights and three different focus treatments across the
 * settings and creative panels. Both controls here share the same geometry:
 * h-8, --r-sm corners, well background (inputs sit IN the page, not on it),
 * border brightens on hover and keeps the global focus ring on keyboard focus.
 *
 * The Select wraps a native <select> rather than a custom dropdown listbox.
 * A hand-rolled listbox means re-implementing typeahead, arrow keys, Home/End,
 * ARIA combobox semantics and outside-click dismissal — several hundred lines
 * of JS for a look a native popup already has. Native also costs nothing at
 * runtime: no portal, no scroll locking, no focus trap. The chevron is painted
 * by this component; the browser's own marker is suppressed in index.css.
 */

const CONTROL = cn(
  'h-8 w-full rounded-sm border border-line-strong bg-well px-3',
  'font-sans text-sm text-ink placeholder:text-ink-4',
  'transition-colors duration-fast ease hover:border-ink-4',
  'disabled:opacity-reject disabled:cursor-not-allowed',
);

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return <input ref={ref} className={cn(CONTROL, className)} {...rest} />;
  },
);

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  /** Options as [value, label] pairs — keeps call sites to one expression. */
  options: readonly (readonly [string, string])[];
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { options, className, ...rest },
  ref,
) {
  return (
    <div className={cn('relative', className)}>
      <select
        ref={ref}
        className={cn(CONTROL, 'cursor-pointer appearance-none pr-8')}
        {...rest}
      >
        {options.map(([value, label]) => (
          <option key={value} value={value}>{label}</option>
        ))}
      </select>
      {/* Pointer-events none: clicks fall through to the select itself. */}
      <ChevronDown
        size={13}
        aria-hidden
        className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-ink-3"
      />
    </div>
  );
});