import type { ReactNode, TextareaHTMLAttributes } from 'react';
import { cn } from '../../lib/cn';

/* Labelled control. The label uses Archivo's width axis (the `.t-label` role),
 * which is how uppercase micro-labels are set everywhere in the app — one
 * treatment rather than the four slightly different letter-spacings the config
 * panel had grown. */

export function Field({
  label,
  hint,
  action,
  children,
  className,
}: {
  label: string;
  hint?: ReactNode;
  /** Optional control aligned to the right of the label (e.g. "Clear all"). */
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="t-label">{label}</span>
        {action}
      </div>
      {hint && <p className="-mt-1 text-xs text-ink-3">{hint}</p>}
      {children}
    </div>
  );
}

/* Focus is a border change plus the global focus ring — the old inputs swapped
 * their border colour with onFocus/onBlur handlers that mutated element style
 * directly, which meant keyboard focus and mouse focus behaved differently. */
export function TextArea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'w-full resize-none rounded-sm border border-line-strong bg-well px-3 py-2',
        'font-sans text-sm text-ink placeholder:text-ink-4',
        'transition-colors duration-fast ease hover:border-ink-4',
        className,
      )}
      {...rest}
    />
  );
}
