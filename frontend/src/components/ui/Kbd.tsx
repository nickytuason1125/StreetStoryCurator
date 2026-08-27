import type { ReactNode } from 'react';
import { cn } from '../../lib/cn';

/* Kbd — a key cap.
 *
 * The signature element of a keyboard-first tool, and the reason it is a
 * primitive rather than a class string: once shortcuts are printed on the
 * controls themselves, the cap appears in the status rail, in every command
 * row, in the palette footer and eventually on toolbar buttons. Four hand-rolled
 * versions of it drifting apart is exactly how the old UI accumulated thirteen
 * border radii.
 *
 * Two details carry most of the effect:
 *
 * 1. `min-w-kbd` — single letters and words like "esc" sit on the same minimum
 *    width, so a column of caps down the status rail aligns instead of ragging.
 *    A cap that changes width as the label changes reads as text, not a key.
 * 2. The inset top-light hairline. It is the same edge --shadow-3 paints on a
 *    dialog and the `solid` Button wears; on something this small it is the
 *    entire difference between "a bordered box" and "a physical key".
 *
 * Neutral by policy: a key cap is chrome, so it never takes --mark (the
 * photographer's colour) or --ai (the machine's voice).
 */

export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        't-num inline-flex h-4 min-w-kbd shrink-0 items-center justify-center px-1',
        'rounded-sm border border-line-strong bg-raised text-xs leading-none text-ink-2',
        '[box-shadow:inset_0_1px_0_rgb(255_255_255/_.07)]',
        className,
      )}
    >
      {children}
    </kbd>
  );
}

/* A shortcut printed next to what it does: cap first, then the action.
 *
 * Used by the status rail and the palette footer. The label is deliberately
 * dimmer than the cap — you scan these for the KEY, and read the word only
 * when the key is unfamiliar. */
export function KbdHint({ keys, label }: { keys: string; label: string }) {
  return (
    <span className="flex shrink-0 items-center gap-1 text-xs text-ink-3">
      <Kbd>{keys}</Kbd>
      {label}
    </span>
  );
}
