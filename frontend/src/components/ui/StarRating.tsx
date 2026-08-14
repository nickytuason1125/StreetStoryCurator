import { useState } from 'react';
import { T } from '../../theme/tokens';
import { cn } from '../../lib/cn';

/* The photographer's own rating — and therefore one of the few things in the
 * app allowed to be warm.
 *
 * The old widget used oklch(70% .18 72), an amber roughly ten degrees in hue
 * from the accent originally proposed for grade badges. At 11px over a
 * photograph those two were indistinguishable, so one colour would have carried
 * two unrelated meanings inside a single 180px cell. Unifying both onto --mark
 * resolves it in the right direction: stars ARE the user's judgement, so they
 * belong to the mark colour, and the machine's grade gives up colour entirely.
 */

export function StarRating({
  stars,
  onSet,
  size = 14,
  className,
}: {
  stars: number;
  onSet: (n: number) => void;
  size?: number;
  className?: string;
}) {
  const [hover, setHover] = useState(0);
  const shown = hover || stars;

  return (
    <div
      className={cn('flex items-center gap-px', className)}
      onMouseLeave={() => setHover(0)}
      role="group"
      aria-label={`Rating: ${stars} of 5 stars`}
    >
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          type="button"
          aria-label={`${n} star${n > 1 ? 's' : ''}`}
          aria-pressed={n <= stars}
          onMouseEnter={() => setHover(n)}
          // Clicking the current rating clears it — the fastest way to undo a
          // misclick while scrubbing a filmstrip.
          onClick={(e) => { e.stopPropagation(); onSet(stars === n ? 0 : n); }}
          className="cursor-pointer rounded-sm border-0 bg-transparent p-0 leading-none transition-opacity duration-fast ease hover:opacity-80"
        >
          <svg width={size} height={size} viewBox="0 0 24 24" strokeWidth="2"
               fill={n <= shown ? T.mark : 'none'}
               stroke={n <= shown ? T.mark : T.ink4}>
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
        </button>
      ))}
    </div>
  );
}
