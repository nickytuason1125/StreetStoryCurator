import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { RefreshCw } from 'lucide-react';
import { thumbUrl } from '../../lib/api';
import { cn } from '../../lib/cn';

/* Thumb — the single image primitive for every thumbnail in the app.
 *
 * Why this exists: /api/thumb answers 204 while a grade is running and 404
 * when a RAW decode fails. A bare <img> renders both as a silent black box —
 * indistinguishable from a genuinely dark frame, which in a photo tool is a
 * lie. This primitive makes every state honest:
 *
 *   loading → skeleton shimmer (the tile is being decoded; wait)
 *   loaded  → the frame, faded in on opacity only (motion policy)
 *   waiting → shimmer persists while auto-retry backs off (1s/2s/4s) — this
 *             is the 204-during-grading case: tiles self-heal once the grade
 *             finishes, with no reload and no user action
 *   error   → parked quietly with a retry affordance (a failed decode will
 *             not fix itself; the user should know and can force one more)
 *
 * Renders a FRAGMENT meant to sit inside the call site's existing
 * `relative overflow-hidden bg-well` wrapper — the skeleton and error
 * overlays are absolutely positioned against it. The error affordance is a
 * span, not a button: every call site nests inside a tile <button>, and
 * buttons may not nest.
 */

const BACKOFF_MS = [1000, 2000, 4000];

export interface ThumbProps {
  /** Absolute photo path — sent as the ?path= query to /api/thumb. */
  path: string;
  /** Classes for the <img> itself (sizing/object-fit stay the call site's job). */
  className?: string;
  /** Above-the-fold thumbs (loupe-adjacent, anchor picker) may load eagerly. */
  eager?: boolean;
  /** Called once the frame has actually painted — loupe uses this for dims. */
  onLoad?: () => void;
  /** Pass-through for the inline-styled call sites (anchor picker, side panel). */
  style?: CSSProperties;
}

export function Thumb({ path, className, eager, onLoad, style }: ThumbProps) {
  const [phase, setPhase] = useState<'loading' | 'loaded' | 'error'>('loading');
  const [attempt, setAttempt] = useState(0);
  const timer = useRef<number | undefined>(undefined);

  // A new path is a new image — reset the whole retry story.
  useEffect(() => {
    setPhase('loading');
    setAttempt(0);
    return () => window.clearTimeout(timer.current);
  }, [path]);

  // Cache-buster on retries: the 204/404 response itself is not cached, but
  // the browser may negative-cache the URL; a fresh query forces a real fetch.
  const src = attempt === 0 ? thumbUrl(path) : `${thumbUrl(path)}&_r=${attempt}`;

  const handleError = () => {
    if (attempt < BACKOFF_MS.length) {
      // Still trying — keep the shimmer up; the tile heals itself if this was
      // a 204 (grading active) and fills in when the grade releases the pool.
      timer.current = window.setTimeout(() => setAttempt(a => a + 1), BACKOFF_MS[attempt]);
    } else {
      setPhase('error');
    }
  };

  if (phase === 'error') {
    return (
      <span
        role="button"
        tabIndex={0}
        aria-label="Thumbnail failed to load — press to retry"
        title="Thumbnail unavailable — click to retry"
        onClick={() => { setAttempt(0); setPhase('loading'); }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setAttempt(0);
            setPhase('loading');
          }
        }}
        className={cn(
          'group absolute inset-0 flex cursor-pointer items-center justify-center',
          'bg-well transition-colors duration-fast ease hover:bg-raised',
        )}
      >
        <RefreshCw size={12} className="text-ink-4 transition-colors duration-fast ease group-hover:text-ink-2" />
      </span>
    );
  }

  return (
    <>
      {phase === 'loading' && <span aria-hidden className="skeleton absolute inset-0" />}
      <img
        src={src}
        alt=""
        decoding="async"
        loading={eager ? 'eager' : 'lazy'}
        style={style}
        onError={handleError}
        onLoad={() => { setPhase('loaded'); onLoad?.(); }}
        className={cn(
          'transition-opacity duration-fast ease',
          phase === 'loading' && 'opacity-0',
          className,
        )}
      />
    </>
  );
}
