import { gradeKey, gradeRule } from '../../theme/tokens';
import { cn } from '../../lib/cn';

/* The machine's verdict, rendered as a 2px rule directly beneath the frame.
 *
 * This replaces the coloured "Strong / Mid / Weak" badge, and it also replaced
 * an earlier idea — a grease-pencil ring for a select. The ring was cut for two
 * reasons worth recording, because both are easy to re-propose:
 *
 * 1. Glyph shape is not preattentive. At thumbnail size the eye scans on hue,
 *    luminance, orientation and position; telling a ring from a slash needs a
 *    fixation. Across 500 cells that is 500 fixations instead of one saccade
 *    per row, which defeats the entire purpose of a contact sheet.
 * 2. A chinagraph mark means "a human looked at this and decided". Drawing one
 *    around a frame a model scored 0.61 claims a judgement that never happened.
 *
 * A rule in a fixed position aligns into visible bands down a column, so runs
 * of Strong are legible without reading anything, and it occludes no image.
 *
 * Mid renders nothing at all. Silence is the correct rendering of "no opinion",
 * and Mid is the majority bucket, so the grid stays quiet. Pending renders a
 * hatch instead, because "not yet seen" must not look identical to "shrugged".
 */

/* "This frame has AI annotations drawn on it."
 *
 * Lives in the metadata row, never on the image — the point of moving the grade
 * off the thumbnail was to stop covering the photograph, and re-adding a floating
 * badge would undo it. Rendered in ink rather than the warm mark colour because
 * the annotations are the model's work, not the photographer's. */
export function AnnotatedMark({ className }: { className?: string }) {
  return (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
         className={cn('shrink-0 text-ink-3', className)}
         role="img" aria-label="Has annotations">
      <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
    </svg>
  );
}

export function GradeRule({ grade, className }: { grade?: string | null; className?: string }) {
  const key = gradeKey(grade);
  const color = gradeRule(grade);

  if (key === 'pending') {
    return (
      <div
        aria-hidden
        className={cn('hatch-pending h-px w-full opacity-60', className)}
        style={{ height: 'var(--rule)' }}
      />
    );
  }

  // Mid: an empty track keeps every cell the same height, so the grid does not
  // reflow between graded and ungraded states.
  return (
    <div
      aria-hidden
      className={cn('w-full', className)}
      style={{ height: 'var(--rule)', background: color ?? 'transparent' }}
    />
  );
}
