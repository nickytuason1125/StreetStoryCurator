/* Cull time estimate for the pre-grade dialog (2026-10-04).
 *
 * Measured with scripts/perf_guard.py on this pipeline (RTX 3060 Laptop,
 * 600 ARWs, real grade_runner): ~0.30 s per photo plus ~25 s of fixed start-up
 * when the machine has >= 3.5 GB free. Below that the encoder waits for memory
 * between chunks and detection can no longer overlap the encode — the same
 * cull measured roughly 2x slower. Re-measure (perf_guard) if the pipeline
 * changes; these are measurements, not targets.
 */
export const SECONDS_PER_PHOTO = 0.30;
export const FIXED_SECONDS = 25;
export const FULL_SPEED_FREE_GB = 3.5;
export const LOW_RAM_FACTOR = 2;

export interface CullEstimate {
  seconds: number;
  slowedByRam: boolean;
  text: string;
}

function fmt(seconds: number): string {
  if (seconds < 90) return 'under 2 minutes';
  const m = Math.round(seconds / 60);
  if (m < 60) return `about ${m} min`;
  const h = Math.floor(m / 60), r = m % 60;
  return `about ${h} h ${r ? `${r} min` : ''}`.trim();
}

export function cullEstimate(photoCount: number, freeGb: number | null): CullEstimate | null {
  if (!photoCount || photoCount <= 0) return null;
  const base = FIXED_SECONDS + SECONDS_PER_PHOTO * photoCount;
  const slowedByRam = freeGb != null && freeGb < FULL_SPEED_FREE_GB;
  const seconds = slowedByRam ? base * LOW_RAM_FACTOR : base;
  const text = slowedByRam
    ? `Estimated time: ${fmt(seconds)} — about twice as long as usual, because only ` +
      `${freeGb!.toFixed(1)} GB of memory is free. Closing a few apps restores full speed (${fmt(base)}).`
    : `Estimated time: ${fmt(seconds)} for ${photoCount.toLocaleString()} photos.`;
  return { seconds, slowedByRam, text };
}
