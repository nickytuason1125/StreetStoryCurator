import { T, gradeKey } from "../theme/tokens";

/**
 * Classify free system RAM into a readiness level for grading. `min` is the
 * server's hard gate (below it a grade is refused); a +1.2 GB margin above
 * that is treated as "tight" (grades, but may drop to lighter CLIP scoring).
 */
export function ramReadiness(gs: any): {
  level: 'clear' | 'tight' | 'critical' | 'unknown';
  free: number | null; total: number | null; percent: number | null; readout: string; tip: string;
} {
  const free    = gs?.ram_free_gb ?? null;
  const total   = gs?.ram_total_gb ?? null;
  const percent = gs?.ram_percent ?? null;
  const min     = gs?.ram_min_gb ?? 1.8;
  if (free == null) return { level: 'unknown', free, total, percent, readout: '', tip: 'System memory unknown' };
  // Headroom is the only number that changes a decision here, and it is the
  // only one the grade floor is expressed in, so it is the only one the chip
  // carries. Printing "% in use" beside it made this the widest element in a
  // toolbar that already needs 2008px of a 1500px window; the percentage is
  // still in the tooltip and in the popover, where there is room for context.
  const readout = percent != null || total != null
    ? `${free.toFixed(1)} GB free`
    : `${free.toFixed(1)} GB`;
  const usedTip = percent != null ? ` (${percent.toFixed(0)}% in use — matches Task Manager)` : '';
  // "clear" requires at least 5 GB free: the SigLIP encode subprocess needs
  // ~2 GB RAM during model load plus the grade worker's baseline ~1 GB, leaving
  // 2 GB breathing room on a 5 GB machine. Below 5 GB is genuinely risky.
  const clearThresh = Math.max(min + 1.2, 5.0);
  if (free < min)           return { level: 'critical', free, total, percent, readout, tip: `Only ${free.toFixed(1)} GB free${usedTip} — grading needs ~${min} GB. Close some apps before grading.` };
  if (free < clearThresh)   return { level: 'tight',    free, total, percent, readout, tip: `${free.toFixed(1)} GB free${usedTip} — enough to grade, but close Chrome or other heavy apps first for a stable cull.` };
  return { level: 'clear', free, total, percent, readout, tip: `${free.toFixed(1)} GB free${usedTip} — clear to grade.` };
}

/** Grade → token colour. Mid is deliberately silent (see tokens.css). */
export function gc(g: string) {
  const k = gradeKey(g);
  if (k === 'strong') return T.gradeStrong;
  if (k === 'weak')   return T.gradeWeak;
  if (k === 'mid')    return T.ink2;   // silent — neutral, never amber
  return T.ink3;
}
