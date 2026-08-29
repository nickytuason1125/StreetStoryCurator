import { T, gradeKey } from "../theme/tokens.ts";

/**
 * Classify free system RAM into a readiness level for grading. `min` is the
 * server's hard gate (below it a grade is refused); a +1.2 GB margin above
 * that is treated as "tight" (grades, but may drop to lighter CLIP scoring).
 */
export function ramReadiness(gs: any): {
  level: 'clear' | 'tight' | 'critical' | 'unknown';
  free: number | null; total: number | null; percent: number | null;
  min: number | null; readout: string; tip: string;
} {
  const free    = gs?.ram_free_gb ?? null;
  const total   = gs?.ram_total_gb ?? null;
  const percent = gs?.ram_percent ?? null;
  // The floor is the SERVER'S, always. It is computed per-machine by
  // run_profile.required_ram_gb() from a measured table, and it has already
  // moved once (1.8 -> 3.8). A literal default here would be a second source
  // of truth that goes stale silently and under-warns — which is how the
  // photographer gets told 1.8 GB is fine and then gets a 503.
  const min: number | null = typeof gs?.ram_min_gb === 'number' ? gs.ram_min_gb : null;
  const unknown = { level: 'unknown' as const, free, total, percent, min,
                    readout: '', tip: 'System memory unknown' };
  if (free == null || min == null) return unknown;
  // Headroom is the only number that changes a decision here, and it is the
  // only one the grade floor is expressed in, so it is the only one the chip
  // carries. Printing "% in use" beside it made this the widest element in a
  // toolbar that already needs 2008px of a 1500px window; the percentage is
  // still in the tooltip and in the popover, where there is room for context.
  const readout = percent != null || total != null
    ? `${free.toFixed(1)} GB free`
    : `${free.toFixed(1)} GB`;
  const usedTip = percent != null ? ` (${percent.toFixed(0)}% in use — matches Task Manager)` : '';
  // "clear" needs headroom above the floor, not merely clearance of it: a cull
  // that starts at exactly the floor has nothing left for the browser beside
  // it. 1.2 GB of margin, with an absolute 5 GB backstop for the case where a
  // low floor would otherwise call a genuinely tight machine clear.
  const clearThresh = Math.max(min + 1.2, 5.0);
  if (free < min)           return { level: 'critical', free, total, percent, min, readout, tip: `Only ${free.toFixed(1)} GB free${usedTip} — grading needs ~${min} GB. Close some apps before grading.` };
  if (free < clearThresh)   return { level: 'tight',    free, total, percent, min, readout, tip: `${free.toFixed(1)} GB free${usedTip} — enough to grade, but close Chrome or other heavy apps first for a stable cull.` };
  return { level: 'clear', free, total, percent, min, readout, tip: `${free.toFixed(1)} GB free${usedTip} — clear to grade.` };
}

/** Grade → token colour. Mid is deliberately silent (see tokens.css). */
export function gc(g: string) {
  const k = gradeKey(g);
  if (k === 'strong') return T.gradeStrong;
  if (k === 'weak')   return T.gradeWeak;
  if (k === 'mid')    return T.ink2;   // silent — neutral, never amber
  return T.ink3;
}
