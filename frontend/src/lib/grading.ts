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

/* ── Aspect calibration ──────────────────────────────────────────────────────
 * The pipeline's final score is a calibrated, gated fusion (archetype weights,
 * anchor floors, hard rejects, rating-anchored thresholds), but the per-aspect
 * breakdown values are RAW head outputs on the CLIP scale — typically 0.35-0.55
 * even for keepers. Comparing raw aspects against the calibrated grade
 * cut-points (Strong >= 0.60 / Weak < 0.41) makes every calculation contradict
 * the grade: a Strong photo read as "Mid/Weak" in every row of its own panel.
 *
 * The fix is a per-photo linear remap: shift the photo's aspect family so its
 * mean lands on the photo's final score, preserving the spread BETWEEN aspects
 * (which aspect carries and which drags) while putting them on the scale the
 * grade actually uses. Raw zeros are preserved — the grader emits 0 for
 * "not scored" (e.g. Human/Culture with no person) and shifting those would
 * manufacture data out of nothing. */
export function calibratedAspects(
  raw: Record<string, number>,
  score: number,
): Record<string, number> {
  const vals = Object.entries(raw)
    .filter(([, v]) => typeof v === 'number' && isFinite(v) && v > 0);
  if (!vals.length || typeof score !== 'number' || !isFinite(score)) return { ...raw };
  const mean = vals.reduce((s, [, v]) => s + v, 0) / vals.length;
  const off  = score - mean;
  const out: Record<string, number> = { ...raw };
  for (const [k, v] of vals) out[k] = Math.max(0, Math.min(1, v + off));
  return out;
}

/** Same cut-points the grade vocabulary documents (tokens.ts): the tier an
 * aspect's CALIBRATED value occupies. */
export function aspectTier(v: number): 'Strong' | 'Mid' | 'Weak' {
  return v >= 0.60 ? 'Strong' : v >= 0.41 ? 'Mid' : 'Weak';
}

/** Grade → token colour. Mid is deliberately silent (see tokens.css). */
export function gc(g: string) {
  const k = gradeKey(g);
  if (k === 'strong') return T.gradeStrong;
  if (k === 'weak')   return T.gradeWeak;
  if (k === 'mid')    return T.ink2;   // silent — neutral, never amber
  return T.ink3;
}

