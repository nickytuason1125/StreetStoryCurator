import { T } from '../theme/tokens';

/* ── Vision-critique region guide ───────────────────────────────────────────
 * One teacher vocabulary shared by every overlay (heatmap glow, box callouts,
 * Analysis panel). Maps each Qwen spatial label → a quality tier, a plain-English
 * title, and a one-line coaching tip a photographer can act on.
 *
 * Extracted verbatim from App.tsx during the views split — the loupe stage,
 * the analysis panel and the critique heatmap all speak this vocabulary.
 */
export type RegionTier = 'strong' | 'refine' | 'fix';
const REGION_GUIDE: Record<string, { tier: RegionTier; title: string; tip: string }> = {
  anchor_subject:     { tier:'strong', title:'Strong anchor',      tip:'The eye lands here first — a clear subject grounds the frame.' },
  composition_anchor: { tier:'strong', title:'Composition anchor', tip:'This element structures the shot — placement is working.' },
  focal_point_miss:   { tier:'refine', title:'Focal point drifts',  tip:'Attention wanders here — simplify or re-frame to hold the eye.' },
  blown_highlight:    { tier:'fix',    title:'Highlights clipped',  tip:'Detail lost in the brights — lower exposure or recover in post.' },
  crushed_shadow:     { tier:'fix',    title:'Shadows crushed',     tip:'Detail lost in the darks — lift shadows to keep texture.' },
  motion_blur:        { tier:'fix',    title:'Motion blur',         tip:'Subject isn’t sharp — raise shutter speed to freeze motion.' },
};
export const regionGuide = (label: string) =>
  REGION_GUIDE[label] ?? { tier: 'refine' as RegionTier, title: (label || 'region').replace(/_/g, ' '), tip: '' };
export const tierColor = (t: RegionTier) =>
  t === 'strong' ? T.gradeStrong : t === 'fix' ? T.gradeWeak : T.ink2;
export const tierIcon  = (t: RegionTier) => t === 'strong' ? '✓' : t === 'fix' ? '!' : '◐';
/* The heatmap is painted ON the photograph, so it may not reach for --mark:
 * that colour is reserved for marks the photographer made himself. This is the
 * one place the alarm tokens apply to image content rather than status chrome,
 * because `fix` and `refine` flag something he actually has to act on.
 *
 * These were #3fb950 / #f85149 / #d8a657 — GitHub's palette, three raw hex
 * literals sitting under a comment claiming they were "kept distinct from theme
 * tokens". Distinct from the tokens is exactly what a stray literal is. */
export const tierHeat  = (t: RegionTier) =>
  t === 'strong' ? T.gradeStrong : t === 'fix' ? T.alarmCrit : T.alarmWarn;
