import { T } from '../theme/tokens';

/* ── Aspect → canonical dimension classifier ───────────────────────
 * The Judge's Eye Evidence Checklist has five fixed photographic rows:
 *   tech → Focus · light → Exposure · human → Subject · auth → Moment · comp → Geometry
 * The Qwen primary grader emits niche-specific axis names (20 niches, ~70 distinct
 * axes), so a hardcoded lookup on 'Lighting'/'Human/Culture'/'Narrative' leaves rows
 * unrated. This maps every known axis (display Title-Case) onto a dimension; a keyword
 * fallback covers anything unseen. Where a niche genuinely lacks a dimension (e.g.
 * Landscape has no human axis) the row stays empty and falls back to a context label.
 *
 * Extracted verbatim from App.tsx — shared by the analysis panel and the loupe. */
export type AspectDim = 'tech'|'light'|'human'|'auth'|'comp'|'';
export const ASPECT_DIM: Record<string,'tech'|'light'|'human'|'auth'|'comp'> = {
  // canonical 5 (SpecVLM fallback + legacy)
  Technical:'tech', Composition:'comp', Lighting:'light', Narrative:'auth', 'Human/Culture':'human',
  // tech
  Detail:'tech', Execution:'tech', 'Depth Of Field':'tech', 'Detail Retention':'tech',
  Cleanliness:'tech', 'News Sharpness':'tech', 'Sharpness & Detail':'tech',
  // comp (geometry / framing / spatial)
  Geometry:'comp', 'Compositional Urgency':'comp', 'City Texture':'comp', 'Landscape Comp':'comp',
  'Depth Scale':'comp', 'Negative Space':'comp', 'Graphic Simplicity':'comp',
  'Visual Abstraction':'comp', 'Pattern Texture':'comp', 'Graphic Impact':'comp',
  'Framing':'comp', 'Geometry & Balance':'comp', 'Framing Instinct':'comp', 'Layered Depth':'comp',
  // light (lighting / mood / colour / atmosphere)
  'Light Atmosphere':'light', 'Light Quality':'light', 'Light Mood':'light', 'Nocturnal Mood':'light',
  'Color Palette':'light', 'Light Painting':'light', 'Color Form':'light', 'Tonal Balance':'light',
  Atmosphere:'light', 'Weather Drama':'light', Mood:'light',
  'Natural Light':'light', 'Mood & Tone':'light', 'Tonal Purity':'light', 'Contrast Purity':'light',
  'Available Light':'light', 'Natural Light Quality':'light',
  // human (subject / figure / expression / emotion)
  Human:'human', Expression:'human', 'Model Expression':'human', 'Subject Behavior':'human',
  'Subject Detail':'human', Emotion:'human', 'Emotional Moment':'human', 'Styling Aesthetic':'human',
  'Sense Of Place':'human', 'Subject Isolation':'human', 'Human Impact':'human',
  'Character Presence':'human', 'Emotional Resonance':'human', 'Scale Element':'human',
  Presence:'human', 'Scale & Life':'human',
  // auth (moment / narrative / concept / authenticity)
  Moment:'auth', 'Narrative Impact':'auth', Authenticity:'auth', Context:'auth', 'News Impact':'auth',
  'Cultural Authenticity':'auth', 'Urban Energy':'auth', 'Motion Quality':'auth', 'Temporal Effect':'auth',
  'Artistic Vision':'auth', 'Visual Poetry':'auth', 'Environmental Context':'auth', 'Habitat Context':'auth',
  'Editorial Mood':'auth', 'Peak Action':'auth', 'Story Telling':'auth', 'Conceptual Strength':'auth',
  'Visual Innovation':'auth', 'Intent Clarity':'auth',
  'Decisive Moment':'auth', 'Cultural Depth':'auth', 'Journalistic Integrity':'auth',
  'Narrative Suggestion':'auth', 'Conceptual Weight':'auth', Reduction:'auth', Immediacy:'auth',
  'Environmental Truth':'auth',
};
export function aspectDim(label: string): AspectDim {
  const hit = ASPECT_DIM[label];
  if (hit) return hit;
  const s = label.toLowerCase();
  if (/sharp|noise|technical|execution|clean|grain|render/.test(s)) return 'tech';
  if (/light|tonal|expos|contrast|atmos|mood|colou?r|nocturn|weather|night|chiaroscuro/.test(s)) return 'light';
  if (/human|subject|express|emotion|character|model|portrait|gesture|presence|behavio|figure|face|styling/.test(s)) return 'human';
  if (/moment|narrative|story|authentic|action|temporal|concept|news|energy|vision|urgenc|context|peak|innovation|intent/.test(s)) return 'auth';
  if (/compos|geometr|framing|negative|graphic|pattern|landscape|depth|abstract|place|texture|scale|spatial/.test(s)) return 'comp';
  return '';
}

