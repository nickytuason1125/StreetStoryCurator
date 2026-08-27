#!/usr/bin/env node
/* Token guard — the part that keeps the design system from rotting.
 *
 * The UI reached 539 inline style blocks, 15 font sizes, 13 border radii and 41
 * stray hex colours purely by accretion. Nobody chose 7px and 32px as a scale;
 * they arrived one commit at a time. A style guide in a document cannot stop
 * that. A check that fails the build can.
 *
 * Two modes, because a big migration needs both:
 *
 *   STRICT   — src/theme, src/components, src/views, src/panels, src/overlays,
 *              src/modals, src/hooks, src/lib. Zero violations allowed. All new
 *              code lands here, so the new surface is clean from day one.
 *
 *   RATCHET  — everything else in src/ (i.e. the not-yet-migrated App.tsx). The
 *              current count is recorded in token-baseline.json and may only go
 *              DOWN. You can't fix 4,868 lines in one pass, but you must never
 *              add to the pile. Lowering the count rewrites the baseline.
 *
 * Run:  npm run lint:tokens          (check)
 *       npm run lint:tokens -- --update   (accept a lower baseline)
 */

import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const SRC = join(ROOT, 'src');
const BASELINE = join(ROOT, 'token-baseline.json');

// Files that are ALLOWED to contain literal values — they are the source of truth.
const EXEMPT = ['src/theme/tokens.css'];

// Directories held to zero.
const STRICT_DIRS = [
  'src/theme', 'src/components', 'src/views',
  'src/panels', 'src/overlays', 'src/modals', 'src/hooks', 'src/lib',
];

// The spacing scale is deliberately sparse (see tailwind.config.js). Anything
// outside it is not a token — it is a class name that silently resolves to
// nothing, which is far worse than an ugly value because it still looks correct
// in the source.
const SPACING_STEPS = new Set(['0', '1', '2', '3', '4', '6', '8', '12']);
const SIZE_PREFIXES =
  'p|px|py|pt|pb|pl|pr|m|mx|my|mt|mb|ml|mr|gap|gap-x|gap-y|w|h|min-w|min-h|space-x|space-y|inset|top|bottom|left|right';

const RULES = [
  { name: 'hex colour',    re: /#[0-9a-fA-F]{3,8}\b/g },
  // Raw pixel sizing in inline styles. `fontSize:13` and `borderRadius:8` are
  // exactly how the 15-size / 13-radius sprawl happened.
  { name: 'raw fontSize',     re: /\bfontSize\s*:\s*['"]?\d/g },
  { name: 'raw borderRadius', re: /\bborderRadius\s*:\s*['"]?\d/g },

  // ── Typography literals ───────────────────────────────────────────────────
  // The guard caught hex and font SIZE but never tracking or leading, so those
  // two drifted freely while everything around them stayed disciplined: nine
  // different letter-spacings for what is one thing (an uppercase micro-label,
  // --track-label) and six different body leadings against one --leading-body.
  // That is why retuning the tokens did not visibly change the UI — the call
  // sites had stopped asking. `var(--x)` passes; a digit, dot or minus does not.
  //
  // These carry their own ratchet (`typographyLiterals`) rather than joining the
  // legacy one, because `legacyViolations` was already at 0 and folding a new
  // category in would have silently surrendered that guarantee. The counter was
  // seeded at 42, the call sites were migrated in the same pass, and it now sits
  // at 0 — so a single new literal anywhere in src/ fails the build, which is
  // the same strictness the other rules get. The separate counter is kept for
  // its error message: it names the token you should have reached for.
  { name: 'raw letterSpacing', re: /\bletterSpacing\s*:\s*['"]?[-.\d]/g, typography: true },
  { name: 'raw lineHeight',    re: /\blineHeight\s*:\s*['"]?[.\d]/g,     typography: true },
  // oklch()/rgb() literals are hex by another name.
  { name: 'raw colour fn', re: /\b(?:oklch|rgba?|hsla?)\s*\(/g },

  // ── Dead-class rules ──────────────────────────────────────────────────────
  // Replacing theme.colors / theme.spacing wholesale deleted every stock
  // Tailwind class name. Code using them keeps compiling and keeps passing
  // type-checks; it just renders unstyled. That shipped twice — the Toast lost
  // its entire background, and three panels lost their widths — so it is worth
  // a build failure rather than a code review.
  {
    name: 'stock Tailwind palette class (deleted by the token theme)',
    re: /\b(?:bg|text|border|outline|ring|divide|from|via|to)-(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/g,
  },
  {
    name: 'radius outside the four-step scale (rounded-sm | rounded-md | rounded-lg | rounded-xl | rounded-full)',
    re: /\brounded-(?:2xl|3xl)\b/g,
  },
];

/** Spacing/size utilities whose step isn't in the scale — they resolve to nothing. */
function offScaleSizes(body) {
  const re = new RegExp(`\\b(?:${SIZE_PREFIXES})-(\\d+(?:\\.5)?)\\b`, 'g');
  const bad = [];
  for (const m of body.matchAll(re)) {
    if (!SPACING_STEPS.has(m[1])) bad.push(m[0]);
  }
  return bad;
}

function walk(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).flatMap((entry) => {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) return walk(p);
    return /\.(tsx?|css)$/.test(p) ? [p] : [];
  });
}

// Comments explain the rules and legitimately quote hex values; counting them
// would make the guard punish its own documentation.
function strip(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
}

const files = walk(SRC);
let strictViolations = [];
let deadClasses = [];      // always fatal, anywhere — these render as nothing
let ratchetCount = 0;
let typographyCount = 0;
const typographyHits = [];   // rel -> sample, for a report that names files
const definedKeyframes = new Set();
const usedKeyframes = new Map();  // name -> first file that uses it

for (const file of files) {
  const rel = relative(ROOT, file).split(sep).join('/');
  const raw = readFileSync(file, 'utf8');
  const body = strip(raw);

  // Collect keyframes across the whole tree before judging any of them.
  for (const m of body.matchAll(/@keyframes\s+([A-Za-z][\w-]*)/g)) definedKeyframes.add(m[1]);
  for (const m of body.matchAll(/animation:\s*['"`]?\s*([A-Za-z][\w-]*)/g)) {
    if (!usedKeyframes.has(m[1])) usedKeyframes.set(m[1], rel);
  }

  if (EXEMPT.includes(rel)) continue;
  const isStrict = STRICT_DIRS.some((d) => rel.startsWith(d + '/'));

  for (const rule of RULES) {
    const hits = body.match(rule.re) || [];
    if (!hits.length) continue;
    // Dead classes are never acceptable — not even in un-migrated code, because
    // the failure is invisible rather than merely untidy.
    if (rule.name.startsWith('stock Tailwind') || rule.name.startsWith('radius outside')) {
      deadClasses.push(`  ${rel}: ${[...new Set(hits)].slice(0, 6).join(', ')}  (${rule.name})`);
    } else if (rule.typography) {
      // Own ratchet, regardless of STRICT/legacy — see the note on the rules.
      typographyCount += hits.length;
      typographyHits.push(`  ${rel}: ${hits.length}x ${rule.name}`);
    } else if (isStrict) {
      strictViolations.push(`  ${rel}: ${hits.length}x ${rule.name} (${[...new Set(hits)].slice(0, 4).join(', ')})`);
    } else {
      ratchetCount += hits.length;
    }
  }

  const offScale = offScaleSizes(body);
  if (offScale.length) {
    deadClasses.push(`  ${rel}: ${[...new Set(offScale)].slice(0, 6).join(', ')}  (spacing step not in the scale)`);
  }
}

// Keyframes live in plain CSS precisely because Tailwind only emits the ones it
// can see in an `animate-*` utility; an inline `animation: spin` is invisible to
// that scan. Catch the mismatch here instead of noticing frozen spinners later.
const KEYFRAME_IGNORE = new Set(['none', 'inherit', 'initial', 'unset', 'var']);
const missingKeyframes = [...usedKeyframes]
  .filter(([name]) => !definedKeyframes.has(name) && !KEYFRAME_IGNORE.has(name));

const saved = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, 'utf8')) : {};
const prev = existsSync(BASELINE) ? saved.legacyViolations : ratchetCount;
// A key that has never been recorded seeds from the current count; from then on
// it may only fall. Seeding is why this is `??` and not `|| typographyCount` —
// a recorded 0 is a real baseline and must not be re-seeded to a higher number.
const prevTypo = saved.typographyLiterals ?? typographyCount;

const updating = process.argv.includes('--update');
let failed = false;

if (deadClasses.length) {
  console.error('\nToken guard FAILED — class names that resolve to NOTHING:\n');
  console.error(deadClasses.join('\n'));
  console.error(
    '\nThese still compile and still type-check; they just render unstyled.\n' +
    'Use a token colour (bg-surface, text-ink-2, border-line) and a scale step\n' +
    '(0 1 2 3 4 6 8 12), or add a named size to tailwind.config.js.\n'
  );
  failed = true;
}

if (missingKeyframes.length) {
  console.error('\nToken guard FAILED — animation with no @keyframes:\n');
  for (const [name, where] of missingKeyframes) {
    console.error(`  ${where}: animation: ${name} — no @keyframes ${name} anywhere in src/`);
  }
  console.error('\nDeclare it in src/index.css. Defining it only in tailwind.config.js\n' +
                'is not enough: Tailwind emits keyframes solely for animate-* utilities\n' +
                'it can see, so inline `animation:` usages silently get nothing.\n');
  failed = true;
}

if (strictViolations.length) {
  console.error('\nToken guard FAILED — literal values in managed code:\n');
  console.error(strictViolations.join('\n'));
  console.error('\nUse a token from src/theme/tokens.css instead.\n');
  failed = true;
}

if (ratchetCount > prev) {
  console.error(
    `\nToken guard FAILED — legacy literals went UP: ${prev} -> ${ratchetCount}.\n` +
    `Un-migrated code may only shrink. Move the code you touched onto tokens.\n`
  );
  failed = true;
}

if (typographyCount > prevTypo) {
  console.error(
    `\nToken guard FAILED — typography literals went UP: ${prevTypo} -> ${typographyCount}.\n\n` +
    typographyHits.join('\n') +
    `\n\nUse --track-label / --track-tight / --track-brand for letterSpacing and\n` +
    `--leading-body / --leading-display for lineHeight. If a value genuinely has\n` +
    `no token yet, add one to tokens.css — that is the decision this guard exists\n` +
    `to force.\n`
  );
  failed = true;
}

if (failed) process.exit(1);

// A missing baseline must be WRITTEN, not quietly tolerated. `prev` falls back
// to ratchetCount when the file is absent, so without this branch the ratchet
// compares 401 against 401 forever: it can never fail, and never records the
// file that would let it fail. The guard's whole second mode was inert.
const record = () => writeFileSync(BASELINE, JSON.stringify(
  { legacyViolations: ratchetCount, typographyLiterals: typographyCount }, null, 2) + '\n');

// A newly-added counter has to be written on its first clean run, or it never
// gets a baseline to ratchet against — the same inert-guard bug the legacy
// counter had before the `!existsSync` branch below was added.
const seedingTypo = saved.typographyLiterals === undefined;

if (!existsSync(BASELINE)) {
  record();
  console.log(`Token guard OK — baseline recorded at ${ratchetCount} legacy, ${typographyCount} typography literals.`);
} else if (ratchetCount < prev || typographyCount < prevTypo || seedingTypo || updating) {
  record();
  const parts = [];
  if (ratchetCount < prev || updating) parts.push(`legacy ${prev} -> ${ratchetCount}`);
  if (seedingTypo) parts.push(`typography baseline seeded at ${typographyCount}`);
  else if (typographyCount < prevTypo) parts.push(`typography ${prevTypo} -> ${typographyCount}`);
  console.log(`Token guard OK — ${parts.join('; ')}.`);
} else {
  console.log(`Token guard OK — managed code clean, legacy ${ratchetCount} (baseline ${prev}), ` +
              `typography ${typographyCount} (baseline ${prevTypo}).`);
}
