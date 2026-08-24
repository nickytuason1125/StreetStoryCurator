/**
 * scripts/check-tdz.mjs — TDZ (temporal dead zone) guard for hook dependency
 * arrays.
 *
 * Why this exists: in a ~5k-line component, a useEffect/useMemo/useCallback can
 * legally list an identifier in its dependency array that is declared LATER in
 * the component body with `const`. Dependency arrays evaluate during every
 * render, so this throws "Cannot access 'x' before initialization" at runtime —
 * but only when that view mounts, and minified into an opaque name like `ma`
 * in production builds. This exact bug shipped once (handleSetStars, 2026-08).
 *
 * The check: for every hook dependency array `[...]` in src/**\/*. {ts,tsx},
 * flag identifiers that have NO declaration before the use site but DO have one
 * after it. Declarations counted: import bindings, `const x =`,
 * `const [a, b] =`, and `function x(`. Anything already declared above is the
 * binding the dep array closes over — safe regardless of later shadowing.
 *
 * Exit code 1 on any finding, so `npm run build` fails fast.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = join(process.cwd(), 'src');
const HOOK_DEPS_RE = /\}\s*,\s*\[([^\]]*)\]\s*\)/g;
const IDENT_RE = /^[A-Za-z_$][\w$]*$/;

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(e =>
    e.isDirectory() ? walk(join(dir, e.name)) : /\.(ts|tsx)$/.test(e.name) ? [join(dir, e.name)] : []
  );
}

// Names bound by a declaration line: `const x =`, `const [a, b] =`,
// `function x(`, or any name inside an import clause.
function declaredNames(line) {
  const names = [];
  let m;
  if ((m = line.match(/^\s*const\s*\[([^\]]*)\]/))) {
    // Split destructure, drop default values and renames ("src: dest").
    for (const part of m[1].split(',')) {
      const n = part.split(':')[0].split('=')[0].trim();
      if (IDENT_RE.test(n)) names.push(n);
    }
  } else if ((m = line.match(/^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/))) {
    names.push(m[1]);
  } else if ((m = line.match(/^\s*function\s+([A-Za-z_$][\w$]*)/))) {
    names.push(m[1]);
  } else if (/^\s*import\b/.test(line)) {
    const clause = line.match(/\{([^}]*)\}/)?.[1] ?? '';
    for (const part of clause.split(',')) {
      const n = part.split(' as ').pop()?.trim();
      if (n && IDENT_RE.test(n)) names.push(n);
    }
    const def = line.match(/import\s+([A-Za-z_$][\w$]*)\s+from/);
    if (def) names.push(def[1]);
  }
  return names;
}

let failures = 0;

for (const file of walk(SRC)) {
  const rel = relative(process.cwd(), file);
  const lines = readFileSync(file, 'utf8').split('\n');

  // decls: name -> sorted list of 1-based line numbers.
  const decls = new Map();
  for (let i = 0; i < lines.length; i++)
    for (const n of declaredNames(lines[i]))
      decls.set(n, [...(decls.get(n) ?? []), i + 1]);

  const text = lines.join('\n');
  for (const m of text.matchAll(HOOK_DEPS_RE)) {
    const useLine = text.slice(0, m.index).split('\n').length;
    for (const raw of m[1].split(',')) {
      const name = raw.trim().replace(/\s*=>\s*\(?\s*\w+\s*\)?$/, ''); // allow "(e) => setX(e)" style? keep simple
      if (!IDENT_RE.test(name)) continue;
      const all = decls.get(name);
      if (!all) continue; // prop, state destructure elsewhere, global — not ours to judge
      const before = all.filter(l => l < useLine).length;
      const after = all.length - before;
      if (before === 0 && after > 0) {
        failures++;
        console.error(
          `✗ ${rel}:${useLine} — dep '${name}' is declared LATER ` +
          `(line ${all.find(l => l > useLine)}) but referenced in a hook ` +
          `dependency array here. Deps evaluate during render → TDZ crash ` +
          `when this view mounts. Move the declaration above this effect.`
        );
      }
    }
  }
}

if (failures) {
  console.error(`\nlint:tdz — ${failures} temporal-dead-zone risk${failures > 1 ? 's' : ''} found.`);
  process.exit(1);
}
console.log('lint:tdz — OK, no forward references in hook dep arrays.');
