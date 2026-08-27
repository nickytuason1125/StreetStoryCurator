#!/usr/bin/env node
/* Frontend test runner — discovers and runs every scripts/test-*.mjs.
 *
 * Deliberately not vitest. This project already guards itself with hand-rolled
 * node scripts wired into `npm run build` (lint-tokens, check-tdz), and those
 * catch the failures that actually shipped here: dead Tailwind classes, missing
 * keyframes, TDZ in hook deps. A test framework would add a dependency tree and
 * a config file to run assertions that are three lines of plain JavaScript.
 *
 * The one thing worth having is DISCOVERY: a single test file wired into the
 * build by name is a file the next person forgets to add a sibling to.
 *
 * Type stripping is passed explicitly so tests can import src/theme/tokens.ts
 * directly rather than testing a compiled copy of it.
 *
 * Run:  npm run test
 */
import { readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

const tests = readdirSync(HERE)
  .filter((f) => /^test-.*\.mjs$/.test(f))
  .sort();

if (!tests.length) {
  console.error('No scripts/test-*.mjs found. Expected at least one.');
  process.exit(1);
}

let failed = 0;
for (const t of tests) {
  const r = spawnSync(process.execPath, ['--experimental-strip-types', join(HERE, t)], {
    stdio: 'inherit',
  });
  if (r.status !== 0) {
    failed++;
    console.error(`FAILED: ${t}`);
  }
}

console.log(
  failed
    ? `\n${failed} of ${tests.length} test file(s) failed.`
    : `\nFrontend tests OK — ${tests.length} file(s).`,
);
process.exit(failed ? 1 : 0);
