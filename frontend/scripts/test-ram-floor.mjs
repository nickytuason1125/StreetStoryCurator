// The cull RAM floor is served by the backend as ram_min_gb (computed by
// run_profile.required_ram_gb). It has moved once already — 1.8 -> 3.8 — and
// the move left three stale literal 1.8s in display code, each of which told
// the photographer that 1.8 GB was enough while the gate refused the cull at
// 3.8. Every one of those failures under-warns, which is the dangerous
// direction.
//
// These lock the rule: ramReadiness is the ONLY reader of ram_min_gb, it
// reports what it read as `min`, and when the server has not said, the answer
// is 'unknown' rather than an invented number.
//
// Run:  node scripts/test-ram-floor.mjs
import { ramReadiness } from '../src/lib/grading.ts';

let failed = 0;
const check = (name, actual, expected) => {
  const ok = Object.is(actual, expected);
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}` + (ok ? '' : `  (got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)})`));
};

// It reports the floor the server sent, verbatim.
check('min is the served floor',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 3.8 }).min, 3.8);
check('min follows the server when the floor moves',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 6.6 }).min, 6.6);

// No floor from the server = unknown. Never a guess.
check('absent floor is unknown',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44 }).level, 'unknown');
check('absent floor reports no min',
      ramReadiness({ ram_free_gb: 9, ram_total_gb: 16, ram_percent: 44 }).min, null);
check('null payload is unknown',
      ramReadiness(null).level, 'unknown');
check('absent free RAM is still unknown',
      ramReadiness({ ram_min_gb: 3.8 }).level, 'unknown');

// The readiness bands are driven by the SERVED floor, not by a baked-in one.
// At a 3.8 floor, 3.0 GB free is below it: critical.
check('3.0 free under a 3.8 floor is critical',
      ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 3.8 }).level, 'critical');
// The same 3.0 GB under the old 1.8 floor was merely tight — proving the band
// actually follows the server rather than a constant.
check('3.0 free under a 1.8 floor is tight',
      ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 1.8 }).level, 'tight');
check('9.0 free under a 3.8 floor is clear',
      ramReadiness({ ram_free_gb: 9.0, ram_total_gb: 16, ram_percent: 44, ram_min_gb: 3.8 }).level, 'clear');

// The tip must quote the served floor, so the popover and the chip cannot
// disagree with each other or with the gate.
const crit = ramReadiness({ ram_free_gb: 3.0, ram_total_gb: 16, ram_percent: 81, ram_min_gb: 3.8 });
check('tip quotes the served floor', crit.tip.includes('3.8'), true);
check('tip does not quote the dead 1.8 floor', crit.tip.includes('1.8'), false);

console.log(failed ? `\n${failed} FAILED` : '\nall green');
process.exit(failed ? 1 : 0);
