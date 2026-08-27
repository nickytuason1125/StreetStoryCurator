// The machine has two verdict channels on a contact-sheet cell: the 2px rule
// under the frame, and the glass chip over it. Both answer the same question -
// "does this grade get a mark?" - and tokens.css, tokens.ts and the JSX comment
// in GridView all give the same answer: Strong and Weak speak, Mid stays silent.
//
// `gradeRule` implemented that. The chip did not: its JSX guarded only on
// !isPending, so Mid fell through and rendered a label in --ink-2. On a real
// folder that is most of the grid (10 Strong / 2 Mid / 0 Weak on the sample
// set), which is exactly the noise "Mid stays silent" exists to prevent.
//
// These lock the two channels to the same rule so they cannot drift again.
//
// Run:  node scripts/test-grade-channels.mjs
import { gradeRule, gradeBadge, gradeKey } from '../src/theme/tokens.ts';

let failed = 0;
const check = (name, actual, expected) => {
  const ok = actual === expected;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}` + (ok ? '' : `  (got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)})`));
};

// The rule channel - the reference implementation, already correct.
check('rule: Strong speaks', typeof gradeRule('Strong ✅'), 'string');
check('rule: Weak speaks', typeof gradeRule('Weak ❌'), 'string');
check('rule: Mid is silent', gradeRule('Mid ⚠️'), null);
check('rule: Pending is silent', gradeRule(null), null);

// The badge channel - must answer identically.
check('badge: Strong speaks', typeof gradeBadge('Strong ✅'), 'string');
check('badge: Weak speaks', typeof gradeBadge('Weak ❌'), 'string');
check('badge: Mid is silent', gradeBadge('Mid ⚠️'), null);
check('badge: Pending is silent', gradeBadge(null), null);
check('badge: undefined is silent', gradeBadge(undefined), null);

// The invariant itself: the two channels agree on WHO speaks, for every grade.
for (const g of ['Strong ✅', 'Mid ⚠️', 'Weak ❌', null, undefined, '', 'garbage']) {
  check(`channels agree for ${JSON.stringify(g)} (${gradeKey(g)})`,
        gradeRule(g) === null, gradeBadge(g) === null);
}

console.log(failed ? `\n${failed} FAILED` : '\nall green');
process.exit(failed ? 1 : 0);
