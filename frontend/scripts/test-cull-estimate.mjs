import assert from 'node:assert/strict';
import { cullEstimate, SECONDS_PER_PHOTO, FIXED_SECONDS } from '../src/lib/cullEstimate.ts';

assert.equal(cullEstimate(0, 8), null);
const ok = cullEstimate(3000, 8);
assert.equal(ok.slowedByRam, false);
assert.equal(ok.seconds, FIXED_SECONDS + SECONDS_PER_PHOTO * 3000);
assert.match(ok.text, /about 15 min/);
const low = cullEstimate(3000, 1.2);
assert.equal(low.slowedByRam, true);
assert.equal(low.seconds, ok.seconds * 2);
assert.match(low.text, /1\.2 GB/);
assert.equal(cullEstimate(100, null).slowedByRam, false);   // unknown RAM: no false alarm
console.log('cull-estimate: ok');
