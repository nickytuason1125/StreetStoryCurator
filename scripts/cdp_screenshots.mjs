/**
 * scripts/cdp_screenshots.mjs — drive the real FrameGrade app headlessly and
 * capture screenshots of its actual UI states (no mocks).
 *
 * Flow: spawn Edge (headless, CDP port) → open the app → click "Resume" so the
 * seeded catalog loads → wait for thumbnails → screenshot grid → open loupe →
 * screenshot. Uses Node's built-in WebSocket (v22+); no dependencies.
 *
 * Usage: node scripts/cdp_screenshots.mjs [outDir]
 */
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const OUT = process.argv[2] ?? 'screenshots';
const PORT = 9223;
const APP = 'http://127.0.0.1:8000/';
const EDGE = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
].find(p => { try { return readdirSync(p).includes('msedge.exe'); } catch { return false; } });

if (!EDGE) { console.error('msedge.exe not found'); process.exit(1); }
mkdirSync(OUT, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));

const edge = spawn(EDGE, [
  '--headless=new', '--disable-gpu',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${join(OUT, '.edge-profile')}`,
  '--window-size=1680,1050', APP,
], { stdio: 'ignore' });
const cleanup = () => { try { edge.kill(); } catch {} };
process.on('exit', cleanup);

// Wait for the CDP target to appear.
let target = null;
for (let i = 0; i < 40 && !target; i++) {
  await sleep(500);
  try {
    const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
    target = list.find(t => t.type === 'page' && t.url.includes('127.0.0.1:8000'));
  } catch {}
}
if (!target) { console.error('CDP target never appeared'); process.exit(1); }

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

let seq = 0;
const pending = new Map();
ws.onmessage = ev => {
  const msg = JSON.parse(ev.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
};
const rpc = (method, params = {}) => new Promise(res => {
  const id = ++seq;
  pending.set(id, res);
  ws.send(JSON.stringify({ id, method, params }));
});
const evalJs = async expression => {
  const r = await rpc('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  return r.result?.result?.value;
};
const shot = async name => {
  const r = await rpc('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(OUT, name), Buffer.from(r.result.data, 'base64'));
  console.log(`captured ${name}`);
};
const waitFor = async (expr, timeoutMs = 30000) => {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (await evalJs(expr)) return true;
    await sleep(500);
  }
  return false;
};

await rpc('Page.enable');
await rpc('Emulation.setDeviceMetricsOverride', { width: 1680, height: 1050, deviceScaleFactor: 1, mobile: false });
await sleep(4000); // let the SPA boot

await shot('now-01-home.png');

// Click "Resume" (restores the seeded catalog → grid with photos).
await evalJs(`
  const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Resume');
  if (b) b.click();
  !!b
`);
const loaded = await waitFor(`document.querySelectorAll('img').length >= 3`, 45000);
if (!loaded) console.warn('warn: fewer than 3 thumbnails appeared');
await sleep(2500); // thumbs + RAM indicator settle

await shot('now-02-grid.png');

// Open the loupe: click the selected/first grid tile, or press E if already loupe.
await evalJs(`
  const tile = document.querySelector('[data-sel="1"]') || document.querySelector('img');
  (tile && tile.click(), !!tile)
`);
await sleep(2000);
await evalJs(`
  const kb = new KeyboardEvent('keydown', {key: 'e'});
  window.dispatchEvent(kb);
  true
`);
await sleep(1500);

await shot('now-03-loupe.png');

ws.close();
cleanup();
console.log('done');
