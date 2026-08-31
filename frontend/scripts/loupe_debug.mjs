// Debug: click first grid thumb, sample img sizes every 250ms for 12s.
import puppeteer from "puppeteer-core";

const edge = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const browser = await puppeteer.launch({
  executablePath: edge,
  headless: true,
  args: ["--no-first-run", "--disable-gpu"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900 });
await page.goto("http://127.0.0.1:8000/", { waitUntil: "domcontentloaded", timeout: 30000 });
// welcome stage gates the gallery — wait for the Resume button, then click
await page.waitForFunction(() => {
  return [...document.querySelectorAll("button, a")].some((x) => /^resume$/i.test((x.textContent || "").trim()));
}, { timeout: 30000, polling: 200 });
await page.evaluate(() => {
  const b = [...document.querySelectorAll("button, a")].find((x) => /^resume$/i.test((x.textContent || "").trim()));
  if (b) b.click();
});
await page.waitForFunction(() => document.querySelectorAll("img").length >= 20, { timeout: 30000, polling: 100 });

// pick a real grid thumbnail (has /api/thumb in src)
const t0 = performance.now();
const clicked = await page.evaluate(() => {
  const imgs = [...document.querySelectorAll("img")].filter(i => (i.currentSrc || i.src).includes("/api/thumb"));
  if (!imgs.length) return false;
  const r = imgs[0].getBoundingClientRect();
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
});
if (!clicked) { console.log("no thumb found"); process.exit(2); }
await page.mouse.click(clicked.x, clicked.y);

for (let i = 0; i < 16; i++) {
  await new Promise(r => setTimeout(r, 500));
  const state = await page.evaluate(() => {
    return [...document.querySelectorAll("img")].map(im => ({
      src: (im.currentSrc || im.src).slice(0, 90),
      w: im.naturalWidth, h: Math.round(im.getBoundingClientRect().height),
      nw: Math.round(im.getBoundingClientRect().width),
      op: (+getComputedStyle(im).opacity).toFixed(2),
      done: im.complete,
    }));
  });
  const t = Math.round(performance.now() - t0);
  console.log(`t=${t}ms`);
  for (const s of state) console.log(`   [${s.w}x${s.h} box=${s.nw}x${s.h} op=${s.op} done=${s.done}] ${s.src}`);
  const full = state.find(s => s.src.includes("/api/photo") && s.op === "1");
  if (full && i > 2) break;
}
await browser.close();

