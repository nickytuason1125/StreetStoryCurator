/**
 * ux_bench.mjs — real-browser UX benchmark (Part 2 of the UX suite).
 *
 * Drives the ACTUAL served UI (http://127.0.0.1:8000) in headless Edge via
 * puppeteer-core and measures what a human feels:
 *   - cold load: navigation timings + first-contentful-paint
 *   - grid populated: time until >= 20 thumbnails exist in the DOM
 *   - scroll smoothness: rAF-sampled FPS over 6 s (idle vs scrolling)
 *   - interaction: click first thumbnail -> loupe rendered
 *   - lightness: JS heap after the scroll stress
 *   - quality: screenshots (also visually confirm branding)
 *
 * Writes reports/ux_bench.json + reports/ux_bench.md + reports/ux_*.png.
 * Requires the backend to be running. Usage:
 *   node scripts/ux_bench.mjs [--base http://127.0.0.1:8000]
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", ".."); // repo root (reports/ lives there)
const REPORTS = path.join(ROOT, "reports");
fs.mkdirSync(REPORTS, { recursive: true });

const argBase = (() => {
  const i = process.argv.indexOf("--base");
  return i > 0 ? process.argv[i + 1] : "http://127.0.0.1:8000";
})();

const EDGE_PATHS = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
];
const edge = EDGE_PATHS.find((p) => fs.existsSync(p));
if (!edge) {
  console.error("Edge not found — install Edge or point EDGE_PATHS at a Chromium.");
  process.exit(1);
}

const R = { base: argBase, generated: new Date().toISOString(), checks: {} };
const ms = (t0) => Math.round(performance.now() - t0);

const browser = await puppeteer.launch({
  executablePath: edge,
  headless: true,
  args: ["--window-size=1680,950", "--disable-extensions", "--no-first-run"],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900 });

  // cold load
  const t0 = performance.now();
  await page.goto(argBase + "/", { waitUntil: "domcontentloaded", timeout: 120000 });
  R.checks.domContentLoaded = ms(t0);
  R.checks.fcp = await page.evaluate(() => {
    const e = performance.getEntriesByName("first-contentful-paint");
    return e.length ? Math.round(e[0].startTime) : null;
  });
  await page.screenshot({ path: path.join(REPORTS, "ux_home.png") });

  // grid populated — the welcome stage gates the gallery; click Resume
  await page.evaluate(() => {
    const b = [...document.querySelectorAll("button, a")].find((x) =>
      /^resume$/i.test((x.textContent || "").trim()),
    );
    if (b) b.click();
  });
  const t1 = performance.now();
  try {
    await page.waitForFunction(
      () => document.querySelectorAll("img").length >= 20,
      { timeout: 120000, polling: 100 },
    );
    R.checks.gridPopulated = ms(t1);
  } catch {
    R.checks.gridPopulated = null;
  }
  R.checks.thumbCount = await page.evaluate(
    () => document.querySelectorAll("img").length,
  );
  await page.screenshot({ path: path.join(REPORTS, "ux_home.png") });

  // scroll smoothness: rAF-sampled 6 s idle vs 6 s active scrolling
  R.checks.scroll = await page.evaluate(async () => {
    const sampleFps = (active) =>
      new Promise((res) => {
        let n = 0;
        const t0 = performance.now();
        const tick = () => {
          n += 1;
          if (active) {
            const el = document.scrollingElement;
            el.scrollTop += 400;
          }
          if (performance.now() - t0 >= 6000) res(n);
          else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    const idleFps = Math.round(((await sampleFps(false)) / 6) * 10) / 10;
    const scrollFps = Math.round(((await sampleFps(true)) / 6) * 10) / 10;
    return { idleFps, scrollFps };
  });

  R.checks.heapAfterScrollMB = await page.evaluate(() =>
    performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null,
  );
  await page.screenshot({ path: path.join(REPORTS, "ux_scrolled.png") });

  // interaction: click first thumbnail, wait for a loupe-sized image
  try {
    const img = await page.$("img");
    if (img) {
      const box = await img.boundingBox();
      if (box) {
        const t2 = performance.now();
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
        await page
          .waitForFunction(
            () => [...document.querySelectorAll("img")]
              .some((i) => i.getBoundingClientRect().height > 400),
            { timeout: 15000, polling: 50 },
          )
          .catch(() => {});
        R.checks.interactionMs = ms(t2);
        await page.screenshot({ path: path.join(REPORTS, "ux_selected.png") });
      }
    }
  } catch (e) {
    R.checks.interactionError = String(e).slice(0, 160);
  }

  R.checks.heapFinalMB = await page.evaluate(() =>
    performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : null,
  );
} finally {
  await browser.close();
}

// ── write reports ──────────────────────────────────────────────────────────
const c = R.checks;
const md = [
  "# UX Benchmark (real browser, headless Edge)",
  "",
  `Base: \`${R.base}\` · Generated: ${R.generated}`,
  "",
  "| Check | Value |",
  "|---|---|",
  `| DOM content loaded | ${c.domContentLoaded} ms |`,
  `| First contentful paint | ${c.fcp} ms |`,
  `| Grid populated (20+ thumbs) | ${c.gridPopulated ?? "timeout"} ms |`,
  `| Thumbnails in DOM | ${c.thumbCount} |`,
  `| Idle FPS (6 s sample) | ${c.scroll?.idleFps} |`,
  `| Scroll FPS (6 s sample) | ${c.scroll?.scrollFps} |`,
  `| Click -> loupe rendered | ${c.interactionMs ?? "n/a"} ms |`,
  `| Heap after scroll | ${c.heapAfterScrollMB} MB |`,
  `| Heap final | ${c.heapFinalMB} MB |`,
  "",
  "Screenshots: ux_home.png, ux_scrolled.png, ux_selected.png",
].join("\n");
fs.writeFileSync(path.join(REPORTS, "ux_bench.json"), JSON.stringify(R, null, 2));
fs.writeFileSync(path.join(REPORTS, "ux_bench.md"), md + "\n");
console.log(md);

if (c.gridPopulated == null) process.exit(2);