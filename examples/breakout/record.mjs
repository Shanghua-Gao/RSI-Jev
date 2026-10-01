// One live Breakout game on the unmodified jev-visual page, played by the model through
// /v1/judge, recorded as a video. Stops when the game is won or lost, at the game's own
// 200-decision budget, or after MAX_S seconds. Nothing is mocked.
//
//   DEMO_URL=http://127.0.0.1:8788 OUT_DIR=out SPEED=slow MAX_S=240 node examples/breakout/record.mjs
//
// Writes OUT_DIR/run.json (summary, every decision without its image, a timeline),
// OUT_DIR/final.png and a .webm video under OUT_DIR/video/.
import { chromium } from 'playwright';
import { writeFile, mkdir } from 'node:fs/promises';
const base = process.env.DEMO_URL || 'http://127.0.0.1:8788';
const speed = process.env.SPEED || 'slow';
const MAX_S = Number(process.env.MAX_S || 240);
const out = process.env.OUT_DIR || 'breakout-out';
await mkdir(`${out}/video`, { recursive: true });
const browser = await chromium.launch({ headless: true });
const size = { width: 1280, height: 1120 };
const context = await browser.newContext({ viewport: size, recordVideo: { dir: `${out}/video`, size } });
const page = await context.newPage();
const errors = [];
page.on('pageerror', e => errors.push(e.message));
try {
  await page.goto(`${base}/demo/breakout/`);
  await page.waitForFunction(() => !!window.breakoutDemo);
  if (speed !== 'slow') await page.selectOption('#speed', speed);
  const t0 = Date.now();
  await page.locator('#ai').click();
  await page.evaluate(() => window.scrollTo(0, 0));
  const timeline = [];
  while (true) {
    const s = await page.evaluate(() => ({ ...window.breakoutDemo.summary(), status: document.querySelector('#status').textContent }));
    timeline.push({ wall_s: (Date.now() - t0) / 1000, ...s });
    if (['won', 'lost'].includes(s.phase) || s.decisions >= 200 || (Date.now() - t0) / 1000 > MAX_S) break;
    await page.waitForTimeout(500);
  }
  await page.waitForTimeout(1500);
  const result = await page.evaluate(() => ({
    summary: window.breakoutDemo.summary(),
    status: document.querySelector('#status').textContent,
    latency: document.querySelector('#latency').textContent,
    records: window.breakoutDemo.records.map(r => { const { input, ...rest } = r; return rest; }),
  }));
  await page.screenshot({ path: `${out}/final.png` });
  Object.assign(result, { timeline, errors, speed, max_s: MAX_S });
  await writeFile(`${out}/run.json`, JSON.stringify(result, null, 1));
  console.log('RUN', JSON.stringify(result.summary), '|', result.status, '|', result.latency);
} finally {
  await context.close();          // flushes the video
  await browser.close();
}
