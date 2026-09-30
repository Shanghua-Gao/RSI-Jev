// Render an animated SVG to PNG frames by seeking its SMIL clock, then a static last frame.
import { chromium } from 'playwright';
import { readFileSync, mkdirSync } from 'node:fs';
const [,, svgPath, outDir, fps = '10', dur = '16'] = process.argv;
mkdirSync(outDir, { recursive: true });
const svg = readFileSync(svgPath, 'utf8');
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 792 }, deviceScaleFactor: 1 });
await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
const n = Math.round(+fps * +dur);
for (let i = 0; i <= n; i++) {
  const t = Math.min(i / +fps, +dur - 0.01);
  await page.evaluate((t) => { const s = document.querySelector('svg'); s.pauseAnimations(); s.setCurrentTime(t); }, t);
  await page.screenshot({ path: `${outDir}/f${String(i).padStart(4, '0')}.png`, clip: { x: 0, y: 0, width: 1440, height: 792 } });
}
await browser.close();
console.log('frames', n + 1);
