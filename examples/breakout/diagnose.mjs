// The lane check: ten fixed Breakout scenes, each drawn by the game's own renderer with a
// known ball position, sent to /v1/judge exactly as the game sends a live frame. The
// known positions are used for drawing and scoring only; the model sees the picture.
//
//   DEMO_URL=http://127.0.0.1:8788 OUT_DIR=out node examples/breakout/diagnose.mjs
//
// Same scenes as jev-visual's own demo/breakout/diagnose.mjs. Writes OUT_DIR/diagnose.json.
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';
const base = process.env.DEMO_URL || 'http://127.0.0.1:8788';
const out = process.env.OUT_DIR || 'breakout-out';
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.BROWSER_CHANNEL ? { channel: process.env.BROWSER_CHANNEL } : {}) });
try {
  const page = await browser.newPage();
  await page.goto(`${base}/demo/breakout/`);
  const scenes = await page.evaluate(async () => {
    const { createGame } = await import('./physics.mjs');
    const { drawBoard } = await import('./render.mjs');
    const { instructions, criteria } = await import('./observation.mjs');
    return [[30,480], [160,180], [290,480], [430,180], [610,480], [90,180], [220,480], [345,180], [490,480], [570,180]].map(([ball, paddle], index) => {
      const game = createGame(); game.paddle = paddle; game.ball.x = ball; game.ball.y = index % 2 ? 360 : 200;
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
      drawBoard(canvas, game);
      const q = c => ({ type: 'choice', scoring: 'label', instructions, criteria: c });
      return { id: index, expected: String(Math.floor(ball / 128) + 1), input: {
        image: canvas.toDataURL('image/png'), mode: 'shared',
        questions: { action: q(criteria) },
      } };
    });
  });
  const records = [];
  for (const scene of scenes) {
    const response = await page.request.post(`${base}/v1/judge`, { data: scene.input, timeout: 120000 });
    if (!response.ok()) throw new Error(await response.text());
    const output = await response.json();
    records.push({ ...scene, output });
    console.log(scene.id, 'expected', scene.expected, 'got', output.answers.action.choice);
  }
  const right = records.filter(r => r.output.answers.action.choice === r.expected).length;
  await writeFile(`${out}/diagnose.json`, JSON.stringify(records, null, 2));
  console.log(`lane check: ${right}/${records.length} -> ${out}/diagnose.json`);
} finally { await browser.close(); }
