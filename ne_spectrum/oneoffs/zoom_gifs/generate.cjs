// One-off captures of the NE-spectrum maps; the original DREAMS GIFs are retained.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const root = path.resolve(__dirname, '../../..');
if (!process.env.PLAYWRIGHT_BROWSERS_PATH && fs.existsSync(path.join(root, '.browser-cache'))) {
  process.env.PLAYWRIGHT_BROWSERS_PATH = path.join(root, '.browser-cache');
}
const { chromium } = require('playwright');
const names = ['fashion_mnist', 'mnist', '20_newsgroups'];
const size = 480;
const fps = 20;
const seconds = 12;
const count = fps * seconds;
const ffmpeg = process.env.FFMPEG_PATH || 'ffmpeg';
const preview = process.argv.includes('--preview');

async function main() {
  execFileSync(ffmpeg, ['-version'], { stdio: 'ignore' });
  const htmls = new Map(names.map(name => [
    '/' + name + '.html', fs.readFileSync(path.join(root, 'ne_spectrum/artifacts', name + '_zoom.html')),
  ]));
  const server = http.createServer((request, response) => {
    const html = htmls.get(request.url);
    response.writeHead(html ? 200 : 404, { 'Content-Type': 'text/html; charset=utf-8' });
    response.end(html || 'Not found');
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'zoommap-gifs-'));
  console.log('Temporary frames: ' + temporary);
  let browser;
  let completed = false;
  try {
    browser = await chromium.launch({
      headless: true,
      args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
    });
    const page = await browser.newPage({
      viewport: { width: size, height: size }, deviceScaleFactor: 1,
    });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const name of names) {
      const frames = path.join(temporary, name);
      fs.mkdirSync(frames);
      await page.goto(`http://127.0.0.1:${server.address().port}/${name}.html`, {
        waitUntil: 'domcontentloaded',
      });
      await page.waitForFunction(() => Boolean(window.zoomPositions), null, { timeout: 60000 });
      await page.addStyleTag({ content: `
        html, body { background: #eff0eb !important; overflow: hidden !important; }
        #dataset-heading, #dataset-legend, #zoom-layout-controls,
        .content-wrapper, .deck-tooltip { display: none !important; }
        #deck-container { z-index: 0 !important; background: #eff0eb !important; }
      ` });
      await page.evaluate(() => {
        datamap.layers = [datamap.pointLayer];
        datamap.deckgl.setProps({ layers: datamap.layers });
      });
      assert.equal(await page.locator('body').innerText(), '');
      assert.equal(await page.evaluate(() => datamap.pointLayer.props.data.length), 5000);
      const [first, last] = await page.evaluate(() => [zoomPositions.zooms[0], zoomPositions.zooms.at(-1)]);
      const focusCount = await page.evaluate(async ({ first, last, size }) => {
        zoomPositions.setZoom(first);
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const originalView = { ...datamap.deckgl.props.initialViewState };
        const projection = datamap.deckgl.getViewports()[0];
        zoomPositions.setZoom(last);
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const viewport = datamap.deckgl.getViewports()[0];
        const positions = zoomPositions.positionsAt(last);
        const pixels = Array.from({ length: positions.length / 2 }, (_, i) =>
          viewport.project([positions[2 * i], positions[2 * i + 1]]));
        // Find the square with the most points at maximum zoom. Leave a 10%
        // margin on each edge, so the focus is a neighborhood rather than a dot.
        const half = size * .4;
        let best = 0, maximum = 0;
        for (let i = 0; i < pixels.length; i++) {
          let count = 0;
          for (const pixel of pixels) {
            if (Math.abs(pixel[0] - pixels[i][0]) < half &&
                Math.abs(pixel[1] - pixels[i][1]) < half) count++;
          }
          if (count > maximum) { best = i; maximum = count; }
        }
        const indices = pixels.flatMap((pixel, i) =>
          Math.abs(pixel[0] - pixels[best][0]) < half &&
          Math.abs(pixel[1] - pixels[best][1]) < half ? [i] : []);
        const mean = [0, 0];
        for (const i of indices) {
          const pixel = projection.project([positions[2 * i], positions[2 * i + 1]]);
          mean[0] += pixel[0] / indices.length;
          mean[1] += pixel[1] / indices.length;
        }
        const center = projection.project([positions[2 * best], positions[2 * best + 1]]);
        window.captureFocus = {
          originalView, projection, indices,
          offset: [center[0] - mean[0], center[1] - mean[1]],
        };
        window.captureCanvas = document.createElement('canvas');
        captureCanvas.width = size;
        captureCanvas.height = size;
        return indices.length;
      }, { first, last, size });
      assert(focusCount > 50, 'The selected local neighborhood is too sparse');
      console.log(`Dense local focus: ${focusCount} points inside the central 80% of the frame`);
      console.log(`Capturing ${name}: ${count} frames, zoom +${first} → +${last} → +${first}`);
      const frameIndices = preview ? [0, 60, 120, 180, count - 1] : Array.from({ length: count }, (_, i) => i);
      for (const index of frameIndices) {
        const t = (1 - Math.cos(2 * Math.PI * index / (count - 1))) / 2;
        const zoom = first + t * (last - first);
        const frame = await page.evaluate(async ({ z, first, last, size }) => {
          const { originalView, projection, indices, offset } = captureFocus;
          const expected = zoomPositions.positionsAt(z);
          const mean = [...offset];
          for (const i of indices) {
            const pixel = projection.project([expected[2 * i], expected[2 * i + 1]]);
            mean[0] += pixel[0] / indices.length;
            mean[1] += pixel[1] / indices.length;
          }
          const center = projection.unproject(mean);
          const progress = (z - first) / (last - first);
          const blend = progress * progress * (3 - 2 * progress);
          const view = {
            ...originalView, zoom: zoomPositions.referenceZoom + z,
            longitude: originalView.longitude * (1 - blend) + center[0] * blend,
            latitude: originalView.latitude * (1 - blend) + center[1] * blend,
            transitionDuration: 0,
          };
          datamap.deckgl.setProps({ initialViewState: view });
          datamap.notifyViewStateChange(view);
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const actual = datamap.pointLayer.props.data.attributes.getPosition.value;
          datamap.deckgl.redraw(true);
          // Read the map canvas directly, avoiding DOM overlays and screenshot
          // compositor timing. Composite its transparent pixels over our background.
          const context = captureCanvas.getContext('2d');
          context.fillStyle = '#eff0eb';
          context.fillRect(0, 0, size, size);
          context.drawImage(datamap.deckgl.getCanvas(), 0, 0, size, size);
          const pixels = context.getImageData(0, 0, size, size).data;
          let foreground = 0;
          for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i] !== 239 || pixels[i + 1] !== 240 || pixels[i + 2] !== 235) foreground++;
          }
          const viewport = datamap.deckgl.getViewports()[0];
          let visible = 0;
          for (let i = 0; i < actual.length; i += 2) {
            const pixel = viewport.project([actual[i], actual[i + 1]]);
            if (pixel[0] >= 0 && pixel[0] < size && pixel[1] >= 0 && pixel[1] < size) visible++;
          }
          return {
            matches: actual.every((value, i) => Math.abs(value - expected[i]) < 3e-4),
            png: captureCanvas.toDataURL('image/png').split(',')[1], foreground, visible,
          };
        }, { z: zoom, first, last, size });
        assert(frame.matches, 'A captured frame did not match its requested layout');
        assert(frame.foreground > 1000, 'The capture did not contain a populated map');
        if (index === 120) {
          assert(frame.visible >= focusCount, 'The close-up missed its dense neighborhood');
          console.log(`  close-up: ${frame.visible} visible points`);
        }
        fs.writeFileSync(path.join(frames, String(index).padStart(4, '0') + '.png'), Buffer.from(frame.png, 'base64'));
        if (index % 60 === 0) console.log(`  ${name}: ${index}/${count}`);
      }
      assert.deepEqual(errors, []);
      if (preview) continue;
      const gif = path.join(temporary, name + '.gif');
      execFileSync(ffmpeg, [
        '-hide_banner', '-loglevel', 'error', '-y',
        '-framerate', String(fps), '-i', path.join(frames, '%04d.png'),
        '-filter_complex',
        '[0:v]split[a][b];[a]palettegen=reserve_transparent=0[p];[b][p]paletteuse=dither=none',
        '-loop', '0', gif,
      ], { stdio: 'inherit' });
      const output = path.join(__dirname, name + '.gif');
      fs.copyFileSync(gif, output);
      console.log(`Saved ${output} (${(fs.statSync(output).size / 1024 ** 2).toFixed(2)} MiB)`);
    }
    completed = true;
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
    if (completed && !preview) fs.rmSync(temporary, { recursive: true, force: true });
    else console.log('Retained preview/debug frames: ' + temporary);
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
