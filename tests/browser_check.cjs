// Run after executing general_example.ipynb and serving artifacts/ on port 8788.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
process.env.PLAYWRIGHT_BROWSERS_PATH ||= path.join(root, '.browser-cache');
const { chromium } = require('playwright');
const baseUrl = process.env.ZOOMMAP_BASE_URL || 'http://127.0.0.1:8788';
const chrome = process.env.CHROME_EXECUTABLE_PATH;
const names = ['fashion_mnist', 'mnist', '20_newsgroups'];
const titles = ['Fashion-MNIST', 'MNIST', '20 Newsgroups · MPNet'];
const neSpectrum = process.env.ZOOMMAP_TECHNIQUE === 'ne-spectrum';
const slogan = neSpectrum ?
  'Zoom from global neighbor structure through NE-spectrum toward local t-SNE structure.' :
  'Zoom from a global PCA layout through DREAMS toward local t-SNE structure.';

async function checkQuality(context, zoom) {
  const result = await context.evaluate(z => {
    const grid = zoomQuality.samples;
    const clamped = Math.max(grid[0].zoom, Math.min(grid.at(-1).zoom, z));
    let right = grid.findIndex(sample => sample.zoom >= clamped);
    if (right === 0) right = 1;
    const a = grid[right - 1], b = grid[right];
    const t = (clamped - a.zoom) / (b.zoom - a.zoom);
    const expected = {
      nn: a.nn_recall * (1 - t) + b.nn_recall * t,
      spearman: a.global_distance_spearman * (1 - t) + b.global_distance_spearman * t,
    };
    const row = document.getElementById('zoom-quality-metrics');
    return {
      expected, actual: { nn: Number(row.dataset.nnRecall), spearman: Number(row.dataset.distanceSpearman) },
      zoom: Number(row.dataset.zoom), clamped,
      metadata: zoomQuality.metadata, samples: grid.length,
      recallText: document.getElementById('zoom-nn-recall').value,
      spearmanText: document.getElementById('zoom-distance-spearman').value,
    };
  }, zoom);
  assert.equal(result.metadata.k, 10);
  assert.equal(result.metadata.distance_subset_size, 1000);
  assert.equal(result.metadata.point_count, 5000);
  assert(result.samples >= 101);
  assert(Math.abs(result.zoom - result.clamped) < 1e-8);
  assert(Math.abs(result.actual.nn - result.expected.nn) < 1e-8);
  assert(Math.abs(result.actual.spearman - result.expected.spearman) < 1e-8);
  assert.equal(result.recallText, '≈ ' + (result.expected.nn * 100).toFixed(1) + '%');
  assert.equal(result.spearmanText, '≈ ' + result.expected.spearman.toFixed(3));
}

async function waitForLayout(context, zoom) {
  await context.waitForFunction(z => {
    const actual = datamap.pointLayer.props.data.attributes.getPosition.value;
    const expected = zoomPositions.positionsAt(z);
    return Math.abs(zoomPositions.zoom - z) < 1e-8 &&
      actual.every((value, index) => Math.abs(value - expected[index]) < 3e-4);
  }, zoom, { timeout: 15000, polling: 100 });
}

(async () => {
  const browser = await chromium.launch({
    executablePath: chrome, headless: true,
    args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const name of names) {
      await page.goto(baseUrl + '/' + name + '_zoom.html', { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => Boolean(window.zoomPositions && datamap.metaData),
        null, { timeout: 60000, polling: 100 });
      const initial = await page.evaluate(() => {
        window.initialAttributes = datamap.pointLayer.props.data.attributes;
        return {
          positions: Array.from(initialAttributes.getPosition.value),
          count: datamap.pointLayer.props.data.length,
        };
      });
      assert.equal(initial.count, 5000);
      const title = 'From global to local · ' + titles[names.indexOf(name)];
      assert.equal(await page.title(), title);
      assert.equal(await page.locator('#dataset-heading h1').innerText(), title);
      assert.equal(await page.locator('.dataset-eyebrow').count(), 0);
      assert.equal(await page.locator('#dataset-heading p').innerText(),
        slogan);
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).backgroundColor),
        'rgb(239, 240, 235)');
      for (const zoom of await page.evaluate(() => zoomPositions.zooms)) {
        await page.evaluate(z => zoomPositions.setZoom(z), zoom);
        await waitForLayout(page, zoom);
        await checkQuality(page, zoom);
        assert(await page.evaluate(() => {
          const p = datamap.pointLayer.props.data.attributes.getPosition.value;
          const viewport = datamap.deckgl.getViewports()[0];
          for (let i = 0; i < p.length; i += 2) {
            const pixel = viewport.project([p[i], p[i + 1]]);
            if (!pixel.every(Number.isFinite)) return false;
          }
          return true;
        }), 'Every point must remain projectable at each keyframe');
        assert(await page.evaluate(() => {
          const attributes = datamap.pointLayer.props.data.attributes;
          return attributes.getFillColor === initialAttributes.getFillColor &&
            attributes.getFilterValue === initialAttributes.getFilterValue;
        }));
      }
      await page.evaluate(() => zoomPositions.setZoom(0));
      await waitForLayout(page, 0);
      await checkQuality(page, 0);
      assert(await page.evaluate(expected => {
        const actual = datamap.pointLayer.props.data.attributes.getPosition.value;
        return actual.every((value, index) => Math.abs(value - expected[index]) < 3e-4);
      }, initial.positions));
      // A real slider input updates both geometry and the sampled quality readout.
      await page.locator('#zoom-layout-slider').evaluate(slider => {
        slider.value = '1.37'; slider.dispatchEvent(new Event('input', { bubbles: true }));
      });
      await waitForLayout(page, 1.37);
      await checkQuality(page, 1.37);
      await page.evaluate(() => zoomPositions.setZoom(0));
      await waitForLayout(page, 0);
      await page.mouse.move(640, 450);
      await page.mouse.wheel(0, -150);
      await page.waitForFunction(() => zoomPositions.zoom > .005, null, { polling: 100 });
      // Wait until the native smooth-wheel gesture stops before scripted picking.
      await page.waitForFunction(() => {
        const z = zoomPositions.zoom, now = performance.now();
        if (!window.lastTestZoom || Math.abs(lastTestZoom.z - z) > 1e-6) {
          window.lastTestZoom = { z, changed: now };
          return false;
        }
        return now - lastTestZoom.changed > 200;
      }, null, { timeout: 10000, polling: 50 });
      await page.evaluate(() => zoomPositions.setZoom(.7));
      await waitForLayout(page, .7);
      await page.evaluate(() => new Promise(resolve =>
        requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const point = await page.evaluate(() => {
        const p = datamap.pointLayer.props.data.attributes.getPosition.value;
        const viewport = datamap.deckgl.getViewports()[0];
        for (let i = 0; i < p.length; i += 2) {
          const xy = viewport.project([p[i], p[i + 1]]);
          const x = Math.round(xy[0]), y = Math.round(xy[1]);
          if (x < 300 || x > 950 || y < 260 || y > 650) continue;
          const picked = datamap.deckgl.pickObject({ x, y, radius: 0, layerIds: ['dataPointLayer'] });
          if (picked?.index >= 0) return { x, y, id: datamap.metaData.sample_id[picked.index] };
        }
        return null;
      });
      assert(point, 'No visible point to inspect in ' + name);
      await page.mouse.move(point.x, point.y);
      const prefix = name === '20_newsgroups' ? 'document ' : 'sample ';
      await page.waitForFunction(text => document.querySelector('.deck-tooltip')?.textContent.includes(text),
        prefix + point.id, { timeout: 10000, polling: 100 });
      const tooltip = page.locator('.deck-tooltip');
      assert.equal(await tooltip.locator('img').count(), name === '20_newsgroups' ? 0 : 1);
      if (name === '20_newsgroups') {
        assert((await tooltip.locator('[style*="white-space:pre-wrap"]').innerText()).length > 0);
      }
      await page.getByRole('button', { name: 'Animate', exact: true }).click();
      await page.getByRole('button', { name: 'Stop', exact: true }).click();
      console.log('PASS ' + name + ': 5,000 points, six keyframes, preserved attributes, reverse, wheel, hover, animation and dynamic metrics');
    }
    const notebook = JSON.parse(fs.readFileSync(path.join(root,
      neSpectrum ? 'ne_spectrum/general_example.ipynb' : 'notebooks/general_example.ipynb')));
    const embeds = notebook.cells.flatMap(cell => (cell.outputs || [])
      .map(output => output.data?.['text/html'])
      .map(html => Array.isArray(html) ? html.join('') : html)
      .filter(html => html?.includes('<iframe')));
    await page.goto('about:blank');
    await page.setContent(embeds.join('\n'), { waitUntil: 'domcontentloaded' });
    assert.equal(await page.locator('iframe').count(), names.length);
    for (let i = 0; i < names.length; i++) {
      const element = page.locator('iframe').nth(i);
      await element.scrollIntoViewIfNeeded();
      const frame = await (await element.elementHandle()).contentFrame();
      await frame.waitForFunction(() => Boolean(window.zoomPositions), null, { timeout: 60000, polling: 100 });
      await frame.evaluate(() => zoomPositions.setZoom(1.25));
      await waitForLayout(frame, 1.25);
      assert((await frame.locator('#zoom-layout-status').innerText()).includes(neSpectrum ? 'ρ=2' : 'λ=0.15'));
      await checkQuality(frame, 1.25);
      assert.equal(await frame.locator('#dataset-heading h1').innerText(),
        'From global to local · ' + titles[i]);
      console.log('PASS notebook iframe: ' + names[i]);
    }
    assert.deepEqual(errors, []);
    console.log('PASS all three notebook displays; no JavaScript page errors');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
