// Demo-only metric display. The position helper and DataMapPlot API are unchanged.
(() => {
  const specification = __ZOOM_QUALITY_SPEC__;
  const samples = specification.samples;
  const panel = document.getElementById('zoom-layout-controls');
  const row = document.createElement('div');
  row.id = 'zoom-quality-metrics';
  row.style.cssText = 'display:flex;justify-content:space-between;gap:16px;margin:0 0 12px;' +
    'padding:10px 0;border-top:1px solid #e1e5ed;border-bottom:1px solid #e1e5ed;font-size:11px';
  row.title = `Quality of all ${specification.metadata.point_count.toLocaleString()} points. ` +
    `Spearman uses ${specification.metadata.distance_subset_size.toLocaleString()} fixed sampled observations. ` +
    'Evaluated along the interpolated layout path; approximate between sampled zooms.';
  row.innerHTML = '<span>NN recall (k=' + specification.metadata.k +
    ') <output id="zoom-nn-recall" style="font-variant-numeric:tabular-nums;font-weight:600"></output></span>' +
    '<span>Global distance Spearman <output id="zoom-distance-spearman" ' +
    'style="font-variant-numeric:tabular-nums;font-weight:600"></output></span>';
  panel.insertBefore(row, panel.querySelector('.dataset-control-footer'));
  const recall = row.querySelector('#zoom-nn-recall');
  const spearman = row.querySelector('#zoom-distance-spearman');
  function update(zoom) {
    const z = Math.max(samples[0].zoom, Math.min(samples.at(-1).zoom, zoom));
    let i = 0;
    while (i < samples.length - 2 && samples[i + 1].zoom < z) i++;
    const left = samples[i], right = samples[i + 1];
    const t = (z - left.zoom) / (right.zoom - left.zoom);
    const nn = left.nn_recall * (1 - t) + right.nn_recall * t;
    const rho = left.global_distance_spearman * (1 - t) + right.global_distance_spearman * t;
    recall.value = '≈ ' + (nn * 100).toFixed(1) + '%';
    spearman.value = '≈ ' + rho.toFixed(3);
    row.dataset.zoom = String(z);
    row.dataset.nnRecall = String(nn);
    row.dataset.distanceSpearman = String(rho);
  }
  document.addEventListener('zoomPositionsChanged', ({ detail }) => update(detail.zoom));
  document.addEventListener('zoomPositionsReady', ({ detail }) => update(detail.zoom));
  update(window.zoomPositions ? window.zoomPositions.zoom : samples[0].zoom);
  // Make the sampling policy inspectable in exported maps.
  window.zoomQuality = specification;
})();
