// Injected through DataMapPlot's existing custom_js parameter.
(async () => {
  const spec = __ZOOMMAP_SPEC__;
  const map = datamap;
  const initialView = { ...map.deckgl.props.initialViewState };

  async function decode(encoded) {
    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Float32Array(await new Response(stream).arrayBuffer());
  }

  const pointReady = new Promise(resolve => {
    if (map.pointLayer) { resolve(); return; }
    function loaded() {
      if (!map.pointLayer) return;
      document.removeEventListener('datamapDataLoaded', loaded);
      resolve();
    }
    document.addEventListener('datamapDataLoaded', loaded);
  });
  const [raw, coefficients] = await Promise.all([
    decode(spec.positions), spec.coefficients ? decode(spec.coefficients) : null, pointReady,
  ]);
  if (window.zoomPositions) throw new Error('A ZoomPositions extension is already installed');
  const base = map.pointLayer.props.data.attributes.getPosition.value;
  const size = spec.count * 2;
  if (base.length !== size || raw.length !== size * spec.zooms.length) {
    throw new Error('ZoomPositions row count does not match the plotted points');
  }

  // Learn the common affine transform from the first raw and rendered layout.
  // This avoids reproducing DataMapPlot's private coordinate-normalization code.
  let rx = 0, ry = 0, bx = 0, by = 0;
  for (let i = 0; i < size; i += 2) {
    rx += raw[i]; ry += raw[i + 1]; bx += base[i]; by += base[i + 1];
  }
  rx /= spec.count; ry /= spec.count; bx /= spec.count; by /= spec.count;
  let numerator = 0, denominator = 0;
  for (let i = 0; i < size; i += 2) {
    numerator += (raw[i] - rx) * (base[i] - bx) + (raw[i + 1] - ry) * (base[i + 1] - by);
    denominator += (raw[i] - rx) ** 2 + (raw[i + 1] - ry) ** 2;
  }
  const scale = numerator / denominator;
  const offset = [bx - scale * rx, by - scale * ry];
  let residual = 0, extent = 0;
  for (let i = 0; i < size; i++) {
    residual = Math.max(residual, Math.abs(raw[i] * scale + offset[i % 2] - base[i]));
    extent = Math.max(extent, Math.abs(base[i]));
  }
  if (!(scale > 0) || !Number.isFinite(scale) || residual > Math.max(1e-4, extent * 1e-5)) {
    throw new Error('The first keyframe must match the coordinates and row order passed to DataMapPlot');
  }
  const coordinates = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) coordinates[i] = raw[i] * scale + offset[i % 2];
  if (coefficients) {
    for (let i = 0; i < coefficients.length; i++) {
      const degree = Math.floor(i / size) % 4;
      coefficients[i] = coefficients[i] * scale + (degree === 3 ? offset[i % 2] : 0);
    }
  }
  const reference = spec.reference === 'initial' ? initialView.zoom : 0;
  let currentView = initialView;
  let frame = null;
  let pendingZoom = initialView.zoom;
  let lastPositionZoom = null;
  let destroyed = false;

  function interval(z) {
    if (z <= spec.zooms[0]) return { index: 0, t: 0, z: spec.zooms[0] };
    const last = spec.zooms.length - 1;
    if (z >= spec.zooms[last]) return { index: last - 1, t: 1, z: spec.zooms[last] };
    let index = 0;
    while (z >= spec.zooms[index + 1]) index++;
    return { index, t: (z - spec.zooms[index]) / (spec.zooms[index + 1] - spec.zooms[index]), z };
  }

  function evaluate(z) {
    const part = interval(z);
    const transition = spec.transitions[part.index];
    let t = part.t;
    if (transition.easing === 'smoothstep') t = t * t * (3 - 2 * t);
    if (transition.easing === 'smootherstep') t = t * t * t * (t * (6 * t - 15) + 10);
    const positions = new Float32Array(size);
    const start = part.index * size;
    // Exact endpoint copies avoid cubic coefficient roundoff at keyframes.
    if (part.t === 0 || part.t === 1) {
      positions.set(coordinates.subarray(start + part.t * size, start + (part.t + 1) * size));
    } else if (transition.interpolation === 'pchip') {
      const u = t * (spec.zooms[part.index + 1] - spec.zooms[part.index]);
      const c = part.index * 4 * size;
      for (let i = 0; i < size; i++) {
        positions[i] = ((coefficients[c + i] * u + coefficients[c + size + i]) * u
          + coefficients[c + 2 * size + i]) * u + coefficients[c + 3 * size + i];
      }
    } else {
      for (let i = 0; i < size; i++) {
        positions[i] = (1 - t) * coordinates[start + i] + t * coordinates[start + size + i];
      }
    }
    return { ...part, positions };
  }

  function apply(absoluteZoom) {
    if (destroyed) return;
    const relativeZoom = absoluteZoom - reference;
    const part = interval(relativeZoom);
    if (part.z !== lastPositionZoom) {
      const { positions } = evaluate(relativeZoom);
      const old = map.pointLayer;
      map.pointLayer = old.clone({ data: {
        ...old.props.data,
        attributes: { ...old.props.data.attributes, getPosition: { value: positions, size: 2 } },
      } });
      for (let i = 0; i < size; i += 2) {
        map.pointData.x[i / 2] = positions[i]; map.pointData.y[i / 2] = positions[i + 1];
      }
      map.layers = map.layers.map(layer => layer.id === old.id ? map.pointLayer : layer);
      map.deckgl.setProps({ layers: [...map.layers] });
      lastPositionZoom = part.z;
    }
    document.dispatchEvent(new CustomEvent('zoomPositionsChanged', { detail: {
      zoom: relativeZoom, absoluteZoom, interval: part.index, progress: part.t,
      left: spec.labels[part.index], right: spec.labels[part.index + 1],
    } }));
  }

  map.onViewStateChange('zoomPositions', ({ viewState }) => {
    currentView = { ...viewState };
    pendingZoom = viewState.zoom;
    // Coalesce multiple input events to one upload per browser frame.
    if (frame === null) frame = requestAnimationFrame(() => { frame = null; apply(pendingZoom); });
  });
  const handle = {
    referenceZoom: reference,
    zooms: [...spec.zooms],
    labels: [...spec.labels],
    transform: { scale, offset },
    positionsAt: z => evaluate(z).positions,
    get zoom() { return currentView.zoom - reference; },
    setZoom(z) {
      if (!Number.isFinite(z)) throw new Error('zoom must be finite');
      currentView = { ...currentView, zoom: z + reference, transitionDuration: 0 };
      map.deckgl.setProps({ initialViewState: currentView });
      map.notifyViewStateChange(currentView);
    },
    destroy() {
      destroyed = true;
      if (frame !== null) cancelAnimationFrame(frame);
      map.offViewStateChange('zoomPositions');
      delete window.zoomPositions;
    },
  };
  window.zoomPositions = handle;
  apply(currentView.zoom);
  document.dispatchEvent(new CustomEvent('zoomPositionsReady', { detail: handle }));
})().catch(error => {
  console.error('ZoomPositions:', error);
  document.dispatchEvent(new CustomEvent('zoomPositionsError', { detail: error.message }));
});
