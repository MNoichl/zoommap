// Optional notebook/demo UI. It subscribes to the helper's public events.
(() => {
  const slider = document.getElementById('zoom-layout-slider');
  const status = document.getElementById('zoom-layout-status');
  const button = document.getElementById('zoom-layout-play');
  let animation = null;
  function stop() {
    if (animation !== null) cancelAnimationFrame(animation);
    animation = null;
    button.textContent = 'Animate';
  }
  document.addEventListener('zoomPositionsReady', ({ detail: motion }) => {
    slider.min = motion.zooms[0]; slider.max = motion.zooms.at(-1);
    slider.value = motion.zoom; slider.disabled = false; button.disabled = false;
    const stops = document.getElementById('zoom-layout-stops');
    for (let i = 0; i < motion.zooms.length; i++) {
      const option = document.createElement('option');
      option.value = motion.zooms[i]; option.label = motion.labels[i]; stops.appendChild(option);
    }
    slider.addEventListener('input', () => { stop(); motion.setZoom(Number(slider.value)); });
    document.getElementById('deck-container').addEventListener('wheel', stop, { passive: true });
    document.getElementById('deck-container').addEventListener('pointerdown', stop);
    button.addEventListener('click', () => {
      if (animation !== null) { stop(); return; }
      const started = performance.now();
      button.textContent = 'Stop';
      function tick(now) {
        const elapsed = Math.min(1, (now - started) / 16000);
        const t = (1 - Math.cos(elapsed * Math.PI * 2)) / 2;
        motion.setZoom(motion.zooms[0] + t * (motion.zooms.at(-1) - motion.zooms[0]));
        if (elapsed < 1) animation = requestAnimationFrame(tick);
        else stop();
      }
      animation = requestAnimationFrame(tick);
    });
  });
  document.addEventListener('zoomPositionsChanged', ({ detail }) => {
    slider.value = detail.zoom;
    const layout = detail.progress === 0 ? detail.left : detail.progress === 1 ? detail.right :
      `${detail.left} → ${detail.right} · ${Math.round(detail.progress * 100)}%`;
    status.textContent = `Zoom +${detail.zoom.toFixed(2)} · ${layout}`;
  });
  document.addEventListener('zoomPositionsError', ({ detail }) => { status.textContent = detail; });
})();
