# Square map GIFs

One-off, text-free captures of the three final interactive maps. Each GIF is
480 × 480 pixels, 20 fps, and loops over 12 seconds from global to local and back.
The captures retain the maps' point colors, sizing, `#eff0eb` background, and
linear layout interpolation. Cosine timing makes the camera slow down at the
ends of the zoom range (offsets 0 to 2.5).

Each close-up centers on a dense square neighborhood in the final local
layout, with a 10% margin on each edge. The camera follows that same group of
observations as their layout changes and returns to the original overview.

- [Fashion-MNIST](fashion_mnist.gif) · [Live GIF](https://mnoichl.github.io/zoommap/oneoffs/zoom_gifs/fashion_mnist.gif)
- [MNIST](mnist.gif) · [Live GIF](https://mnoichl.github.io/zoommap/oneoffs/zoom_gifs/mnist.gif)
- [20 Newsgroups · MPNet](20_newsgroups.gif) · [Live GIF](https://mnoichl.github.io/zoommap/oneoffs/zoom_gifs/20_newsgroups.gif)

The generator reads the existing HTML exports and captures their point layer
directly from the map canvas, without text or controls. It writes PNG frames to
a temporary folder that is removed after encoding. The notebook, HTML maps, and
library stay unchanged.

To regenerate, install FFmpeg and run from the `zoommap/` root:

```sh
npm install
npx playwright install chromium
node oneoffs/zoom_gifs/generate.cjs
```

Pass `--preview` to render five checkpoint PNGs per dataset into a temporary
folder for inspection, without encoding or replacing the GIFs.

CDN access is needed to load the maps' JavaScript dependencies. `FFMPEG_PATH`
can specify an FFmpeg executable outside `PATH`.
