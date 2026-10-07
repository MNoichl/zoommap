NE-spectrum zoom captures
========================

Three text-free, 480x480 GIFs capture the exported NE-spectrum maps: 12 seconds,
20 frames per second, seamless infinite loops, and the #eff0eb background.

* `Fashion-MNIST <https://mnoichl.github.io/zoommap/ne_spectrum/oneoffs/zoom_gifs/fashion_mnist.gif>`_
* `MNIST <https://mnoichl.github.io/zoommap/ne_spectrum/oneoffs/zoom_gifs/mnist.gif>`_
* `20 Newsgroups / MPNet <https://mnoichl.github.io/zoommap/ne_spectrum/oneoffs/zoom_gifs/20_newsgroups.gif>`_

Run from zoommap/ after executing ne_spectrum/general_example.ipynb::

    node ne_spectrum/oneoffs/zoom_gifs/generate.cjs

Requires the root Playwright setup and FFmpeg. ``--preview`` captures five PNG
frames per map in a temporary directory without writing GIFs.

The camera follows the densest square neighborhood of the final local layout,
tracking its same point IDs throughout the morph. The captured canvas contains
only points. Its direct canvas readback avoids DOM overlays and checks that each
frame matches its requested interpolated layout, contains populated pixels, and
shows the selected dense neighborhood at maximum zoom. Cosine timing drives the
zoom in and out; spatial interpolation remains the map's linear interpolation.

The original DREAMS captures and generator remain in oneoffs/zoom_gifs/.
