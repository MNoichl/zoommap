From global to local
====================

``zoommap`` moves points between ordered layouts as the viewer zooms. It uses
DataMapPlot's existing ``custom_js`` parameter; the DataMapPlot source and public
plotting API stay unchanged.

``notebooks/general_example.ipynb`` produces four final interactive maps:

* ``artifacts/fashion_mnist_zoom.html`` — Fashion-MNIST, PCA-50 pixel features.
* ``artifacts/mnist_zoom.html`` — MNIST digits, PCA-50 pixel features.
* ``artifacts/20_newsgroups_zoom.html`` — 20 Newsgroups, MPNet sentence embeddings.
* ``artifacts/mammoth_zoom.html`` — Mammoth, original XYZ point-cloud distances.

Each dataset uses 5,000 observations (balanced image/topic samples, uniformly
sampled Mammoth points), perplexity 150, and six
layouts at lambda = 1, 0.5, 0.25, 0.15, 0.05, 0. The PCA endpoint is analytic;
intermediate layouts use the authors' DREAMS fork, and the local endpoint is t-SNE.
Each optimized layout initializes the next while the PCA regularization anchor
stays fixed. Original row IDs, colors, and hover metadata follow the same points.

Setup and execution
-------------------

Clone this repository and run from its root::

    python3.12 -m venv .venv
    .venv/bin/python -m pip install -r requirements-notebook.txt
    .venv/bin/python -m ipykernel install --prefix .venv --name zoommap --display-name "ZoomMap research"
    .venv/bin/python -u notebooks/execute.py

Or open ``notebooks/general_example.ipynb`` in Jupyter with the **ZoomMap research**
kernel. The requirements install unmodified DataMapPlot and DREAMS directly from
pinned Git revisions; no sibling checkouts are required. DREAMS compiles native
extensions, so a C/C++ compiler is required (Xcode Command Line Tools on macOS).

The first run downloads the training datasets and MPNet weights, encodes the text,
and fits the layouts. Later runs reuse caches under ``data/``. Dataset downloads,
model weights, optimizer caches, virtual environments and browser tools are local
and ignored by Git. The four final maps, coordinate archives, preprocessing
settings and quality reports are kept in ``artifacts/`` and versioned.

The exported HTML maps need no Python or notebook server to view. Like normal
DataMapPlot exports, they load JavaScript dependencies from CDNs.

GitHub Pages
------------

The root ``index.html`` links to all four datasets with DREAMS at 150 and 30
and NE-spectrum at 30. Enable Pages under repository Settings → Pages, choose **Deploy from a branch**, then select
``main`` and ``/(root)``. The root ``.nojekyll`` file serves the generated HTML
directly, without Jekyll processing. No build or notebook execution is needed.

Once Pages is enabled, the landing page is https://mnoichl.github.io/zoommap/:

* `Fashion-MNIST <https://mnoichl.github.io/zoommap/artifacts/fashion_mnist_zoom.html>`_
* `MNIST <https://mnoichl.github.io/zoommap/artifacts/mnist_zoom.html>`_
* `20 Newsgroups · MPNet <https://mnoichl.github.io/zoommap/artifacts/20_newsgroups_zoom.html>`_
* `Mammoth <https://mnoichl.github.io/zoommap/artifacts/mammoth_zoom.html>`_

Rerunning the notebook replaces the same HTML files; pushing the updated maps
to ``main`` updates their existing Pages links.

Reproduction settings
---------------------

Mammoth uses the original ``mammoth_a.csv`` from
`Noichl's dataset repository <https://github.com/MNoichl/UMAP-examples-mammoth>`_,
pinned to revision ``d98f5ba768e88d51662406f240e0c47e15c10bb7``. A seeded,
uniform sample without replacement retains original CSV row IDs. A full PCA
rotation keeps all three XYZ dimensions, without whitening or per-axis scaling,
so original Euclidean distances are preserved. The same points and features are
used in all three variants. Twelve Ward spatial regions provide display colors
only; they are not anatomical labels or optimizer inputs. For samples larger
than 5,000, regions are fitted to a seeded 5,000-point reference and extended by
10-neighbor voting. Hover shows original XYZ coordinates, and each notebook also
shows the original 3D cloud. Provenance records the CSV checksum and source.
Dataset citation: `Noichl (2025) <https://doi.org/10.5281/zenodo.17290165>`_.
Original scan: `Smithsonian Institution, Mammuthus primigenius
<https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12>`_.

Images use grayscale pixels divided by 255 and seeded PCA-50. 20 Newsgroups uses
the training split with headers, footers and quotes removed; empty cleaned texts
are excluded before balanced sampling. ``sentence-transformers/all-mpnet-base-v2``
is pinned to model commit ``e8c3b32edf5434bc2275fc9bab85f82640a19130``. Its normalized
768-dimensional vectors are centered and rotated with full PCA, preserving their
Euclidean distances. Documents are truncated at 384 wordpieces including special
tokens; the model embeds cleaned text, not the shorter hover excerpt.

The sentence embedding cache records document contents and row order, model
revision, token policy and package versions. A cache hit does not load the model.
SentenceTransformers chooses an available accelerator or CPU; set
``EMBEDDING_DEVICE="cpu"`` in the notebook to choose CPU explicitly.

The warm-start optimizer cache records the feature arrays, original IDs, ordered
lambda sequence, perplexity and fit policy. Every cached layout also records its
actual starting coordinates and PCA reference. The first fit uses scaled PCA and
early exaggeration; subsequent fits continue from raw coordinates and skip early
exaggeration. The pinned DREAMS checkout forwards an unused ``X`` parameter from
``prepare_initial``; the helper removes it from the new embedding and uses the
public optimization API. Neither upstream repository is patched.

Edit ``DATASETS`` to run a subset, or ``SAMPLE_SIZE`` for a different sample size.
``PARALLEL_DATASETS`` controls concurrent sweeps and ``N_JOBS`` optimizer threads.
Changing zoom spacing or interpolation reuses existing fitted layouts.

Plot styling uses ``opinionated`` and its ``opinionated_j`` (Jost) preset. The
notebook registers the fonts bundled with the package and applies its Matplotlib
stylesheet; the interactive maps use Jost through DataMapPlot's existing font
parameters and the example CSS, with the bundled Jost font embedded in each HTML
map. An example helper removes duplicate CDN font declarations from the generated
HTML before saving and displaying it. Figures have no overall title and use
11-point subplot titles; figures and maps share the ``#eff0eb`` background. The
notebook environment pins Opinionated 0.0.3.0,
Matplotlib below 3.11 (the package uses ``matplotlib.style.core``), and setuptools
below 81 (the package imports ``pkg_resources``).

Repository structure
--------------------

* ``src/zoommap/`` — reusable Python keyframe API and JavaScript position updates.
* ``notebooks/general_example.ipynb`` — the executed example for all four datasets.
* ``notebooks/dataset_helpers.py`` — dataset loading, sampling, image PCA and hover metadata.
* ``notebooks/text_embeddings.py`` — pinned sentence embeddings and document cache.
* ``notebooks/experiment_helpers.py`` — DREAMS fits, alignment, quality scores and map UI.
* ``notebooks/controls.js`` — zoom slider and animation controls.
* ``notebooks/execute.py`` — execute and save the notebook.
* ``tests/`` — position, dataset, embedding cache, continuation and browser checks.
* ``artifacts/`` — the four final maps and their coordinate/provenance/quality records.

Minimal plotting API::

    from zoommap import ZoomPositions
    import datamapplot

    motion = ZoomPositions(
        [(0, global_positions), (.8, intermediate_positions), (2.5, local_positions)],
        interpolation="linear",
    )
    figure = datamapplot.create_interactive_plot(
        global_positions,
        marker_color_array=colors,
        hover_text=descriptions,
        custom_js=motion.to_javascript(),
    )
    figure.save("moving_map.html")

For an existing ``custom_js`` string, use ``**motion.plot_options(**options)``
to compose it. Keyframes can be ``(zoom, array)`` pairs or ``ZoomKeyframe`` objects
with human-readable labels. Every array must be finite, have shape (N, 2), and
contain the same observations in the same row order. Zooms are strictly increasing
and need not be equally spaced. ``zoom_reference="initial"`` uses offsets from
the initial fitted camera (one zoom unit doubles magnification); ``"absolute"``
uses deck.gl's zoom values. Outside the keyframe range, coordinates hold at the
nearest endpoint, while the camera can continue zooming.

Interpolation and easing
------------------------

Spatial interpolation is ``"linear"`` or ``"pchip"``. PCHIP is a shape-preserving
cubic on the actual zoom grid, estimated across all keyframes. It avoids coordinate
overshoot and has continuous derivatives when used throughout with linear easing.
Easing is independently ``"linear"``, ``"smoothstep"``, or ``"smootherstep"``.
Smoothstep/smootherstep stop at keyframe boundaries, which can be desirable for
individual transitions but distracting in a dense continuous sequence.

``transitions`` optionally takes one ``Transition`` per interval; ``None`` inherits
the defaults. For example::

    motion = ZoomPositions(
        [(0, A), (.4, B), (2, C)],
        transitions=[Transition(easing="smoothstep"), Transition(interpolation="pchip")],
    )

The helper also provides ``motion.positions_at(zoom)`` for Python-side inspection.
Only built-in serializable interpolation/easing options are accepted. Browser data
is gzip-compressed Float32 binary, rather than nested JSON coordinates. CPU updates
are coalesced to one position upload per animation frame. Point colors, sizes,
selection filter attributes, IDs and hover metadata are retained. The browser learns
the shared centering/scaling transform from DataMapPlot's rendered first layout,
and rejects a first layout that disagrees with the supplied coordinates.

Display policy and scope
------------------------

The notebook explicitly centers layouts, equalizes RMS size, and aligns rotations
to the shared PCA reference, without reflections. Set ``EQUALIZE_SCALE=False`` to
retain differences in their overall size. This is display preprocessing outside
the helper; raw embeddings remain available. Interpolated positions are visual
approximations between the computed embeddings, rather than freshly optimized
DREAMS solutions. The notebook compares local/global quality at the real keyframes.

Both DREAMS and the separate ``ne_spectrum/`` implementation now display the
paper's quality metrics inside their slider boxes: 10-neighbor recall over all
observations and Spearman distance correlation among 1,000 fixed sampled
observations. The same feature arrays and sampled original IDs are used for both
techniques. Scores are evaluated along a dense grid of actual interpolated
layouts, including every keyframe. Browser lookup between samples is approximate
and marked with an approx sign. Camera cropping does not change these whole-layout
scores. Metric curves and provenance live in ``artifacts/quality_metrics/``.
The original endpoint CSV reports retain their legacy 700-observation sample.

The independent NE-spectrum notebook, environment, maps and GIFs are documented
in ``ne_spectrum/README.rst``. It uses the authors' native attraction-exaggeration
sweep and upstream openTSNE, leaving the DREAMS fits and environment intact.

A separate ``dreams_p30/`` notebook and four maps repeat DREAMS at perplexity 30
while preserving the original p=150 outputs. All other fitting and display
settings match. Its ``artifacts/perplexity_comparison.csv`` compares both versions
using their identical 1,000-observation slider metric references. See
``dreams_p30/README.rst`` for execution and browser validation. The root overview
links to DREAMS at both perplexities and NE-spectrum at perplexity 30.

The examples use class/topic or spatial-region colors and image/document/XYZ
tooltips. Position-dependent labels, hulls, edges, density tiles, minimaps, annotations and lasso indices require
their own updates and should not be enabled yet. Updating coordinates globally can
move a cluster away from the location being zoomed into; the helper leaves ordinary
camera behavior intact. A future camera-anchor policy can address that separately.

Validation::

    .venv/bin/python -m pytest

Browser checks use this repository's own Playwright dependency::

    npm ci
    PLAYWRIGHT_BROWSERS_PATH="$PWD/.browser-cache" npx playwright install chromium --only-shell
    .venv/bin/python -m http.server 8788 --bind 127.0.0.1 --directory artifacts

In another terminal from the project root::

    npm run test:browser

These checks inspect all four maps and the executed notebook's iframe outputs:
six keyframes, reverse motion, wheel zoom, stable point attributes, image/document/XYZ
hover content, animation and the page titles. ``CHROME_EXECUTABLE_PATH`` and
``ZOOMMAP_BASE_URL`` override the browser executable and preview URL.

References: https://github.com/berenslab/DREAMS,
https://arxiv.org/abs/2508.13747,
https://github.com/zalandoresearch/fashion-mnist,
https://www.tensorflow.org/datasets/catalog/mnist,
https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset,
https://huggingface.co/sentence-transformers/all-mpnet-base-v2,
https://github.com/MNoichl/opinionated.
