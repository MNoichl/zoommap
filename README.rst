From global to local
====================

``zoommap`` moves points between ordered layouts as the viewer zooms. It uses
DataMapPlot's existing ``custom_js`` parameter; the DataMapPlot source and public
plotting API stay unchanged.

``notebooks/general_example.ipynb`` produces the three final interactive maps:

* ``artifacts/fashion_mnist_zoom.html`` — Fashion-MNIST, PCA-50 pixel features.
* ``artifacts/mnist_zoom.html`` — MNIST digits, PCA-50 pixel features.
* ``artifacts/20_newsgroups_zoom.html`` — 20 Newsgroups, MPNet sentence embeddings.

Each dataset uses 5,000 balanced training observations, perplexity 150, and six
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
and ignored by Git. The three final maps, coordinate archives, preprocessing
settings and quality reports are kept in ``artifacts/`` and versioned.

The exported HTML maps need no Python or notebook server to view. Like normal
DataMapPlot exports, they load JavaScript dependencies from CDNs.

Reproduction settings
---------------------

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
* ``notebooks/general_example.ipynb`` — the single executed example for all three datasets.
* ``notebooks/dataset_helpers.py`` — dataset loading, sampling, image PCA and hover metadata.
* ``notebooks/text_embeddings.py`` — pinned sentence embeddings and document cache.
* ``notebooks/experiment_helpers.py`` — DREAMS fits, alignment, quality scores and map UI.
* ``notebooks/controls.js`` — zoom slider and animation controls.
* ``notebooks/execute.py`` — execute and save the notebook.
* ``tests/`` — position, dataset, embedding cache, continuation and browser checks.
* ``artifacts/`` — the three final maps and their coordinate/provenance/quality records.

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

The examples use class/topic colors and image/document tooltips. Position-dependent
labels, hulls, edges, density tiles, minimaps, annotations and lasso indices require
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

These checks inspect all three maps and the executed notebook's iframe outputs:
six keyframes, reverse motion, wheel zoom, stable point attributes, image/document
hover content, animation and the page titles. ``CHROME_EXECUTABLE_PATH`` and
``ZOOMMAP_BASE_URL`` override the browser executable and preview URL.

References: https://github.com/berenslab/DREAMS,
https://arxiv.org/abs/2508.13747,
https://github.com/zalandoresearch/fashion-mnist,
https://www.tensorflow.org/datasets/catalog/mnist,
https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset,
https://huggingface.co/sentence-transformers/all-mpnet-base-v2,
https://github.com/MNoichl/opinionated.
