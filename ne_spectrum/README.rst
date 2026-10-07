From global to local with NE-spectrum
====================================

This is a separate experiment using the authors' unmodified
`NE-spectrum package <https://github.com/sciai-lab/ne-spectrum>`_.
Its native ``TSNESpectrum`` backend varies t-SNE attraction exaggeration
through ``[8, 6, 3, 2, 1.5, 1]``. The first optimized layout starts from
scaled PCA. Subsequent slides continue from the preceding raw embedding and
reuse its affinities, with 500 iterations per slide and perplexity 150.
The global endpoint is an optimized neighbor embedding; rho=1 is ordinary
t-SNE. The moderate rho=8 start keeps the global layouts two-dimensional.

`Open the three maps <https://mnoichl.github.io/zoommap/ne_spectrum/>`_.
Their original DREAMS counterparts remain at the root landing page.
Both versions have the same Jost / Opinionated aesthetic, #eff0eb background,
5,000 observations, colors, hover metadata and unequal zoom offsets.
Coordinates use linear interpolation with linear easing.

Installation and execution
--------------------------

Use a separate Python 3.12 environment: upstream openTSNE cannot coexist with
the DREAMS fork under the same import name. Run these commands from zoommap/::

    UV_CACHE_DIR="$PWD/ne_spectrum/data/uv-cache" uv venv ne_spectrum/.venv --python 3.12
    UV_CACHE_DIR="$PWD/ne_spectrum/data/uv-cache" uv pip install --python ne_spectrum/.venv/bin/python -r ne_spectrum/requirements.txt
    ne_spectrum/.venv/bin/python -m ipykernel install --prefix ne_spectrum/.venv --name zoommap-ne-spectrum --display-name "ZoomMap NE-spectrum research"
    ne_spectrum/.venv/bin/python ne_spectrum/execute.py

Alternatively, open ``general_example.ipynb`` with the **ZoomMap NE-spectrum
research** kernel and run all cells. The notebook requires the shared
``../notebooks/`` helper files and installed ZoomPositions package.
NE-spectrum is pinned to c980702a2594ee628fc86c36902f8c699c80b67e,
upstream openTSNE to 1.0.4, and DataMapPlot to the same revision as DREAMS.

The shared dataset loaders reuse root ``data/`` downloads and normalized MPNet
vectors. Matching preprocessing dependencies make the feature arrays and row
IDs identical to DREAMS. NE-spectrum optimizer caches, plotting caches and
runtime files live in ``ne_spectrum/data/``; its outputs live in
``ne_spectrum/artifacts/``. The native sequence is cached atomically, keyed by
processed features, ordered row IDs, ordered attraction settings, optimizer
settings and dependency versions. No DREAMS optimizer cache is read or written.

Display and quality
-------------------

Raw coordinates are preserved in the archives. Display layouts are centered,
equalized to the first layout's RMS radius and aligned to its rotation, without
reflections. This display preprocessing preserves neighbor and distance ranks.
No library is patched; the same existing ``custom_js`` hook feeds ZoomPositions
and optional demo controls to DataMapPlot.

The slider boxes implement the definitions in
`section 5.1 of the DREAMS paper <https://arxiv.org/html/2508.13747#S5.SS1>`_:

* NN recall: fraction of the original feature-space 10 nearest neighbors that
  remain neighbors in the 2D layout, averaged over every observation, excluding
  the query observation itself.
* Global distance Spearman: rank correlation of Euclidean pairwise distances
  among 1,000 fixed randomly selected observations, against their original
  feature-space distances. All 499,500 unordered pairs are included.

Seed 42 selects the same 1,000 original row IDs for both techniques. Labels are
unused. We evaluate the actual interpolated coordinates on 101 evenly spaced
zoom samples, plus every knot. The browser linearly looks up this dense curve;
values between evaluated samples are approximate and marked with an approx sign.
Scores describe the entire layout, independent of the camera crop or pan.
The JSON and CSV curves under ``artifacts/quality_metrics/`` include provenance
and can be inspected independently of the maps. NE-spectrum endpoint reports
use these same scores. The original DREAMS endpoint CSVs retain their earlier
700-observation distance sample; its new slider curves use the paper's 1,000.

Validation and GIFs
-------------------

From zoommap/::

    ne_spectrum/.venv/bin/python -m pytest tests ne_spectrum/test_helpers.py
    ne_spectrum/.venv/bin/python -m http.server 8789 --bind 127.0.0.1

In another terminal::

    ZOOMMAP_TECHNIQUE=ne-spectrum ZOOMMAP_BASE_URL=http://127.0.0.1:8789/ne_spectrum/artifacts npm run test:browser
    node ne_spectrum/oneoffs/zoom_gifs/generate.cjs

Browser checks require the root Playwright setup described in ../README.rst.
The GIF generator additionally requires FFmpeg. Its separate folder contains
three square, text-free, dense-region zoom-in/out captures. DREAMS GIFs are kept
unchanged in ../oneoffs/zoom_gifs/.
