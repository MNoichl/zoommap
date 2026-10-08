From global to local with DREAMS at perplexity 30
================================================

This controlled variant preserves the original perplexity-150 DREAMS maps.
It uses the same 5,000 observations, preprocessing, seed 42, lambda sequence
``[1, .5, .25, .15, .05, 0]``, 500 iterations per optimized layout, warm-start
policy, fixed PCA reference, display alignment and zoom settings. Perplexity
is 30. The existing DREAMS Python environment and shared notebook helpers are
reused; no library code is changed.

Run from zoommap/ after installing the root research environment::

    mkdir -p dreams_p30/data
    IPYTHONDIR="$PWD/dreams_p30/data/ipython" JUPYTER_RUNTIME_DIR="$PWD/dreams_p30/data/jupyter-runtime" .venv/bin/python notebooks/execute.py dreams_p30/general_example.ipynb

Alternatively, run ``general_example.ipynb`` using the **ZoomMap research**
kernel. Downloads and normalized MPNet vectors are shared from root ``data/``.
Optimizer, quality and plotting caches are isolated under ``dreams_p30/data/``;
the executed notebook and final outputs live under ``dreams_p30/``.

Fashion-MNIST, MNIST, 20 Newsgroups and Mammoth share the same observations and
features across all variants. Mammoth uses original 3D distances, a uniform sample
and display-only spatial-region colors, as documented in ../README.rst.

The slider and endpoint reports use 10-neighbor recall over all observations
and global distance Spearman over 1,000 fixed sampled observations. The
``artifacts/perplexity_comparison.csv`` table compares the original p=150 slider
curves against p=30 at matching lambda settings. It verifies identical feature
hashes, sample IDs, distance subsets and metric definitions before comparison.
The original p=150 endpoint CSVs retain their legacy 700-observation scores.

Both versions use the same Jost / Opinionated aesthetic and #eff0eb background.
The root overview links directly to both versions and the NE-spectrum p=30 maps.

With zoommap/ served at http://127.0.0.1:8791, run browser checks from zoommap/::

    ZOOMMAP_TECHNIQUE=dreams-p30 ZOOMMAP_BASE_URL=http://127.0.0.1:8791/dreams_p30/artifacts npm run test:browser
