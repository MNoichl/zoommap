"""Shared DREAMS fitting and display utilities for the research notebooks."""

from pathlib import Path
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

DREAMS_COMMIT = "eeea6a6985ded4fa926e4176a9dbd002eae14228"


def fit_dreams_spectrum(features, sample_ids, lambdas, cache_directory, *,
                        seed=42, n_iter=500, n_jobs=4, perplexity=30, warm_start=True):
    """Fit a global-to-local sequence, keeping the PCA regularization anchor fixed.

    lambda=1 is evaluated analytically as PCA, the exact limiting layout.
    lambda=0 runs ordinary t-SNE; all intermediate values run the DREAMS fork.
    Warm starts use the preceding optimized layout at its original scale, without
    repeating early exaggeration. The first optimization starts from scaled PCA.
    Set warm_start=False to initialize every fit independently from scaled PCA.
    """
    import openTSNE
    from openTSNE import TSNE

    if "reg_lambda" not in inspect.signature(TSNE).parameters:
        raise RuntimeError("Install the DREAMS fork of openTSNE, not upstream openTSNE. See README.rst.")
    lambdas = [float(strength) for strength in lambdas]
    if not lambdas or any(not np.isfinite(strength) or not 0 <= strength <= 1 for strength in lambdas):
        raise ValueError("DREAMS regularization strengths must be a nonempty sequence in [0, 1]")
    if warm_start and any(left <= right for left, right in zip(lambdas[:-1], lambdas[1:])):
        raise ValueError("Warm-start strengths must be strictly decreasing from global to local")
    if not np.isfinite(perplexity) or not 0 < perplexity < len(features):
        raise ValueError("perplexity must be positive and smaller than the number of observations")
    # Canonicalize integer-valued settings so existing independent caches remain valid.
    perplexity = int(perplexity) if float(perplexity).is_integer() else float(perplexity)
    settings = {
        "seed": seed, "n_iter": n_iter, "n_jobs": n_jobs, "perplexity": perplexity,
        "early_exaggeration_iter": 250, "negative_gradient_method": "bh",
        "dreams_commit": DREAMS_COMMIT, "opentsne_version": openTSNE.__version__,
        "fit_policy": "warm-start-fixed-pca-v1" if warm_start else "independent-shared-pca-v2",
        "compatibility": "omit unused X optimizer parameter from DREAMS prepare_initial",
    }
    if warm_start:
        # A layout now depends on every preceding lambda, including their order.
        settings.update(lambda_sequence=lambdas, warm_start_early_exaggeration_iter=0)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(features, dtype="<f8").tobytes())
    digest.update(np.asarray(sample_ids, dtype="<i8").tobytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    cache_directory = Path(cache_directory) / digest.hexdigest()[:16]
    cache_directory.mkdir(parents=True, exist_ok=True)
    reference = features[:, :2].copy()
    reference -= reference.mean(axis=0)
    pca_initialization = reference / reference[:, 0].std() * 1e-4
    results, records = [], []
    previous, previous_strength = None, None
    for strength in lambdas:
        continuing = warm_start and previous is not None and strength != 1
        initialization = previous.copy() if continuing else pca_initialization.copy()
        initialization_source = f"lambda={previous_strength:g}" if continuing else "scaled PCA"
        early_iterations = 0 if continuing or strength == 1 else 250
        if strength == 1:
            initialization = reference.copy()
            initialization_source = "analytic PCA"
        target = cache_directory / f"lambda-{strength:.8g}.npz"
        started = time.perf_counter()
        cached = target.exists()
        if cached:
            with np.load(target) as saved:
                # If a missing prefix was recomputed, never reuse a suffix fitted
                # from different predecessor coordinates.
                if warm_start:
                    cached = (np.array_equal(saved["initial_positions"], initialization)
                              and np.array_equal(saved["reg_reference"], pca_initialization))
                if cached:
                    layout = saved["positions"].copy()
                    compute_seconds = float(saved["compute_seconds"])
            if cached:
                print(f"λ={strength:g}: loaded cached layout ({initialization_source})", flush=True)
        if not cached:
            print(f"λ={strength:g}: {'PCA endpoint' if strength == 1 else 'optimizing from ' + initialization_source}", flush=True)
            if strength == 1:
                layout = reference.copy()
            else:
                model = TSNE(
                    initialization=initialization.copy(),
                    reg_embedding=pca_initialization.copy(),
                    regularization=strength > 0, reg_lambda=strength,
                    reg_scaling="norm", reg_scaling_dims="one",
                    random_state=seed, n_jobs=n_jobs, n_iter=n_iter,
                    perplexity=perplexity, early_exaggeration_iter=early_iterations,
                    negative_gradient_method="bh", verbose=False,
                )
                # The pinned authors' checkout adds X to optimizer parameters,
                # but gradient_descent does not accept it. Use its public
                # preparation/optimization API after removing that unused parameter.
                embedding = model.prepare_initial(features)
                embedding.gradient_descent_params.pop("X", None)
                if model.early_exaggeration_iter:
                    embedding.optimize(
                        n_iter=model.early_exaggeration_iter,
                        exaggeration=model.early_exaggeration,
                        momentum=model.initial_momentum,
                        inplace=True, propagate_exception=True,
                    )
                embedding.optimize(
                    n_iter=model.n_iter, exaggeration=model.exaggeration,
                    momentum=model.final_momentum,
                    inplace=True, propagate_exception=True,
                )
                layout = np.asarray(embedding).copy()
            compute_seconds = time.perf_counter() - started
            if not np.isfinite(layout).all():
                raise RuntimeError(f"Non-finite coordinates at λ={strength:g}")
            np.savez_compressed(target, positions=layout, sample_ids=sample_ids,
                                compute_seconds=compute_seconds,
                                initial_positions=initialization,
                                reg_reference=pca_initialization)
        if layout.shape != reference.shape or not np.isfinite(layout).all():
            raise RuntimeError(f"Invalid coordinates at λ={strength:g}")
        results.append(layout)
        # The analytic PCA endpoint seeds the first fit at the standard tiny scale.
        # All subsequent optimized layouts are passed on unchanged, with copies.
        if strength != 1:
            previous, previous_strength = layout.copy(), strength
        records.append({"lambda": strength, "method": "PCA endpoint" if strength == 1 else
                        ("t-SNE endpoint" if strength == 0 else "DREAMS"),
                        "cached": cached, "compute_seconds": compute_seconds,
                        "initialization": initialization_source,
                        "early_exaggeration_iter": early_iterations,
                        "perplexity": perplexity})
    (cache_directory / "settings.json").write_text(json.dumps(settings, indent=2))
    return results, pd.DataFrame(records), cache_directory


def align_for_display(layouts, *, equalize_scale=True):
    """Explicit display policy: center, optionally equalize RMS size, rotate.

    Rotations align to the first layout; reflections are disabled. Raw DREAMS
    coordinates are never overwritten. No nonlinear registration is performed.
    """
    reference = np.asarray(layouts[0], dtype=np.float64)
    reference = reference - reference.mean(axis=0)
    target_rms = np.sqrt(np.mean(np.sum(reference ** 2, axis=1)))
    result = []
    for layout in layouts:
        centered = np.asarray(layout, dtype=np.float64) - np.mean(layout, axis=0)
        rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
        if rms <= 0:
            raise ValueError("Cannot align a layout with zero extent")
        if equalize_scale:
            centered *= target_rms / rms
        rotation, _ = orthogonal_procrustes(centered, reference)
        if np.linalg.det(rotation) < 0:
            # Constrain the fit to a proper rotation, never a mirror image.
            left, _, right = np.linalg.svd(centered.T @ reference)
            left[:, -1] *= -1
            rotation = left @ right
        result.append((centered @ rotation).astype(np.float32))
    return result


def quality_scores(features, layouts, seed=42, k=10):
    """kNN recall and sampled Spearman distance correlation; labels unused."""
    high = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(return_distance=False)[:, :k]
    # kneighbors(X=None) already excludes each query point itself.
    rng = np.random.default_rng(seed)
    subset = rng.choice(len(features), min(700, len(features)), replace=False)
    high_distances = pdist(features[subset])
    records = []
    for layout in layouts:
        low = NearestNeighbors(n_neighbors=k + 1).fit(layout).kneighbors(return_distance=False)[:, :k]
        recall = np.mean([len(set(a) & set(b)) / k for a, b in zip(high, low)])
        correlation = spearmanr(high_distances, pdist(np.asarray(layout)[subset])).statistic
        records.append({"knn_recall": recall, "distance_correlation": correlation})
    return pd.DataFrame(records)


EXAMPLE_CSS = """
body { background: #eff0eb; color: #182238; font-family: "Jost", sans-serif; }
#dataset-heading { position: fixed; top: 30px; left: 36px; pointer-events: none;
  max-width: calc(100vw - 320px); padding: 10px 18px 12px 0;
  background: linear-gradient(90deg, #eff0ebf5 75%, #eff0eb00); }
#dataset-heading h1 { font-size: clamp(23px, 3vw, 34px); font-weight: 550; line-height: 1.2; margin: 12px 0; }
#dataset-heading p { font-size: 12px; color: #75829a; margin: 8px 0; }
#dataset-legend { position: fixed; right: 30px; top: 40px; background: #eff0ebe8;
  padding: 12px 16px; border-radius: 10px; pointer-events: none; }
.dataset-legend-row { display: flex; align-items: center; gap: 9px; font-size: 11px; padding: 4px 0; }
.dataset-legend-row span { width: 8px; height: 8px; border-radius: 50%; }
#zoom-layout-controls { position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
  width: min(510px, calc(100vw - 40px)); box-sizing: border-box; padding: 20px 24px;
  background: #ffffffed; border: 1px solid #e1e5ed; border-radius: 14px; box-shadow: 0 8px 30px #1822380a; }
.dataset-endpoints, .dataset-control-footer { display: flex; align-items: center; justify-content: space-between; }
.dataset-endpoints { font-size: 12px; font-weight: 600; }
#zoom-layout-slider { width: 100%; margin: 14px 0; accent-color: #5568cb; cursor: pointer; }
#zoom-layout-status { font-size: 11px; font-variant-numeric: tabular-nums; color: #75829a; }
#zoom-layout-play { font-size: 11px; border: 1px solid #dbe0ef; border-radius: 6px;
  padding: 5px 11px; color: #4355ac; background: #f4f6ff; cursor: pointer; font-family: inherit; }
#zoom-layout-controls small { display: block; margin-top: 10px; font-size: 10px; color: #8590a4; }
.dataset-sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }
@media (max-width: 650px) { #dataset-heading { top: 18px; left: 20px; max-width: calc(100vw - 40px); }
  #dataset-legend { display: none; } #dataset-heading p { max-width: 230px; }
  #zoom-layout-controls { padding: 16px; bottom: 16px; } }
"""


def plot_ui(title, count, noun, class_names, colors, perplexity):
    """Optional example UI; the reusable position helper needs no dataset UI."""
    from base64 import b64encode
    from html import escape
    from importlib.metadata import distribution

    # Embed Opinionated's bundled Jost font so notebook iframes do not need a
    # network request before DataMapPlot's text layers can use it.
    fonts = Path(distribution("opinionated").locate_file("opinionated/fonts"))
    font_data = b64encode((fonts / "Jost-VariableFont_wght.ttf").read_bytes()).decode("ascii")
    license_text = (fonts / "OFL.txt").read_text()
    license_text = license_text[license_text.index("SIL OPEN FONT LICENSE"):]
    font_css = f"""/*
Copyright 2020 The Jost Project Authors (https://github.com/indestructible-type/Jost)
{license_text}
*/
@font-face {{
  font-family: "Jost";
  src: url("data:font/ttf;base64,{font_data}") format("truetype");
  font-weight: 100 900;
  font-style: normal;
  font-display: swap;
}}
"""
    legend = "".join(
        f'<div class="dataset-legend-row"><span style="background:{escape(color)}"></span>{escape(name)}</div>'
        for name, color in zip(class_names, colors)
    )
    inspection = "its image" if noun == "images" else "a document excerpt"
    html = f"""
<script>document.fonts.load('12px "Jost"');</script>
<div id="dataset-heading">
  <h1>From global to local · {escape(title)}</h1>
  <p>Zoom from a global PCA layout through DREAMS toward local t-SNE structure.</p>
</div>
<div id="dataset-legend">{legend}</div>
<div id="zoom-layout-controls">
  <div class="dataset-endpoints"><span>Global · PCA</span><span>Local · t-SNE</span></div>
  <label class="dataset-sr-only" for="zoom-layout-slider">Zoom and embedding layout</label>
  <input id="zoom-layout-slider" type="range" step="0.001" value="0" list="zoom-layout-stops" disabled>
  <datalist id="zoom-layout-stops"></datalist>
  <div class="dataset-control-footer">
    <span id="zoom-layout-status">Loading layouts…</span>
    <button id="zoom-layout-play" disabled>Animate</button>
  </div>
  <small>Scroll, pinch, or use the slider. Hover a point to inspect {inspection}.</small>
</div>"""
    return html, font_css + EXAMPLE_CSS


def use_bundled_jost(figure):
    """Keep the embedded font instead of DataMapPlot's duplicate CDN faces.

    Rewrap the generated HTML using the existing InteractiveFigure constructor;
    DataMapPlot and its API remain unchanged.
    """
    import re

    head, body = str(figure).split("</head>", 1)
    head = re.sub(
        r'<link\b[^>]*href="https://fonts\.(?:googleapis|gstatic)\.com[^\"]*"[^>]*>\s*',
        "", head,
    )
    head = re.sub(
        r"@font-face\s*\{[^{}]*font-family:\s*['\"]Jost['\"];[^{}]*"
        r"src:\s*url\(https://fonts\.gstatic\.com/[^)]*\)[^{}]*\}\s*",
        "", head,
    )
    return type(figure)(
        head + "</head>" + body, width=figure.width, height=figure.height,
        api_token=figure.api_token,
    )
