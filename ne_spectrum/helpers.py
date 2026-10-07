"""Native NE-spectrum fits; display utilities are shared with the DREAMS demo."""

from pathlib import Path
import hashlib
import inspect
import json
import time
from importlib.metadata import version

import numpy as np
import pandas as pd

NE_SPECTRUM_COMMIT = "c980702a2594ee628fc86c36902f8c699c80b67e"


def fit_ne_spectrum(features, sample_ids, exaggerations, cache_directory, *,
                    seed=42, n_iter=500, n_jobs=4, perplexity=150):
    """Use the package's native continuation, reusing affinities and prior fits.

    High attraction exaggeration emphasizes global neighbor structure. rho=1
    gives ordinary t-SNE. The first slide starts from tiny scaled PCA; later
    slides continue from the preceding optimized embedding without new early
    exaggeration. This function requires upstream openTSNE in its own venv.
    """
    from openTSNE import TSNE
    from ne_spectrum import TSNESpectrum

    if "reg_lambda" in inspect.signature(TSNE).parameters:
        raise RuntimeError("Use ne_spectrum/.venv: this pipeline requires upstream openTSNE, not DREAMS")
    features = np.asarray(features, dtype=np.float64)
    sample_ids = np.asarray(sample_ids)
    strengths = [float(value) for value in exaggerations]
    if (features.ndim != 2 or features.shape[1] < 2 or not np.isfinite(features).all()
            or sample_ids.shape != (len(features),)
            or len(np.unique(sample_ids)) != len(features)):
        raise ValueError("Finite feature rows must have unique, matching sample IDs")
    if (len(strengths) < 2 or any(not np.isfinite(s) or s < 1 for s in strengths)
            or any(a <= b for a, b in zip(strengths[:-1], strengths[1:]))):
        raise ValueError("Exaggerations must strictly decrease from global attraction toward t-SNE")
    if not np.isfinite(perplexity) or not 0 < perplexity < len(features):
        raise ValueError("perplexity must be positive and smaller than the observation count")
    if not isinstance(n_iter, int) or n_iter <= 0:
        raise ValueError("n_iter must be a positive integer")
    reference = features[:, :2] - features[:, :2].mean(axis=0)
    scale = reference[:, 0].std()
    if scale == 0:
        raise ValueError("The first PCA component must have nonzero extent")
    initialization = reference / scale * 1e-4
    settings = {
        "ne_spectrum_commit": NE_SPECTRUM_COMMIT,
        "ne_spectrum_version": version("ne-spectrum"),
        "opentsne_version": version("openTSNE"),
        "numpy_version": version("numpy"), "scipy_version": version("scipy"),
        "fit_policy": "native-tsne-spectrum-continuation-v1",
        "exaggerations": strengths, "seed": seed, "n_iter": n_iter,
        "n_jobs": n_jobs, "perplexity": float(perplexity),
        "negative_gradient_method": "bh", "early_exaggeration_iter": 0,
        "initialization": "centered first two PCA features; first-axis std=1e-4",
        "use_previous_as_init": True,
    }
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(features, dtype="<f8").tobytes())
    digest.update(np.asarray(sample_ids, dtype="<i8").tobytes())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    directory = Path(cache_directory) / digest.hexdigest()[:16]
    target = directory / "spectrum.npz"
    cached = target.exists()
    if cached:
        with np.load(target, allow_pickle=False) as saved:
            if not np.array_equal(saved["sample_ids"], sample_ids):
                raise RuntimeError("Cached sample identity is inconsistent")
            layouts = saved["positions"].copy()
            compute_seconds = float(saved["compute_seconds"])
        print(f"NE-spectrum: loaded all {len(strengths)} cached layouts", flush=True)
    else:
        # Explicit per-slide n_iter avoids the package's default 50-iteration
        # continuation override. No patches or private optimizer calls needed.
        model = TSNESpectrum(
            num_slides=len(strengths), use_previous_as_init=True,
            kwarg_list=[{"n_iter": n_iter, "exaggeration": s} for s in strengths],
            early_exaggeration=0, seed=seed, verbose=False,
            initialization=initialization.copy(), perplexity=perplexity,
            n_jobs=n_jobs, negative_gradient_method="bh",
        )
        started = time.perf_counter()
        model.fit(features)
        compute_seconds = time.perf_counter() - started
        layouts = np.asarray(model.get_embeddings(), dtype=np.float64)
    if layouts.shape != (len(strengths), len(features), 2) or not np.isfinite(layouts).all():
        raise RuntimeError("NE-spectrum returned invalid coordinates")
    directory.mkdir(parents=True, exist_ok=True)
    if not cached:
        temporary = directory / "spectrum.partial.npz"
        np.savez_compressed(temporary, positions=layouts, sample_ids=sample_ids,
                            exaggerations=strengths, initial_positions=initialization,
                            compute_seconds=compute_seconds)
        temporary.replace(target)
    (directory / "settings.json").write_text(json.dumps(settings, indent=2))
    report = pd.DataFrame({
        "exaggeration": strengths,
        "method": ["t-SNE endpoint" if s == 1 else "NE-spectrum" for s in strengths],
        "cached": cached, "n_iter": n_iter, "perplexity": perplexity,
        "initialization": ["scaled PCA"] + [f"rho={s:g}" for s in strengths[:-1]],
        "sequence_compute_seconds": compute_seconds,
    })
    return list(layouts), report, directory


def spectrum_plot_ui(*args, global_exaggeration=8, **kwargs):
    from experiment_helpers import plot_ui

    html, css = plot_ui(*args, **kwargs)
    html = html.replace(
        "Zoom from a global PCA layout through DREAMS toward local t-SNE structure.",
        "Zoom from global neighbor structure through NE-spectrum toward local t-SNE structure.",
    ).replace("Global · PCA", f"Global · ρ={global_exaggeration:g}")
    return html, css
