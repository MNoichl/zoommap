"""DREAMS-paper quality metrics evaluated on the actual interpolated layouts.

The browser reads a dense sampled curve; it does not run nearest-neighbor
searches. Its values between samples are explicitly approximate.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

PAPER = "https://arxiv.org/html/2508.13747#S5.SS1"


def array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _unit_ranks(distances):
    ranks = rankdata(distances, method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    if norm == 0:
        raise ValueError("Distance correlation requires nonconstant pairwise distances")
    return ranks / norm


class QualityReference:
    """One shared high-dimensional reference, with labels never consulted.

    Recall averages overlap of self-excluding k-neighbor sets over all points.
    Spearman compares all unordered distances among a fixed random subset of
    1,000 observations, matching section 5.1 of the DREAMS paper.
    """

    def __init__(self, features, sample_ids, *, k=10, subset_size=1000, seed=42):
        features = np.asarray(features, dtype=np.float64)
        sample_ids = np.asarray(sample_ids)
        if (features.ndim != 2 or not np.isfinite(features).all()
                or sample_ids.shape != (len(features),)
                or len(np.unique(sample_ids)) != len(features)):
            raise ValueError("Finite feature rows must have unique, matching sample IDs")
        if not isinstance(k, int) or not 0 < k < len(features):
            raise ValueError("k must be positive and smaller than the observation count")
        if not isinstance(subset_size, int) or subset_size < 3:
            raise ValueError("subset_size must be at least three")
        self.k, self.count = k, len(features)
        # X=None excludes the query observation itself, including duplicate rows.
        self.neighbors = NearestNeighbors(n_neighbors=k).fit(features).kneighbors(
            return_distance=False,
        )
        self.subset = np.random.default_rng(seed).choice(
            len(features), min(subset_size, len(features)), replace=False,
        )
        self.distance_ranks = _unit_ranks(pdist(features[self.subset]))
        self.metadata = {
            "definition_version": 1, "paper": PAPER,
            "k": k, "point_count": len(features),
            "distance_subset_size": len(self.subset), "seed": seed,
            "distance_subset_ids": sample_ids[self.subset].tolist(),
            "features_sha256": array_digest(features),
            "sample_ids_sha256": array_digest(sample_ids),
            "neighbor_policy": "all observations; exclude self; Euclidean distance",
            "distance_policy": "Spearman; all unordered pairs within fixed random subset",
        }

    def score(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self.count, 2) or not np.isfinite(positions).all():
            raise ValueError("Layout must contain one finite 2D row per observation")
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(positions).kneighbors(
            return_distance=False,
        )
        overlap = (self.neighbors[:, :, None] == neighbors[:, None, :]).any(axis=2)
        recall = float(overlap.mean())
        correlation = float(np.dot(self.distance_ranks, _unit_ranks(pdist(positions[self.subset]))))
        return {"nn_recall": recall, "global_distance_spearman": float(np.clip(correlation, -1, 1))}


def evaluate_zoom_quality(features, sample_ids, motion, cache_directory, *,
                          k=10, subset_size=1000, seed=42, grid_size=101):
    """Evaluate a dense grid plus every knot, caching by data and motion identity."""
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be at least two")
    regular_grid = np.linspace(motion.zooms[0], motion.zooms[-1], grid_size)
    # Prefer the exact knot when linspace differs by a floating-point ulp.
    regular_grid = regular_grid[~np.isclose(
        regular_grid[:, None], np.asarray(motion.zooms)[None, :], rtol=0, atol=1e-12,
    ).any(axis=1)]
    grid = np.sort(np.concatenate([regular_grid, motion.zooms]))
    identity = {
        "version": 1, "features": array_digest(np.asarray(features, dtype=np.float64)),
        "ids": array_digest(sample_ids), "grid": grid.tolist(),
        "positions": [array_digest(frame.positions) for frame in motion.keyframes],
        "transitions": [vars(t) for t in motion.transitions],
        "k": k, "subset_size": subset_size, "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = Path(cache_directory) / f"quality-{digest[:16]}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved.get("identity") == identity:
            print(f"Quality metrics: loaded {len(grid)} cached zoom samples", flush=True)
            return saved
    reference = QualityReference(features, sample_ids, k=k, subset_size=subset_size, seed=seed)
    records = []
    for i, zoom in enumerate(grid):
        records.append({"zoom": float(zoom), **reference.score(motion.positions_at(zoom))})
        if i % 25 == 0 or i == len(grid) - 1:
            print(f"Quality metrics: evaluated {i + 1}/{len(grid)} zoom samples", flush=True)
    result = {
        "identity": identity, "metadata": reference.metadata, "samples": records,
        "browser_policy": "linear lookup between densely evaluated zoom samples; approximate",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(target)
    return result


def quality_javascript(metrics):
    template = Path(__file__).with_name("metrics.js").read_text()
    encoded = json.dumps(metrics, allow_nan=False).replace("<", "\\u003c")
    return template.replace("__ZOOM_QUALITY_SPEC__", encoded)
