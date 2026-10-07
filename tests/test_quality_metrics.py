"""Check the paper definitions independently and the evaluated zoom path."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

pytest.importorskip("sklearn")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from quality_metrics import QualityReference, evaluate_zoom_quality
from zoommap import ZoomPositions, ZoomKeyframe, Transition


def brute_recall(features, positions, k):
    high, low = cdist(features, features), cdist(positions, positions)
    np.fill_diagonal(high, np.inf)
    np.fill_diagonal(low, np.inf)
    high, low = np.argsort(high, axis=1)[:, :k], np.argsort(low, axis=1)[:, :k]
    return np.mean([len(set(a) & set(b)) / k for a, b in zip(high, low)])


def test_metrics_match_independent_self_excluding_definition_and_spearman():
    rng = np.random.default_rng(3)
    features, positions = rng.normal(size=(40, 7)), rng.normal(size=(40, 2))
    reference = QualityReference(features, np.arange(40), k=4, subset_size=20)
    actual = reference.score(positions)
    assert actual["nn_recall"] == pytest.approx(brute_recall(features, positions, 4))
    expected = spearmanr(pdist(features[reference.subset]), pdist(positions[reference.subset])).statistic
    assert actual["global_distance_spearman"] == pytest.approx(expected)
    assert not np.any(reference.neighbors == np.arange(40)[:, None])


def test_identical_layout_and_similarities_preserve_both_scores():
    features = np.random.default_rng(8).normal(size=(35, 2))
    reference = QualityReference(features, np.arange(35), k=5)
    scores = reference.score(features)
    assert scores == pytest.approx({"nn_recall": 1, "global_distance_spearman": 1})
    rotated = features @ np.array([[0, -1], [1, 0]]) * 7 + [13, -8]
    assert reference.score(rotated) == pytest.approx(scores)


def test_paper_subset_is_shared_and_sample_identity_is_retained():
    features = np.random.default_rng(3).normal(size=(1100, 3))
    ids = np.arange(1100) + 5000
    a, b = QualityReference(features, ids), QualityReference(features.copy(), ids.copy())
    assert len(a.subset) == 1000
    assert a.metadata == b.metadata
    assert a.metadata["distance_subset_ids"] == ids[a.subset].tolist()


def test_dense_grid_evaluates_the_real_motion_and_invalidates_on_policy_change(tmp_path):
    rng = np.random.default_rng(17)
    features = rng.normal(size=(30, 4))
    frames = [ZoomKeyframe(z, rng.normal(size=(30, 2))) for z in [0, .7, 2.5]]
    motion = ZoomPositions(frames)
    result = evaluate_zoom_quality(features, np.arange(30), motion, tmp_path,
                                   grid_size=6, k=3, subset_size=15)
    reference = QualityReference(features, np.arange(30), k=3, subset_size=15)
    assert .7 in [sample["zoom"] for sample in result["samples"]]
    middle = next(sample for sample in result["samples"] if sample["zoom"] == 1)
    expected = reference.score(motion.positions_at(1))
    assert {key: middle[key] for key in expected} == pytest.approx(expected)
    assert evaluate_zoom_quality(features, np.arange(30), motion, tmp_path,
                                 grid_size=6, k=3, subset_size=15) == result
    changed = evaluate_zoom_quality(features, np.arange(30),
        ZoomPositions(frames, transitions=[Transition(easing="smoothstep"), Transition()]),
        tmp_path, grid_size=6, k=3, subset_size=15)
    assert result["identity"] != changed["identity"]
    assert len(list(tmp_path.glob("quality-*.json"))) == 2
