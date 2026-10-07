"""Check continuation and cache provenance without running the expensive optimizer."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# These tests exercise the optional notebook environment.
for dependency in ("pandas", "PIL", "requests", "sklearn"):
    pytest.importorskip(dependency)
source = Path(__file__).resolve().parents[1] / "notebooks" / "experiment_helpers.py"
spec = importlib.util.spec_from_file_location("experiment_helpers", source)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture
def experiment(monkeypatch):
    calls = []

    class Embedding(np.ndarray):
        def __new__(cls, model):
            result = np.array(model.initialization).view(cls)
            result.model = model
            result.gradient_descent_params = {"X": object()}
            return result

        def optimize(self, **parameters):
            assert "X" not in self.gradient_descent_params
            self.model.phases.append(parameters)
            # Deliberately mutate the embedding, as the real optimizer does.
            offsets = np.arange(len(self), dtype=float)[:, None] * np.array([[1, -1]])
            self[:] += (1 + self.model.reg_lambda) * offsets
            return self

    class RecordingTSNE:
        def __init__(self, reg_lambda, **parameters):
            self.__dict__.update(parameters)
            self.reg_lambda = reg_lambda
            self.early_exaggeration = 12
            self.initial_momentum = .5
            self.final_momentum = .8
            self.exaggeration = None
            self.phases = []
            calls.append(self)

        def prepare_initial(self, features):
            return Embedding(self)

    monkeypatch.setitem(sys.modules, "openTSNE",
                        SimpleNamespace(TSNE=RecordingTSNE, __version__="test"))
    features = np.random.default_rng(42).normal(size=(8, 3))
    return features, np.arange(len(features)), calls


def test_predecessor_initializes_next_fit_and_pca_anchor_stays_fixed(experiment, tmp_path):
    features, ids, calls = experiment
    original = features.copy()
    layouts, report, directory = helpers.fit_dreams_spectrum(
        features, ids, [1, .5, .25, 0], tmp_path, perplexity=2,
    )
    reference = features[:, :2] - features[:, :2].mean(axis=0)
    pca_initialization = reference / reference[:, 0].std() * 1e-4
    np.testing.assert_array_equal(layouts[0], reference)
    np.testing.assert_array_equal(calls[0].initialization, pca_initialization)
    for index, model in enumerate(calls):
        np.testing.assert_array_equal(model.reg_embedding, pca_initialization)
        if index:
            np.testing.assert_array_equal(model.initialization, layouts[index])
        assert len(model.phases) == (2 if index == 0 else 1)
        assert model.perplexity == 2
        with np.load(directory / f"lambda-{model.reg_lambda:.8g}.npz") as saved:
            np.testing.assert_array_equal(saved["initial_positions"], model.initialization)
            np.testing.assert_array_equal(saved["reg_reference"], pca_initialization)
    assert not calls[-1].regularization
    assert list(report.early_exaggeration_iter) == [0, 250, 0, 0]
    np.testing.assert_array_equal(features, original)


def test_partial_cache_resumes_from_cached_predecessor(experiment, tmp_path):
    features, ids, calls = experiment
    expected, _, directory = helpers.fit_dreams_spectrum(
        features, ids, [1, .5, .25, 0], tmp_path, perplexity=2,
    )
    (directory / "lambda-0.25.npz").unlink()
    calls.clear()
    actual, report, _ = helpers.fit_dreams_spectrum(
        features, ids, [1, .5, .25, 0], tmp_path, perplexity=2,
    )
    assert [model.reg_lambda for model in calls] == [.25]
    np.testing.assert_array_equal(calls[0].initialization, expected[1])
    assert list(report.cached) == [True, True, False, True]
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)


def test_cache_depends_on_lambda_sequence_perplexity_and_policy(experiment, tmp_path):
    features, ids, calls = experiment
    directories = []
    for strengths, options in [
        ([1, .5, .25, 0], {}),
        ([1, .25, 0], {}),
        ([1, .5, .25, 0], {"perplexity": 3}),
        ([1, .5, .25, 0], {"warm_start": False}),
    ]:
        calls.clear()
        _, _, directory = helpers.fit_dreams_spectrum(
            features, ids, strengths, tmp_path, **({"perplexity": 2} | options),
        )
        directories.append(directory)
        settings = json.loads((directory / "settings.json").read_text())
        if options.get("warm_start") is False:
            assert settings["fit_policy"] == "independent-shared-pca-v2"
            for model in calls:
                np.testing.assert_array_equal(model.initialization, model.reg_embedding)
                assert len(model.phases) == 2
        else:
            assert settings["lambda_sequence"] == strengths
    assert len(set(directories)) == 4


def test_changed_cached_predecessor_invalidates_the_remaining_chain(experiment, tmp_path):
    features, ids, calls = experiment
    _, _, directory = helpers.fit_dreams_spectrum(
        features, ids, [1, .5, .25, 0], tmp_path, perplexity=2,
    )
    predecessor = directory / "lambda-0.5.npz"
    with np.load(predecessor) as saved:
        replacement = {name: saved[name].copy() for name in saved.files}
    replacement["positions"][0, 0] += 1
    np.savez_compressed(predecessor, **replacement)
    calls.clear()
    _, report, _ = helpers.fit_dreams_spectrum(
        features, ids, [1, .5, .25, 0], tmp_path, perplexity=2,
    )
    assert [model.reg_lambda for model in calls] == [.25, 0]
    np.testing.assert_array_equal(calls[0].initialization, replacement["positions"])
    assert list(report.cached) == [True, True, False, False]


@pytest.mark.parametrize("strengths", [[1, .5, .5], [0, .5], [1, np.nan], []])
def test_invalid_continuation_sequence_rejected_before_fitting(experiment, tmp_path, strengths):
    features, ids, calls = experiment
    with pytest.raises(ValueError):
        helpers.fit_dreams_spectrum(features, ids, strengths, tmp_path, perplexity=2)
    assert calls == []
