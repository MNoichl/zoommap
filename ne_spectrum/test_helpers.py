"""Exercise native package continuation and cache provenance without long fits."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ne_spectrum import TSNESpectrum

spec = importlib.util.spec_from_file_location("ne_helpers", Path(__file__).with_name("helpers.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture
def native_sequence(monkeypatch):
    calls, optimized = [], []

    class Embedding(np.ndarray):
        def optimize(self, **kwargs):
            optimized.append((self.copy(), kwargs))
            displacement = np.arange(len(self))[:, None] * np.array([[.1, -.2]])
            return (np.asarray(self) + displacement / kwargs["exaggeration"]).view(Embedding)

    class RecordingTSNE:
        def __init__(self, **kwargs):
            self.options = kwargs
            calls.append(self)

        def fit(self, features):
            return self.options["initialization"].copy().view(Embedding)

    monkeypatch.setattr(TSNESpectrum, "embedder_class", RecordingTSNE)
    features = np.random.default_rng(42).normal(size=(25, 4))
    return features, np.arange(25), calls, optimized


def test_native_continuation_keeps_previous_coordinates_and_explicit_iterations(native_sequence, tmp_path):
    features, ids, calls, optimized = native_sequence
    layouts, report, directory = helpers.fit_ne_spectrum(
        features, ids, [30, 6, 1], tmp_path, perplexity=5, n_iter=500,
    )
    assert len(calls) == 1  # Only one fit/affinity construction for the native sweep.
    assert calls[0].options["exaggeration"] == 30
    assert calls[0].options["early_exaggeration_iter"] == 0
    assert calls[0].options["n_iter"] == 500
    assert calls[0].options["perplexity"] == 5
    assert [options for _, options in optimized] == [
        {"n_iter": 500, "exaggeration": 6}, {"n_iter": 500, "exaggeration": 1},
    ]
    for previous, (initial, _) in zip(layouts, optimized):
        np.testing.assert_array_equal(previous, initial)
    again, cached_report, _ = helpers.fit_ne_spectrum(
        features, ids, [30, 6, 1], tmp_path, perplexity=5, n_iter=500,
    )
    np.testing.assert_array_equal(layouts, again)
    assert cached_report.cached.all() and len(calls) == 1
    with np.load(directory / "spectrum.npz") as saved:
        np.testing.assert_array_equal(saved["sample_ids"], ids)


def test_sequence_perplexity_features_and_row_order_invalidate_cache(native_sequence, tmp_path):
    features, ids, calls, _ = native_sequence
    cases = [
        (features, ids, [30, 6, 1], 5),
        (features, ids, [30, 3, 1], 5),
        (features, ids, [30, 6, 1], 6),
        (features + .1, ids, [30, 6, 1], 5),
        (features[::-1], ids[::-1], [30, 6, 1], 5),
    ]
    directories = [helpers.fit_ne_spectrum(x, rows, rhos, tmp_path, perplexity=p)[2]
                   for x, rows, rhos, p in cases]
    assert len(set(directories)) == len(cases)


@pytest.mark.parametrize("rhos", [[30, 30, 1], [1, 3], [30, 0], [np.nan, 1], []])
def test_invalid_sequence_does_not_call_optimizer(native_sequence, tmp_path, rhos):
    features, ids, calls, _ = native_sequence
    with pytest.raises(ValueError):
        helpers.fit_ne_spectrum(features, ids, rhos, tmp_path, perplexity=5)
    assert calls == []
