"""Check dataset identity, sampling, text geometry, and safe hover content."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.distance import pdist

for dependency in ("pandas", "PIL", "requests", "sklearn", "matplotlib"):
    pytest.importorskip(dependency)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import dataset_helpers as helpers


def mammoth_source(directory, count=120):
    directory.mkdir(parents=True, exist_ok=True)
    points = np.random.default_rng(7).normal(size=(count, 3)) * [3, 10, 40]
    points += [75, 150, 200]
    helpers.pd.DataFrame(points, columns=["x", "y", "z"]).to_csv(
        directory / f"mammoth-{helpers.MAMMOTH_REVISION}.csv", index=False,
    )
    return points


def test_mammoth_preserves_xyz_geometry_identity_and_uniform_sample(tmp_path):
    points = mammoth_source(tmp_path / "mammoth")
    dataset = helpers.load_example_dataset("mammoth", tmp_path, sample_size=60, seed=42)
    expected_ids = np.sort(np.random.default_rng(42).choice(len(points), 60, replace=False))
    np.testing.assert_array_equal(dataset.sample_ids, expected_ids)
    np.testing.assert_allclose(pdist(dataset.features), pdist(points[expected_ids]), atol=1e-12)
    assert dataset.features.shape == (60, 3)
    assert dataset.noun == "points"
    assert len(dataset.class_names) == len(dataset.colors) == 12
    np.testing.assert_array_equal(np.unique(dataset.labels), np.arange(12))
    np.testing.assert_array_equal(dataset.extra_data.sample_id, expected_ids)
    for i, axis in enumerate(["x", "y", "z"]):
        assert dataset.extra_data[axis].tolist() == [f"{v:.3f}" for v in points[expected_ids, i]]
    repeat = helpers.load_example_dataset("mammoth", tmp_path, sample_size=60, seed=42)
    np.testing.assert_array_equal(dataset.features, repeat.features)
    np.testing.assert_array_equal(dataset.labels, repeat.labels)
    assert dataset.preprocessing == repeat.preprocessing
    assert dataset.preprocessing["source_revision"] == helpers.MAMMOTH_REVISION
    assert len(dataset.preprocessing["source_sha256"]) == 64
    assert "not anatomical labels" in dataset.tooltip


def test_mammoth_full_sample_keeps_original_rows(tmp_path):
    points = mammoth_source(tmp_path, count=36)
    dataset = helpers.load_mammoth(tmp_path, sample_size=None)
    np.testing.assert_array_equal(dataset.sample_ids, np.arange(len(points)))
    np.testing.assert_allclose(pdist(dataset.features), pdist(points), atol=1e-12)


@pytest.mark.parametrize("size", [0, 11, 121, 30.5])
def test_mammoth_rejects_invalid_samples(tmp_path, size):
    mammoth_source(tmp_path)
    with pytest.raises(ValueError, match="sample_size"):
        helpers.load_mammoth(tmp_path, sample_size=size)


def test_mammoth_rejects_nonfinite_source(tmp_path):
    mammoth_source(tmp_path)
    target = tmp_path / f"mammoth-{helpers.MAMMOTH_REVISION}.csv"
    source = helpers.pd.read_csv(target)
    source.loc[0, "z"] = np.inf
    source.to_csv(target, index=False)
    with pytest.raises(ValueError, match="finite XYZ"):
        helpers.load_mammoth(tmp_path)


def test_balanced_sample_is_reproducible_sorted_and_keeps_original_indices():
    labels = np.repeat(np.arange(4), [6, 9, 8, 10])
    ids = helpers.balanced_indices(labels, 12, seed=42)
    np.testing.assert_array_equal(ids, helpers.balanced_indices(labels, 12, seed=42))
    np.testing.assert_array_equal(np.bincount(labels[ids]), [3, 3, 3, 3])
    assert np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(helpers.balanced_indices(labels, None), np.arange(len(labels)))
    np.testing.assert_array_equal(helpers.balanced_indices(labels, len(labels)), np.arange(len(labels)))


@pytest.mark.parametrize("size", [0, 5, 20, 3.5])
def test_invalid_or_impossible_balanced_sample_rejected(size):
    with pytest.raises(ValueError):
        helpers.balanced_indices(np.repeat(np.arange(4), 3), size)


def test_text_hover_content_is_escaped_and_retains_identity():
    data = helpers.text_metadata(
        ["<script>alert('x')</script>" + "a" * 100],
        np.array([0]), np.array([97]), ["<unsafe topic>"], preview_length=35,
    )
    assert data.sample_id[0] == 97
    assert data.class_name[0] == "&lt;unsafe topic&gt;"
    assert "<script>" not in data.preview[0]
    assert data.preview[0].startswith("&lt;script&gt;")
    assert data.preview[0].endswith("…")


def test_mpnet_uses_cleaned_texts_and_preserves_embedding_distances(monkeypatch, tmp_path):
    texts = ["alpha", " ", "the", "beta", "gamma", "delta", "epsilon"]
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    source = SimpleNamespace(data=texts, target=labels, target_names=["a", "b"])
    observed = {}
    def fetch(**options):
        observed["fetch_options"] = options
        return source

    monkeypatch.setattr(helpers, "fetch_20newsgroups", fetch)

    def encode(selected_texts, ids, directory, **options):
        observed.update(texts=selected_texts, ids=ids.copy(), options=options)
        semantic = np.random.default_rng(42).normal(size=(len(ids), 4))
        semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
        observed["semantic"] = semantic
        return semantic, {"settings": {"model": options["model_name"]},
                          "cache_file": "cache.npz", "device": "cpu"}

    monkeypatch.setattr(helpers, "sentence_embeddings", encode)
    dataset = helpers.load_newsgroups(tmp_path, sample_size=6)
    np.testing.assert_array_equal(dataset.sample_ids, [0, 2, 3, 4, 5, 6])
    assert observed["texts"] == [texts[i] for i in dataset.sample_ids]
    assert observed["fetch_options"]["subset"] == "train"
    assert observed["fetch_options"]["shuffle"] is False
    assert observed["fetch_options"]["remove"] == ("headers", "footers", "quotes")
    np.testing.assert_array_equal(dataset.labels, labels[dataset.sample_ids])
    np.testing.assert_array_equal(dataset.extra_data.sample_id.to_numpy(), dataset.sample_ids)
    assert "the" in observed["texts"]  # Semantic embeddings do not filter stop words.
    assert observed["options"]["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    np.testing.assert_allclose(pdist(dataset.features), pdist(observed["semantic"]), atol=1e-12)
    assert dataset.preprocessing["text_representation"] == "mpnet"
    assert dataset.preprocessing["pca_components"] == 4
