"""Embedding cache identity and resume behavior, without downloading a model."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import text_embeddings as helper


@pytest.fixture
def fake_model(monkeypatch):
    calls = []

    class Model:
        max_seq_length = 384
        device = "cpu"

        def encode(self, texts, **options):
            calls.append((texts, options))
            values = np.random.default_rng(42).normal(size=(len(texts), 4))
            return values / np.linalg.norm(values, axis=1, keepdims=True)

    monkeypatch.setattr(helper, "version", lambda name: "test-version")
    monkeypatch.setattr(helper, "_load_sentence_model", lambda *args: Model())
    return calls


def test_embedding_cache_reuses_vectors_and_original_row_order(fake_model, monkeypatch, tmp_path):
    texts, ids = ["document one", "document two"], [17, 9]
    first, report = helper.sentence_embeddings(texts, ids, tmp_path)
    assert not report["cached"]
    assert fake_model[0][1]["normalize_embeddings"] is True

    def no_model(*args):
        raise AssertionError("A cache hit must not load the model")

    monkeypatch.setattr(helper, "_load_sentence_model", no_model)
    second, cached = helper.sentence_embeddings(texts, ids, tmp_path)
    np.testing.assert_array_equal(second, first)
    assert cached["cached"]
    with np.load(cached["cache_file"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_ids"], ids)


@pytest.mark.parametrize("change", ["text", "ids", "revision", "tokens", "version"])
def test_embedding_cache_invalidates_changed_inputs(change, fake_model, monkeypatch, tmp_path):
    texts, ids = ["document one", "document two"], [17, 9]
    _, first = helper.sentence_embeddings(texts, ids, tmp_path)
    options = {}
    if change == "text":
        texts = ["different words", texts[1]]
    elif change == "ids":
        ids = ids[::-1]
    elif change == "revision":
        options["revision"] = "a" * 40
    elif change == "tokens":
        options["max_seq_length"] = 128
    else:
        monkeypatch.setattr(helper, "version", lambda name: "new-version")
    _, second = helper.sentence_embeddings(texts, ids, tmp_path, **options)
    assert first["cache_file"] != second["cache_file"]
    assert len(fake_model) == 2
    assert not second["cached"]


def test_corrupted_nonfinite_cached_vectors_are_rejected(fake_model, tmp_path):
    _, first = helper.sentence_embeddings(["one", "two"], [0, 1], tmp_path)
    path = Path(first["cache_file"])
    with np.load(path, allow_pickle=False) as saved:
        contents = {key: saved[key].copy() for key in saved.files}
    contents["embeddings"][0, 0] = np.nan
    np.savez_compressed(path, **contents)
    with pytest.raises(ValueError, match="finite"):
        helper.sentence_embeddings(["one", "two"], [0, 1], tmp_path)


def test_unpinned_model_revision_and_overlong_context_are_rejected(fake_model, tmp_path):
    with pytest.raises(ValueError, match="commit hash"):
        helper.sentence_embeddings(["one", "two"], [0, 1], tmp_path, revision="main")
    with pytest.raises(ValueError, match="configured limit"):
        helper.sentence_embeddings(["one", "two"], [0, 1], tmp_path, max_seq_length=512)
