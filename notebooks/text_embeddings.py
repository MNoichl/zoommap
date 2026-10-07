"""Pinned sentence embeddings with a cache tied to document contents and order."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

import numpy as np


MPNET_MODEL = "sentence-transformers/all-mpnet-base-v2"
MPNET_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def _load_sentence_model(model_name, revision, directory, device):
    # Import only on a cache miss: cached vectors do not load model weights.
    os.environ.setdefault("HF_HOME", str(Path(directory).resolve().parent / "huggingface"))
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name, revision=revision, cache_folder=str(directory),
        device=device, trust_remote_code=False,
    )


def _validate_embeddings(embeddings, count):
    if (embeddings.ndim != 2 or embeddings.shape[0] != count
            or embeddings.shape[1] < 2 or not np.isfinite(embeddings).all()):
        raise ValueError("Sentence embeddings must be a finite document-by-feature matrix")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise ValueError("Sentence embeddings must have unit length")


def sentence_embeddings(texts, sample_ids, directory, *, model_name=MPNET_MODEL,
                        revision=MPNET_REVISION, max_seq_length=384,
                        batch_size=32, device=None):
    """Encode full cleaned texts, truncating at the stated model token limit.

    Model weights and normalized Float32 vectors stay in ``directory``. Cache
    identity includes the immutable model revision, token policy, package
    versions, actual text contents, and original document IDs in row order.
    ``device=None`` lets SentenceTransformers choose an available accelerator.
    """
    texts = list(texts)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if not texts or sample_ids.shape != (len(texts),):
        raise ValueError("texts and sample_ids must contain the same nonzero number of rows")
    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Every document must contain nonempty text")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin revision to a full Hugging Face model commit hash")
    for name, value in (("max_seq_length", max_seq_length), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    text_digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        text_digest.update(len(encoded).to_bytes(8, "little"))
        text_digest.update(encoded)
    settings = {
        "model": model_name, "model_revision": revision,
        "max_seq_length": max_seq_length,
        "long_document_policy": "truncate each cleaned document to max_seq_length wordpieces",
        "normalize_embeddings": True, "embedding_dtype": "float32",
        "documents": len(texts), "texts_sha256": text_digest.hexdigest(),
        "sample_ids_sha256": hashlib.sha256(sample_ids.astype("<i8").tobytes()).hexdigest(),
        "package_versions": {name: version(name) for name in
                             ("sentence-transformers", "transformers", "torch")},
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"embeddings-{fingerprint}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["settings"] != settings or not np.array_equal(saved["sample_ids"], sample_ids):
                raise ValueError("Sentence embedding cache does not match its document/model settings")
            embeddings = saved["embeddings"].copy()
        _validate_embeddings(embeddings, len(texts))
        print(f"Sentence embeddings: reused {cache.name}", flush=True)
        return embeddings, dict(metadata, cached=True, cache_file=str(cache))

    print(f"Sentence embeddings: {model_name} ({revision[:12]}), {len(texts):,} documents", flush=True)
    started = time.perf_counter()
    model = _load_sentence_model(model_name, revision, directory / "models", device)
    # Do not silently extend a model beyond its configured context length.
    if max_seq_length > model.max_seq_length:
        raise ValueError(f"max_seq_length exceeds this model's configured limit ({model.max_seq_length})")
    model.max_seq_length = max_seq_length
    print(f"Sentence embeddings: device={model.device}, token limit={max_seq_length}", flush=True)
    embeddings = np.asarray(model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ), dtype=np.float32)
    _validate_embeddings(embeddings, len(texts))
    metadata = {
        "settings": settings, "embedding_dimensions": embeddings.shape[1],
        "device": str(model.device), "batch_size": batch_size,
        "compute_seconds": time.perf_counter() - started,
    }
    temporary = cache.with_suffix(".part")
    with temporary.open("wb") as file:
        np.savez_compressed(file, embeddings=embeddings, sample_ids=sample_ids,
                            metadata=json.dumps(metadata, sort_keys=True))
    temporary.replace(cache)
    return embeddings, dict(metadata, cached=False, cache_file=str(cache))
