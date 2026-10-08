"""Dataset preparation and hover metadata for the general example notebook."""

from dataclasses import dataclass
from pathlib import Path
import base64
import gzip
import hashlib
from html import escape
import io
import struct

import numpy as np
import pandas as pd
import requests
from PIL import Image
from matplotlib import colormaps
from matplotlib.colors import to_hex
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from text_embeddings import MPNET_MODEL, MPNET_REVISION, sentence_embeddings


FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
           "Shirt", "Sneaker", "Bag", "Ankle boot"]
COLORS = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee",
          "#aa3377", "#7755bb", "#dd8844", "#558877", "#777777"]
FASHION_DATA_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
MAMMOTH_REVISION = "d98f5ba768e88d51662406f240e0c47e15c10bb7"
MAMMOTH_REPOSITORY = "https://github.com/MNoichl/UMAP-examples-mammoth"
MAMMOTH_DATA_URL = (
    "https://raw.githubusercontent.com/MNoichl/UMAP-examples-mammoth/"
    + MAMMOTH_REVISION + "/mammoth_a.csv"
)


def load_fashion_mnist(directory, sample_size=5000, seed=42):
    """Download official IDX training data; return a fixed class-balanced subset."""
    if sample_size < 100 or sample_size > 60000 or sample_size % 10:
        raise ValueError("sample_size must be a multiple of 10 between 100 and 60000")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    loaded = []
    for name in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
        target = directory / name
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            with requests.get(FASHION_DATA_URL + name, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                temporary = target.with_suffix(".part")
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(1024 * 1024):
                        file.write(chunk)
                temporary.replace(target)
        loaded.append(gzip.decompress(target.read_bytes()))
    image_magic, count, rows, columns = struct.unpack(">IIII", loaded[0][:16])
    label_magic, label_count = struct.unpack(">II", loaded[1][:8])
    if (image_magic, label_magic, count, label_count, rows, columns) != (2051, 2049, 60000, 60000, 28, 28):
        raise ValueError("Unexpected Fashion-MNIST IDX headers")
    images = np.frombuffer(loaded[0][16:], dtype=np.uint8).reshape(count, rows, columns)
    labels = np.frombuffer(loaded[1][8:], dtype=np.uint8)
    if len(labels) != count:
        raise ValueError("Unexpected Fashion-MNIST label count")
    rng = np.random.default_rng(seed)
    ids = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == category), sample_size // 10, replace=False)
        for category in range(10)
    ]))
    return images[ids].copy(), labels[ids].copy(), ids


def pca_features(images, seed=42):
    pixels = images.reshape(len(images), -1).astype(np.float64) / 255.0
    pca = PCA(n_components=50, svd_solver="randomized", random_state=seed)
    features = pca.fit_transform(pixels)
    return features, pca


IMAGE_TOOLTIP = """<div style="padding:10px;font-family:inherit;text-align:center">
<img src="{thumbnail}" width="84" height="84" style="image-rendering:pixelated;border-radius:5px">
<div style="margin-top:8px;font-weight:600">{class_name}</div>
<div style="font-size:11px;opacity:0.65">{dataset_name} sample {sample_id}</div></div>"""

TEXT_TOOLTIP = """<div style="padding:12px;font-family:inherit;max-width:380px">
<strong>{class_name}</strong><div style="font-size:11px;opacity:0.65;margin:5px 0">
20 Newsgroups document {sample_id}</div>
<div style="font-size:12px;line-height:1.45;white-space:pre-wrap;max-height:260px;overflow:auto">{preview}</div>
</div>"""

POINT_TOOLTIP = """<div style="padding:12px;font-family:inherit">
<strong>{class_name}</strong><div style="font-size:11px;opacity:0.65;margin:5px 0">
Mammoth sample {sample_id}</div>
<div style="font-size:12px">Original 3D coordinates: ({x}, {y}, {z})</div>
<div style="font-size:11px;opacity:0.65;margin-top:5px">Colors mark spatial regions, not anatomical labels.</div>
</div>"""


@dataclass
class ExampleDataset:
    name: str
    title: str
    noun: str
    features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    class_names: list[str]
    colors: list[str]
    extra_data: pd.DataFrame
    tooltip: str
    preprocessing: dict


def balanced_indices(labels, sample_size, seed=42):
    """Select equal class counts; None or the full size retains all observations."""
    labels = np.asarray(labels)
    if sample_size is None or sample_size == len(labels):
        return np.arange(len(labels))
    classes, counts = np.unique(labels, return_counts=True)
    if not len(classes) or not isinstance(sample_size, (int, np.integer)):
        raise ValueError("sample_size must be an integer or None")
    if sample_size <= 0 or sample_size % len(classes):
        raise ValueError(f"sample_size must be a positive multiple of {len(classes)}")
    per_class = sample_size // len(classes)
    if per_class > counts.min():
        raise ValueError(f"At most {int(counts.min() * len(classes))} observations can be sampled with equal class counts")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == label), per_class, replace=False)
        for label in classes
    ]))


def load_mnist(directory, sample_size=5000, seed=42):
    """Load the original 60,000 training digits from Google's MNIST IDX mirror."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = []
    for name in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
        target = directory / name
        if not target.exists():
            print(f"MNIST: downloading {name}", flush=True)
            url = f"https://storage.googleapis.com/cvdf-datasets/mnist/{name}"
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                temporary = target.with_suffix(".part")
                with temporary.open("wb") as file:
                    for chunk in response.iter_content(1024 * 1024):
                        file.write(chunk)
                temporary.replace(target)
        raw.append(gzip.decompress(target.read_bytes()))
    image_header = struct.unpack(">IIII", raw[0][:16])
    label_header = struct.unpack(">II", raw[1][:8])
    if image_header != (2051, 60000, 28, 28) or label_header != (2049, 60000):
        raise ValueError("Unexpected MNIST IDX headers")
    images = np.frombuffer(raw[0][16:], dtype=np.uint8).reshape(60000, 28, 28)
    labels = np.frombuffer(raw[1][8:], dtype=np.uint8)
    if labels.shape != (60000,):
        raise ValueError("Unexpected MNIST label count")
    ids = balanced_indices(labels, sample_size, seed)
    return images[ids].copy(), labels[ids].copy(), ids


def image_metadata(images, labels, sample_ids, class_names, dataset_name):
    thumbnails = []
    for pixels in images:
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG", optimize=True)
        thumbnails.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return pd.DataFrame({
        "thumbnail": thumbnails,
        "sample_id": sample_ids,
        "class_name": [escape(class_names[int(label)]) for label in labels],
        "dataset_name": escape(dataset_name),
    })


def text_metadata(texts, labels, sample_ids, class_names, preview_length=1200):
    # Escaping occurs after truncation so raw document markup is always data.
    return pd.DataFrame({
        "preview": [escape(text[:preview_length] + ("…" if len(text) > preview_length else "")) for text in texts],
        "sample_id": sample_ids,
        "class_name": [escape(class_names[int(label)]) for label in labels],
    })


def load_newsgroups(directory, sample_size=5000, seed=42, *,
                    model_name=MPNET_MODEL, model_revision=MPNET_REVISION,
                    embedding_batch_size=32, embedding_device=None,
                    max_seq_length=384):
    """Cleaned training documents -> unit sentence embeddings -> full PCA rotation.

    Topic labels determine sampling, hover content and colors only.
    """
    source = fetch_20newsgroups(
        data_home=str(directory), subset="train", shuffle=False,
        remove=("headers", "footers", "quotes"),
    )
    usable = np.array([i for i, text in enumerate(source.data) if text.strip()], dtype=int)
    if len(np.unique(source.target[usable])) != len(source.target_names):
        raise ValueError("Some topics have no usable documents after text preprocessing")
    chosen = balanced_indices(source.target[usable], sample_size, seed)
    ids = usable[chosen]
    labels = source.target[ids].copy()
    texts = [source.data[int(index)] for index in ids]
    semantic, metadata = sentence_embeddings(
        texts, ids, Path(directory) / "text-embeddings", model_name=model_name,
        revision=model_revision, batch_size=embedding_batch_size,
        device=embedding_device, max_seq_length=max_seq_length,
    )
    # Fit in Float64 for stable PCA coordinates and optimizer cache identity.
    semantic = semantic.astype(np.float64)
    dimensions = min(semantic.shape)
    if dimensions < 2:
        raise ValueError("At least two usable text dimensions and documents are required")
    # All available components preserve Euclidean distances between normalized
    # semantic vectors, with a genuine PCA global reference in the first two.
    pca = PCA(n_components=dimensions, svd_solver="full")
    features = pca.fit_transform(semantic)
    names = list(source.target_names)
    representation = "MPNet" if model_name == MPNET_MODEL else model_name.rsplit("/", 1)[-1]
    return ExampleDataset(
        "20_newsgroups", f"20 Newsgroups · {representation}", "documents", features, labels, ids,
        names, [to_hex(color) for color in colormaps["tab20"].colors],
        text_metadata(texts, labels, ids, names), TEXT_TOOLTIP,
        {"source": "scikit-learn fetch_20newsgroups", "subset": "train",
         "remove": ["headers", "footers", "quotes"], "original_documents": len(source.data),
         "usable_documents": len(usable), "selected_documents": len(ids),
         "text_representation": "mpnet", "document_filter": "nonempty cleaned text",
         **metadata["settings"], "embedding_dimensions": semantic.shape[1],
         "embedding_cache": metadata["cache_file"], "embedding_device": metadata["device"],
         "pca_components": dimensions, "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
         "pca_policy": "center and rotate all normalized semantic dimensions", "seed": seed},
    )


def load_mammoth(directory, sample_size=5000, seed=42):
    """Sample the original 3D cloud uniformly; use spatial clusters for color only.

    A full, unwhitened PCA rotation preserves original XYZ Euclidean distances
    and supplies the global reference. There are no supervised class labels.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"mammoth-{MAMMOTH_REVISION}.csv"
    if not target.exists():
        print("Mammoth: downloading the original 3D point cloud", flush=True)
        with requests.get(MAMMOTH_DATA_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            temporary = target.with_suffix(".part")
            with temporary.open("wb") as file:
                for chunk in response.iter_content(1024 * 1024):
                    file.write(chunk)
            temporary.replace(target)
    source = pd.read_csv(target)
    if list(source.columns) != ["x", "y", "z"]:
        raise ValueError("Unexpected Mammoth CSV columns; expected x, y, z")
    xyz = source.to_numpy(dtype=np.float64)
    if len(xyz) < 12 or not np.isfinite(xyz).all():
        raise ValueError("Mammoth requires at least 12 finite XYZ observations")
    if sample_size is None:
        sample_size = len(xyz)
    if (not isinstance(sample_size, (int, np.integer))
            or not 12 <= sample_size <= len(xyz)):
        raise ValueError("Mammoth sample_size must be an integer between 12 and the source size, or None")
    ids = (np.arange(len(xyz)) if sample_size == len(xyz) else
           np.sort(np.random.default_rng(seed).choice(len(xyz), sample_size, replace=False)))
    points = xyz[ids]
    pca = PCA(n_components=3, svd_solver="full")
    features = pca.fit_transform(points)
    color_ids = (np.arange(len(points)) if len(points) <= 5000 else
                 np.sort(np.random.default_rng(seed).choice(len(points), 5000, replace=False)))
    color_points = points[color_ids]
    labels = AgglomerativeClustering(n_clusters=12, linkage="ward").fit_predict(color_points)
    # Name regions in original centroid XYZ order, rather than arbitrary cluster IDs.
    centroids = np.array([color_points[labels == i].mean(axis=0) for i in range(12)])
    order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
    region_ids = np.empty(12, dtype=int)
    region_ids[order] = np.arange(12)
    labels = region_ids[labels]
    if len(color_ids) < len(points):
        labels = KNeighborsClassifier(n_neighbors=10).fit(color_points, labels).predict(points)
    names = [f"Spatial region {i + 1}" for i in range(12)]
    colors = list(COLORS) + ["#999933", "#882255"]
    extra_data = pd.DataFrame({
        "sample_id": ids, "class_name": [names[int(label)] for label in labels],
        **{axis: [f"{value:.3f}" for value in points[:, i]]
           for i, axis in enumerate(["x", "y", "z"])},
    })
    return ExampleDataset(
        "mammoth", "Mammoth", "points", features, labels, ids, names, colors,
        extra_data, POINT_TOOLTIP,
        {"source": MAMMOTH_REPOSITORY, "source_revision": MAMMOTH_REVISION,
         "source_url": MAMMOTH_DATA_URL,
         "source_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
         "source_credit": "Smithsonian Institution, Mammuthus primigenius 3D scan",
         "source_model": "https://3d.si.edu/object/3d/mammuthus-primigenius-blumbach:341c96cd-f967-4540-8ed1-d3fc56d31f12",
         "source_citation": "https://doi.org/10.5281/zenodo.17290165",
         "original_points": len(xyz), "selected_points": len(ids),
         "sampling": "uniform without replacement, original CSV row order",
         "pca_components": 3, "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
         "pca_policy": "center and rotate all XYZ dimensions, no whitening or per-axis scaling",
         "color_policy": "12 Ward spatial regions, ordered by original XYZ centroids; display only",
         "color_reference_points": len(color_ids), "color_extension": "10-neighbor vote when sample exceeds 5000",
         "seed": seed},
    )


def load_example_dataset(name, directory, sample_size=5000, seed=42, **text_options):
    directory = Path(directory)
    if name == "mammoth":
        return load_mammoth(directory / name, sample_size, seed)
    if name == "20_newsgroups":
        return load_newsgroups(directory / name, sample_size, seed, **text_options)
    if name == "fashion_mnist":
        images, labels, ids = load_fashion_mnist(
            directory / "fashion-mnist", 60000 if sample_size is None else sample_size, seed,
        )
        title, names = "Fashion-MNIST", FASHION_CLASSES
    elif name == "mnist":
        images, labels, ids = load_mnist(directory / name, sample_size, seed)
        title, names = "MNIST", [f"Digit {digit}" for digit in range(10)]
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    features, pca = pca_features(images, seed)
    return ExampleDataset(
        name, title, "images", features, labels, ids, list(names), list(COLORS),
        image_metadata(images, labels, ids, names, title), IMAGE_TOOLTIP,
        {"subset": "train", "pixel_scale": "divide by 255", "pca_components": 50,
         "pca_explained_variance": float(pca.explained_variance_ratio_.sum()), "seed": seed},
    )
