"""Fashion-MNIST dataset download and preprocessing."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FASHION_MNIST_URLS = {
    "train_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


@dataclass
class DatasetSplit:
    images: np.ndarray
    labels: np.ndarray


@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    class_names: list[str]


def _download_if_missing(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    urllib.request.urlretrieve(url, destination)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, num_images, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(num_images, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, num_items = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(num_items)


def _split_train_val(
    images: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[DatasetSplit, DatasetSplit]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    val_size = int(len(images) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return (
        DatasetSplit(images[train_indices], labels[train_indices]),
        DatasetSplit(images[val_indices], labels[val_indices]),
    )


def _preprocess_images(images: np.ndarray, normalize: bool) -> np.ndarray:
    flattened = images.reshape(images.shape[0], -1).astype(np.float32)
    if normalize:
        flattened /= 255.0
    return flattened


def load_fashion_mnist(data_dir: str | Path, val_ratio: float = 0.1, normalize: bool = True, seed: int = 42) -> DatasetBundle:
    data_dir = Path(data_dir)
    for key, url in FASHION_MNIST_URLS.items():
        _download_if_missing(url, data_dir / f"{key}.gz")

    train_images = _read_idx_images(data_dir / "train_images.gz")
    train_labels = _read_idx_labels(data_dir / "train_labels.gz")
    test_images = _read_idx_images(data_dir / "test_images.gz")
    test_labels = _read_idx_labels(data_dir / "test_labels.gz")

    train_split, val_split = _split_train_val(train_images, train_labels, val_ratio, seed)
    return DatasetBundle(
        train=DatasetSplit(_preprocess_images(train_split.images, normalize), train_split.labels.astype(np.int64)),
        val=DatasetSplit(_preprocess_images(val_split.images, normalize), val_split.labels.astype(np.int64)),
        test=DatasetSplit(_preprocess_images(test_images, normalize), test_labels.astype(np.int64)),
        class_names=list(CLASS_NAMES),
    )
