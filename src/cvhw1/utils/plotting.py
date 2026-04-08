"""Reusable plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def save_curve_plot(
    series: dict[str, Sequence[float]],
    title: str,
    ylabel: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in series.items():
        plt.plot(range(1, len(values) + 1), values, label=label, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_weight_grid(weights: np.ndarray, path: str | Path, max_items: int = 64) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_items = min(len(weights), max_items)
    cols = 8
    rows = int(np.ceil(num_items / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = np.array(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < num_items:
            ax.imshow(weights[idx], cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_error_cases(
    images: np.ndarray,
    true_labels: Iterable[str],
    pred_labels: Iterable[str],
    path: str | Path,
    max_items: int = 16,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images[:max_items]
    true_labels = list(true_labels)[:max_items]
    pred_labels = list(pred_labels)[:max_items]
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx < len(images):
            ax.imshow(images[idx], cmap="gray")
            ax.set_title(f"T: {true_labels[idx]}\nP: {pred_labels[idx]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter_2d(
    points: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    title: str,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if np.any(mask):
            plt.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.65, label=class_name)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=2, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
