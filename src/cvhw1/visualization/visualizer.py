"""Report-oriented visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cvhw1.utils.io import load_json
from cvhw1.utils.plotting import (
    save_curve_plot,
    save_error_cases,
    save_heatmap,
    save_scatter_2d,
    save_weight_grid,
)


class Visualizer:
    def __init__(self, class_names: Sequence[str]) -> None:
        self.class_names = list(class_names)

    def plot_history(self, history_path: str | Path, output_dir: str | Path) -> None:
        history = load_json(history_path)
        output_dir = Path(output_dir)
        save_curve_plot(
            {"train_loss": history["train_loss"], "val_loss": history["val_loss"]},
            title="Training and Validation Loss",
            ylabel="Loss",
            path=output_dir / "loss_curves.png",
        )
        save_curve_plot(
            {"val_accuracy": history["val_accuracy"]},
            title="Validation Accuracy",
            ylabel="Accuracy",
            path=output_dir / "val_accuracy_curve.png",
        )
        save_curve_plot(
            {"learning_rate": history["learning_rate"]},
            title="Learning Rate Schedule",
            ylabel="Learning Rate",
            path=output_dir / "learning_rate_curve.png",
        )

    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, output_path: str | Path) -> None:
        save_heatmap(confusion_matrix, self.class_names, "Confusion Matrix", output_path)

    def plot_first_layer_weights(
        self,
        first_layer_weights: np.ndarray,
        output_path: str | Path,
        max_items: int = 64,
    ) -> None:
        weight_maps = first_layer_weights.T.reshape(-1, 28, 28)
        save_weight_grid(weight_maps, output_path, max_items=max_items)

    def plot_error_cases(
        self,
        images: np.ndarray,
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
        output_path: str | Path,
        max_items: int = 16,
    ) -> None:
        save_error_cases(
            images=images,
            true_labels=[self.class_names[idx] for idx in true_labels],
            pred_labels=[self.class_names[idx] for idx in pred_labels],
            path=output_path,
            max_items=max_items,
        )

    def plot_tsne(self, embeddings: np.ndarray, labels: np.ndarray, output_path: str | Path) -> None:
        save_scatter_2d(
            points=embeddings,
            labels=labels,
            class_names=self.class_names,
            title="t-SNE of Hidden Representations",
            path=output_path,
        )
