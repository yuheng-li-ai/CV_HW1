"""Evaluation helpers for checkpoints and derived artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.manifold import TSNE

from cvhw1.data import DatasetSplit
from cvhw1.models import MLPClassifier
from cvhw1.nn import CrossEntropyLoss
from cvhw1.utils.io import save_json
from cvhw1.utils.metrics import accuracy_score, confusion_matrix_array, per_class_accuracy


@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    predictions: np.ndarray
    confusion_matrix: np.ndarray
    embeddings: np.ndarray | None


class Evaluator:
    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.loss_fn = CrossEntropyLoss()

    def load_checkpoint(self, model: MLPClassifier, checkpoint_path: str | Path) -> None:
        loaded = np.load(Path(checkpoint_path), allow_pickle=True)
        state = {
            key: loaded[key]
            for key in loaded.files
            if not key.startswith("__")
        }
        model.load_state_dict(state)

    def evaluate(
        self,
        model: MLPClassifier,
        split: DatasetSplit,
        output_dir: str | Path | None = None,
        compute_tsne: bool = False,
        tsne_sample_size: int = 2000,
    ) -> EvaluationResult:
        model.eval()
        logits = model(split.images)
        predictions = np.argmax(logits, axis=1)
        loss = self.loss_fn.forward(logits, split.labels)
        confusion = confusion_matrix_array(split.labels, predictions, len(self.class_names))
        metrics = {
            "loss": loss,
            "accuracy": accuracy_score(split.labels, predictions),
            "per_class_accuracy": per_class_accuracy(split.labels, predictions, len(self.class_names)),
        }

        embeddings = None
        if compute_tsne:
            sample_size = min(tsne_sample_size, len(split.images))
            features = model.extract_features(split.images[:sample_size])
            embeddings = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=min(30, max(5, sample_size // 100)),
                random_state=42,
            ).fit_transform(features)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(metrics, output_dir / "evaluation_metrics.json")
            np.save(output_dir / "predictions.npy", predictions)
            np.save(output_dir / "confusion_matrix.npy", confusion)
            if embeddings is not None:
                np.save(output_dir / "tsne_embeddings.npy", embeddings)
                np.save(output_dir / "tsne_labels.npy", split.labels[: len(embeddings)])

        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            confusion_matrix=confusion,
            embeddings=embeddings,
        )

    def collect_error_cases(
        self,
        split: DatasetSplit,
        predictions: np.ndarray,
        max_items: int = 16,
    ) -> Dict[str, np.ndarray]:
        mismatch = np.where(split.labels != predictions)[0][:max_items]
        images = split.images[mismatch].reshape(-1, 28, 28)
        return {
            "images": images,
            "true_labels": split.labels[mismatch],
            "pred_labels": predictions[mismatch],
        }
