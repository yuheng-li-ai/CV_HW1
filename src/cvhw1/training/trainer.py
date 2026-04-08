"""Training loop and checkpoint management."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from cvhw1.data import DatasetSplit
from cvhw1.models import MLPClassifier
from cvhw1.nn import CrossEntropyLoss
from cvhw1.optim import SGD, build_scheduler
from cvhw1.utils.io import ensure_dir, save_json
from cvhw1.utils.metrics import accuracy_score


def iterate_minibatches(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(images))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield images[batch_indices], labels[batch_indices]


@dataclass
class TrainingArtifacts:
    run_dir: Path
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path


class Trainer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.loss_fn = CrossEntropyLoss()

    def fit(self, model: MLPClassifier, train_split: DatasetSplit, val_split: DatasetSplit) -> Dict[str, Any]:
        train_cfg = self.config["train"]
        run_artifacts = self._build_artifact_paths()
        optimizer = SGD(
            model.parameters(),
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        scheduler = build_scheduler(optimizer, train_cfg.get("lr_decay", {"name": "none"}), int(train_cfg["epochs"]))

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        best_val_accuracy = -1.0
        best_epoch = -1
        start_time = time.time()

        for epoch in range(int(train_cfg["epochs"])):
            model.train()
            train_losses = []
            progress = tqdm(
                iterate_minibatches(
                    train_split.images,
                    train_split.labels,
                    int(train_cfg["batch_size"]),
                    shuffle=True,
                    seed=int(self.config["seed"]) + epoch,
                ),
                total=int(np.ceil(len(train_split.images) / int(train_cfg["batch_size"]))),
                desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}",
                leave=False,
                disable=self._disable_progress(),
            )
            for batch_images, batch_labels in progress:
                logits = model(batch_images)
                data_loss = self.loss_fn.forward(logits, batch_labels)
                l2_loss = self._l2_penalty(model, optimizer.weight_decay)
                loss = data_loss + l2_loss

                optimizer.zero_grad()
                model.backward(self.loss_fn.backward())
                optimizer.step()

                train_losses.append(loss)
                progress.set_postfix(loss=f"{loss:.4f}", lr=f"{optimizer.learning_rate:.5f}")

            val_metrics = self.evaluate_split(model, val_split)
            current_lr = scheduler.step()
            history["train_loss"].append(float(np.mean(train_losses)))
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["learning_rate"].append(current_lr)

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = epoch + 1
                self.save_checkpoint(model, run_artifacts.checkpoint_path, best_epoch, best_val_accuracy)

        metrics = {
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "training_seconds": round(time.time() - start_time, 2),
        }
        save_json(history, run_artifacts.history_path)
        save_json(metrics, run_artifacts.metrics_path)
        return {
            "history": history,
            "metrics": metrics,
            "artifacts": run_artifacts,
        }

    def evaluate_split(self, model: MLPClassifier, split: DatasetSplit) -> Dict[str, float]:
        model.eval()
        logits = model(split.images)
        loss = self.loss_fn.forward(logits, split.labels)
        preds = np.argmax(logits, axis=1)
        return {"loss": loss, "accuracy": accuracy_score(split.labels, preds)}

    def save_checkpoint(self, model: MLPClassifier, path: Path, epoch: int, val_accuracy: float) -> None:
        state = model.state_dict()
        payload: Dict[str, Any] = {
            **state,
            "__epoch__": np.array([epoch], dtype=np.int64),
            "__val_accuracy__": np.array([val_accuracy], dtype=np.float32),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    def _build_artifact_paths(self) -> TrainingArtifacts:
        run_dir = ensure_dir(Path(self.config["logging"]["output_dir"]) / self.config["logging"]["run_name"])
        checkpoint_dir = ensure_dir(self.config["checkpoint"]["save_dir"])
        return TrainingArtifacts(
            run_dir=run_dir,
            checkpoint_path=checkpoint_dir / "best_model.npz",
            history_path=run_dir / "history.json",
            metrics_path=run_dir / "train_metrics.json",
        )

    @staticmethod
    def _l2_penalty(model: MLPClassifier, weight_decay: float) -> float:
        if weight_decay <= 0:
            return 0.0
        penalty = 0.0
        for parameter in model.parameters():
            penalty += float(np.sum(parameter.data ** 2))
        return 0.5 * weight_decay * penalty

    @staticmethod
    def _disable_progress() -> bool:
        env_value = os.environ.get("CVHW1_DISABLE_TQDM", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        return not sys.stderr.isatty()
