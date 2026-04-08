"""Metrics utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for class_idx in range(num_classes):
        mask = y_true == class_idx
        result[class_idx] = float(np.mean(y_pred[mask] == y_true[mask])) if np.any(mask) else 0.0
    return result


def confusion_matrix_array(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
