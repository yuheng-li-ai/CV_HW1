"""Loss functions."""

from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._targets: np.ndarray | None = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        batch_indices = np.arange(targets.shape[0])
        loss = -np.log(np.clip(probs[batch_indices, targets], 1e-12, 1.0)).mean()
        self._probs = probs
        self._targets = targets
        return float(loss)

    def backward(self) -> np.ndarray:
        if self._probs is None or self._targets is None:
            raise RuntimeError("CrossEntropyLoss.backward called before forward")
        grad = self._probs.copy()
        grad[np.arange(self._targets.shape[0]), self._targets] -= 1.0
        return grad
