"""MLP model definitions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cvhw1.core.base import Module
from cvhw1.nn import Linear, build_activation


class MLPClassifier(Module):
    """Three-layer MLP with two hidden nonlinear stages."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | Sequence[int],
        output_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim, hidden_dim]
        else:
            hidden_dims = list(hidden_dim)
            if len(hidden_dims) != 2:
                raise ValueError("hidden_dim sequence must contain exactly two hidden sizes")
        self.layer1 = Linear(input_dim, hidden_dims[0])
        self.act1 = build_activation(activation)
        self.layer2 = Linear(hidden_dims[0], hidden_dims[1])
        self.act2 = build_activation(activation)
        self.layer3 = Linear(hidden_dims[1], output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return self.layer3(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_output = self.layer3.backward(grad_output)
        grad_output = self.act2.backward(grad_output)
        grad_output = self.layer2.backward(grad_output)
        grad_output = self.act1.backward(grad_output)
        return self.layer1.backward(grad_output)

    def extract_features(self, x: np.ndarray) -> np.ndarray:
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return x
