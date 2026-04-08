"""Neural-network layers and activations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cvhw1.core.base import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        bias = np.zeros(out_features, dtype=np.float32)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError("Linear.backward called before forward")
        batch_size = self._input.shape[0]
        self.weight.grad += (self._input.T @ grad_output) / batch_size
        self.bias.grad += grad_output.mean(axis=0)
        return grad_output @ self.weight.data.T


class Activation(Module):
    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        grad_fn: Callable[[np.ndarray], np.ndarray],
        name: str,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.grad_fn = grad_fn
        self.name = name
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return self.fn(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input is None:
            raise RuntimeError(f"{self.name}.backward called before forward")
        return grad_output * self.grad_fn(self._input)


def relu() -> Activation:
    return Activation(
        fn=lambda x: np.maximum(0.0, x),
        grad_fn=lambda x: (x > 0).astype(np.float32),
        name="relu",
    )


def sigmoid() -> Activation:
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    return Activation(
        fn=_sigmoid,
        grad_fn=lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)),
        name="sigmoid",
    )


def tanh() -> Activation:
    return Activation(
        fn=np.tanh,
        grad_fn=lambda x: 1.0 - np.tanh(x) ** 2,
        name="tanh",
    )


def leaky_relu(negative_slope: float = 0.01) -> Activation:
    return Activation(
        fn=lambda x: np.where(x > 0.0, x, negative_slope * x),
        grad_fn=lambda x: np.where(x > 0.0, 1.0, negative_slope).astype(np.float32),
        name="leaky_relu",
    )


def elu(alpha: float = 1.0) -> Activation:
    return Activation(
        fn=lambda x: np.where(x > 0.0, x, alpha * (np.exp(np.clip(x, -50, 50)) - 1.0)),
        grad_fn=lambda x: np.where(x > 0.0, 1.0, alpha * np.exp(np.clip(x, -50, 50))).astype(np.float32),
        name="elu",
    )


def softplus() -> Activation:
    return Activation(
        fn=lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0),
        grad_fn=lambda x: (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32),
        name="softplus",
    )


def swish() -> Activation:
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    return Activation(
        fn=lambda x: x * _sigmoid(x),
        grad_fn=lambda x: (_sigmoid(x) + x * _sigmoid(x) * (1.0 - _sigmoid(x))).astype(np.float32),
        name="swish",
    )


ACTIVATION_FACTORIES = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "softplus": softplus,
    "swish": swish,
}


def build_activation(name: str) -> Activation:
    if name not in ACTIVATION_FACTORIES:
        available = ", ".join(sorted(ACTIVATION_FACTORIES))
        raise ValueError(f"Unsupported activation '{name}'. Available: {available}")
    return ACTIVATION_FACTORIES[name]()
