"""Optimization algorithms."""

from __future__ import annotations

from typing import Iterable

from cvhw1.core.base import Parameter


class SGD:
    def __init__(self, parameters: Iterable[Parameter], learning_rate: float, weight_decay: float = 0.0) -> None:
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        for parameter in self.parameters:
            update = parameter.grad + self.weight_decay * parameter.data
            parameter.data -= self.learning_rate * update
