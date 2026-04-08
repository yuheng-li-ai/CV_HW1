"""Core abstractions for trainable modules and parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator

import numpy as np


@dataclass
class Parameter:
    """A trainable NumPy parameter with an attached gradient buffer."""

    data: np.ndarray
    grad: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float32)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    """Minimal module base class for manual neural-network implementations."""

    def __init__(self) -> None:
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def train(self) -> None:
        self.training = True
        for module in self.children():
            module.train()

    def eval(self) -> None:
        self.training = False
        for module in self.children():
            module.eval()

    def children(self) -> Iterator["Module"]:
        for value in self.__dict__.values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield item

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in self.__dict__.items():
            full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            if isinstance(value, Parameter):
                yield full_name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(full_name)
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, Parameter):
                        yield f"{full_name}.{idx}", item
                    elif isinstance(item, Module):
                        yield from item.named_parameters(f"{full_name}.{idx}")

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter.data.copy()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        for name, parameter in self.named_parameters():
            if name not in state_dict:
                raise KeyError(f"Missing parameter '{name}' in state_dict")
            parameter.data[...] = state_dict[name]
