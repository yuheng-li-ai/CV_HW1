"""Learning-rate schedulers."""

from __future__ import annotations

import math

from cvhw1.optim.optimizers import SGD


class LRScheduler:
    def __init__(self, optimizer: SGD) -> None:
        self.optimizer = optimizer
        self.epoch = 0

    def step(self) -> float:
        self.epoch += 1
        self.optimizer.learning_rate = self.get_lr()
        return self.optimizer.learning_rate

    def get_lr(self) -> float:
        return self.optimizer.learning_rate


class NoDecayScheduler(LRScheduler):
    pass


class StepLRScheduler(LRScheduler):
    def __init__(self, optimizer: SGD, step_size: int, gamma: float) -> None:
        super().__init__(optimizer)
        self.base_lr = optimizer.learning_rate
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        factor = self.gamma ** (self.epoch // self.step_size)
        return self.base_lr * factor


class ExponentialLRScheduler(LRScheduler):
    def __init__(self, optimizer: SGD, gamma: float) -> None:
        super().__init__(optimizer)
        self.base_lr = optimizer.learning_rate
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * (self.gamma ** self.epoch)


class CosineLRScheduler(LRScheduler):
    def __init__(self, optimizer: SGD, total_epochs: int, min_lr: float = 0.0) -> None:
        super().__init__(optimizer)
        self.base_lr = optimizer.learning_rate
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def get_lr(self) -> float:
        progress = min(self.epoch / max(self.total_epochs, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


def build_scheduler(optimizer: SGD, config: dict, total_epochs: int) -> LRScheduler:
    name = config.get("name", "none")
    if name == "none":
        return NoDecayScheduler(optimizer)
    if name == "step":
        return StepLRScheduler(
            optimizer=optimizer,
            step_size=int(config.get("step_size", 5)),
            gamma=float(config.get("gamma", 0.5)),
        )
    if name == "exponential":
        return ExponentialLRScheduler(
            optimizer=optimizer,
            gamma=float(config.get("gamma", 0.95)),
        )
    if name == "cosine":
        return CosineLRScheduler(
            optimizer=optimizer,
            total_epochs=total_epochs,
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported lr scheduler '{name}'")
