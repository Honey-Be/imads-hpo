"""Early stopping rules for HPO trials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class EarlyStoppingRule(Protocol):
    """Protocol for early stopping decisions during training."""

    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool:
        """Return True if training should be stopped early."""
        ...

    def reset(self) -> None:
        """Reset state for a new trial."""
        ...


@dataclass
class PatientStopper:
    """Stop if no improvement for `patience` epochs."""

    patience: int = 10
    min_delta: float = 1e-4
    best: float = field(default=float("inf"), init=False)
    wait: int = field(default=0, init=False)

    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def reset(self) -> None:
        self.best = float("inf")
        self.wait = 0


@dataclass
class MedianStopper:
    """Stop if current trial is worse than median of completed trials at same epoch."""

    completed_curves: list[list[float]] = field(default_factory=list)
    min_epochs: int = 5

    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool:
        if epoch < self.min_epochs:
            return False
        # Collect values at this epoch from completed curves.
        values_at_epoch = [
            curve[epoch] for curve in self.completed_curves if len(curve) > epoch
        ]
        if len(values_at_epoch) < 2:
            return False
        median = sorted(values_at_epoch)[len(values_at_epoch) // 2]
        return metric > median

    def reset(self) -> None:
        pass


@dataclass
class ThresholdStopper:
    """Stop if metric exceeds a fixed threshold."""

    threshold: float
    min_epochs: int = 1

    def should_stop(self, epoch: int, metric: float, history: list[float]) -> bool:
        if epoch < self.min_epochs:
            return False
        return metric > self.threshold

    def reset(self) -> None:
        pass


__all__ = [
    "EarlyStoppingRule",
    "MedianStopper",
    "PatientStopper",
    "ThresholdStopper",
]
