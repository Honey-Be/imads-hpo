"""Constraint helpers for resource-aware HPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GpuMemoryConstraint:
    """Constraint: peak GPU memory <= max_gb."""
    max_gb: float

    def __call__(self, actual_gb: float) -> float:
        return actual_gb - self.max_gb  # <= 0 means feasible


@dataclass(frozen=True)
class LatencyConstraint:
    """Constraint: inference latency <= max_ms."""
    max_ms: float

    def __call__(self, actual_ms: float) -> float:
        return actual_ms - self.max_ms


@dataclass(frozen=True)
class ModelSizeConstraint:
    """Constraint: model parameter count <= max_params."""
    max_params: int

    def __call__(self, actual_params: int) -> float:
        return float(actual_params - self.max_params)


__all__ = ["GpuMemoryConstraint", "LatencyConstraint", "ModelSizeConstraint"]
