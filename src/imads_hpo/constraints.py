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


@dataclass(frozen=True)
class ConditionalConstraint:
    """Constraint active only when predicate(params) is True."""

    predicate: Callable[[dict[str, Any]], bool]
    inner: Callable[[float], float]
    inactive_value: float = -1.0  # feasible when inactive

    def __call__(self, actual: float, params: dict[str, Any]) -> float:
        if self.predicate(params):
            return self.inner(actual)
        return self.inactive_value


@dataclass(frozen=True)
class DynamicBoundConstraint:
    """Constraint where the bound depends on hyperparameters."""

    bound_fn: Callable[[dict[str, Any]], float]

    def __call__(self, actual: float, params: dict[str, Any]) -> float:
        return actual - self.bound_fn(params)


@dataclass(frozen=True)
class CompositeConstraint:
    """Max of multiple constraints (all must be feasible)."""

    constraints: tuple[Callable[..., float], ...]

    def __call__(self, actuals: list[float], params: dict[str, Any]) -> float:
        return max(c(a, params) for a, c in zip(actuals, self.constraints))


__all__ = [
    "GpuMemoryConstraint",
    "LatencyConstraint",
    "ModelSizeConstraint",
    "ConditionalConstraint",
    "DynamicBoundConstraint",
    "CompositeConstraint",
]
