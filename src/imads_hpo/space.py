"""Search space primitives for hyperparameter optimization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Real:
    """Continuous hyperparameter in [low, high]."""
    low: float
    high: float

    def __post_init__(self):
        assert self.low < self.high, f"Real: low ({self.low}) must be < high ({self.high})"


@dataclass(frozen=True)
class LogReal:
    """Log-scale continuous hyperparameter in [low, high] (both > 0)."""
    low: float
    high: float

    def __post_init__(self):
        assert 0 < self.low < self.high, f"LogReal: need 0 < low < high, got ({self.low}, {self.high})"


@dataclass(frozen=True)
class Integer:
    """Integer hyperparameter in [low, high] with optional step."""
    low: int
    high: int
    step: int = 1

    def __post_init__(self):
        assert self.low < self.high, f"Integer: low ({self.low}) must be < high ({self.high})"
        assert self.step >= 1, f"Integer: step must be >= 1, got {self.step}"


@dataclass(frozen=True)
class Categorical:
    """Categorical hyperparameter with a fixed set of choices.

    Encoded as int64 indices: 0, 1, ..., len(choices)-1.
    """
    choices: Sequence[Any]

    def __post_init__(self):
        assert len(self.choices) >= 2, f"Categorical: need >= 2 choices, got {len(self.choices)}"


Dimension = Real | LogReal | Integer | Categorical


class Space:
    """Search space: a named collection of hyperparameter dimensions.

    Example::

        space = Space({
            "lr":         LogReal(1e-5, 1e-1),
            "batch_size": Integer(16, 256, step=16),
            "optimizer":  Categorical(["adam", "sgd", "adamw"]),
        })
    """

    def __init__(self, dimensions: dict[str, Dimension]):
        self._dims = dict(dimensions)
        self._names = list(self._dims.keys())

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def dimensions(self) -> dict[str, Dimension]:
        return dict(self._dims)

    def __len__(self) -> int:
        return len(self._dims)

    def __getitem__(self, name: str) -> Dimension:
        return self._dims[name]

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._dims.items())
        return f"Space({{{items}}})"


__all__ = [
    "Categorical",
    "Dimension",
    "Integer",
    "LogReal",
    "Real",
    "Space",
]
