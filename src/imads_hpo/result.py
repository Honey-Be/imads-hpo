"""Result container for HPO runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrialRecord:
    """Record of a single evaluated trial."""
    params: dict[str, Any]
    value: float
    constraints: list[float]
    feasible: bool
    mesh_coords: list[int]
    truth_eval: bool


@dataclass
class Result:
    """Result of an HPO optimization run."""

    best_params: Optional[dict[str, Any]] = None
    best_value: Optional[float] = None
    best_mesh_coords: Optional[list[int]] = None
    trials: list[TrialRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def feasible_trials(self) -> list[TrialRecord]:
        return [t for t in self.trials if t.feasible]

    def __repr__(self) -> str:
        return (
            f"Result(best_value={self.best_value}, "
            f"best_params={self.best_params}, "
            f"n_trials={self.n_trials})"
        )


__all__ = ["Result", "TrialRecord"]
