"""Callback hooks for HPO run monitoring."""

from __future__ import annotations

from typing import Any, Protocol


class Callback(Protocol):
    """Protocol for HPO callbacks."""

    def on_trial_start(self, trial_id: int, params: dict[str, Any]) -> None: ...
    def on_trial_end(self, trial_id: int, value: float, feasible: bool) -> None: ...
    def on_iteration_end(self, iteration: int, best_value: float | None) -> None: ...


class PrintCallback:
    """Simple callback that prints trial results."""

    def on_trial_start(self, trial_id: int, params: dict[str, Any]) -> None:
        print(f"[Trial {trial_id}] params={params}")

    def on_trial_end(self, trial_id: int, value: float, feasible: bool) -> None:
        status = "feasible" if feasible else "infeasible"
        print(f"[Trial {trial_id}] value={value:.6f} ({status})")

    def on_iteration_end(self, iteration: int, best_value: float | None) -> None:
        print(f"[Iter {iteration}] best={best_value}")


__all__ = ["Callback", "PrintCallback"]
