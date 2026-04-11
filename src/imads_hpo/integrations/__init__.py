"""Dashboard integration protocols and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from imads_hpo.fp import RunLog


class DashboardSink(Protocol):
    """Protocol for dashboard/experiment tracker integration."""

    def on_trial_start(self, trial_id: str, params: dict) -> None: ...
    def on_trial_metric(self, trial_id: str, epoch: int, metrics: Mapping[str, float]) -> None: ...
    def on_trial_end(self, trial_id: str, result: float, constraints: list[float]) -> None: ...
    def on_study_end(self, best_params: dict, best_value: float) -> None: ...
    def flush(self) -> None: ...


class RunLogAdapter:
    """Route RunLog events to a DashboardSink."""

    def __init__(self, sink: DashboardSink) -> None:
        self._sink = sink

    def consume(self, trial_id: str, log: RunLog) -> None:
        for event in log.events:
            evt_type = event.get("event", "")
            if evt_type == "objective_evaluated":
                self._sink.on_trial_end(
                    trial_id,
                    event.get("value", float("inf")),
                    list(event.get("constraints", [])),
                )
            elif evt_type == "early_stopped":
                self._sink.on_trial_metric(
                    trial_id,
                    int(event.get("epoch", 0)),
                    {"value": event.get("metric", float("inf"))},
                )


__all__ = ["DashboardSink", "RunLogAdapter"]
