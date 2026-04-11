"""RunLog sink protocols and implementations."""

from __future__ import annotations

from typing import Protocol

from imads_hpo.fp import RunLog


class RunLogSink(Protocol):
    """Protocol for consuming RunLog events from trial execution."""

    def consume(self, trial_id: str, log: RunLog) -> None: ...


class NullSink:
    """Discards all log events."""

    def consume(self, trial_id: str, log: RunLog) -> None:
        pass


class ListSink:
    """Accumulates log events into a list (useful for testing)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, RunLog]] = []

    def consume(self, trial_id: str, log: RunLog) -> None:
        self.entries.append((trial_id, log))

    @property
    def all_events(self) -> list[dict]:
        result = []
        for _, log in self.entries:
            for event in log.events:
                result.append(dict(event))
        return result


__all__ = ["ListSink", "NullSink", "RunLogSink"]
