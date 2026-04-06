"""Functional-programming primitives for the HPO runtime.

This module provides two core abstractions:

* ``curry`` — automatic partial application based on ``inspect.signature``.
* ``Run`` — a Reader/Writer/State monad that models HPO trial execution as
  deterministic state transitions with structured event logging.

The design follows the patterns established in
`framesmoothie <https://github.com/Honey-Be/framesmoothie>`_, adapted for
hyperparameter optimization rather than panoptic segmentation training.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import update_wrapper
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

EnvT = TypeVar("EnvT")
StateT = TypeVar("StateT")
A = TypeVar("A")
B = TypeVar("B")


# ---------------------------------------------------------------------------
# RunLog — append-only event log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLog:
    """Immutable, append-only event log carried by ``Run`` programs."""

    events: tuple[Mapping[str, Any], ...] = ()

    def append(self, event: Mapping[str, Any]) -> RunLog:
        return RunLog(self.events + (dict(event),))

    def extend(self, events: Iterable[Mapping[str, Any]]) -> RunLog:
        acc = self
        for event in events:
            acc = acc.append(event)
        return acc

    def __len__(self) -> int:
        return len(self.events)


# ---------------------------------------------------------------------------
# Run monad — Reader/Writer/State
# ---------------------------------------------------------------------------


class Run(Generic[EnvT, StateT, A]):
    """A tiny Reader/Writer/State monad.

    ``Run[Env, State, A]`` wraps a function::

        (env, state) -> (new_state, value, log)

    This lets us describe HPO trial execution as a chain of deterministic
    state transitions with structured logging, without side effects leaking
    into the composition layer.
    """

    def __init__(
        self,
        fn: Callable[[EnvT, StateT], tuple[StateT, A, RunLog]],
    ) -> None:
        self._fn = fn

    def execute(self, env: EnvT, state: StateT) -> tuple[StateT, A, RunLog]:
        return self._fn(env, state)

    __call__ = execute

    # -- Functor --

    def map(self, fn: Callable[[A], B]) -> Run[EnvT, StateT, B]:
        def _mapped(env: EnvT, state: StateT) -> tuple[StateT, B, RunLog]:
            state2, value, log = self.execute(env, state)
            return state2, fn(value), log

        return Run(_mapped)

    # -- Monad --

    def bind(self, fn: Callable[[A], Run[EnvT, StateT, B]]) -> Run[EnvT, StateT, B]:
        def _bound(env: EnvT, state: StateT) -> tuple[StateT, B, RunLog]:
            state2, value, log1 = self.execute(env, state)
            state3, value2, log2 = fn(value).execute(env, state2)
            return state3, value2, RunLog(log1.events + log2.events)

        return Run(_bound)

    def then(self, nxt: Run[EnvT, StateT, B]) -> Run[EnvT, StateT, B]:
        """Sequence: run self, discard its value, then run *nxt*."""
        return self.bind(lambda _: nxt)

    @staticmethod
    def pure(value: A) -> Run[EnvT, StateT, A]:
        """Lift a plain value into Run (no state change, no log)."""
        return Run(lambda _env, state: (state, value, RunLog()))


# ---------------------------------------------------------------------------
# Reader / Writer / State primitives
# ---------------------------------------------------------------------------


def ask() -> Run[EnvT, StateT, EnvT]:
    """Read the environment."""
    return Run(lambda env, state: (state, env, RunLog()))


def asks(fn: Callable[[EnvT], A]) -> Run[EnvT, StateT, A]:
    """Read the environment through a projection."""
    return ask().map(fn)


def get_state() -> Run[EnvT, StateT, StateT]:
    """Read the current state."""
    return Run(lambda _env, state: (state, state, RunLog()))


def put_state(state: StateT) -> Run[EnvT, StateT, None]:
    """Replace the current state."""
    return Run(lambda _env, _state: (state, None, RunLog()))


def modify_state(fn: Callable[[StateT], StateT]) -> Run[EnvT, StateT, None]:
    """Apply a pure function to the current state."""

    def _modify(_env: EnvT, state: StateT) -> tuple[StateT, None, RunLog]:
        return fn(state), None, RunLog()

    return Run(_modify)


def tell(event: Mapping[str, Any]) -> Run[EnvT, StateT, None]:
    """Emit a single log event."""
    return Run(lambda _env, state: (state, None, RunLog((dict(event),))))


def sequence(programs: Iterable[Run[EnvT, StateT, Any]]) -> Run[EnvT, StateT, list[Any]]:
    """Run a sequence of programs, collecting their values into a list."""

    def _run(env: EnvT, state: StateT) -> tuple[StateT, list[Any], RunLog]:
        acc_state = state
        out: list[Any] = []
        log = RunLog()
        for program in programs:
            acc_state, value, step_log = program.execute(env, acc_state)
            out.append(value)
            log = RunLog(log.events + step_log.events)
        return acc_state, out, log

    return Run(_run)


# ---------------------------------------------------------------------------
# curry — automatic partial application
# ---------------------------------------------------------------------------


@dataclass
class Curried:
    """Callable wrapper returned by :func:`curry`.

    Accumulates arguments until every required parameter is provided,
    then executes the wrapped function.
    """

    fn: Callable[..., Any]
    signature: inspect.Signature
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged_args = self.args + args
        merged_kwargs = dict(self.kwargs)
        merged_kwargs.update(kwargs)
        bound = self.signature.bind_partial(*merged_args, **merged_kwargs)
        missing = [
            p
            for p in self.signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.name not in bound.arguments
        ]
        if missing:
            return Curried(self.fn, self.signature, merged_args, merged_kwargs)
        return self.fn(*merged_args, **merged_kwargs)


def curry(fn: Callable[..., A]) -> Callable[..., Any]:
    """Return a curried version of *fn*.

    The returned callable accumulates positional and keyword arguments until
    every required parameter has been provided, then executes *fn*.

    Example::

        @curry
        def add(a, b, c):
            return a + b + c

        add(1)(2)(3)  # => 6
        add(1, 2)(3)  # => 6
    """
    curried = Curried(fn=fn, signature=inspect.signature(fn))
    update_wrapper(curried, fn)
    return curried


__all__ = [
    "Curried",
    "Run",
    "RunLog",
    "ask",
    "asks",
    "curry",
    "get_state",
    "modify_state",
    "put_state",
    "sequence",
    "tell",
]
