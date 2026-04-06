"""Objective function decorator and trial execution via Run monad."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

from imads_hpo.checkpoint import CheckpointManager, TrainingState
from imads_hpo.encoding import SpaceEncoder
from imads_hpo.fidelity import EpochFidelity, Fidelity
from imads_hpo.fp import Run, RunLog, ask, get_state, modify_state, put_state, tell
from imads_hpo.repro import (
    DeterminismConfig,
    RngRegistry,
    SeedPath,
    capture_rng_snapshot,
    configure_determinism,
    restore_rng_snapshot,
)
from imads_hpo.space import Space


@dataclass(frozen=True)
class TrialEnv:
    """Read-only environment for a single trial execution."""

    params: dict[str, Any]
    fidelity: Fidelity
    seed_path: SeedPath
    determinism_config: DeterminismConfig
    mesh_coords: list[int]
    checkpoint_mgr: CheckpointManager


@dataclass
class TrialState:
    """Mutable state threaded through Run monad during trial execution."""

    training_state: TrainingState
    rng_registry: RngRegistry


# Type alias for trial Run programs
TrialRun = Run[TrialEnv, TrialState, Any]


def _load_or_init_checkpoint() -> TrialRun:
    """Load best prefix checkpoint or initialize fresh state."""

    def _run(env: TrialEnv, state: TrialState) -> tuple[TrialState, None, RunLog]:
        prefix = env.checkpoint_mgr.load_best_prefix(
            env.mesh_coords, env.fidelity.seed_index, env.fidelity.epochs
        )
        if prefix is not None:
            state.training_state = prefix
            if prefix.rng_snapshot is not None:
                state.rng_registry.restore(prefix.rng_snapshot)
            log = RunLog().append(
                {"event": "checkpoint_loaded", "epoch": prefix.epoch}
            )
        else:
            log = RunLog().append({"event": "fresh_init"})
        return state, None, log

    return Run(_run)


def _save_checkpoint() -> TrialRun:
    """Save current training state as checkpoint."""

    def _run(env: TrialEnv, state: TrialState) -> tuple[TrialState, None, RunLog]:
        state.training_state.rng_snapshot = state.rng_registry.snapshot()
        path = env.checkpoint_mgr.save(
            env.mesh_coords, env.fidelity.seed_index, state.training_state
        )
        log = RunLog().append(
            {"event": "checkpoint_saved", "epoch": state.training_state.epoch, "path": str(path)}
        )
        return state, None, log

    return Run(_run)


ObjectiveFn = Callable[[dict[str, Any], Fidelity], tuple[float, list[float]]]


@dataclass
class WrappedObjective:
    """Wraps a user objective function for use with the IMADS evaluator."""

    fn: ObjectiveFn
    space: Space
    encoder: SpaceEncoder
    fidelity_config: EpochFidelity | None
    num_constraints: int
    checkpoint_mgr: CheckpointManager | None

    def evaluate(
        self,
        mesh_coords: list[int],
        tau: int,
        smc: int,
        k: int,
        master_seed: int,
    ) -> tuple[float, list[float]]:
        """Execute a single trial evaluation (called from IMADS Evaluator)."""

        params = self.encoder.decode(mesh_coords)

        if self.fidelity_config is not None:
            fidelity = self.fidelity_config.resolve_fidelity(tau, smc, k)
        else:
            fidelity = Fidelity(epochs=1, seed_index=k, tau=tau, smc=smc)

        seed_path = SeedPath(master_seed).child("trial", hash(tuple(mesh_coords)), "sample", k)
        det_config = DeterminismConfig(master_seed=seed_path.seed())
        configure_determinism(det_config)

        return self.fn(params, fidelity)


def objective(
    space: Space,
    num_constraints: int = 0,
) -> Callable[[ObjectiveFn], WrappedObjective]:
    """Decorator to wrap a user objective function.

    Example::

        @objective(space, num_constraints=1)
        def train(params, fidelity):
            ...
            return val_loss, [gpu_mem - 8.0]
    """

    def decorator(fn: ObjectiveFn) -> WrappedObjective:
        return WrappedObjective(
            fn=fn,
            space=space,
            encoder=SpaceEncoder(space),
            fidelity_config=None,  # set by minimize()
            num_constraints=num_constraints,
            checkpoint_mgr=None,  # set by minimize()
        )

    return decorator


__all__ = [
    "ObjectiveFn",
    "TrialEnv",
    "TrialRun",
    "TrialState",
    "WrappedObjective",
    "objective",
]
