"""Objective function decorator and trial execution via Run monad."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from imads_hpo.sink import NullSink, RunLogSink
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


def _configure_determinism() -> TrialRun:
    """Configure determinism settings based on the trial environment."""

    def _run(env: TrialEnv, state: TrialState) -> tuple[TrialState, dict[str, Any], RunLog]:
        report = configure_determinism(env.determinism_config)
        log = RunLog().append({"event": "determinism_configured", **report})
        return state, report, log

    return Run(_run)


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


def _execute_objective(fn: ObjectiveFn) -> TrialRun:
    """Execute the user's objective function."""

    def _run(env: TrialEnv, state: TrialState) -> tuple[TrialState, tuple[float, list[float]], RunLog]:
        result = fn(env.params, env.fidelity)
        log = RunLog().append({
            "event": "objective_evaluated",
            "value": result[0],
            "constraints": result[1],
            "epochs": env.fidelity.epochs,
        })
        return state, result, log

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


def build_trial_program(fn: ObjectiveFn, checkpoint_enabled: bool = True) -> TrialRun:
    """Build the full trial execution pipeline as a Run monad program.

    Steps:
    1. Configure determinism
    2. Load checkpoint (if enabled)
    3. Execute objective function
    4. Save checkpoint (if enabled)
    """
    pipeline = _configure_determinism().then(_execute_objective(fn))

    if checkpoint_enabled:
        pipeline = (
            _configure_determinism()
            .then(_load_or_init_checkpoint())
            .then(_execute_objective(fn))
            .bind(lambda result: _save_checkpoint().map(lambda _: result))
        )

    return pipeline


ObjectiveFn = Callable[[dict[str, Any], Fidelity], tuple[float, list[float]]]


@dataclass
class WrappedObjective:
    """Wraps a user objective function for use with the IMADS evaluator."""

    fn: ObjectiveFn
    space: Space
    encoder: SpaceEncoder
    fidelity_config: EpochFidelity | None
    num_constraints: int
    num_objectives: int = 1
    checkpoint_mgr: CheckpointManager | None = None
    log_sink: RunLogSink = field(default_factory=NullSink)

    def evaluate(
        self,
        mesh_coords: list[int],
        tau: int,
        smc: int,
        k: int,
        master_seed: int,
    ) -> tuple[float, list[float]]:
        """Execute a single trial evaluation (called from IMADS Evaluator).

        Uses the Run monad pipeline for structured, deterministic execution
        with automatic event logging.
        """

        params = self.encoder.decode(mesh_coords)

        if self.fidelity_config is not None:
            fidelity = self.fidelity_config.resolve_fidelity(tau, smc, k)
        else:
            fidelity = Fidelity(epochs=1, seed_index=k, tau=tau, smc=smc)

        seed_path = SeedPath(master_seed).child("trial", hash(tuple(mesh_coords)), "sample", k)
        det_config = DeterminismConfig(master_seed=seed_path.seed())

        # If no checkpoint manager, create a minimal one and skip checkpointing.
        checkpoint_enabled = self.checkpoint_mgr is not None
        if self.checkpoint_mgr is None:
            import tempfile
            from imads_hpo.checkpoint import CheckpointManager
            checkpoint_mgr = CheckpointManager(tempfile.mkdtemp(prefix="imads_hpo_trial_"))
        else:
            checkpoint_mgr = self.checkpoint_mgr

        # Build and execute the trial pipeline.
        trial_env = TrialEnv(
            params=params,
            fidelity=fidelity,
            seed_path=seed_path,
            determinism_config=det_config,
            mesh_coords=mesh_coords,
            checkpoint_mgr=checkpoint_mgr,
        )
        trial_state = TrialState(
            training_state=TrainingState(),
            rng_registry=RngRegistry(),
        )

        pipeline = build_trial_program(self.fn, checkpoint_enabled=checkpoint_enabled)
        _, result, log = pipeline.execute(trial_env, trial_state)

        # Forward log events to sink.
        trial_id = f"x={mesh_coords}_k={k}"
        self.log_sink.consume(trial_id, log)

        return result


def objective(
    space: Space,
    num_constraints: int = 0,
    num_objectives: int = 1,
) -> Callable[[ObjectiveFn], WrappedObjective]:
    """Decorator to wrap a user objective function.

    Example::

        @objective(space, num_constraints=1)
        def train(params, fidelity):
            ...
            return val_loss, [gpu_mem - 8.0]

        # Multi-objective:
        @objective(space, num_objectives=2, num_constraints=1)
        def train_multi(params, fidelity):
            ...
            return (val_loss, model_size), [gpu_mem - 8.0]
    """

    def decorator(fn: ObjectiveFn) -> WrappedObjective:
        return WrappedObjective(
            fn=fn,
            space=space,
            encoder=SpaceEncoder(space),
            fidelity_config=None,  # set by minimize()
            num_constraints=num_constraints,
            num_objectives=num_objectives,
        )

    return decorator


__all__ = [
    "ObjectiveFn",
    "TrialEnv",
    "TrialRun",
    "TrialState",
    "WrappedObjective",
    "build_trial_program",
    "objective",
]
