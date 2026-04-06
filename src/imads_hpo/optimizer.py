"""Top-level HPO API: minimize()."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import imads

from imads_hpo.checkpoint import CheckpointManager
from imads_hpo.encoding import SpaceEncoder
from imads_hpo.evaluator import HpoEvaluator
from imads_hpo.fidelity import EpochFidelity
from imads_hpo.objective import WrappedObjective
from imads_hpo.result import Result


def minimize(
    objective: WrappedObjective,
    *,
    preset: str = "balanced",
    max_evals: int = 100,
    workers: int = 1,
    fidelity: EpochFidelity | None = None,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> Result:
    """Run hyperparameter optimization.

    Args:
        objective: A ``WrappedObjective`` created by ``@hpo.objective(...)``.
        preset: IMADS preset name ("balanced", "conservative", "throughput").
        max_evals: Approximate evaluation budget.
        workers: Number of parallel workers (1 = single-threaded).
        fidelity: Epoch-based fidelity configuration. If None, single-fidelity.
        seed: Master random seed for deterministic execution.
        checkpoint_dir: Directory for training checkpoints. If None, uses tempdir.

    Returns:
        A ``Result`` with the best parameters and trial history.
    """

    # Set up fidelity
    if fidelity is not None:
        objective.fidelity_config = fidelity

    # Set up checkpoint manager
    if checkpoint_dir is None:
        checkpoint_dir = Path(tempfile.mkdtemp(prefix="imads_hpo_"))
    objective.checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Create IMADS engine
    cfg = imads.EngineConfig.from_preset(preset)
    env = imads.Env(
        run_id=seed,
        config_hash=hash(preset) & 0xFFFFFFFFFFFFFFFF,
        data_snapshot_id=0,
        rng_master_seed=seed,
    )

    evaluator = HpoEvaluator(wrapped=objective, master_seed=seed)
    engine = imads.Engine()

    output = engine.run(
        cfg, env,
        workers=workers,
        evaluator=evaluator,
        num_constraints=objective.num_constraints,
    )

    # Decode result
    encoder = objective.encoder
    best_params = None
    best_mesh = None
    if output.x_best is not None:
        best_mesh = list(output.x_best)
        best_params = encoder.decode(best_mesh)

    return Result(
        best_params=best_params,
        best_value=output.f_best,
        best_mesh_coords=best_mesh,
        stats={
            "truth_evals": output.truth_evals,
            "partial_steps": output.partial_steps,
            "cheap_rejects": output.cheap_rejects,
            "invalid_eval_rejects": output.invalid_eval_rejects,
        },
    )


__all__ = ["minimize"]
