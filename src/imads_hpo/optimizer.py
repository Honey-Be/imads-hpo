"""Top-level HPO API: minimize()."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional

import imads

from imads_hpo.checkpoint import CheckpointManager
from imads_hpo.encoding import SpaceEncoder
from imads_hpo.evaluator import HpoEvaluator
from imads_hpo.fidelity import FidelitySchedule
from imads_hpo.objective import WrappedObjective
from imads_hpo.result import Result, TrialRecord


def minimize(
    objective: WrappedObjective,
    *,
    preset: str = "balanced",
    max_evals: int = 100,
    workers: int = 1,
    fidelity: FidelitySchedule | None = None,
    seed: int = 42,
    checkpoint_dir: str | Path | None = None,
) -> Result:
    """Run hyperparameter optimization.

    Args:
        objective: A ``WrappedObjective`` created by ``@hpo.objective(...)``.
        preset: IMADS preset name ("balanced", "conservative", "throughput").
        max_evals: Approximate evaluation budget.
        workers: Number of parallel workers (1 = single-threaded).
        fidelity: Fidelity schedule configuration. If None, single-fidelity.
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
    # Propagate the user-facing eval budget into the engine config.
    # In v1.0.1 ``max_evals`` was a dead parameter — the preset's default
    # ``max_iters`` was always used. v1.0.2 forwards it through the new
    # ``EngineConfig.max_iters`` setter. Since one engine iteration may
    # produce multiple truth evaluations, this value is an upper bound on
    # iterations rather than a strict eval count.
    if max_evals is not None and max_evals > 0:
        cfg.max_iters = int(max_evals)

    env = imads.Env(
        run_id=seed,
        config_hash=hash(preset) & 0xFFFFFFFFFFFFFFFF,
        data_snapshot_id=0,
        rng_master_seed=seed,
    )

    # Collect per-trial records via the HpoEvaluator sink. The IMADS
    # engine core's ``EngineOutput`` does not carry per-trial history
    # (it only reports ``x_best`` / ``f_best`` / ``stats``), so we derive
    # the ``Result.trials`` list from the evaluator side — every
    # ``mc_sample`` call corresponds to one engine evaluation.
    trials: List[TrialRecord] = []

    evaluator = HpoEvaluator(
        wrapped=objective,
        master_seed=seed,
        trial_sink=trials,
    )
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

    # best_value: SO → float (output.f_best), MO → list[float] (output.f_best_all).
    n_obj = objective.num_objectives
    if n_obj > 1:
        f_best_all = getattr(output, "f_best_all", None)
        best_value: Any = list(f_best_all) if f_best_all is not None else None
    else:
        best_value = output.f_best

    # Pareto front: non-dominated feasible trials under minimisation.
    pareto: List[TrialRecord] = []
    if n_obj > 1 and trials:
        pareto = _nondominated(trials)

    return Result(
        best_params=best_params,
        best_value=best_value,
        best_mesh_coords=best_mesh,
        trials=trials,
        pareto_front=pareto,
        stats={
            "truth_evals": output.truth_evals,
            "partial_steps": output.partial_steps,
            "cheap_rejects": output.cheap_rejects,
            "invalid_eval_rejects": output.invalid_eval_rejects,
        },
    )


def _nondominated(trials: List[TrialRecord]) -> List[TrialRecord]:
    """Return the Pareto-optimal subset of ``trials`` under minimisation.

    Only feasible trials participate. A trial ``a`` is dominated by ``b``
    when every objective of ``b`` is ≤ the corresponding objective of
    ``a`` and at least one is strictly less.
    """
    feasible = [t for t in trials if t.feasible]

    def _vec(t: TrialRecord) -> List[float]:
        v = t.value
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return [float(v)]

    kept: List[TrialRecord] = []
    for i, ti in enumerate(feasible):
        vi = _vec(ti)
        dominated = False
        for j, tj in enumerate(feasible):
            if i == j:
                continue
            vj = _vec(tj)
            if len(vi) != len(vj):
                continue
            if all(b <= a for a, b in zip(vi, vj)) and any(
                b < a for a, b in zip(vi, vj)
            ):
                dominated = True
                break
        if not dominated:
            kept.append(ti)
    return kept


__all__ = ["minimize"]
