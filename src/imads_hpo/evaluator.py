"""IMADS Evaluator implementation that bridges to HPO objectives."""

from __future__ import annotations

from typing import Any, List, Optional

from imads_hpo.encoding import SpaceEncoder
from imads_hpo.objective import WrappedObjective


class HpoEvaluator:
    """Implements the ``imads.Evaluator`` protocol for HPO.

    This is the bridge between the IMADS engine's ``mc_sample`` interface
    and the user's PyTorch training function.

    Parameters
    ----------
    wrapped:
        The ``WrappedObjective`` produced by ``@hpo.objective``.
    master_seed:
        Master RNG seed for deterministic execution.
    trial_sink:
        Optional list that receives a :class:`~imads_hpo.result.TrialRecord`
        per ``mc_sample`` invocation. Used by ``imads_hpo.minimize`` to
        populate ``Result.trials`` (and, for multi-objective runs, to
        derive ``Result.pareto_front``). The engine core's
        ``EngineOutput`` does not carry per-trial history, so the history
        is captured here instead. Single-objective values are stored as a
        ``float``; multi-objective values as ``list[float]``.
    """

    def __init__(
        self,
        wrapped: WrappedObjective,
        master_seed: int,
        trial_sink: Optional[List[Any]] = None,
    ):
        self._wrapped = wrapped
        self._master_seed = master_seed
        self._trial_sink = trial_sink

    def mc_sample(
        self, x: list[float], tau: int, smc: int, k: int
    ) -> tuple[float, list[float]]:
        """Called by IMADS engine for each evaluation.

        Args:
            x: Continuous coordinates from IMADS (will be quantized to mesh).
            tau: Tolerance level.
            smc: Sample count level.
            k: 0-based sample index.

        Returns:
            (objective_value, [constraint_0, constraint_1, ...])
        """
        encoder = self._wrapped.encoder
        # Convert continuous x to mesh coords, then decode to params
        mesh_coords = [round(xi / encoder.mesh_base_step) for xi in x]

        result = self._wrapped.evaluate(
            mesh_coords=mesh_coords,
            tau=tau,
            smc=smc,
            k=k,
            master_seed=self._master_seed,
        )

        if self._trial_sink is not None:
            # Lazy import to avoid circular dependency (result imports nothing
            # heavy, but keeping it local makes this evaluator module
            # import-cheap).
            from imads_hpo.result import TrialRecord

            value, constraints = result[0], result[1]
            try:
                params = encoder.decode(mesh_coords)
            except Exception:
                params = {}
            cons_list = list(constraints) if constraints is not None else []
            feasible = all(c <= 0.0 for c in cons_list)
            self._trial_sink.append(
                TrialRecord(
                    params=params,
                    value=value,
                    constraints=cons_list,
                    feasible=feasible,
                    mesh_coords=list(mesh_coords),
                    # mc_sample does not expose whether this is the engine's
                    # final truth evaluation; downstream users interested in
                    # distinguishing partial vs truth can filter by tau/smc
                    # against the engine config.
                    truth_eval=True,
                )
            )

        return result

    def cheap_constraints(self, x: list[float]) -> bool:
        """Optional fast rejection (always accept by default)."""
        return True

    def solver_bias(
        self, x: list[float], tau: int
    ) -> tuple[float, list[float]]:
        """Return zero solver bias for HPO trials.

        The IMADS engine queries this for tau-dependent residual effects
        (e.g. when the underlying solver introduces a bias that depends on
        the tolerance level). HPO trials are independent black-box
        evaluations with no such bias, so we always return zero objectives
        and zero constraint slacks.

        This was missing in v1.0.1, which caused imads-core to fall back
        to its default ``unimplemented!()`` panic
        ("solver_bias requires explicit implementation for multi-objective
        evaluators") for any single-objective HPO run.
        """
        n_obj = self.num_objectives
        n_cons = self.num_constraints
        if n_obj <= 1:
            return (0.0, [0.0] * n_cons)
        # multi-objective path: return zero vector of objectives
        return ([0.0] * n_obj, [0.0] * n_cons)  # type: ignore[return-value]

    @property
    def num_constraints(self) -> int:
        return self._wrapped.num_constraints

    @property
    def num_objectives(self) -> int:
        """Number of objectives this evaluator returns.

        HPO is single-objective unless ``@objective(..., num_objectives=N)``
        was used. Required by the IMADS Evaluator protocol; was missing in
        v1.0.1.
        """
        return self._wrapped.num_objectives

    def search_dim(self) -> int:
        """Return the number of search dimensions from the Space encoder."""
        return self._wrapped.encoder.search_dim


__all__ = ["HpoEvaluator"]
