"""IMADS Evaluator implementation that bridges to HPO objectives."""

from __future__ import annotations

from typing import Any

from imads_hpo.encoding import SpaceEncoder
from imads_hpo.objective import WrappedObjective


class HpoEvaluator:
    """Implements the ``imads.Evaluator`` protocol for HPO.

    This is the bridge between the IMADS engine's ``mc_sample`` interface
    and the user's PyTorch training function.
    """

    def __init__(
        self,
        wrapped: WrappedObjective,
        master_seed: int,
    ):
        self._wrapped = wrapped
        self._master_seed = master_seed

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

        return self._wrapped.evaluate(
            mesh_coords=mesh_coords,
            tau=tau,
            smc=smc,
            k=k,
            master_seed=self._master_seed,
        )

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
