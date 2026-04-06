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

    @property
    def num_constraints(self) -> int:
        return self._wrapped.num_constraints

    def search_dim(self) -> int:
        """Return the number of search dimensions from the Space encoder."""
        return self._wrapped.encoder.search_dim


__all__ = ["HpoEvaluator"]
