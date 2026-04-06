"""Fidelity mapping: IMADS (tau, S) <-> PyTorch training (epochs, seeds)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fidelity:
    """Concrete fidelity level passed to the user's objective function."""

    epochs: int
    seed_index: int
    tau: int
    smc: int


@dataclass(frozen=True)
class EpochFidelity:
    """Map IMADS fidelity ladder to epoch-based training.

    - ``tau``: inversely proportional to epoch count.
      ``tau = max_epochs / epochs`` (loose = few epochs, tight = full training).
    - ``S``: number of independent training seeds.

    Args:
        min_epochs: Cheapest evaluation epoch count (tau_loose).
        max_epochs: Full training epoch count (tau_tight = TRUTH).
        num_seeds: Maximum number of independent seeds (S_max).
    """

    min_epochs: int = 5
    max_epochs: int = 100
    num_seeds: int = 3

    def __post_init__(self):
        assert 1 <= self.min_epochs < self.max_epochs
        assert self.num_seeds >= 1

    @property
    def tau_levels(self) -> list[int]:
        """Generate tau levels from loose to tight.

        tau = max_epochs / epochs, so larger tau = fewer epochs (looser).
        Generates geometrically spaced epoch counts, then converts to tau.
        """
        levels: list[int] = []
        epochs = self.min_epochs
        while epochs < self.max_epochs:
            tau = max(1, self.max_epochs // epochs)
            if not levels or tau != levels[-1]:
                levels.append(tau)
            epochs *= 2
        levels.append(1)  # TRUTH: tau=1 means full max_epochs
        # Sort descending (loose first = large tau)
        return sorted(set(levels), reverse=True)

    @property
    def smc_levels(self) -> list[int]:
        """Generate S levels: [1, num_seeds]."""
        if self.num_seeds <= 1:
            return [1]
        return [1, self.num_seeds]

    def tau_to_epochs(self, tau: int) -> int:
        """Convert tau level to epoch count."""
        tau = max(1, tau)
        return max(self.min_epochs, self.max_epochs // tau)

    def resolve_fidelity(self, tau: int, smc: int, k: int) -> Fidelity:
        """Create a concrete Fidelity from IMADS fidelity parameters."""
        return Fidelity(
            epochs=self.tau_to_epochs(tau),
            seed_index=k,
            tau=tau,
            smc=smc,
        )


__all__ = ["EpochFidelity", "Fidelity"]
