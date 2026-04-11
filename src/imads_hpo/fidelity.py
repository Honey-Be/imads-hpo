"""Fidelity mapping: IMADS (tau, S) <-> PyTorch training (epochs, seeds)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Fidelity:
    """Concrete fidelity level passed to the user's objective function."""

    epochs: int
    seed_index: int
    tau: int
    smc: int
    data_fraction: float = 1.0


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


@runtime_checkable
class FidelitySchedule(Protocol):
    """Protocol for fidelity schedule implementations."""

    @property
    def tau_levels(self) -> list[int]: ...

    @property
    def smc_levels(self) -> list[int]: ...

    def resolve_fidelity(self, tau: int, smc: int, k: int) -> Fidelity: ...


@dataclass(frozen=True)
class DataFractionFidelity:
    """Data subsampling-based fidelity."""

    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
    num_seeds: int = 1

    @property
    def tau_levels(self) -> list[int]:
        # tau = round(1.0 / fraction), descending
        return sorted(
            {max(1, round(1.0 / f)) for f in self.fractions}, reverse=True
        )

    @property
    def smc_levels(self) -> list[int]:
        return [1, self.num_seeds] if self.num_seeds > 1 else [1]

    def resolve_fidelity(self, tau: int, smc: int, k: int) -> Fidelity:
        fraction = min(1.0, 1.0 / max(1, tau))
        # Snap to nearest defined fraction
        best = min(self.fractions, key=lambda f: abs(f - fraction))
        return Fidelity(epochs=1, seed_index=k, tau=tau, smc=smc, data_fraction=best)


@dataclass(frozen=True)
class ExplicitFidelity:
    """User-specified explicit epoch steps."""

    epoch_steps: tuple[int, ...] = (5, 20, 50, 100)
    num_seeds: int = 1

    def __post_init__(self):
        assert len(self.epoch_steps) >= 1
        assert all(e > 0 for e in self.epoch_steps)

    @property
    def tau_levels(self) -> list[int]:
        max_e = max(self.epoch_steps)
        return sorted(
            {max(1, max_e // e) for e in self.epoch_steps}, reverse=True
        )

    @property
    def smc_levels(self) -> list[int]:
        return [1, self.num_seeds] if self.num_seeds > 1 else [1]

    def resolve_fidelity(self, tau: int, smc: int, k: int) -> Fidelity:
        max_e = max(self.epoch_steps)
        target = max(1, max_e // max(1, tau))
        best = min(self.epoch_steps, key=lambda e: abs(e - target))
        return Fidelity(epochs=best, seed_index=k, tau=tau, smc=smc)


__all__ = [
    "DataFractionFidelity",
    "EpochFidelity",
    "ExplicitFidelity",
    "Fidelity",
    "FidelitySchedule",
]
