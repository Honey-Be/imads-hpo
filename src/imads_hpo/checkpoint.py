"""PyTorch checkpoint management for multi-fidelity prefix reuse.

When evaluating at fidelity (tau_tight, S), the checkpoint from a previous
(tau_loose, S) evaluation of the same candidate is loaded and training
resumes from where it left off. This implements MC prefix reuse at the
epoch level.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from imads_hpo.repro import RngRegistry, RngSnapshot, capture_rng_snapshot, restore_rng_snapshot


def _trial_key(mesh_coords: list[int], seed_index: int) -> str:
    """Deterministic hash key for a (candidate, seed) pair."""
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(tuple(mesh_coords)).encode())
    h.update(repr(seed_index).encode())
    return h.hexdigest()


@dataclass
class TrainingState:
    """Serializable training state for checkpoint resume."""
    model_state: dict[str, Any] = field(default_factory=dict)
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    scheduler_state: dict[str, Any] | None = None
    epoch: int = 0
    rng_snapshot: RngSnapshot | None = None


class CheckpointManager:
    """Manages per-trial checkpoints for multi-fidelity prefix reuse.

    Directory layout::

        checkpoint_dir/
        ├── <trial_key>/
        │   ├── epoch_005.pt
        │   ├── epoch_010.pt
        │   └── epoch_020.pt
        ...
    """

    def __init__(self, checkpoint_dir: str | Path):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _trial_dir(self, trial_key: str) -> Path:
        d = self._dir / trial_key
        d.mkdir(exist_ok=True)
        return d

    def save(
        self,
        mesh_coords: list[int],
        seed_index: int,
        state: TrainingState,
    ) -> Path:
        """Save a training checkpoint."""
        key = _trial_key(mesh_coords, seed_index)
        path = self._trial_dir(key) / f"epoch_{state.epoch:06d}.pt"
        torch.save(
            {
                "model_state": state.model_state,
                "optimizer_state": state.optimizer_state,
                "scheduler_state": state.scheduler_state,
                "epoch": state.epoch,
                "rng_snapshot": state.rng_snapshot,
            },
            path,
        )
        return path

    def load_best_prefix(
        self,
        mesh_coords: list[int],
        seed_index: int,
        target_epochs: int,
    ) -> TrainingState | None:
        """Load the best checkpoint for prefix reuse.

        Returns the checkpoint with the most epochs <= target_epochs,
        or None if no checkpoint exists.
        """
        key = _trial_key(mesh_coords, seed_index)
        trial_dir = self._dir / key
        if not trial_dir.exists():
            return None

        best_path: Path | None = None
        best_epoch = -1
        for path in trial_dir.glob("epoch_*.pt"):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if epoch <= target_epochs and epoch > best_epoch:
                best_epoch = epoch
                best_path = path

        if best_path is None:
            return None

        data = torch.load(best_path, map_location="cpu", weights_only=False)
        return TrainingState(
            model_state=data["model_state"],
            optimizer_state=data["optimizer_state"],
            scheduler_state=data.get("scheduler_state"),
            epoch=data["epoch"],
            rng_snapshot=data.get("rng_snapshot"),
        )


__all__ = ["CheckpointManager", "TrainingState"]
