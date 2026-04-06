"""imads-hpo — PyTorch hyperparameter optimization via IMADS.

Example::

    import imads_hpo as hpo

    space = hpo.Space({
        "lr": hpo.LogReal(1e-5, 1e-1),
        "batch_size": hpo.Integer(16, 256, step=16),
        "optimizer": hpo.Categorical(["adam", "sgd", "adamw"]),
    })

    @hpo.objective(space, num_constraints=1)
    def train(params, fidelity):
        ...
        return val_loss, [gpu_mem - 8.0]

    result = hpo.minimize(train, preset="balanced", workers=4, seed=42)
    print(result.best_params)
"""

from imads_hpo.callbacks import Callback, PrintCallback
from imads_hpo.checkpoint import CheckpointManager, TrainingState
from imads_hpo.constraints import GpuMemoryConstraint, LatencyConstraint, ModelSizeConstraint
from imads_hpo.encoding import SpaceEncoder
from imads_hpo.fidelity import EpochFidelity, Fidelity
from imads_hpo.fp import Run, RunLog, ask, asks, curry, get_state, modify_state, put_state, sequence, tell
from imads_hpo.objective import WrappedObjective, objective
from imads_hpo.optimizer import minimize
from imads_hpo.repro import (
    DeterminismConfig,
    RngRegistry,
    RngSnapshot,
    SeedPath,
    configure_determinism,
    derive_seed,
)
from imads_hpo.result import Result, TrialRecord
from imads_hpo.space import Categorical, Integer, LogReal, Real, Space

__all__ = [
    # Space
    "Categorical", "Integer", "LogReal", "Real", "Space",
    # Encoding
    "SpaceEncoder",
    # Fidelity
    "EpochFidelity", "Fidelity",
    # Objective
    "WrappedObjective", "objective",
    # Optimizer
    "minimize",
    # Result
    "Result", "TrialRecord",
    # FP
    "Run", "RunLog", "ask", "asks", "curry", "get_state", "modify_state",
    "put_state", "sequence", "tell",
    # Reproducibility
    "DeterminismConfig", "RngRegistry", "RngSnapshot", "SeedPath",
    "configure_determinism", "derive_seed",
    # Checkpoint
    "CheckpointManager", "TrainingState",
    # Constraints
    "GpuMemoryConstraint", "LatencyConstraint", "ModelSizeConstraint",
    # Callbacks
    "Callback", "PrintCallback",
]
