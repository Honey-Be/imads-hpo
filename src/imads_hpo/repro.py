"""Deterministic runtime and RNG-state utilities.

Provides hierarchical seed derivation, explicit RNG stream management,
and PyTorch determinism configuration — following the patterns from
`framesmoothie <https://github.com/Honey-Be/framesmoothie>`_.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch

from imads_hpo.fp import curry


def _stable_bytes(parts: tuple[Any, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\0")
    return h.digest()


@curry
def derive_seed(
    master_seed: int,
    namespace: str,
    rank: int = 0,
    worker: int | None = None,
    epoch: int | None = None,
    step: int | None = None,
) -> int:
    """Derive a stable 64-bit child seed from a master seed and coordinates.

    Curried: ``derive_seed(42)("train")(0)(None)(5)(100)``
    """
    return int.from_bytes(
        _stable_bytes((int(master_seed), namespace, rank, worker, epoch, step))[:8],
        byteorder="little",
        signed=False,
    )


@dataclass(frozen=True)
class SeedPath:
    """Hierarchical seed namespace.

    Models the HPO trial as a tree of RNG streams::

        SeedPath(42).child("trial", 7).child("sample", 0).child("train")
    """

    master_seed: int
    parts: tuple[str, ...] = ()

    def child(self, *parts: Any) -> SeedPath:
        return SeedPath(self.master_seed, self.parts + tuple(str(p) for p in parts))

    def seed(self) -> int:
        return derive_seed(self.master_seed, "/".join(self.parts), 0, None, None, None)

    def torch_generator(self, *, device: str | torch.device = "cpu") -> torch.Generator:
        gen = torch.Generator(device=str(device))
        gen.manual_seed(self.seed())
        return gen

    def numpy_generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed())

    def python_random(self) -> random.Random:
        rng = random.Random()
        rng.seed(self.seed())
        return rng


@dataclass(frozen=True)
class DeterminismConfig:
    """Configuration for fully deterministic PyTorch execution.

    When ``master_seed`` and dataset are identical, results are identical
    regardless of hardware execution order or speed.
    """

    master_seed: int = 0
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    allow_tf32: bool = False
    matmul_precision: str = "highest"
    cublas_workspace_config: str | None = ":4096:2"
    python_hash_seed: int | None = None


def configure_determinism(config: DeterminismConfig) -> dict[str, Any]:
    """Apply deterministic runtime settings globally. Returns an environment report."""

    desired_hash_seed = str(
        config.python_hash_seed if config.python_hash_seed is not None else config.master_seed
    )
    os.environ.setdefault("PYTHONHASHSEED", desired_hash_seed)

    if config.cublas_workspace_config:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", config.cublas_workspace_config)

    random.seed(config.master_seed)
    torch.manual_seed(config.master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.master_seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(config.deterministic_algorithms, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(config.cudnn_deterministic)
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.matmul_precision)

    return {
        "master_seed": int(config.master_seed),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": config.deterministic_algorithms,
    }


@dataclass
class RngSnapshot:
    """Serializable snapshot of all RNG states for checkpoint restore."""

    python_state: Any = None
    numpy_global_state: Any = None
    numpy_named_states: dict[str, Any] = field(default_factory=dict)
    torch_cpu_state: torch.Tensor | None = None
    torch_cuda_states: list[torch.Tensor] = field(default_factory=list)
    torch_named_states: dict[str, torch.Tensor] = field(default_factory=dict)


class RngRegistry:
    """Explicit registry of named RNG streams.

    Checkpoints serialize only registered generators plus global framework RNGs.
    """

    def __init__(self) -> None:
        self.numpy_generators: dict[str, np.random.Generator] = {}
        self.torch_generators: dict[str, torch.Generator] = {}

    def register_numpy(self, name: str, generator: np.random.Generator) -> np.random.Generator:
        self.numpy_generators[name] = generator
        return generator

    def register_torch(self, name: str, generator: torch.Generator) -> torch.Generator:
        self.torch_generators[name] = generator
        return generator

    def snapshot(self) -> RngSnapshot:
        return capture_rng_snapshot(
            numpy_generators=self.numpy_generators,
            torch_generators=self.torch_generators,
        )

    def restore(self, snapshot: RngSnapshot) -> None:
        restore_rng_snapshot(
            snapshot,
            numpy_generators=self.numpy_generators,
            torch_generators=self.torch_generators,
        )


def capture_rng_snapshot(
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> RngSnapshot:
    return RngSnapshot(
        python_state=random.getstate(),
        numpy_global_state=np.random.get_state(),
        numpy_named_states={
            name: gen.bit_generator.state for name, gen in (numpy_generators or {}).items()
        },
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        torch_named_states={
            name: gen.get_state() for name, gen in (torch_generators or {}).items()
        },
    )


def restore_rng_snapshot(
    snapshot: RngSnapshot,
    *,
    numpy_generators: MutableMapping[str, np.random.Generator] | None = None,
    torch_generators: MutableMapping[str, torch.Generator] | None = None,
) -> None:
    numpy_generators = numpy_generators or {}
    torch_generators = torch_generators or {}

    if snapshot.python_state is not None:
        random.setstate(snapshot.python_state)
    if snapshot.numpy_global_state is not None:
        np.random.set_state(snapshot.numpy_global_state)

    for name, gen in numpy_generators.items():
        if name in snapshot.numpy_named_states:
            gen.bit_generator.state = snapshot.numpy_named_states[name]

    if snapshot.torch_cpu_state is not None:
        torch.set_rng_state(snapshot.torch_cpu_state)
    if torch.cuda.is_available() and snapshot.torch_cuda_states:
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_states)

    for name, gen in torch_generators.items():
        if name in snapshot.torch_named_states:
            gen.set_state(snapshot.torch_named_states[name])


__all__ = [
    "DeterminismConfig",
    "RngRegistry",
    "RngSnapshot",
    "SeedPath",
    "capture_rng_snapshot",
    "configure_determinism",
    "derive_seed",
    "restore_rng_snapshot",
]
