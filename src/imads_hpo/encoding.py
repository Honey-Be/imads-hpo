"""Encode/decode between hyperparameter dicts and IMADS mesh coordinates.

All dimensions are mapped to 64-bit integer mesh coordinates. Categorical
variables use direct int64 indexing (0, 1, 2, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from imads_hpo.space import Categorical, Integer, LogReal, Real, Space


@dataclass(frozen=True)
class DimEncoding:
    """Encoding metadata for one dimension."""

    name: str
    kind: str  # "real", "logreal", "integer", "categorical"
    base_step: float
    offset: float  # continuous = offset + mesh_coord * base_step
    # For categorical / integer: number of valid values
    n_values: int | None = None
    # For logreal: log-space offset and step
    log_low: float | None = None
    # For integer: step in original space
    int_step: int | None = None
    # For categorical: original choices
    choices: tuple[Any, ...] | None = None
    # Lower / upper bound in the *original* space (Real / LogReal). Used
    # by ``decode`` to clamp mesh coordinates that fall outside the
    # declared search range — IMADS may poll points beyond the bounds
    # during its mesh exploration, and v1.0.1 propagated those raw values
    # directly, producing impossible HP values like ``weight_decay=4.0``
    # from a ``Real(0.0, 0.1)`` declaration or ``math.exp`` overflows
    # from ``LogReal``.
    low: float | None = None
    high: float | None = None
    log_high: float | None = None


class SpaceEncoder:
    """Converts a ``Space`` to IMADS mesh encoding and back.

    Each dimension maps to one mesh axis. The encoder computes per-dimension
    ``base_step`` values such that the mesh covers the search space with
    reasonable resolution.

    Args:
        space: The search space to encode.
        resolution: Approximate number of distinct mesh points per continuous
            dimension (default 1000).
    """

    def __init__(self, space: Space, resolution: int = 1000):
        self._space = space
        self._encodings: list[DimEncoding] = []

        for name in space.names:
            dim = space[name]

            if isinstance(dim, Real):
                step = (dim.high - dim.low) / max(resolution, 1)
                self._encodings.append(
                    DimEncoding(
                        name=name, kind="real", base_step=step,
                        offset=dim.low, low=dim.low, high=dim.high,
                    )
                )

            elif isinstance(dim, LogReal):
                log_low = math.log(dim.low)
                log_high = math.log(dim.high)
                step = (log_high - log_low) / max(resolution, 1)
                self._encodings.append(
                    DimEncoding(
                        name=name, kind="logreal", base_step=step,
                        offset=log_low, log_low=log_low, log_high=log_high,
                        low=dim.low, high=dim.high,
                    )
                )

            elif isinstance(dim, Integer):
                n_values = (dim.high - dim.low) // dim.step + 1
                self._encodings.append(
                    DimEncoding(
                        name=name, kind="integer", base_step=1.0,
                        offset=float(dim.low), n_values=n_values, int_step=dim.step,
                    )
                )

            elif isinstance(dim, Categorical):
                self._encodings.append(
                    DimEncoding(
                        name=name, kind="categorical", base_step=1.0,
                        offset=0.0, n_values=len(dim.choices),
                        choices=tuple(dim.choices),
                    )
                )

    @property
    def search_dim(self) -> int:
        """Number of mesh dimensions (one per hyperparameter)."""
        return len(self._encodings)

    @property
    def mesh_base_step(self) -> float:
        """Global base_step for IMADS EngineConfig.

        Uses the minimum across all dimensions (IMADS uses a single scalar).
        Integer/categorical dimensions use 1.0, so this is always >= the finest
        continuous resolution.
        """
        return min(e.base_step for e in self._encodings)

    def encode(self, params: dict[str, Any]) -> list[int]:
        """Encode hyperparameters to mesh coordinates (list of int64)."""
        coords: list[int] = []
        for enc in self._encodings:
            val = params[enc.name]

            if enc.kind == "real":
                coords.append(round((float(val) - enc.offset) / enc.base_step))

            elif enc.kind == "logreal":
                log_val = math.log(max(float(val), 1e-300))
                coords.append(round((log_val - enc.offset) / enc.base_step))

            elif enc.kind == "integer":
                idx = (int(val) - int(enc.offset)) // (enc.int_step or 1)
                coords.append(max(0, min(idx, (enc.n_values or 1) - 1)))

            elif enc.kind == "categorical":
                assert enc.choices is not None
                try:
                    coords.append(enc.choices.index(val))
                except ValueError:
                    coords.append(0)

        return coords

    def decode(self, mesh_coords: list[int]) -> dict[str, Any]:
        """Decode mesh coordinates back to hyperparameters.

        Mesh coordinates are clamped to the declared search bounds before
        being returned. IMADS may poll points outside the original bounds
        during its mesh exploration phase; without clamping these would
        leak into user objectives as out-of-range values (and in the
        ``logreal`` case, ``math.exp`` would overflow).
        """
        params: dict[str, Any] = {}
        for enc, coord in zip(self._encodings, mesh_coords):
            if enc.kind == "real":
                val = enc.offset + coord * enc.base_step
                if enc.low is not None and enc.high is not None:
                    val = max(enc.low, min(enc.high, val))
                params[enc.name] = val

            elif enc.kind == "logreal":
                log_val = enc.offset + coord * enc.base_step
                if enc.log_high is not None:
                    log_val = max(enc.log_low, min(enc.log_high, log_val))
                val = math.exp(log_val)
                # Float precision: ``math.exp(math.log(low))`` may be ε
                # below ``low``. Clamp once more in linear space against
                # the *original* bounds (not their re-exp'd versions) so
                # the result is guaranteed to satisfy ``low <= val <= high``.
                if enc.low is not None and enc.high is not None:
                    val = max(enc.low, min(enc.high, val))
                params[enc.name] = val

            elif enc.kind == "integer":
                step = enc.int_step or 1
                val = int(enc.offset) + coord * step
                if enc.n_values is not None:
                    max_val = int(enc.offset) + (enc.n_values - 1) * step
                    val = max(int(enc.offset), min(max_val, val))
                params[enc.name] = val

            elif enc.kind == "categorical":
                assert enc.choices is not None
                idx = max(0, min(coord, len(enc.choices) - 1))
                params[enc.name] = enc.choices[idx]

        return params

    @property
    def encodings(self) -> list[DimEncoding]:
        return list(self._encodings)


__all__ = ["DimEncoding", "SpaceEncoder"]
