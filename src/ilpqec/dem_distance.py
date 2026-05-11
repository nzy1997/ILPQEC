"""Exact distance helpers for Stim detector error models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ilpqec.decoder import Decoder
from ilpqec.distance_ilp import (
    minimize_nonzero_logical_operator,
    minimize_weight_with_fixed_syndrome,
)


@dataclass(frozen=True)
class DEMDistanceResult:
    """Minimum-weight DEM fault representative."""

    distance: int
    fault_vector: np.ndarray
    fault_indices: list[int]
    observable_mask: np.ndarray


def _validated_target_observables(
    target_observables,
    *,
    num_observables: int,
) -> np.ndarray:
    target = np.asarray(target_observables)
    if target.ndim != 1:
        raise ValueError("target_observables must be one-dimensional")
    if target.shape[0] != num_observables:
        raise ValueError("target_observables length must match the DEM observable count")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("target_observables must contain only binary values")

    target = np.asarray(target, dtype=np.uint8)
    if not np.any(target):
        raise ValueError("target_observables must be a nonzero observable mask")
    return target


def dem_distance(
    dem,
    *,
    target_observables=None,
    merge_parallel_edges: bool = True,
    flatten_dem: bool = True,
    solver: Optional[str] = None,
    **solver_options: Any,
) -> DEMDistanceResult:
    """Compute an exact minimum-weight logical fault for a detector error model."""
    h_matrix, obs_matrix, _ = Decoder()._parse_dem(
        dem,
        merge_parallel=merge_parallel_edges,
        flatten_dem=flatten_dem,
    )
    if obs_matrix.shape[0] == 0:
        raise ValueError("DEM distance requires at least one logical observable")

    if target_observables is None:
        result = minimize_nonzero_logical_operator(
            h_matrix,
            obs_matrix,
            solver=solver,
            **solver_options,
        )
    else:
        target = _validated_target_observables(
            target_observables,
            num_observables=obs_matrix.shape[0],
        )
        matrix = np.vstack([h_matrix, obs_matrix])
        syndrome = np.concatenate([np.zeros(h_matrix.shape[0], dtype=np.uint8), target])
        result = minimize_weight_with_fixed_syndrome(
            matrix,
            syndrome,
            solver=solver,
            **solver_options,
        )

    fault_vector = np.array(result.vector, dtype=np.uint8, copy=True)
    observable_mask = np.array((obs_matrix @ fault_vector) % 2, dtype=np.uint8, copy=True)
    return DEMDistanceResult(
        distance=result.weight,
        fault_vector=fault_vector,
        fault_indices=[int(index) for index in np.flatnonzero(fault_vector)],
        observable_mask=observable_mask,
    )
