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

    if target_observables is None:
        result = minimize_nonzero_logical_operator(
            h_matrix,
            obs_matrix,
            solver=solver,
            **solver_options,
        )
    else:
        target = np.asarray(target_observables, dtype=np.uint8)
        matrix = np.vstack([h_matrix, obs_matrix])
        syndrome = np.concatenate([np.zeros(h_matrix.shape[0], dtype=np.uint8), target])
        result = minimize_weight_with_fixed_syndrome(
            matrix,
            syndrome,
            solver=solver,
            **solver_options,
        )

    observable_mask = (obs_matrix @ result.vector) % 2
    return DEMDistanceResult(
        distance=result.weight,
        fault_vector=result.vector,
        fault_indices=[int(index) for index in np.flatnonzero(result.vector)],
        observable_mask=np.asarray(observable_mask, dtype=np.uint8),
    )
