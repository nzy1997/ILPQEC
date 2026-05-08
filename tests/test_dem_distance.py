"""Tests for exact DEM distance calculations."""

from itertools import product

import numpy as np
import pytest

from ilpqec import DEMDistanceResult, Decoder, dem_distance
from ilpqec.solver import get_available_solvers


pytestmark = pytest.mark.skipif(
    "highs" not in get_available_solvers(),
    reason="DEM distance tests require the HiGHS backend",
)


def brute_force_dem_distance(h_matrix, obs_matrix, target=None):
    h_matrix = np.asarray(h_matrix, dtype=np.uint8)
    obs_matrix = np.asarray(obs_matrix, dtype=np.uint8)
    target = None if target is None else np.asarray(target, dtype=np.uint8)
    n = h_matrix.shape[1]

    best = None
    for bits in product([0, 1], repeat=n):
        vector = np.array(bits, dtype=np.uint8)
        if np.any((h_matrix @ vector) % 2):
            continue
        observable_mask = (obs_matrix @ vector) % 2
        if target is None:
            if not np.any(observable_mask):
                continue
        elif not np.array_equal(observable_mask, target):
            continue
        weight = int(vector.sum())
        if best is None or weight < best[0]:
            best = (weight, vector.copy(), observable_mask.copy())

    if best is None:
        raise AssertionError("No feasible DEM fault found")
    return best


def test_dem_distance_default_mode_matches_bruteforce():
    dem = """
        error(0.1) D0
        error(0.1) D0 L0
        error(0.1) L0
    """

    h_matrix, obs_matrix, _ = Decoder()._parse_dem(
        dem,
        merge_parallel=True,
        flatten_dem=True,
    )
    expected_weight, expected_vector, expected_mask = brute_force_dem_distance(
        h_matrix,
        obs_matrix,
    )

    result = dem_distance(dem, solver="highs")

    assert isinstance(result, DEMDistanceResult)
    assert result.distance == expected_weight
    np.testing.assert_array_equal(result.fault_vector, expected_vector)
    np.testing.assert_array_equal(result.observable_mask, expected_mask)
    assert result.fault_indices == list(np.flatnonzero(expected_vector))


def test_dem_distance_targeted_mode_matches_bruteforce():
    dem = """
        error(0.1) D0 L0
        error(0.1) D0
        error(0.1) L0
    """
    target = np.array([1], dtype=np.uint8)

    h_matrix, obs_matrix, _ = Decoder()._parse_dem(
        dem,
        merge_parallel=True,
        flatten_dem=True,
    )
    expected_weight, expected_vector, expected_mask = brute_force_dem_distance(
        h_matrix,
        obs_matrix,
        target=target,
    )

    result = dem_distance(dem, target_observables=target, solver="highs")

    assert result.distance == expected_weight
    np.testing.assert_array_equal(result.fault_vector, expected_vector)
    np.testing.assert_array_equal(result.observable_mask, expected_mask)
    assert result.fault_indices == list(np.flatnonzero(expected_vector))


def test_dem_distance_rotated_surface_code_distance_three():
    stim = pytest.importorskip("stim")
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    default_result = dem_distance(dem, solver="highs")
    targeted_result = dem_distance(dem, target_observables=[1], solver="highs")

    assert default_result.distance == 3
    assert targeted_result.distance == 3
    np.testing.assert_array_equal(targeted_result.observable_mask, np.array([1], dtype=np.uint8))
