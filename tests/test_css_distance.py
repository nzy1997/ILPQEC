"""Tests for exact CSS distance and logical reduction ILPs."""

from itertools import product

import numpy as np
import pytest

from ilpqec.distance_ilp import (
    minimize_nonzero_logical_operator,
    minimize_weight_with_fixed_syndrome,
)
from ilpqec.solver import get_available_solvers


pytestmark = pytest.mark.skipif(
    "highs" not in get_available_solvers(),
    reason="CSS distance ILP tests require the HiGHS backend",
)


def brute_force_fixed(checks, duals, rhs):
    checks = np.asarray(checks, dtype=np.uint8)
    duals = np.asarray(duals, dtype=np.uint8)
    rhs = np.asarray(rhs, dtype=np.uint8)
    n = checks.shape[1] if checks.size else duals.shape[1]
    for weight in range(n + 1):
        for bits in product([0, 1], repeat=n):
            vector = np.array(bits, dtype=np.uint8)
            if int(vector.sum()) != weight:
                continue
            if checks.size and np.any((checks @ vector) % 2):
                continue
            if np.array_equal((duals @ vector) % 2, rhs):
                return vector
    raise AssertionError("No feasible vector found")


def test_minimize_weight_with_fixed_syndrome_matches_bruteforce():
    checks = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    duals = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    rhs = np.array([1, 0], dtype=np.uint8)
    matrix = np.vstack([checks, duals])
    syndrome = np.concatenate([np.zeros(checks.shape[0], dtype=np.uint8), rhs])

    result = minimize_weight_with_fixed_syndrome(matrix, syndrome, solver="highs")

    expected = brute_force_fixed(checks, duals, rhs)
    np.testing.assert_array_equal((matrix @ result.vector) % 2, syndrome)
    assert result.weight == int(expected.sum())


def test_minimize_nonzero_logical_operator_finds_weight_one_operator():
    checks = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    duals = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    result = minimize_nonzero_logical_operator(checks, duals, solver="highs")

    assert result.weight == 1
    assert not np.any((checks @ result.vector) % 2)
    assert np.any((duals @ result.vector) % 2)


def test_exact_ilp_rejects_positive_gap():
    matrix = np.array([[1, 1]], dtype=np.uint8)
    syndrome = np.array([0], dtype=np.uint8)

    with pytest.raises(ValueError, match="exact"):
        minimize_weight_with_fixed_syndrome(matrix, syndrome, solver="highs", gap=0.1)


def test_nonzero_logical_operator_rejects_positive_gap():
    checks = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    duals = np.array([[0, 0, 1, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="exact"):
        minimize_nonzero_logical_operator(checks, duals, solver="highs", gap=0.1)


def test_exact_ilp_rejects_non_binary_matrix_input():
    matrix = np.array([[1, 2]], dtype=np.int64)
    syndrome = np.array([0], dtype=np.uint8)

    with pytest.raises(ValueError, match="binary"):
        minimize_weight_with_fixed_syndrome(matrix, syndrome, solver="highs")
