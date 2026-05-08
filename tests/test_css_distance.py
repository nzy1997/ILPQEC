"""Tests for exact CSS distance and logical reduction ILPs."""

from itertools import product

import numpy as np
import pytest

from ilpqec import CSSCode
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


def steane_check_matrix():
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def assert_valid_distance_result(code, result):
    assert result.d == min(result.dx, result.dz)
    assert result.dx == int(result.shortest_x.sum())
    assert result.dz == int(result.shortest_z.sum())
    assert not np.any((code.hz @ result.shortest_x) % 2)
    assert not np.any((code.hx @ result.shortest_z) % 2)


def test_css_code_distance_for_steane_code():
    h = steane_check_matrix()
    code = CSSCode.from_parity_check_matrices(h, h)

    result = code.distance(solver="highs")

    assert result.d == 3
    assert result.dx == 3
    assert result.dz == 3
    assert_valid_distance_result(code, result)


def test_css_code_distance_for_two_logical_toy_code():
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    result = code.distance(solver="highs")

    assert result.d == 1
    assert result.dx == 1
    assert result.dz == 1
    assert_valid_distance_result(code, result)


def test_reduced_logical_basis_is_paired_and_fixed_coset_minimal():
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    canonical = code.logical_basis(reduce=False)
    basis = code.logical_basis(reduce=True, solver="highs")

    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(2, dtype=np.uint8))
    for index in range(code.k):
        rhs = np.zeros(code.k, dtype=np.uint8)
        rhs[index] = 1

        assert not np.any((hz @ basis.x[index]) % 2)
        assert not np.any((hx @ basis.z[index]) % 2)
        np.testing.assert_array_equal((canonical.z @ basis.x[index]) % 2, rhs)
        np.testing.assert_array_equal((canonical.x @ basis.z[index]) % 2, rhs)

        expected_x = brute_force_fixed(hz, canonical.z, rhs)
        expected_z = brute_force_fixed(hx, canonical.x, rhs)
        assert int(basis.x[index].sum()) == int(expected_x.sum())
        assert int(basis.z[index].sum()) == int(expected_z.sum())
