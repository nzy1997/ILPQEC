"""Tests for CSSCode construction and solver-free APIs."""

import numpy as np
import pytest

from ilpqec import CSSCode


def steane_check_matrix():
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def test_css_code_properties_and_unreduced_logical_basis():
    h = steane_check_matrix()

    code = CSSCode.from_parity_check_matrices(h, h)
    basis = code.logical_basis(reduce=False)

    assert code.n == 7
    assert code.k == 1
    assert code.rank_x == 3
    assert code.rank_z == 3
    np.testing.assert_array_equal(code.hx, h)
    np.testing.assert_array_equal(code.hz, h)
    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(1, dtype=np.uint8))


def test_css_code_copies_matrices_for_read_only_properties():
    hx = np.array([[1, 1, 0]], dtype=np.uint8)
    hz = np.zeros((0, 3), dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    hx_copy = code.hx
    hx_copy[0, 0] = 0

    np.testing.assert_array_equal(code.hx, np.array([[1, 1, 0]], dtype=np.uint8))


def test_css_code_copies_input_matrices_on_construction():
    hx = np.array([[1, 1, 0]], dtype=np.uint8)
    hz = np.zeros((0, 3), dtype=np.uint8)

    code = CSSCode(hx, hz)
    hx[0, 0] = 0

    np.testing.assert_array_equal(code.hx, np.array([[1, 1, 0]], dtype=np.uint8))


def test_css_code_rejects_non_binary_input():
    hx = np.array([[2, 0, 1]], dtype=int)
    hz = np.zeros((0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="binary"):
        CSSCode.from_parity_check_matrices(hx, hz)


def test_css_code_rejects_shape_mismatch():
    hx = np.zeros((1, 3), dtype=np.uint8)
    hz = np.zeros((1, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="same number of columns"):
        CSSCode.from_parity_check_matrices(hx, hz)


def test_css_code_rejects_non_commuting_checks_by_default():
    hx = np.array([[1, 0]], dtype=np.uint8)
    hz = np.array([[1, 1]], dtype=np.uint8)

    with pytest.raises(ValueError, match="commute"):
        CSSCode.from_parity_check_matrices(hx, hz)


def test_css_code_can_skip_commutation_validation():
    hx = np.array([[1, 0]], dtype=np.uint8)
    hz = np.array([[1, 1]], dtype=np.uint8)

    code = CSSCode.from_parity_check_matrices(hx, hz, validate_commutation=False)

    assert code.n == 2


def test_logical_basis_raises_for_zero_logical_qubits():
    hx = np.eye(2, dtype=np.uint8)
    hz = np.zeros((0, 2), dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    with pytest.raises(ValueError, match="no logical qubits"):
        code.logical_basis()


def test_logical_basis_rejects_solver_arguments_before_ilp_is_implemented():
    h = steane_check_matrix()
    code = CSSCode.from_parity_check_matrices(h, h)

    with pytest.raises(TypeError, match="only supported when reduce=True"):
        code.logical_basis(reduce=False, solver="highs")


def test_distance_raises_for_zero_logical_qubits():
    hx = np.eye(2, dtype=np.uint8)
    hz = np.zeros((0, 2), dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    with pytest.raises(ValueError, match="no logical qubits"):
        code.distance()
