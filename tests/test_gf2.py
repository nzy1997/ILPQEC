"""Tests for GF(2) linear algebra helpers."""

import numpy as np
import pytest

from ilpqec.gf2 import (
    binary_inverse,
    css_logical_basis,
    extend_independent_rows,
    nullspace,
    quotient_basis,
    rank,
    row_basis,
    row_reduce,
)


def test_row_reduce_returns_reduced_echelon_form_and_pivots():
    matrix = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    reduced = row_reduce(matrix)

    expected = np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(reduced.matrix, expected)
    assert reduced.pivot_columns == (0, 1)
    assert reduced.rank == 2


def test_row_reduce_normalizes_python_ints_before_reducing():
    matrix = np.array(
        [
            [-1, 256],
            [2, 3],
        ],
        dtype=object,
    )

    reduced = row_reduce(matrix)

    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(reduced.matrix, expected)
    assert reduced.rank == 2


def test_rank_and_row_basis_ignore_dependent_rows():
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )

    assert rank(matrix) == 2
    basis = row_basis(matrix)
    assert basis.shape == (2, 3)
    assert rank(basis) == 2
    assert rank(np.vstack([basis, matrix])) == 2


def test_nullspace_vectors_are_a_basis():
    matrix = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    kernel = nullspace(matrix)

    assert kernel.shape == (2, 4)
    np.testing.assert_array_equal((matrix @ kernel.T) % 2, np.zeros((2, 2), dtype=np.uint8))
    assert rank(kernel) == 2


def test_binary_inverse_inverts_full_rank_matrix():
    matrix = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )

    inverse = binary_inverse(matrix)

    np.testing.assert_array_equal((matrix @ inverse) % 2, np.eye(3, dtype=np.uint8))
    np.testing.assert_array_equal((inverse @ matrix) % 2, np.eye(3, dtype=np.uint8))


def test_binary_inverse_rejects_singular_matrix():
    matrix = np.array(
        [
            [1, 0],
            [1, 0],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="singular"):
        binary_inverse(matrix)


def test_binary_inverse_rejects_non_square_matrix():
    matrix = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="square"):
        binary_inverse(matrix)


def test_extend_independent_rows_selects_independent_candidates():
    base = np.array(
        [
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    candidates = [
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 1], dtype=np.uint8),
    ]

    selected = extend_independent_rows(base, candidates, count=2)

    expected = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(selected, expected)
    assert rank(np.vstack([base, selected])) == 3


def test_extend_independent_rows_raises_when_not_enough_independent_rows():
    base = np.array(
        [
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    candidates = [
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 0, 0], dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="independent quotient representatives"):
        extend_independent_rows(base, candidates, count=1)


def test_extend_independent_rows_normalizes_candidate_rows_before_selection():
    base = np.array(
        [
            [1, 0],
        ],
        dtype=np.uint8,
    )
    candidates = [
        np.array([2, 3], dtype=object),
    ]

    selected = extend_independent_rows(base, candidates, count=1)

    expected = np.array([[0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(selected, expected)


def test_extend_independent_rows_rejects_negative_count():
    base = np.array([[1, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="count"):
        extend_independent_rows(base, [], count=-1)


@pytest.mark.parametrize("candidates", [[np.array([1, 0, 1], dtype=np.uint8)], []])
def test_extend_independent_rows_zero_count_returns_empty_matrix(candidates):
    base = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    selected = extend_independent_rows(base, candidates, count=0)

    assert selected.shape == (0, 3)
    assert selected.dtype == np.uint8


def steane_check_matrix():
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def test_css_logical_basis_for_steane_code_is_paired():
    h = steane_check_matrix()

    basis = css_logical_basis(h, h)

    assert basis.x.shape == (1, 7)
    assert basis.z.shape == (1, 7)
    np.testing.assert_array_equal((h @ basis.x.T) % 2, np.zeros((3, 1), dtype=np.uint8))
    np.testing.assert_array_equal((h @ basis.z.T) % 2, np.zeros((3, 1), dtype=np.uint8))
    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(1, dtype=np.uint8))


def test_css_logical_basis_for_two_logical_code_is_paired():
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[0, 0, 1, 1]], dtype=np.uint8)

    basis = css_logical_basis(hx, hz)

    assert basis.x.shape == (2, 4)
    assert basis.z.shape == (2, 4)
    np.testing.assert_array_equal((hz @ basis.x.T) % 2, np.zeros((1, 2), dtype=np.uint8))
    np.testing.assert_array_equal((hx @ basis.z.T) % 2, np.zeros((1, 2), dtype=np.uint8))
    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(2, dtype=np.uint8))


def test_css_logical_basis_rejects_non_commuting_checks():
    hx = np.array([[1, 0, 0]], dtype=np.uint8)
    hz = np.array([[1, 0, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="commute"):
        css_logical_basis(hx, hz)


def test_css_logical_basis_repairs_non_identity_initial_pairing():
    hx = np.array([[1, 1, 1]], dtype=np.uint8)
    hz = np.zeros((0, 3), dtype=np.uint8)

    basis = css_logical_basis(hx, hz)

    assert basis.x.shape == (2, 3)
    assert basis.z.shape == (2, 3)
    np.testing.assert_array_equal((hx @ basis.z.T) % 2, np.zeros((1, 2), dtype=np.uint8))
    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(2, dtype=np.uint8))

    raw_z = quotient_basis(nullspace(hx), row_basis(hz), dimension=2)
    np.testing.assert_array_equal(
        (basis.x @ raw_z.T) % 2,
        np.array([[1, 1], [1, 0]], dtype=np.uint8),
    )
    assert not np.array_equal(basis.z, raw_z)


def test_css_logical_basis_returns_empty_bases_for_zero_logical_qubits():
    hx = np.eye(2, dtype=np.uint8)
    hz = np.zeros((0, 2), dtype=np.uint8)

    basis = css_logical_basis(hx, hz)

    assert basis.x.shape == (0, 2)
    assert basis.z.shape == (0, 2)
