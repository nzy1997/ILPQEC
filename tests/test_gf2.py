"""Tests for GF(2) linear algebra helpers."""

import numpy as np
import pytest

from ilpqec.gf2 import binary_inverse, nullspace, rank, row_basis, row_reduce


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
