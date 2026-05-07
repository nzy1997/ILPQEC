"""Small GF(2) linear algebra helpers used by CSS code analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RowReduction:
    """Reduced row-echelon form data over GF(2)."""

    matrix: np.ndarray
    pivot_columns: tuple[int, ...]
    rank: int


@dataclass(frozen=True)
class CSSLogicalBasisData:
    """Paired binary CSS logical bases."""

    x: np.ndarray
    z: np.ndarray


def _as_gf2_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.int64) % 2
    if array.ndim != 2:
        raise ValueError("GF(2) matrix must be two-dimensional")
    return array.astype(np.uint8, copy=True)


def row_reduce(matrix: np.ndarray) -> RowReduction:
    """Return the reduced row-echelon form of a binary matrix over GF(2)."""
    reduced = _as_gf2_matrix(matrix)
    rows, cols = reduced.shape
    pivot_columns = []
    pivot_row = 0

    for col in range(cols):
        candidates = np.flatnonzero(reduced[pivot_row:, col])
        if candidates.size == 0:
            continue

        source_row = pivot_row + int(candidates[0])
        if source_row != pivot_row:
            reduced[[pivot_row, source_row]] = reduced[[source_row, pivot_row]]

        for row in range(rows):
            if row != pivot_row and reduced[row, col]:
                reduced[row] ^= reduced[pivot_row]

        pivot_columns.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    return RowReduction(
        matrix=reduced,
        pivot_columns=tuple(pivot_columns),
        rank=len(pivot_columns),
    )


def rank(matrix: np.ndarray) -> int:
    """Return the GF(2) rank of a binary matrix."""
    return row_reduce(matrix).rank


def row_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a reduced independent row basis for the rowspace of a binary matrix."""
    reduced = row_reduce(matrix)
    return reduced.matrix[: reduced.rank].copy()


def nullspace(matrix: np.ndarray) -> np.ndarray:
    """Return row vectors forming a basis for the right nullspace over GF(2)."""
    reduced = row_reduce(matrix)
    rows, cols = reduced.matrix.shape
    pivots = set(reduced.pivot_columns)
    free_columns = [col for col in range(cols) if col not in pivots]

    kernel = np.zeros((len(free_columns), cols), dtype=np.uint8)
    for out_row, free_col in enumerate(free_columns):
        kernel[out_row, free_col] = 1
        for pivot_row, pivot_col in enumerate(reduced.pivot_columns):
            if pivot_row < rows and reduced.matrix[pivot_row, free_col]:
                kernel[out_row, pivot_col] = 1

    return kernel


def binary_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a full-rank square binary matrix over GF(2)."""
    matrix = _as_gf2_matrix(matrix)
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Only square binary matrices can be inverted")

    augmented = np.hstack([matrix, np.eye(rows, dtype=np.uint8)])
    reduced = row_reduce(augmented)
    left = reduced.matrix[:, :cols]
    if reduced.rank != rows or not np.array_equal(left, np.eye(rows, dtype=np.uint8)):
        raise ValueError("Matrix is singular over GF(2)")

    return reduced.matrix[:, cols:].copy()


def extend_independent_rows(
    base: np.ndarray,
    candidates: Iterable[np.ndarray],
    count: int,
) -> np.ndarray:
    """Select candidate rows that extend the rowspace of base by count dimensions."""
    if count < 0:
        raise ValueError("count must be non-negative")

    current = row_basis(base)
    if count == 0:
        return np.zeros((0, current.shape[1]), dtype=np.uint8)

    selected = []
    current_rank = rank(current)

    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=np.int64) % 2
        if candidate.ndim == 2 and candidate.shape[0] == 1:
            candidate = candidate[0]
        if candidate.ndim != 1:
            raise ValueError("Candidate rows must be one-dimensional")
        candidate = candidate.astype(np.uint8, copy=False).reshape(1, -1)
        trial = np.vstack([current, candidate])
        trial_rank = rank(trial)
        if trial_rank > current_rank:
            selected.append(candidate.reshape(-1))
            current = row_basis(trial)
            current_rank = trial_rank
            if len(selected) == count:
                break

    if len(selected) != count:
        raise ValueError("Could not find enough independent quotient representatives")
    return np.asarray(selected, dtype=np.uint8)


def quotient_basis(super_space: np.ndarray, sub_space: np.ndarray, dimension: int) -> np.ndarray:
    """Return independent representatives for super_space / sub_space.

    Rows of ``super_space`` are treated as candidate representatives. For CSS
    logical-basis use, ``sub_space`` is assumed to lie within that rowspace, and
    ``dimension`` independent rows are selected modulo ``sub_space``.
    """
    return extend_independent_rows(sub_space, super_space, dimension)


def css_logical_basis(hx: np.ndarray, hz: np.ndarray) -> CSSLogicalBasisData:
    """Compute paired X and Z logical bases for a binary CSS code."""
    hx = _as_gf2_matrix(hx)
    hz = _as_gf2_matrix(hz)
    if hx.shape[1] != hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns")
    if np.any((hx @ hz.T) % 2):
        raise ValueError("CSS check matrices must commute")

    n = hx.shape[1]
    k = n - rank(hx) - rank(hz)
    if k < 0:
        raise ValueError("Computed a negative number of logical qubits")
    if k == 0:
        return CSSLogicalBasisData(
            x=np.zeros((0, n), dtype=np.uint8),
            z=np.zeros((0, n), dtype=np.uint8),
        )

    x_candidates = nullspace(hz)
    z_candidates = nullspace(hx)
    lx = quotient_basis(x_candidates, row_basis(hx), k)
    lz = quotient_basis(z_candidates, row_basis(hz), k)

    pairing = (lx @ lz.T) % 2
    inverse_pairing = binary_inverse(pairing)
    paired_lz = (inverse_pairing.T @ lz) % 2

    if not np.array_equal((lx @ paired_lz.T) % 2, np.eye(k, dtype=np.uint8)):
        raise ValueError("Failed to construct paired CSS logical bases")

    return CSSLogicalBasisData(x=lx.astype(np.uint8), z=paired_lz.astype(np.uint8))
