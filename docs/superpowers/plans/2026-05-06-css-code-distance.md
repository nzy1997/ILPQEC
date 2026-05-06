# CSS Code Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add binary CSS-code analysis APIs that compute exact code distance, shortest X/Z logical operators, and optionally reduced paired logical bases from `Hx` and `Hz`.

**Architecture:** Keep code analysis separate from syndrome decoding by adding `CSSCode` plus focused GF(2) and ILP helper modules. Compute paired logical bases with pure GF(2) Gaussian elimination, then use exact ILP models for global distance and per-coset logical reduction.

**Tech Stack:** Python 3.9+, NumPy, optional SciPy sparse inputs, direct HiGHS via `highspy`, optional Pyomo/Gurobi conventions inherited from existing solver configuration, pytest.

---

## File Structure

- Create `src/ilpqec/gf2.py`: pure GF(2) row reduction, rank, nullspace, binary matrix inverse, quotient-basis extraction, and paired CSS logical basis construction.
- Create `src/ilpqec/distance_ilp.py`: exact minimum-Hamming-weight binary ILP routines for fixed parity syndromes and nonzero logical pairing constraints.
- Create `src/ilpqec/css_code.py`: `CSSCode`, `CSSDistanceResult`, `CSSLogicalBasis`, input normalization, validation, caching, and public API orchestration.
- Modify `src/ilpqec/__init__.py`: export `CSSCode`, `CSSDistanceResult`, and `CSSLogicalBasis`.
- Create `tests/test_gf2.py`: solver-free tests for GF(2) primitives and paired logical bases.
- Create `tests/test_css_code.py`: validation and unreduced CSS API tests that do not require a solver.
- Create `tests/test_css_distance.py`: solver-gated distance and reduced logical basis tests.
- Create `docs/css_code.md`: documentation page for CSS code analysis.
- Modify `README.md`, `docs/index.md`, and `mkdocs.yml`: add short user-facing links and examples.

---

### Task 1: GF(2) Linear Algebra Helpers

**Files:**
- Create: `src/ilpqec/gf2.py`
- Create: `tests/test_gf2.py`

- [ ] **Step 1: Write failing tests for row reduction, rank, nullspace, and binary inverse**

Create `tests/test_gf2.py` with this initial content:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_gf2.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'ilpqec.gf2'`.

- [ ] **Step 3: Implement GF(2) primitives**

Create `src/ilpqec/gf2.py`:

```python
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


def _as_gf2_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.uint8) % 2
    if array.ndim != 2:
        raise ValueError("GF(2) matrix must be two-dimensional")
    return array.copy()


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
    current = row_basis(base)
    selected = []
    current_rank = rank(current)

    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=np.uint8).reshape(1, -1) % 2
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
```

- [ ] **Step 4: Run GF(2) primitive tests**

Run:

```bash
uv run pytest tests/test_gf2.py -v
```

Expected: PASS for the five tests in `tests/test_gf2.py`.

- [ ] **Step 5: Commit GF(2) primitives**

Run:

```bash
git add src/ilpqec/gf2.py tests/test_gf2.py
git commit -m "Add GF2 linear algebra helpers"
```

Expected: commit succeeds and `git status --short` only shows unrelated `refs/` if it is still untracked.

---

### Task 2: Paired CSS Logical Basis Construction

**Files:**
- Modify: `src/ilpqec/gf2.py`
- Modify: `tests/test_gf2.py`

- [ ] **Step 1: Add failing tests for paired CSS logical bases**

Append to `tests/test_gf2.py`:

```python
from ilpqec.gf2 import css_logical_basis


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_gf2.py::test_css_logical_basis_for_steane_code_is_paired tests/test_gf2.py::test_css_logical_basis_for_two_logical_code_is_paired -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `css_logical_basis` is not implemented.

- [ ] **Step 3: Implement `CSSLogicalBasisData` and `css_logical_basis`**

Append these definitions to `src/ilpqec/gf2.py`:

```python

@dataclass(frozen=True)
class CSSLogicalBasisData:
    """Paired binary CSS logical bases."""

    x: np.ndarray
    z: np.ndarray


def quotient_basis(super_space: np.ndarray, sub_space: np.ndarray, dimension: int) -> np.ndarray:
    """Return representatives for super_space / sub_space."""
    return extend_independent_rows(sub_space, super_space, dimension)


def css_logical_basis(hx: np.ndarray, hz: np.ndarray) -> CSSLogicalBasisData:
    """Compute paired X and Z logical bases for a binary CSS code."""
    hx = _as_gf2_matrix(hx)
    hz = _as_gf2_matrix(hz)
    if hx.shape[1] != hz.shape[1]:
        raise ValueError("Hx and Hz must have the same number of columns")

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
```

- [ ] **Step 4: Run all GF(2) tests**

Run:

```bash
uv run pytest tests/test_gf2.py -v
```

Expected: PASS for all GF(2) tests.

- [ ] **Step 5: Commit paired logical basis helpers**

Run:

```bash
git add src/ilpqec/gf2.py tests/test_gf2.py
git commit -m "Add paired CSS logical basis helpers"
```

Expected: commit succeeds.

---

### Task 3: CSSCode API, Validation, and Unreduced Logical Basis

**Files:**
- Create: `src/ilpqec/css_code.py`
- Modify: `src/ilpqec/__init__.py`
- Create: `tests/test_css_code.py`

- [ ] **Step 1: Write failing tests for CSSCode construction and validation**

Create `tests/test_css_code.py`:

```python
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
```

- [ ] **Step 2: Run CSSCode tests to verify they fail**

Run:

```bash
uv run pytest tests/test_css_code.py -v
```

Expected: FAIL during import because `CSSCode` is not exported.

- [ ] **Step 3: Implement CSSCode and result dataclasses**

Create `src/ilpqec/css_code.py`:

```python
"""Binary CSS code analysis APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ilpqec.gf2 import css_logical_basis, rank


@dataclass(frozen=True)
class CSSDistanceResult:
    """Exact CSS code distance result."""

    d: int
    dx: int
    dz: int
    shortest_x: np.ndarray
    shortest_z: np.ndarray


@dataclass(frozen=True)
class CSSLogicalBasis:
    """Paired X/Z logical operators for a CSS code."""

    x: np.ndarray
    z: np.ndarray


def _to_binary_matrix(matrix, name: str) -> np.ndarray:
    try:
        from scipy.sparse import spmatrix  # type: ignore
    except Exception:
        spmatrix = None

    if spmatrix is not None and isinstance(matrix, spmatrix):
        array = np.asarray(matrix.toarray())
    else:
        if spmatrix is None and hasattr(matrix, "toarray"):
            raise ImportError(
                "Sparse CSS parity-check matrices require SciPy. Install with: pip install scipy"
            )
        array = np.asarray(matrix)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional binary matrix")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be binary")
    return array.astype(np.uint8, copy=True)


class CSSCode:
    """Binary CSS code built from X- and Z-type parity-check matrices."""

    def __init__(self, hx: np.ndarray, hz: np.ndarray):
        self._hx = hx
        self._hz = hz
        self._rank_x = rank(hx)
        self._rank_z = rank(hz)
        self._k = self.n - self._rank_x - self._rank_z
        if self._k < 0:
            raise ValueError("Computed a negative number of logical qubits")
        self._logical_basis: Optional[CSSLogicalBasis] = None

    @classmethod
    def from_parity_check_matrices(
        cls,
        Hx,
        Hz,
        *,
        validate_commutation: bool = True,
    ) -> "CSSCode":
        """Create a binary CSS code from X- and Z-type parity-check matrices."""
        hx = _to_binary_matrix(Hx, "Hx")
        hz = _to_binary_matrix(Hz, "Hz")
        if hx.shape[1] != hz.shape[1]:
            raise ValueError("Hx and Hz must have the same number of columns")
        if validate_commutation and np.any((hx @ hz.T) % 2):
            raise ValueError("CSS parity checks must commute: Hx @ Hz.T must be zero mod 2")
        return cls(hx, hz)

    @property
    def hx(self) -> np.ndarray:
        """X-type parity-check matrix."""
        return self._hx.copy()

    @property
    def hz(self) -> np.ndarray:
        """Z-type parity-check matrix."""
        return self._hz.copy()

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return int(self._hx.shape[1])

    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return int(self._k)

    @property
    def rank_x(self) -> int:
        """Rank of the X-check matrix."""
        return int(self._rank_x)

    @property
    def rank_z(self) -> int:
        """Rank of the Z-check matrix."""
        return int(self._rank_z)

    def _require_logicals(self) -> None:
        if self.k == 0:
            raise ValueError("CSS code has no logical qubits; logical operators are undefined")

    def _canonical_logical_basis(self) -> CSSLogicalBasis:
        if self._logical_basis is None:
            data = css_logical_basis(self._hx, self._hz)
            self._logical_basis = CSSLogicalBasis(x=data.x.copy(), z=data.z.copy())
        return self._logical_basis

    def logical_basis(
        self,
        *,
        reduce: bool = False,
        solver: str = None,
        **solver_options,
    ) -> CSSLogicalBasis:
        """Return paired logical bases, optionally reduced with exact ILP."""
        self._require_logicals()
        if reduce:
            raise NotImplementedError("Reduced logical basis is implemented in the ILP task")
        basis = self._canonical_logical_basis()
        return CSSLogicalBasis(x=basis.x.copy(), z=basis.z.copy())

    def distance(
        self,
        *,
        solver: str = None,
        **solver_options,
    ) -> CSSDistanceResult:
        """Return exact CSS code distance and shortest X/Z logicals."""
        self._require_logicals()
        raise NotImplementedError("Distance is implemented in the ILP task")
```

- [ ] **Step 4: Export CSSCode symbols**

Modify `src/ilpqec/__init__.py` so the imports and `__all__` include the new API:

```python
from ilpqec.css_code import CSSCode, CSSDistanceResult, CSSLogicalBasis
from ilpqec.decoder import Decoder
from ilpqec.solver import get_available_solvers, get_default_solver, SolverConfig

__version__ = "0.1.0"
__all__ = [
    "CSSCode",
    "CSSDistanceResult",
    "CSSLogicalBasis",
    "Decoder",
    "get_available_solvers",
    "get_default_solver",
    "SolverConfig",
]
```

- [ ] **Step 5: Run solver-free CSSCode tests**

Run:

```bash
uv run pytest tests/test_css_code.py -v
```

Expected: PASS for construction, validation, and unreduced logical basis tests.

- [ ] **Step 6: Commit CSSCode API skeleton**

Run:

```bash
git add src/ilpqec/css_code.py src/ilpqec/__init__.py tests/test_css_code.py
git commit -m "Add CSSCode API and validation"
```

Expected: commit succeeds.

---

### Task 4: Exact ILP Helper for Fixed and Nonzero Logical Constraints

**Files:**
- Create: `src/ilpqec/distance_ilp.py`
- Create: `tests/test_css_distance.py`

- [ ] **Step 1: Write failing ILP helper tests**

Create `tests/test_css_distance.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_css_distance.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'ilpqec.distance_ilp'`.

- [ ] **Step 3: Implement direct HiGHS ILP helper**

Create `src/ilpqec/distance_ilp.py`:

```python
"""Exact ILP helpers for CSS distance calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ilpqec.solver import SolverConfig, get_default_solver


@dataclass(frozen=True)
class MinWeightResult:
    """Minimum-weight binary vector returned by an exact ILP."""

    vector: np.ndarray
    weight: int
    objective: float
    status: str


def _as_binary_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.uint8) % 2
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return array


def _as_binary_vector(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.uint8) % 2
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _exact_solver_config(solver: Optional[str], options: dict[str, Any]) -> SolverConfig:
    solver_name = (solver or get_default_solver()).lower()
    gap = options.pop("gap", None)
    if gap not in (None, 0, 0.0):
        raise ValueError(
            "CSS distance APIs require exact optimization; positive gap is not allowed"
        )

    direct = options.pop("direct", None)
    if direct is None:
        direct = solver_name == "highs"
    if solver_name != "highs" or not direct:
        raise ValueError("CSS distance ILP currently supports the direct HiGHS backend")

    return SolverConfig(
        name=solver_name,
        time_limit=options.pop("time_limit", None),
        gap=0.0,
        threads=options.pop("threads", None),
        verbose=options.pop("verbose", False),
        direct=True,
        options=options,
    )


def minimize_weight_with_fixed_syndrome(
    parity_check_matrix: np.ndarray,
    syndrome: np.ndarray,
    *,
    solver: str = None,
    **solver_options,
) -> MinWeightResult:
    """Minimize Hamming weight subject to Mx = syndrome over GF(2)."""
    matrix = _as_binary_matrix(parity_check_matrix, "parity_check_matrix")
    rhs = _as_binary_vector(syndrome, "syndrome")
    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError("Syndrome length must match the number of parity-check rows")

    config = _exact_solver_config(solver, dict(solver_options))
    return _solve_direct_highs(matrix, rhs, selector_matrix=None, config=config)


def minimize_nonzero_logical_operator(
    check_matrix: np.ndarray,
    dual_logicals: np.ndarray,
    *,
    solver: str = None,
    **solver_options,
) -> MinWeightResult:
    """Minimize Hamming weight with zero syndrome and nonzero dual-logical pairing."""
    checks = _as_binary_matrix(check_matrix, "check_matrix")
    duals = _as_binary_matrix(dual_logicals, "dual_logicals")
    if checks.shape[1] != duals.shape[1]:
        raise ValueError("check_matrix and dual_logicals must have the same number of columns")
    if duals.shape[0] == 0:
        raise ValueError("At least one dual logical is required")

    rhs = np.zeros(checks.shape[0], dtype=np.uint8)
    config = _exact_solver_config(solver, dict(solver_options))
    return _solve_direct_highs(checks, rhs, selector_matrix=duals, config=config)


def _solve_direct_highs(
    fixed_matrix: np.ndarray,
    fixed_rhs: np.ndarray,
    *,
    selector_matrix: Optional[np.ndarray],
    config: SolverConfig,
) -> MinWeightResult:
    try:
        from highspy import (
            Highs,
            HighsLp,
            HighsModelStatus,
            HighsSparseMatrix,
            HighsStatus,
            HighsVarType,
            MatrixFormat,
        )
    except Exception as exc:
        raise ImportError(
            "Direct HiGHS backend requires highspy. Install with: pip install highspy"
        ) from exc

    fixed_matrix = np.asarray(fixed_matrix, dtype=np.uint8)
    fixed_rhs = np.asarray(fixed_rhs, dtype=np.uint8)
    selector_matrix = (
        None if selector_matrix is None else np.asarray(selector_matrix, dtype=np.uint8)
    )

    n = fixed_matrix.shape[1]
    num_fixed = fixed_matrix.shape[0]
    num_selectors = 0 if selector_matrix is None else selector_matrix.shape[0]
    num_aux = num_fixed + num_selectors
    num_cols = n + num_aux + num_selectors
    sum_selector_row = 1 if num_selectors else 0
    num_rows = num_fixed + num_selectors + sum_selector_row

    highs = Highs()
    _set_highs_option(highs, HighsStatus, "output_flag", bool(config.verbose))
    if config.time_limit is not None:
        _set_highs_option(highs, HighsStatus, "time_limit", float(config.time_limit))
    if config.threads is not None:
        _set_highs_option(highs, HighsStatus, "threads", int(config.threads))
    _set_highs_option(highs, HighsStatus, "mip_rel_gap", 0.0)
    for key, value in config.options.items():
        _set_highs_option(highs, HighsStatus, key, value)

    col_cost = [0.0] * num_cols
    col_lower = [0.0] * num_cols
    col_upper = [0.0] * num_cols
    integrality = [HighsVarType.kInteger] * num_cols

    for col in range(n):
        col_cost[col] = 1.0
        col_upper[col] = 1.0

    for row in range(num_aux):
        col_upper[n + row] = float(n)

    selector_offset = n + num_aux
    for row in range(num_selectors):
        col_upper[selector_offset + row] = 1.0

    row_lower = [0.0] * num_rows
    row_upper = [0.0] * num_rows

    for row in range(num_fixed):
        rhs = float(fixed_rhs[row])
        row_lower[row] = rhs
        row_upper[row] = rhs

    for row in range(num_selectors):
        idx = num_fixed + row
        row_lower[idx] = 0.0
        row_upper[idx] = 0.0

    if num_selectors:
        row_lower[-1] = 1.0
        row_upper[-1] = float(num_selectors)

    entries = [[] for _ in range(num_cols)]
    for row in range(num_fixed):
        for col in np.flatnonzero(fixed_matrix[row]):
            entries[int(col)].append((row, 1.0))
        entries[n + row].append((row, -2.0))

    for row in range(num_selectors):
        constraint_row = num_fixed + row
        for col in np.flatnonzero(selector_matrix[row]):
            entries[int(col)].append((constraint_row, 1.0))
        entries[n + num_fixed + row].append((constraint_row, -2.0))
        entries[selector_offset + row].append((constraint_row, -1.0))
        entries[selector_offset + row].append((num_rows - 1, 1.0))

    starts = [0]
    indices = []
    values = []
    for col_entries in entries:
        for row, value in col_entries:
            indices.append(int(row))
            values.append(float(value))
        starts.append(len(indices))

    matrix = HighsSparseMatrix()
    matrix.num_row_ = num_rows
    matrix.num_col_ = num_cols
    matrix.start_ = starts
    matrix.index_ = indices
    matrix.value_ = values
    matrix.format_ = MatrixFormat.kColwise

    lp = HighsLp()
    lp.num_col_ = num_cols
    lp.num_row_ = num_rows
    lp.col_cost_ = col_cost
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.integrality_ = integrality
    lp.a_matrix_ = matrix

    status = highs.passModel(lp)
    if status != HighsStatus.kOk:
        raise RuntimeError("Failed to initialize HiGHS model")

    status = highs.run()
    if status != HighsStatus.kOk:
        raise RuntimeError("HiGHS failed to solve CSS distance model")

    model_status = highs.getModelStatus()
    if model_status != HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS did not prove optimality: {model_status}")

    solution = highs.getSolution()
    vector = (np.asarray(solution.col_value[:n], dtype=float) > 0.5).astype(np.uint8)
    objective = float(highs.getObjectiveValue())
    return MinWeightResult(
        vector=vector,
        weight=int(vector.sum()),
        objective=objective,
        status=str(model_status),
    )


def _set_highs_option(highs, highs_status, key: str, value: Any) -> None:
    status = highs.setOptionValue(key, value)
    if status != highs_status.kOk:
        raise ValueError(f"HiGHS rejected option '{key}'")
```

- [ ] **Step 4: Run ILP helper tests**

Run:

```bash
uv run pytest tests/test_css_distance.py -v
```

Expected: PASS when HiGHS is available; otherwise the tests are skipped.

- [ ] **Step 5: Commit ILP helper**

Run:

```bash
git add src/ilpqec/distance_ilp.py tests/test_css_distance.py
git commit -m "Add exact CSS distance ILP helper"
```

Expected: commit succeeds.

---

### Task 5: Wire Exact Distance and Reduced Logical Basis into CSSCode

**Files:**
- Modify: `src/ilpqec/css_code.py`
- Modify: `tests/test_css_distance.py`
- Modify: `tests/test_css_code.py`

- [ ] **Step 1: Add failing public API tests for distance and reduced basis**

Append to `tests/test_css_distance.py`:

```python
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

    basis = code.logical_basis(reduce=True, solver="highs")

    np.testing.assert_array_equal((basis.x @ basis.z.T) % 2, np.eye(2, dtype=np.uint8))
    for index in range(code.k):
        rhs = np.zeros(code.k, dtype=np.uint8)
        rhs[index] = 1
        expected_x = brute_force_fixed(hz, basis.z, rhs)
        expected_z = brute_force_fixed(hx, basis.x, rhs)
        assert int(basis.x[index].sum()) == int(expected_x.sum())
        assert int(basis.z[index].sum()) == int(expected_z.sum())
```

Append to `tests/test_css_code.py`:

```python

def test_distance_raises_for_zero_logical_qubits():
    hx = np.eye(2, dtype=np.uint8)
    hz = np.zeros((0, 2), dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)

    with pytest.raises(ValueError, match="no logical qubits"):
        code.distance()
```

- [ ] **Step 2: Run public API tests to verify they fail**

Run:

```bash
uv run pytest tests/test_css_code.py::test_distance_raises_for_zero_logical_qubits tests/test_css_distance.py::test_css_code_distance_for_steane_code tests/test_css_distance.py::test_reduced_logical_basis_is_paired_and_fixed_coset_minimal -v
```

Expected: FAIL with `NotImplementedError` from `CSSCode.distance()` or `CSSCode.logical_basis(reduce=True)`.

- [ ] **Step 3: Implement `CSSCode.distance()` and reduced logical basis**

Replace the `logical_basis` and `distance` methods in `src/ilpqec/css_code.py` with:

```python
    def logical_basis(
        self,
        *,
        reduce: bool = False,
        solver: str = None,
        **solver_options,
    ) -> CSSLogicalBasis:
        """Return paired logical bases, optionally reduced with exact ILP."""
        self._require_logicals()
        basis = self._canonical_logical_basis()
        if not reduce:
            return CSSLogicalBasis(x=basis.x.copy(), z=basis.z.copy())

        from ilpqec.distance_ilp import minimize_weight_with_fixed_syndrome

        reduced_x = np.zeros_like(basis.x)
        reduced_z = np.zeros_like(basis.z)
        for index in range(self.k):
            rhs = np.zeros(self.k, dtype=np.uint8)
            rhs[index] = 1

            x_matrix = np.vstack([self._hz, basis.z])
            x_syndrome = np.concatenate([np.zeros(self._hz.shape[0], dtype=np.uint8), rhs])
            reduced_x[index] = minimize_weight_with_fixed_syndrome(
                x_matrix, x_syndrome, solver=solver, **solver_options
            ).vector

            z_matrix = np.vstack([self._hx, basis.x])
            z_syndrome = np.concatenate([np.zeros(self._hx.shape[0], dtype=np.uint8), rhs])
            reduced_z[index] = minimize_weight_with_fixed_syndrome(
                z_matrix, z_syndrome, solver=solver, **solver_options
            ).vector

        return CSSLogicalBasis(x=reduced_x, z=reduced_z)

    def distance(
        self,
        *,
        solver: str = None,
        **solver_options,
    ) -> CSSDistanceResult:
        """Return exact CSS code distance and shortest X/Z logicals."""
        self._require_logicals()

        from ilpqec.distance_ilp import minimize_nonzero_logical_operator

        basis = self._canonical_logical_basis()
        shortest_x = minimize_nonzero_logical_operator(
            self._hz, basis.z, solver=solver, **solver_options
        )
        shortest_z = minimize_nonzero_logical_operator(
            self._hx, basis.x, solver=solver, **solver_options
        )
        return CSSDistanceResult(
            d=min(shortest_x.weight, shortest_z.weight),
            dx=shortest_x.weight,
            dz=shortest_z.weight,
            shortest_x=shortest_x.vector.copy(),
            shortest_z=shortest_z.vector.copy(),
        )
```

- [ ] **Step 4: Run CSS distance tests**

Run:

```bash
uv run pytest tests/test_css_code.py tests/test_css_distance.py -v
```

Expected: PASS when HiGHS is available; solver-gated tests skip if HiGHS is unavailable.

- [ ] **Step 5: Commit public distance and reduced logical basis APIs**

Run:

```bash
git add src/ilpqec/css_code.py tests/test_css_code.py tests/test_css_distance.py
git commit -m "Wire CSS distance and logical reduction APIs"
```

Expected: commit succeeds.

---

### Task 6: Documentation and Package Navigation

**Files:**
- Create: `docs/css_code.md`
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Write documentation page**

Create `docs/css_code.md`:

```markdown
# CSS Code Distance

ILPQEC can analyze binary CSS stabilizer codes from X- and Z-type parity-check
matrices. This is separate from syndrome decoding: the API computes exact code
distance and logical operators of the code itself.

## Input Convention

For a CSS code on `n` qubits:

- `Hx` is the matrix of X-type stabilizer checks.
- `Hz` is the matrix of Z-type stabilizer checks.
- Both matrices must have `n` columns.
- By default ILPQEC checks `Hx @ Hz.T == 0 mod 2`.

## Distance

```python
import numpy as np
from ilpqec import CSSCode

H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

code = CSSCode.from_parity_check_matrices(H, H)
result = code.distance()

print(result.d)          # 3
print(result.dx, result.dz)
print(result.shortest_x) # one minimum-weight X logical
print(result.shortest_z) # one minimum-weight Z logical
```

The distance calculation solves exact integer programs. If the solver does not prove
optimality, ILPQEC raises an error instead of returning an approximate distance.

## Logical Bases

```python
basis = code.logical_basis(reduce=False)
print(basis.x @ basis.z.T % 2)
```

`reduce=False` returns a paired basis from Gaussian elimination. It is valid, but not
necessarily low weight.

```python
reduced = code.logical_basis(reduce=True)
```

`reduce=True` uses exact ILP to reduce every fixed logical coset to a minimum-weight
representative. This does not search over all possible logical-basis choices; it reduces
the canonical cosets selected by Gaussian elimination.

## Scaling

Exact distance is NP-hard in general. This feature is intended for correctness-focused
baselines, small-to-medium code studies, and checking other constructions. Large LDPC
instances may require long solve times.
```

- [ ] **Step 2: Add README quickstart section**

Insert this section in `README.md` after the parity-check matrix decoding quickstart and before Stim DEM decoding:

````markdown
### CSS Code Distance

```python
import numpy as np
from ilpqec import CSSCode

H = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

code = CSSCode.from_parity_check_matrices(H, H)
distance = code.distance()

print(distance.d)
print(distance.shortest_x)
print(distance.shortest_z)
```

For paired logical generators, use `code.logical_basis(reduce=False)` for the Gaussian
elimination basis or `code.logical_basis(reduce=True)` to minimize each fixed logical
coset with ILP.
````

- [ ] **Step 3: Link docs index and nav**

Add a bullet to `docs/index.md` under Documentation Map:

```markdown
- CSS code distance and logical operators: `css_code.md`
```

Add a nav entry to `mkdocs.yml` after Mathematical Formulation:

```yaml
  - CSS Code Distance: css_code.md
```

- [ ] **Step 4: Run documentation-adjacent checks**

Run:

```bash
uv run pytest tests/test_css_code.py tests/test_css_distance.py -v
```

Expected: PASS for solver-free tests and PASS or SKIP for solver-gated tests depending on HiGHS availability.

- [ ] **Step 5: Commit docs**

Run:

```bash
git add docs/css_code.md README.md docs/index.md mkdocs.yml
git commit -m "Document CSS code distance API"
```

Expected: commit succeeds.

---

### Task 7: Full Verification and Cleanup

**Files:**
- Modify only if verification finds a defect: files touched in earlier tasks

- [ ] **Step 1: Run the full test suite**

Run:

```bash
uv run pytest tests/ -v
```

Expected: PASS, with any optional solver tests skipped only when their backend is unavailable.

- [ ] **Step 2: Run lint if available in the environment**

Run:

```bash
uv run ruff check src/ilpqec
```

Expected: PASS. If ruff is unavailable in the environment, install dev dependencies with `uv sync --extra dev` and rerun.

- [ ] **Step 3: Run type checking if available in the environment**

Run:

```bash
uv run mypy src/ilpqec
```

Expected: PASS. If mypy reports missing optional imports for solver backends, follow the existing repository configuration rather than adding broad ignores.

- [ ] **Step 4: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional changes are present. The pre-existing untracked `refs/` directory may remain untracked and should not be added.

- [ ] **Step 5: Commit verification fixes only if needed**

If Step 1, 2, or 3 required code or test fixes, run:

```bash
git add src/ilpqec tests docs README.md mkdocs.yml
git commit -m "Fix CSS code distance verification issues"
```

Expected: commit succeeds. Skip this commit if no fixes were needed.

---

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover GF(2) basis construction, Task 3 covers public API and validation, Tasks 4-5 cover exact ILP distance and per-coset reduction, Task 6 covers docs, and Task 7 covers verification.
- Exactness: Task 4 rejects positive MIP gaps and requires HiGHS optimal status before returning results.
- Scope: The plan implements binary CSS stabilizer codes only and does not add non-CSS, qudit, subsystem, randomized, or approximate-distance APIs.
- Type consistency: The plan consistently uses `CSSCode`, `CSSDistanceResult`, `CSSLogicalBasis`, `distance()`, `logical_basis()`, `hx`, `hz`, `n`, `k`, `rank_x`, and `rank_z`.
- Worktree hygiene: The pre-existing untracked `refs/` directory is explicitly left untracked.
