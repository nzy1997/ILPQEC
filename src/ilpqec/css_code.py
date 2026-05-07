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
                "Sparse CSS parity-check matrices require SciPy. "
                "Install with: pip install scipy"
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
        del solver, solver_options
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
        del solver, solver_options
        self._require_logicals()
        raise NotImplementedError("Distance is implemented in the ILP task")
