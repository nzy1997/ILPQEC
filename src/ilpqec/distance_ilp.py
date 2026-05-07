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
