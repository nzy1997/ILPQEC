"""Extra coverage tests for CSS analysis helpers and edge cases."""

from __future__ import annotations

import builtins
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import ilpqec.css_code as css_code_module
import ilpqec.distance_ilp as distance_ilp_module
from ilpqec import CSSCode
from ilpqec.distance_ilp import (
    _exact_solver_config,
    _set_highs_option,
    minimize_nonzero_logical_operator,
    minimize_weight_with_fixed_syndrome,
)
from ilpqec.solver import SolverConfig


def steane_check_matrix():
    return np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def _install_fake_highspy(
    monkeypatch,
    *,
    option_status=0,
    pass_status=0,
    run_status=0,
    model_status=1,
    objective=2.0,
    col_value=None,
):
    module = types.ModuleType("highspy")

    class FakeHighsStatus:
        kOk = 0
        kError = -1

    class FakeHighsModelStatus:
        kOptimal = 1
        kTimeLimit = 2

    class FakeHighsVarType:
        kInteger = 0

    class FakeMatrixFormat:
        kColwise = 0

    class FakeHighsSparseMatrix:
        pass

    class FakeHighsLp:
        pass

    class FakeSolution:
        def __init__(self, values):
            self.col_value = values

    class FakeHighs:
        def __init__(self):
            self.options = {}
            self.num_cols = 0

        def setOptionValue(self, key, value):
            self.options[key] = value
            return option_status

        def passModel(self, lp):
            self.num_cols = lp.num_col_
            return pass_status

        def run(self):
            return run_status

        def getModelStatus(self):
            return model_status

        def getSolution(self):
            if col_value is None:
                values = [1.0] + [0.0] * max(0, self.num_cols - 1)
            else:
                values = list(col_value)
            return FakeSolution(values)

        def getObjectiveValue(self):
            return objective

    module.Highs = FakeHighs
    module.HighsLp = FakeHighsLp
    module.HighsModelStatus = FakeHighsModelStatus
    module.HighsSparseMatrix = FakeHighsSparseMatrix
    module.HighsStatus = FakeHighsStatus
    module.HighsVarType = FakeHighsVarType
    module.MatrixFormat = FakeMatrixFormat
    monkeypatch.setitem(sys.modules, "highspy", module)
    return FakeHighsStatus, FakeHighsModelStatus


def test_to_binary_matrix_rejects_non_two_dimensional_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        css_code_module._to_binary_matrix(np.array([1, 0, 1]), "Hx")


def test_to_binary_matrix_accepts_scipy_sparse_if_available():
    scipy = pytest.importorskip("scipy.sparse")
    matrix = scipy.eye(2, dtype=np.uint8, format="csr")

    out = css_code_module._to_binary_matrix(matrix, "Hx")

    np.testing.assert_array_equal(out, np.eye(2, dtype=np.uint8))


def test_to_binary_matrix_requires_scipy_for_sparse_like_objects(monkeypatch):
    class DummySparse:
        def toarray(self):
            return np.eye(2, dtype=np.uint8)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("no scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="Sparse CSS parity-check matrices require SciPy"):
        css_code_module._to_binary_matrix(DummySparse(), "Hx")


def test_css_code_init_rejects_negative_logical_qubit_count():
    hx = np.eye(2, dtype=np.uint8)
    hz = np.eye(2, dtype=np.uint8)

    with pytest.raises(ValueError, match="negative number of logical qubits"):
        CSSCode(hx, hz)


def test_css_code_init_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same number of columns"):
        CSSCode(np.zeros((1, 3), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8))


def test_canonical_logical_basis_is_cached_once(monkeypatch):
    calls = []

    def fake_basis(hx, hz):
        calls.append((hx.copy(), hz.copy()))
        return SimpleNamespace(
            x=np.array([[1, 0, 0]], dtype=np.uint8),
            z=np.array([[0, 1, 0]], dtype=np.uint8),
        )

    monkeypatch.setattr(css_code_module, "css_logical_basis", fake_basis)
    code = CSSCode(np.array([[1, 1, 0]], dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8))

    basis1 = code.logical_basis(reduce=False)
    basis2 = code.logical_basis(reduce=False)
    basis1.x[0, 0] = 0
    basis1.z[0, 1] = 0

    assert len(calls) == 1
    np.testing.assert_array_equal(basis2.x, np.array([[1, 0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(basis2.z, np.array([[0, 1, 0]], dtype=np.uint8))


def test_distance_forwards_solver_options_and_returns_copies(monkeypatch):
    h = steane_check_matrix()
    code = CSSCode.from_parity_check_matrices(h, h)
    x_vec = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    z_vec = np.array([0, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    calls = []

    def fake_minimize(check_matrix, dual_logicals, *, solver=None, **solver_options):
        calls.append((check_matrix.copy(), dual_logicals.copy(), solver, dict(solver_options)))
        if len(calls) % 2 == 1:
            return SimpleNamespace(vector=x_vec, weight=3)
        return SimpleNamespace(vector=z_vec, weight=2)

    monkeypatch.setattr(css_code_module, "minimize_nonzero_logical_operator", fake_minimize)

    result = code.distance(solver="highs", threads=2, verbose=True)
    result.shortest_x[0] = 0
    result.shortest_z[3] = 0
    again = code.distance(solver="highs", threads=2, verbose=True)

    assert result.d == 2
    assert result.dx == 3
    assert result.dz == 2
    assert len(calls) == 4
    assert all(call[2] == "highs" for call in calls)
    assert all(call[3] == {"threads": 2, "verbose": True} for call in calls)
    np.testing.assert_array_equal(again.shortest_x, x_vec)
    np.testing.assert_array_equal(again.shortest_z, z_vec)


def test_logical_basis_reduce_forwards_solver_options(monkeypatch):
    hx = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    hz = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    code = CSSCode.from_parity_check_matrices(hx, hz)
    calls = []
    x_outputs = [
        np.array([1, 0, 1, 0], dtype=np.uint8),
        np.array([0, 1, 0, 1], dtype=np.uint8),
    ]
    z_outputs = [
        np.array([1, 0, 0, 0], dtype=np.uint8),
        np.array([0, 0, 1, 0], dtype=np.uint8),
    ]
    outputs = [x_outputs[0], z_outputs[0], x_outputs[1], z_outputs[1]]

    def fake_minimize(matrix, syndrome, *, solver=None, **solver_options):
        calls.append((matrix.copy(), syndrome.copy(), solver, dict(solver_options)))
        return SimpleNamespace(vector=outputs[len(calls) - 1])

    monkeypatch.setattr(css_code_module, "minimize_weight_with_fixed_syndrome", fake_minimize)

    basis = code.logical_basis(reduce=True, solver="highs", time_limit=5.0)

    assert len(calls) == 4
    assert all(call[2] == "highs" for call in calls)
    assert all(call[3] == {"time_limit": 5.0} for call in calls)
    np.testing.assert_array_equal(basis.x, np.vstack(x_outputs))
    np.testing.assert_array_equal(basis.z, np.vstack(z_outputs))


def test_exact_solver_config_defaults_to_highs_and_preserves_options(monkeypatch):
    monkeypatch.setattr(distance_ilp_module, "get_default_solver", lambda: "highs")

    config = _exact_solver_config(
        None,
        {"time_limit": 1.5, "threads": 4, "verbose": True, "node_limit": 7},
    )

    assert config == SolverConfig(
        name="highs",
        time_limit=1.5,
        gap=0.0,
        threads=4,
        verbose=True,
        direct=True,
        options={"node_limit": 7},
    )


def test_exact_solver_config_rejects_non_highs_and_non_direct():
    with pytest.raises(ValueError, match="direct HiGHS"):
        _exact_solver_config("cbc", {})
    with pytest.raises(ValueError, match="direct HiGHS"):
        _exact_solver_config("highs", {"direct": False})


def test_minimize_weight_with_fixed_syndrome_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="two-dimensional"):
        minimize_weight_with_fixed_syndrome(
            np.array([1, 0], dtype=np.uint8),
            np.zeros(2, dtype=np.uint8),
            solver="highs",
        )
    with pytest.raises(ValueError, match="binary values"):
        minimize_weight_with_fixed_syndrome(
            np.eye(2, dtype=np.uint8),
            np.array([0, 2], dtype=np.int64),
            solver="highs",
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        minimize_weight_with_fixed_syndrome(
            np.eye(2, dtype=np.uint8),
            np.zeros((2, 1), dtype=np.uint8),
            solver="highs",
        )
    with pytest.raises(ValueError, match="Syndrome length must match"):
        minimize_weight_with_fixed_syndrome(
            np.eye(2, dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            solver="highs",
        )


def test_minimize_nonzero_logical_operator_rejects_invalid_duals():
    with pytest.raises(ValueError, match="binary values"):
        minimize_nonzero_logical_operator(
            np.eye(2, dtype=np.uint8),
            np.array([[0, 2]], dtype=np.int64),
            solver="highs",
        )
    with pytest.raises(ValueError, match="same number of columns"):
        minimize_nonzero_logical_operator(
            np.eye(2, dtype=np.uint8),
            np.ones((1, 3), dtype=np.uint8),
            solver="highs",
        )
    with pytest.raises(ValueError, match="At least one dual logical"):
        minimize_nonzero_logical_operator(
            np.eye(2, dtype=np.uint8),
            np.zeros((0, 2), dtype=np.uint8),
            solver="highs",
        )


def test_solve_direct_highs_raises_without_highspy(monkeypatch):
    monkeypatch.delitem(sys.modules, "highspy", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "highspy":
            raise ImportError("no highspy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="requires highspy"):
        distance_ilp_module._solve_direct_highs(
            np.eye(1, dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            selector_matrix=None,
            config=SolverConfig(name="highs", direct=True),
        )


def test_set_highs_option_rejects_invalid_option():
    class FakeStatus:
        kOk = 0

    class FakeHighs:
        def setOptionValue(self, key, value):
            return -1

    with pytest.raises(ValueError, match="rejected option 'bad_option'"):
        _set_highs_option(FakeHighs(), FakeStatus, "bad_option", 3)


def test_solve_direct_highs_rejects_passmodel_failure(monkeypatch):
    _install_fake_highspy(monkeypatch, pass_status=-1)
    with pytest.raises(RuntimeError, match="initialize HiGHS model"):
        distance_ilp_module._solve_direct_highs(
            np.eye(1, dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            selector_matrix=None,
            config=SolverConfig(name="highs", direct=True),
        )


def test_solve_direct_highs_rejects_run_failure(monkeypatch):
    _install_fake_highspy(monkeypatch, run_status=-1)
    with pytest.raises(RuntimeError, match="failed to solve CSS distance model"):
        distance_ilp_module._solve_direct_highs(
            np.eye(1, dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            selector_matrix=None,
            config=SolverConfig(name="highs", direct=True),
        )


def test_solve_direct_highs_requires_optimal_status(monkeypatch):
    _, model_status = _install_fake_highspy(monkeypatch, model_status=2)
    with pytest.raises(RuntimeError, match=str(model_status.kTimeLimit)):
        distance_ilp_module._solve_direct_highs(
            np.eye(1, dtype=np.uint8),
            np.zeros(1, dtype=np.uint8),
            selector_matrix=None,
            config=SolverConfig(name="highs", direct=True),
        )


def test_solve_direct_highs_passes_solver_configuration(monkeypatch):
    _install_fake_highspy(monkeypatch, objective=4.0, col_value=[1.0, 0.0, 0.0, 0.0])

    result = distance_ilp_module._solve_direct_highs(
        np.array([[1, 0]], dtype=np.uint8),
        np.array([1], dtype=np.uint8),
        selector_matrix=np.array([[0, 1]], dtype=np.uint8),
        config=SolverConfig(
            name="highs",
            time_limit=3.0,
            threads=2,
            verbose=True,
            direct=True,
            options={"presolve": "off"},
        ),
    )

    np.testing.assert_array_equal(result.vector, np.array([1, 0], dtype=np.uint8))
    assert result.weight == 1
    assert result.objective == 4.0
