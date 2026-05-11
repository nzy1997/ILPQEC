"""Coverage tests for DEM distance API validation and wiring."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from ilpqec import dem_distance

dem_distance_module = importlib.import_module("ilpqec.dem_distance")


class FakeDem:
    """Minimal DEM-like object for validation-only tests."""

    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        return self._text

    def flattened(self) -> "FakeDem":
        return self


def test_dem_distance_rejects_dem_without_logical_observables():
    dem = FakeDem("error(0.1) D0\n")
    with pytest.raises(ValueError, match="at least one logical observable"):
        dem_distance(dem, solver="highs")


def test_dem_distance_rejects_target_length_mismatch():
    dem = FakeDem("error(0.1) D0 L0\n")

    with pytest.raises(ValueError, match="length"):
        dem_distance(dem, target_observables=[1, 0], solver="highs")


def test_dem_distance_rejects_non_binary_target():
    dem = FakeDem("error(0.1) D0 L0\n")

    with pytest.raises(ValueError, match="binary"):
        dem_distance(dem, target_observables=[2], solver="highs")


def test_dem_distance_rejects_zero_target_mask():
    dem = FakeDem("error(0.1) D0 L0\n")

    with pytest.raises(ValueError, match="nonzero"):
        dem_distance(dem, target_observables=[0], solver="highs")


def test_dem_distance_rejects_non_1d_target():
    dem = FakeDem("error(0.1) D0 L0\n")

    with pytest.raises(ValueError, match="one-dimensional"):
        dem_distance(dem, target_observables=np.array([[1]], dtype=np.uint8), solver="highs")


def test_dem_distance_targeted_mode_builds_augmented_matrix_and_syndrome(monkeypatch):
    h_matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    obs_matrix = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    parser_calls = {}
    solver_calls = {}

    def fake_parse_dem(self, dem, merge_parallel, flatten_dem):
        parser_calls["dem"] = dem
        parser_calls["merge_parallel"] = merge_parallel
        parser_calls["flatten_dem"] = flatten_dem
        return h_matrix.copy(), obs_matrix.copy(), np.zeros(3)

    def fake_minimize_weight_with_fixed_syndrome(matrix, syndrome, *, solver=None, **solver_options):
        solver_calls["matrix"] = matrix.copy()
        solver_calls["syndrome"] = syndrome.copy()
        solver_calls["solver"] = solver
        solver_calls["solver_options"] = dict(solver_options)
        return SimpleNamespace(vector=np.array([1, 0, 1], dtype=np.uint8), weight=2)

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse_dem)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_weight_with_fixed_syndrome",
        fake_minimize_weight_with_fixed_syndrome,
    )

    result = dem_distance(
        "fake dem",
        target_observables=[1, 0],
        merge_parallel_edges=False,
        flatten_dem=False,
        solver="highs",
        threads=4,
    )

    np.testing.assert_array_equal(
        solver_calls["matrix"],
        np.vstack([h_matrix, obs_matrix]),
    )
    np.testing.assert_array_equal(
        solver_calls["syndrome"],
        np.array([0, 0, 1, 0], dtype=np.uint8),
    )
    assert solver_calls["solver"] == "highs"
    assert solver_calls["solver_options"] == {"threads": 4}
    assert parser_calls == {
        "dem": "fake dem",
        "merge_parallel": False,
        "flatten_dem": False,
    }
    np.testing.assert_array_equal(result.observable_mask, np.array([1, 1], dtype=np.uint8))


def test_dem_distance_exact_solver_validation_errors_use_generic_wording():
    dem = FakeDem("error(0.1) D0 L0\n")

    with pytest.raises(ValueError, match="Exact distance APIs require exact optimization"):
        dem_distance(dem, solver="highs", gap=0.1)

    with pytest.raises(ValueError, match="Exact distance ILP currently supports the direct HiGHS backend"):
        dem_distance(dem, solver="cbc")


def test_dem_distance_targeted_mode_surfaces_runtime_error_from_exact_solve(monkeypatch):
    h_matrix = np.array([[1, 0, 1]], dtype=np.uint8)
    obs_matrix = np.array([[1, 1, 0]], dtype=np.uint8)

    def fake_parse_dem(self, dem, merge_parallel, flatten_dem):
        return h_matrix.copy(), obs_matrix.copy(), np.zeros(3)

    def fake_minimize_weight_with_fixed_syndrome(matrix, syndrome, *, solver=None, **solver_options):
        raise RuntimeError("HiGHS did not prove optimality: kInfeasible")

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse_dem)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_weight_with_fixed_syndrome",
        fake_minimize_weight_with_fixed_syndrome,
    )

    with pytest.raises(RuntimeError, match="did not prove optimality"):
        dem_distance("fake dem", target_observables=[1], solver="highs")


def test_dem_distance_default_mode_surfaces_generic_run_failure(monkeypatch):
    h_matrix = np.array([[1, 0, 1]], dtype=np.uint8)
    obs_matrix = np.array([[1, 1, 0]], dtype=np.uint8)

    def fake_parse_dem(self, dem, merge_parallel, flatten_dem):
        return h_matrix.copy(), obs_matrix.copy(), np.zeros(3)

    def fake_minimize_nonzero_logical_operator(check_matrix, dual_logicals, *, solver=None, **solver_options):
        raise RuntimeError("HiGHS failed to solve exact distance model")

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse_dem)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_nonzero_logical_operator",
        fake_minimize_nonzero_logical_operator,
    )

    with pytest.raises(RuntimeError, match="failed to solve exact distance model"):
        dem_distance("fake dem", solver="highs")


def test_dem_distance_returns_defensive_copies(monkeypatch):
    shared_vector = np.array([1, 0, 1], dtype=np.uint8)
    h_matrix = np.zeros((1, 3), dtype=np.uint8)
    obs_matrix = np.array([[1, 1, 0]], dtype=np.uint8)

    def fake_parse_dem(self, dem, merge_parallel, flatten_dem):
        return h_matrix.copy(), obs_matrix.copy(), np.zeros(3)

    def fake_minimize_nonzero_logical_operator(check_matrix, dual_logicals, *, solver=None, **solver_options):
        return SimpleNamespace(vector=shared_vector, weight=2)

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse_dem)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_nonzero_logical_operator",
        fake_minimize_nonzero_logical_operator,
    )

    first = dem_distance("fake dem", solver="highs")
    first.fault_vector[0] = 0
    first.observable_mask[0] = 0

    second = dem_distance("fake dem", solver="highs")

    np.testing.assert_array_equal(second.fault_vector, np.array([1, 0, 1], dtype=np.uint8))
    np.testing.assert_array_equal(second.observable_mask, np.array([1], dtype=np.uint8))
