"""Additional tests to raise coverage on edge cases and helper paths."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import ilpdecoder.decoder as decoder_module
import ilpdecoder.solver as solver_module
from ilpdecoder import Decoder
from pyomo.environ import TerminationCondition


class FakeDem:
    """Minimal DEM-like object for parser edge cases."""

    def __init__(self, text: str):
        self._text = text
        self.flatten_called = False

    def __str__(self) -> str:
        return self._text

    def flattened(self) -> "FakeDem":
        self.flatten_called = True
        return self


def test_parity_check_sparse_and_scalar_weights():
    H = csr_matrix(np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8))
    decoder = Decoder.from_parity_check_matrix(H, weights=2.5, solver="highs")
    np.testing.assert_array_equal(decoder.get_parity_check_matrix(), H.toarray() % 2)
    np.testing.assert_array_equal(decoder.get_weights(), np.array([2.5, 2.5, 2.5]))


def test_from_stim_dem_file_reads(tmp_path):
    dem_path = tmp_path / "demo.dem"
    dem_path.write_text("error(0.1) D0 L0\n")
    decoder = Decoder.from_stim_dem_file(dem_path, solver="highs")
    assert decoder.num_errors == 1
    assert decoder.num_observables == 1


def test_decode_requires_configuration():
    with pytest.raises(RuntimeError, match="not configured"):
        Decoder().decode([0])


def test_decode_batch_dem_with_weights():
    dem_str = "error(0.1) D0 L0\nerror(0.1) D1 L1\n"
    decoder = Decoder.from_stim_dem(dem_str, solver="highs")
    responses = [
        (np.array([0, 0], dtype=np.uint8), np.array([1, 0], dtype=np.uint8), 1.0),
        (np.array([0, 0], dtype=np.uint8), np.array([0, 1], dtype=np.uint8), 2.0),
    ]
    it = iter(responses)
    decoder.decode = lambda _, return_weight=False: next(it)  # type: ignore[assignment]
    outputs, weights = decoder.decode_batch(np.array([[0, 0], [1, 1]]), return_weights=True)
    np.testing.assert_array_equal(outputs, np.array([[1, 0], [0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(weights, np.array([1.0, 2.0]))


def test_decode_batch_parity_with_weights():
    H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    decoder = Decoder.from_parity_check_matrix(H, solver="highs")
    responses = [
        (np.array([1, 0], dtype=np.uint8), 3.0),
        (np.array([0, 1], dtype=np.uint8), 4.0),
    ]
    it = iter(responses)
    decoder.decode = lambda _, return_weight=False: next(it)  # type: ignore[assignment]
    outputs, weights = decoder.decode_batch(np.array([[0, 0], [1, 1]]), return_weights=True)
    np.testing.assert_array_equal(outputs, np.array([[1, 0], [0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(weights, np.array([3.0, 4.0]))


def test_set_solver_defaults(monkeypatch):
    decoder = Decoder()
    monkeypatch.setattr(decoder_module, "get_default_solver", lambda: "cbc")
    decoder.set_solver()
    assert decoder.solver_name == "cbc"


def test_solve_ilp_solver_unavailable(monkeypatch):
    decoder = Decoder.from_parity_check_matrix(np.array([[1]], dtype=np.uint8), solver="highs")

    class FakeSolver:
        def __init__(self):
            self.options = {}

        def available(self) -> bool:
            return False

    monkeypatch.setattr(decoder_module, "SolverFactory", lambda _: FakeSolver())
    monkeypatch.setattr(decoder_module, "get_available_solvers", lambda: ["dummy"])
    with pytest.raises(RuntimeError, match="not available"):
        decoder._solve_ilp(np.array([0], dtype=np.uint8))


def test_solve_ilp_termination_error(monkeypatch):
    decoder = Decoder.from_parity_check_matrix(np.array([[1]], dtype=np.uint8), solver="highs")

    class FakeSolver:
        def __init__(self):
            self.options = {}

        def available(self) -> bool:
            return True

        def solve(self, model, tee=False):
            return SimpleNamespace(
                solver=SimpleNamespace(termination_condition=TerminationCondition.infeasible)
            )

    monkeypatch.setattr(decoder_module, "SolverFactory", lambda _: FakeSolver())
    with pytest.raises(RuntimeError, match="terminated"):
        decoder._solve_ilp(np.array([0], dtype=np.uint8))
    assert decoder.last_objective is None


def test_solve_ilp_success_sets_status_and_objective(monkeypatch):
    H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    decoder = Decoder.from_parity_check_matrix(H, weights=[2.0, 3.0], solver="highs")
    decoder.set_solver("highs", time_limit=5, gap=0.1, threads=2)

    class FakeSolver:
        def __init__(self):
            self.options = {}

        def available(self) -> bool:
            return True

        def solve(self, model, tee=False):
            for j in model.e:
                model.e[j].value = 1 if j == 0 else 0
            return SimpleNamespace(
                solver=SimpleNamespace(termination_condition=TerminationCondition.optimal)
            )

    solver = FakeSolver()
    monkeypatch.setattr(decoder_module, "SolverFactory", lambda _: solver)
    correction, objective = decoder._solve_ilp(np.array([0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(correction, np.array([1, 0], dtype=np.uint8))
    assert objective == 2.0
    assert decoder.last_status == str(TerminationCondition.optimal)
    assert solver.options["time_limit"] == 5
    assert solver.options["mip_rel_gap"] == 0.1
    assert solver.options["threads"] == 2


def test_probabilities_to_weights_scalar_and_invalid():
    decoder = Decoder()
    weights = decoder._probabilities_to_weights(0.1, 2)
    assert weights.shape == (2,)
    with pytest.raises(ValueError, match="open interval"):
        decoder._probabilities_to_weights([0.0, 0.1], 2)
    with pytest.raises(ValueError, match="open interval"):
        decoder._probabilities_to_weights([1.0, 0.1], 2)


def test_parse_dem_merge_parallel_edges():
    dem_str = "error(0.1) D0 L0\nerror(0.2) D0 L0\n"
    decoder = Decoder.from_stim_dem(dem_str, solver="highs", merge_parallel_edges=True)
    assert decoder.num_errors == 1
    p1, p2 = 0.1, 0.2
    p_combined = p1 * (1 - p2) + p2 * (1 - p1)
    expected_weight = np.log((1 - p_combined) / p_combined)
    np.testing.assert_allclose(decoder.get_weights(), np.array([expected_weight]))


def test_parse_dem_shift_detectors_invalid_instruction():
    decoder = Decoder()
    fake = FakeDem("shift_detectors 1 2\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="Invalid shift_detectors instruction"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_shift_detectors_invalid_value():
    decoder = Decoder()
    fake = FakeDem("shift_detectors foo\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="Invalid shift_detectors value"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_shift_detectors_negative():
    decoder = Decoder()
    fake = FakeDem("shift_detectors -1\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="non-negative"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_invalid_error_instruction():
    decoder = Decoder()
    fake = FakeDem("error 0.1 D0\n")
    with pytest.raises(ValueError, match="Invalid error instruction"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_detector_separator():
    decoder = Decoder()
    fake = FakeDem("detector_separator\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="detector_separator"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_unknown_instruction():
    decoder = Decoder()
    fake = FakeDem("unknown_instruction 1\nerror(0.1) D0\n")
    with pytest.raises(ValueError, match="Unsupported DEM instruction"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_invalid_targets_skip():
    decoder = Decoder()
    fake = FakeDem("error(0.1) Dfoo Lbar\n")
    with pytest.raises(ValueError, match="No valid error mechanisms"):
        decoder._parse_dem(fake, merge_parallel=True, flatten_dem=False)


def test_parse_dem_no_valid_errors():
    with pytest.raises(ValueError, match="No valid error mechanisms"):
        Decoder.from_stim_dem("error(1) D0\n", solver="highs")


def test_parse_dem_flatten_called():
    decoder = Decoder()
    fake = FakeDem("error(0.1) D0\n")
    decoder._parse_dem(fake, merge_parallel=True, flatten_dem=True)
    assert fake.flatten_called


def test_repr_variants():
    assert repr(Decoder()) == "<Decoder (not configured)>"
    parity = Decoder.from_parity_check_matrix(np.array([[1]], dtype=np.uint8), solver="highs")
    assert "checks" in repr(parity)
    dem = Decoder.from_stim_dem("error(0.1) D0 L0\n", solver="highs")
    assert "observables" in repr(dem)


def test_solver_config_options_various():
    cfg = solver_module.SolverConfig(name="gurobi", time_limit=5, gap=0.1, threads=2)
    assert cfg.to_pyomo_options() == {"TimeLimit": 5, "MIPGap": 0.1, "Threads": 2}
    cfg = solver_module.SolverConfig(name="cplex", time_limit=5, gap=0.1, threads=2)
    assert cfg.to_pyomo_options() == {"timelimit": 5, "mip_tolerances_mipgap": 0.1, "threads": 2}
    cfg = solver_module.SolverConfig(name="cbc", time_limit=5, gap=0.1)
    assert cfg.to_pyomo_options() == {"seconds": 5, "ratioGap": 0.1}
    cfg = solver_module.SolverConfig(name="glpk", time_limit=5, gap=0.1)
    assert cfg.to_pyomo_options() == {"tmlim": 5}


def test_get_available_solvers_fallback(monkeypatch):
    import pyomo.environ as pe

    monkeypatch.setattr(solver_module.shutil, "which", lambda _: None)

    class FakeSolver:
        def __init__(self, exe):
            self._exe = exe

        def available(self) -> bool:
            return self._exe == "cbc"

    monkeypatch.setattr(pe, "SolverFactory", lambda exe: FakeSolver(exe))
    available = solver_module.get_available_solvers()
    assert available == ["cbc"]


def test_get_default_solver_fallback(monkeypatch):
    monkeypatch.setattr(solver_module, "get_available_solvers", lambda: ["glpk"])
    assert solver_module.get_default_solver() == "glpk"
    monkeypatch.setattr(solver_module, "get_available_solvers", lambda: [])
    with pytest.raises(RuntimeError):
        solver_module.get_default_solver()


def test_get_pyomo_solver_name_fallback(monkeypatch):
    import pyomo.environ as pe

    class FakeSolver:
        def available(self) -> bool:
            return False

    monkeypatch.setattr(pe, "SolverFactory", lambda _: FakeSolver())
    assert solver_module.get_pyomo_solver_name("highs") == "highs"
