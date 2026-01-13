"""Direct Gurobi backend for ILPDecoder."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ilpdecoder.solver import SolverConfig


class DirectGurobiSolver:
    """Direct Gurobi solver that reuses a single model across shots."""

    def __init__(self, H: np.ndarray, weights: np.ndarray, config: SolverConfig):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as exc:
            raise ImportError(
                "Direct Gurobi backend requires gurobipy. "
                "Install with: pip install ilpdecoder[gurobi]"
            ) from exc

        self._gp = gp
        self._GRB = GRB
        self._num_rows, self._num_errors = H.shape
        self._env = self._start_env(config)
        self._model = gp.Model(env=self._env)
        self._configure_options(config)
        self._build_model(H, weights)

    def _start_env(self, config: SolverConfig):
        try:
            env = self._gp.Env(empty=True)
            env.setParam("OutputFlag", int(bool(config.verbose)))
            env.start()
            return env
        except self._gp.GurobiError as exc:
            msg = str(exc).lower()
            err = getattr(exc, "errno", None)
            if err in {10009, 10010, 10011, 10015} or "license" in msg:
                raise RuntimeError(
                    "Gurobi license not found or not valid. "
                    "Set GRB_LICENSE_FILE or run grbgetkey."
                ) from exc
            raise

    def _configure_options(self, config: SolverConfig) -> None:
        self._set_param("OutputFlag", int(bool(config.verbose)))
        if config.time_limit is not None:
            self._set_param("TimeLimit", float(config.time_limit))
        if config.gap is not None:
            self._set_param("MIPGap", float(config.gap))
        if config.threads is not None:
            self._set_param("Threads", int(config.threads))
        for key, value in config.options.items():
            self._set_param(key, value)

    def _set_param(self, key: str, value: Any) -> None:
        try:
            self._model.setParam(key, value)
        except Exception as exc:
            raise ValueError(f"Gurobi rejected parameter '{key}'") from exc

    def _build_model(self, H: np.ndarray, weights: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.uint8)
        weights = np.asarray(weights, dtype=float)
        m, n = H.shape
        row_sums = H.sum(axis=1).astype(int)

        self._e = self._model.addVars(n, vtype=self._GRB.BINARY, name="e")
        self._a = self._model.addVars(m, vtype=self._GRB.INTEGER, lb=0, name="a")
        for i in range(m):
            self._a[i].UB = int(row_sums[i] // 2)

        obj = self._gp.quicksum(weights[j] * self._e[j] for j in range(n))
        self._model.setObjective(obj, self._GRB.MINIMIZE)

        self._constraints = []
        for i in range(m):
            cols = np.flatnonzero(H[i])
            expr = self._gp.quicksum(self._e[j] for j in cols) - 2 * self._a[i]
            constraint = self._model.addConstr(expr == 0.0, name=f"syndrome_{i}")
            self._constraints.append(constraint)

        self._model.update()

    def solve(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float, str]:
        if syndrome.shape[0] != self._num_rows:
            raise ValueError(
                f"Syndrome length {syndrome.shape[0]} does not match {self._num_rows}"
            )
        for i in range(self._num_rows):
            self._constraints[i].RHS = float(syndrome[i])

        self._model.optimize()
        status = self._model.Status

        if status not in (
            self._GRB.OPTIMAL,
            self._GRB.SUBOPTIMAL,
            self._GRB.TIME_LIMIT,
        ):
            raise RuntimeError(f"Gurobi terminated with status {status}")
        if self._model.SolCount == 0:
            raise RuntimeError("Gurobi did not return a feasible solution.")

        values = np.array([self._e[j].X for j in range(self._num_errors)], dtype=float)
        correction = (values > 0.5).astype(np.uint8)
        objective = float(self._model.ObjVal)
        return correction, objective, str(status)
