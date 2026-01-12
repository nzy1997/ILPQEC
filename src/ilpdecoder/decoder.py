"""
Main Decoder class for ILP-based decoding.

This module provides the primary user-facing interface for ILP-based
quantum error correction decoding. Uses a direct HiGHS backend by
default, with an optional direct Gurobi backend and a Pyomo backend
for other solvers.

Key Features:
- Direct HiGHS backend for fast default decoding
- Optional direct Gurobi backend for licensed users
- Optional Pyomo modeling for other solvers
- Multiple construction methods (from_parity_check_matrix, from_stim_dem)
- Easy solver switching (scip, highs, gurobi, cplex, cbc, glpk)
- Maximum-likelihood and minimum-weight decoding

Example Usage:
    # From parity-check matrix (uses HiGHS by default)
    decoder = Decoder.from_parity_check_matrix(H)
    correction = decoder.decode(syndrome)
    
    # Use different solver (requires Pyomo extra)
    decoder = Decoder.from_parity_check_matrix(H, solver="scip", direct=False)
    
    # Change solver at runtime
    decoder.set_solver("gurobi", time_limit=60)
    
    # From Stim DEM
    decoder = Decoder.from_stim_dem(dem)
    correction, observables = decoder.decode(detector_outcomes)
"""

from __future__ import annotations

from typing import Union, List, Optional, Tuple, Dict, Any, TYPE_CHECKING
from pathlib import Path
import math

import numpy as np

from ilpdecoder.solver import (
    SolverConfig,
    get_default_solver,
    get_pyomo_solver_name,
    get_available_solvers,
    require_pyomo,
    is_pyomo_available,
    is_gurobi_available,
)

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class Decoder:
    """
    ILP-based quantum error correction decoder.
    
    This class provides a PyMatching-like API for decoding using
    Integer Linear Programming. It uses a direct HiGHS backend by
    default, with an optional direct Gurobi backend and a Pyomo backend
    for other solvers.
    
    Solver switching is trivial - just call set_solver() with a different
    solver name. No need to rebuild the model.
    
    Supported Solvers:
        - highs: HiGHS solver (default, direct backend)
        - gurobi: Gurobi solver (direct backend, requires license)
        - scip: SCIP solver (Pyomo required)
        - cplex: IBM CPLEX (requires license, Pyomo required)
        - cbc: COIN-OR CBC (Pyomo required)
        - glpk: GNU Linear Programming Kit (Pyomo required)
    
    Attributes:
        num_detectors: Number of parity checks / detectors
        num_errors: Number of error mechanisms
        num_observables: Number of logical observables (for DEM)
    """
    
    def __init__(self):
        """
        Initialize an empty decoder.
        
        Use the class methods `from_parity_check_matrix` or `from_stim_dem`
        to create a configured decoder.
        """
        # Decoding data
        self._H: Optional[np.ndarray] = None  # Parity check matrix
        self._weights: Optional[np.ndarray] = None  # Error weights
        self._observable_matrix: Optional[np.ndarray] = None  # For DEM
        
        # Solver configuration
        self._solver_config = SolverConfig()
        self._direct_highs_solver = None
        self._direct_gurobi_solver = None
        self._pyomo_model = None
        self._pyomo_solver = None
        self._pyomo_solver_name = None

        # Last solution info
        self._last_objective: Optional[float] = None
        self._last_status: Optional[str] = None
    
    # =========================================================================
    # Construction Methods
    # =========================================================================
    
    @classmethod
    def from_parity_check_matrix(
        cls,
        parity_check_matrix: Union[np.ndarray, spmatrix, List[List[int]]],
        weights: Union[float, np.ndarray, List[float]] = None,
        error_probabilities: Union[float, np.ndarray, List[float]] = None,
        solver: str = None,
        **solver_options
    ) -> 'Decoder':
        """
        Create a decoder from a binary parity-check matrix.
        
        Args:
            parity_check_matrix: Binary m×n matrix H where m is the number
                of parity checks and n is the number of error mechanisms.
            weights: Weights for each error. If float, same for all.
                If None, computed from error_probabilities or set to 1.0.
            error_probabilities: Error probability for each mechanism.
                Used to compute log-likelihood weights if weights not given.
                Probabilities must be in (0, 0.5].
            solver: Solver name ("highs", "scip", "gurobi", etc.)
                Default is "highs" with the direct HiGHS backend.
            **solver_options: Solver options (time_limit, gap, verbose, direct, etc.)
            
        Returns:
            Configured Decoder instance
            
        Example:
            >>> H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
            >>> decoder = Decoder.from_parity_check_matrix(H)
            >>> correction = decoder.decode([1, 0, 1])
            
            >>> # With different solver
            >>> decoder = Decoder.from_parity_check_matrix(H, solver="highs")
            >>> # Use Pyomo-backed solvers
            >>> decoder = Decoder.from_parity_check_matrix(H, solver="scip", direct=False)
        """
        decoder = cls()
        
        # Convert to numpy array
        try:
            from scipy.sparse import spmatrix  # type: ignore
        except Exception:
            spmatrix = None
        if spmatrix is not None and isinstance(parity_check_matrix, spmatrix):
            H = parity_check_matrix.toarray()
        else:
            if spmatrix is None and hasattr(parity_check_matrix, "toarray"):
                raise ImportError(
                    "Sparse parity-check matrices require SciPy. "
                    "Install with: pip install scipy"
                )
            H = np.asarray(parity_check_matrix)
        
        decoder._H = H % 2  # Ensure binary
        n = H.shape[1]  # Number of errors
        
        # Process weights
        if weights is None:
            if error_probabilities is not None:
                weights = decoder._probabilities_to_weights(error_probabilities, n)
            else:
                weights = np.ones(n)
        elif isinstance(weights, (int, float)):
            weights = np.ones(n) * float(weights)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (n,):
                raise ValueError(f"weights must have length {n} (got {weights.shape})")
        
        decoder._weights = weights
        
        # Set solver
        decoder.set_solver(solver, **solver_options)
        
        return decoder
    
    @classmethod
    def from_stim_dem(
        cls,
        dem: Union['stim.DetectorErrorModel', str],
        solver: str = None,
        merge_parallel_edges: bool = True,
        flatten_dem: bool = True,
        **solver_options
    ) -> 'Decoder':
        """
        Create a decoder from a Stim DetectorErrorModel.
        
        Args:
            dem: A stim.DetectorErrorModel or its string representation
            solver: Solver name ("highs", "scip", etc.). Default is "highs".
            merge_parallel_edges: If True, merge parallel error mechanisms
            flatten_dem: If True, call dem.flattened() to inline repeats and
                apply detector shifts (may increase DEM size).
            **solver_options: Solver options (time_limit, gap, verbose, direct, etc.)
        
        Note:
            This parser reads only error(p) lines (tags are ignored). It ignores
            detector/logical_observable metadata and applies shift_detectors
            offsets. It raises on unsupported instructions such as repeat or
            detector_separator. The '^' separator is treated as whitespace.
            Repeat blocks are handled by flatten_dem=True (default), but can
            cause large DEM expansions. Set flatten_dem=False to fail fast.
            
        Returns:
            Configured Decoder instance
            
        Example:
            >>> import stim
            >>> circuit = stim.Circuit.generated("surface_code:rotated_memory_x",
            ...                                  distance=3, rounds=3,
            ...                                  after_clifford_depolarization=0.01)
            >>> dem = circuit.detector_error_model(decompose_errors=True)
            >>> decoder = Decoder.from_stim_dem(dem)
            >>> # Pyomo-backed solver
            >>> decoder = Decoder.from_stim_dem(dem, solver="scip", direct=False)
            >>> correction, observables = decoder.decode(detector_outcomes)
        """
        decoder = cls()
        
        # Parse DEM
        H, obs_matrix, weights = decoder._parse_dem(dem, merge_parallel_edges, flatten_dem)
        
        decoder._H = H
        decoder._weights = weights
        decoder._observable_matrix = obs_matrix
        
        # Set solver
        decoder.set_solver(solver, **solver_options)
        
        return decoder
    
    @classmethod
    def from_stim_dem_file(
        cls,
        dem_path: Union[str, Path],
        solver: str = None,
        flatten_dem: bool = True,
        **solver_options
    ) -> 'Decoder':
        """
        Create a decoder from a Stim DEM file.
        
        Args:
            dem_path: Path to the .dem file
            solver: Solver name
            flatten_dem: If True, call dem.flattened() to inline repeats and
                apply detector shifts (may increase DEM size).
            **solver_options: Solver options
            
        Returns:
            Configured Decoder instance
        """
        dem_path = Path(dem_path)
        dem_str = dem_path.read_text()
        return cls.from_stim_dem(
            dem_str, solver=solver, flatten_dem=flatten_dem, **solver_options
        )
    
    # =========================================================================
    # Solver Configuration
    # =========================================================================
    
    def set_solver(
        self,
        solver: str = None,
        time_limit: float = None,
        gap: float = None,
        threads: int = None,
        verbose: bool = False,
        direct: Optional[bool] = None,
        **options
    ):
        """
        Set or change the solver.
        
        You can switch solvers at any time without rebuilding the model.
        
        Args:
            solver: Solver name. Options:
                - "highs": HiGHS solver (default, direct backend)
                - "gurobi": Gurobi solver (direct backend, requires license)
                - "scip": SCIP solver (Pyomo required)
                - "cplex": IBM CPLEX (requires license, Pyomo required)
                - "cbc": COIN-OR CBC (Pyomo required)
                - "glpk": GNU Linear Programming Kit (Pyomo required)
            time_limit: Maximum solving time in seconds
            gap: Relative ILP gap tolerance
            threads: Number of threads (solver-dependent)
            verbose: Print solver output
            direct: Use a direct backend when available (HiGHS/Gurobi).
                Defaults to True for HiGHS. For Gurobi, defaults to True
                when gurobipy is installed.
            **options: Additional solver-specific options
            
        Example:
            >>> decoder.set_solver("scip", time_limit=30)
            >>> decoder.set_solver("highs", threads=4)
            >>> decoder.set_solver("gurobi", gap=0.01, verbose=True)
        """
        if solver is None:
            solver = get_default_solver()
        solver_name = solver.lower()
        if direct is None:
            if solver_name == "highs":
                direct = True
            elif solver_name == "gurobi":
                direct = is_gurobi_available()
            else:
                direct = False
        if direct:
            if solver_name == "highs":
                pass
            elif solver_name == "gurobi":
                if not is_gurobi_available():
                    raise ImportError(
                        "Direct Gurobi backend requires gurobipy. "
                        "Install with: pip install ilpdecoder[gurobi]"
                    )
            else:
                raise ValueError("Direct backend currently supports HiGHS and Gurobi only.")
        if not direct and not is_pyomo_available():
            if solver_name == "highs":
                message = (
                    "Pyomo is required for the Pyomo HiGHS backend. "
                    "Install with: pip install ilpdecoder[pyomo]"
                )
            elif solver_name == "gurobi":
                message = (
                    "Gurobi requires gurobipy (direct backend) or Pyomo. "
                    "Install with: pip install ilpdecoder[gurobi] or ilpdecoder[pyomo]"
                )
            else:
                message = (
                    "Pyomo is required for non-HiGHS solvers. "
                    "Install with: pip install ilpdecoder[pyomo]"
                )
            raise ImportError(message)

        self._solver_config = SolverConfig(
            name=solver_name,
            time_limit=time_limit,
            gap=gap,
            threads=threads,
            verbose=verbose,
            direct=direct,
            options=options
        )
        self._direct_highs_solver = None
        self._direct_gurobi_solver = None
        self._pyomo_solver = None
        self._pyomo_solver_name = None
    
    def get_solver_options(self) -> Dict[str, Any]:
        """Get current solver configuration as a dictionary."""
        return {
            "solver": self._solver_config.name,
            "time_limit": self._solver_config.time_limit,
            "gap": self._solver_config.gap,
            "threads": self._solver_config.threads,
            "verbose": self._solver_config.verbose,
            "direct": self._solver_config.direct,
            **self._solver_config.options
        }
    
    # =========================================================================
    # Decoding Methods
    # =========================================================================
    
    def decode(
        self,
        syndrome: Union[np.ndarray, List[int]],
        return_weight: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
        """
        Decode a syndrome using ILP.
        
        The behavior depends on how the decoder was constructed:
        - From parity-check matrix: Returns correction vector
        - From Stim DEM: Returns (correction, observable_predictions)
        
        Args:
            syndrome: Binary syndrome vector or detector outcomes
            return_weight: If True, also return the solution weight
            
        Returns:
            For parity-check matrix:
                correction: Binary vector of errors
                weight (if return_weight): Total weight of solution
                
            For Stim DEM:
                correction: Binary vector of errors  
                observables: Binary vector of observable predictions
                weight (if return_weight): Total weight of solution
                
        Example:
            >>> correction = decoder.decode([1, 0, 1])
            >>> correction, weight = decoder.decode([1, 0, 1], return_weight=True)

        Raises:
            RuntimeError: If the solver fails to find a feasible solution.
        """
        if self._H is None:
            raise RuntimeError("Decoder not configured. Use from_parity_check_matrix() or from_stim_dem().")
        
        syndrome = np.asarray(syndrome, dtype=np.uint8) % 2
        
        # Solve ILP (direct backend or Pyomo backend)
        if self._solver_config.direct:
            if self._solver_config.name == "highs":
                correction, objective = self._solve_direct_highs(syndrome)
            elif self._solver_config.name == "gurobi":
                correction, objective = self._solve_direct_gurobi(syndrome)
            else:
                raise ValueError("Direct backend currently supports HiGHS and Gurobi only.")
        else:
            correction, objective = self._solve_ilp(syndrome)
        
        # For DEM, compute observable predictions
        if self._observable_matrix is not None:
            observables = (self._observable_matrix @ correction) % 2
            observables = observables.astype(np.uint8)
            
            if return_weight:
                return correction, observables, objective
            return correction, observables
        else:
            if return_weight:
                return correction, objective
            return correction
    
    def decode_batch(
        self,
        syndromes: np.ndarray,
        return_weights: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Decode multiple syndromes.
        
        Args:
            syndromes: 2D array of shape (num_shots, num_detectors)
            return_weights: If True, also return weights
            
        Returns:
            For parity-check: corrections array
            For DEM: observable predictions array
            weights (if return_weights): 1D array of weights
        """
        syndromes = np.asarray(syndromes)
        if syndromes.ndim == 1:
            syndromes = syndromes.reshape(1, -1)
        
        num_shots = syndromes.shape[0]
        
        # Determine output shape
        if self._observable_matrix is not None:
            num_outputs = self._observable_matrix.shape[0]
        else:
            num_outputs = self._H.shape[1]
        
        results = np.zeros((num_shots, num_outputs), dtype=np.uint8)
        weights = np.zeros(num_shots, dtype=float) if return_weights else None
        
        for i in range(num_shots):
            result = self.decode(syndromes[i], return_weight=return_weights)
            
            if self._observable_matrix is not None:
                if return_weights:
                    _, obs, w = result
                    results[i] = obs
                    weights[i] = w
                else:
                    _, obs = result
                    results[i] = obs
            else:
                if return_weights:
                    corr, w = result
                    results[i] = corr
                    weights[i] = w
                else:
                    results[i] = result
        
        if return_weights:
            return results, weights
        return results
    
    # =========================================================================
    # ILP Model Building and Solving (Pyomo)
    # =========================================================================
    
    def _solve_ilp(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Build and solve the ILP model using Pyomo.
        
        ILP Formulation:
            Variables:
                e[j] ∈ {0,1} for j = 0,...,n-1 (error indicators)
                a[i] ∈ Z≥0 for i = 0,...,m-1 (auxiliary for mod-2)
            
            Objective:
                minimize Σ_j weight[j] * e[j]
            
            Constraints (mod-2 linearization):
                Σ_j H[i,j] * e[j] = syndrome[i] + 2 * a[i]  for all i
        """
        require_pyomo()
        from pyomo.environ import SolverFactory, TerminationCondition, value

        H = self._H
        m, n = H.shape
        
        if self._pyomo_model is None:
            self._pyomo_model = self._build_pyomo_model()
        model = self._pyomo_model
        for i in range(m):
            model.syndrome_rhs[i] = int(syndrome[i])
        
        # Get solver
        solver_name = get_pyomo_solver_name(self._solver_config.name)
        solver = self._pyomo_solver
        if solver is None or solver_name != self._pyomo_solver_name:
            solver = SolverFactory(solver_name)
            if hasattr(solver, "_version_timeout"):
                # SCIP can take a few seconds to answer --version on some installs.
                solver._version_timeout = max(getattr(solver, "_version_timeout", 0), 10)
            if not solver.available():
                raise RuntimeError(
                    f"Solver '{self._solver_config.name}' is not available. "
                    f"Available solvers: {get_available_solvers()}"
                )
            options = self._solver_config.to_pyomo_options()
            for key, val in options.items():
                solver.options[key] = val
            self._pyomo_solver = solver
            self._pyomo_solver_name = solver_name
        
        # Solve
        tee = self._solver_config.verbose
        results = solver.solve(model, tee=tee)
        
        # Check status
        self._last_status = str(results.solver.termination_condition)
        
        if results.solver.termination_condition not in (
            TerminationCondition.optimal,
            TerminationCondition.feasible,
            TerminationCondition.maxTimeLimit,
        ):
            self._last_objective = None
            raise RuntimeError(
                f"Solver terminated with status {results.solver.termination_condition}"
            )
        
        # Extract solution
        correction = np.zeros(n, dtype=np.uint8)
        for j in range(n):
            val = value(model.e[j])
            correction[j] = 1 if val is not None and val > 0.5 else 0
        
        self._last_objective = value(model.obj)
        
        return correction, self._last_objective

    def _build_pyomo_model(self) -> Any:
        require_pyomo()
        from pyomo.environ import (
            ConcreteModel,
            Var,
            Param,
            Constraint,
            Objective,
            Binary,
            NonNegativeIntegers,
            minimize,
        )

        H = self._H
        weights = self._weights
        m, n = H.shape
        row_sums = np.sum(H, axis=1).astype(int)

        model = ConcreteModel()
        model.e = Var(range(n), within=Binary)

        def aux_bounds(model, i):
            return (0, int(row_sums[i] // 2))

        model.a = Var(range(m), within=NonNegativeIntegers, bounds=aux_bounds)
        model.syndrome_rhs = Param(range(m), initialize=0, mutable=True)

        model.obj = Objective(
            expr=sum(weights[j] * model.e[j] for j in range(n)),
            sense=minimize
        )

        def syndrome_constraint(model, i):
            lhs = sum(int(H[i, j]) * model.e[j] for j in range(n) if H[i, j] != 0)
            return lhs == model.syndrome_rhs[i] + 2 * model.a[i]

        model.syndrome_cons = Constraint(range(m), rule=syndrome_constraint)
        return model

    def _solve_direct_highs(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve the ILP using a direct HiGHS model reused across shots."""
        if self._direct_highs_solver is None:
            self._direct_highs_solver = _DirectHighsSolver(
                self._H, self._weights, self._solver_config
            )
        correction, objective, status = self._direct_highs_solver.solve(syndrome)
        self._last_status = status
        self._last_objective = objective
        return correction, objective

    def _solve_direct_gurobi(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float]:
        """Solve the ILP using a direct Gurobi model reused across shots."""
        if self._direct_gurobi_solver is None:
            self._direct_gurobi_solver = _DirectGurobiSolver(
                self._H, self._weights, self._solver_config
            )
        correction, objective, status = self._direct_gurobi_solver.solve(syndrome)
        self._last_status = status
        self._last_objective = objective
        return correction, objective
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _probabilities_to_weights(
        self, 
        probs: Union[float, np.ndarray, List[float]], 
        n: int
    ) -> np.ndarray:
        """Convert error probabilities to log-likelihood ratio weights (p in (0, 0.5])."""
        if isinstance(probs, (int, float)):
            probs = np.ones(n) * float(probs)
        else:
            probs = np.asarray(probs, dtype=float)

        if probs.shape != (n,):
            raise ValueError(f"error_probabilities must have length {n} (got {probs.shape})")
        if np.any(probs <= 0) or np.any(probs >= 1):
            raise ValueError("error_probabilities must be in the open interval (0, 1)")
        if np.any(probs > 0.5):
            raise ValueError(
                "error_probabilities must be <= 0.5; pass explicit weights for p > 0.5"
            )

        # Clip to avoid numerical issues
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Log-likelihood ratio: weight = log((1-p)/p)
        weights = np.log((1 - probs) / probs)
        
        return np.maximum(weights, 0.0)  # Keep non-negative for minimization
    
    def _parse_dem(
        self,
        dem: Union['stim.DetectorErrorModel', str],
        merge_parallel: bool,
        flatten_dem: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse a Stim DEM into H matrix, observable matrix, and weights."""
        try:
            import stim
        except ImportError:
            raise ImportError("stim is required. Install with: pip install stim")
        
        if isinstance(dem, str):
            dem = stim.DetectorErrorModel(dem)
        if flatten_dem:
            dem = dem.flattened()
        
        # Extract error mechanisms
        errors = []
        seen = {}  # For merging parallel edges
        
        dem_str = str(dem)
        detector_offset = 0
        for line in dem_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line_lower = line.lower()
            if not line_lower.startswith('error'):
                if line_lower.startswith('shift_detectors'):
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError(f"Invalid shift_detectors instruction: {line}")
                    try:
                        shift = int(parts[1])
                    except ValueError as exc:
                        raise ValueError(f"Invalid shift_detectors value: {line}") from exc
                    if shift < 0:
                        raise ValueError(f"shift_detectors must be non-negative: {line}")
                    detector_offset += shift
                    continue
                if line_lower.startswith('repeat') or line == '}':
                    raise ValueError(
                        "Unsupported DEM instruction: repeat. "
                        "Flatten the DEM first (e.g., dem = dem.flattened())."
                    )
                if line_lower.startswith('detector_separator'):
                    raise ValueError(
                        "Unsupported DEM instruction: detector_separator. "
                        "Only error(p) lines are supported."
                    )
                if line_lower.startswith('detector') or line_lower.startswith('logical_observable'):
                    continue
                raise ValueError(f"Unsupported DEM instruction: {line}")
            
            # Parse error(p) D... L...
            try:
                prob_start = line.index('(') + 1
                prob_end = line.index(')')
            except ValueError as exc:
                raise ValueError(f"Invalid error instruction: {line}") from exc
            prob = float(line[prob_start:prob_end])
            
            if prob <= 0 or prob >= 1:
                continue
            
            targets_str = line[prob_end + 1:].strip()
            detectors = set()
            observables = set()
            
            for target in targets_str.replace("^", " ").split():
                target = target.strip()
                if target.startswith('D'):
                    try:
                        det_id = int(target[1:]) + detector_offset
                    except ValueError:
                        continue
                    if det_id in detectors:
                        detectors.remove(det_id)
                    else:
                        detectors.add(det_id)
                elif target.startswith('L'):
                    try:
                        obs_id = int(target[1:])
                    except ValueError:
                        continue
                    if obs_id in observables:
                        observables.remove(obs_id)
                    else:
                        observables.add(obs_id)
            
            if not detectors and not observables:
                continue
            
            if merge_parallel:
                key = (tuple(sorted(detectors)), tuple(sorted(observables)))
                if key in seen:
                    # Merge probabilities
                    idx = seen[key]
                    p1 = errors[idx][0]
                    p_combined = p1 * (1 - prob) + prob * (1 - p1)
                    errors[idx] = (p_combined, detectors, observables)
                    continue
                else:
                    seen[key] = len(errors)
            
            errors.append((prob, detectors, observables))
        
        if not errors:
            raise ValueError("No valid error mechanisms found in DEM")
        
        # Determine dimensions
        num_errors = len(errors)
        if hasattr(dem, "num_detectors"):
            num_detectors = int(dem.num_detectors)
        else:
            num_detectors = (
                max(max(e[1]) for e in errors if e[1]) + 1 if any(e[1] for e in errors) else 0
            )
        if hasattr(dem, "num_observables"):
            num_observables = int(dem.num_observables)
        else:
            num_observables = (
                max(max(e[2]) for e in errors if e[2]) + 1 if any(e[2] for e in errors) else 0
            )
        
        # Build matrices
        H = np.zeros((num_detectors, num_errors), dtype=np.uint8)
        obs_matrix = np.zeros((num_observables, num_errors), dtype=np.uint8)
        weights = np.zeros(num_errors)
        
        for j, (prob, dets, obs) in enumerate(errors):
            for d in dets:
                H[d, j] = 1
            for o in obs:
                obs_matrix[o, j] = 1
            
            # Weight = log((1-p)/p)
            prob = max(1e-15, min(prob, 1 - 1e-15))
            weights[j] = math.log((1 - prob) / prob)
        
        return H, obs_matrix, weights
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def num_detectors(self) -> int:
        """Number of detectors / parity checks."""
        return self._H.shape[0] if self._H is not None else 0
    
    @property
    def num_errors(self) -> int:
        """Number of error mechanisms."""
        return self._H.shape[1] if self._H is not None else 0
    
    @property
    def num_observables(self) -> int:
        """Number of logical observables (for DEM)."""
        return self._observable_matrix.shape[0] if self._observable_matrix is not None else 0
    
    @property
    def solver_name(self) -> str:
        """Name of the configured solver."""
        return self._solver_config.name
    
    @property
    def last_objective(self) -> Optional[float]:
        """Objective value from the last decode() call."""
        return self._last_objective
    
    @property
    def last_status(self) -> Optional[str]:
        """Solver status from the last decode() call."""
        return self._last_status
    
    def get_parity_check_matrix(self) -> Optional[np.ndarray]:
        """Get the parity-check matrix."""
        return self._H
    
    def get_weights(self) -> Optional[np.ndarray]:
        """Get the weights for each error mechanism."""
        return self._weights
    
    def __repr__(self) -> str:
        if self._H is None:
            return "<Decoder (not configured)>"
        
        if self._observable_matrix is not None:
            return (
                f"<Decoder: {self.num_detectors} detectors, {self.num_errors} errors, "
                f"{self.num_observables} observables, solver={self.solver_name}>"
            )
        return (
            f"<Decoder: {self.num_detectors} checks, {self.num_errors} errors, "
            f"solver={self.solver_name}>"
        )


class _DirectHighsSolver:
    def __init__(self, H: np.ndarray, weights: np.ndarray, config: SolverConfig):
        try:
            from highspy import (
                Highs,
                HighsLp,
                HighsSparseMatrix,
                HighsVarType,
                MatrixFormat,
                HighsStatus,
                HighsModelStatus,
            )
        except Exception as exc:
            raise ImportError(
                "Direct HiGHS backend requires highspy. Install with: pip install highspy"
            ) from exc

        self._HighsStatus = HighsStatus
        self._HighsModelStatus = HighsModelStatus
        self._HighsLp = HighsLp
        self._HighsSparseMatrix = HighsSparseMatrix
        self._HighsVarType = HighsVarType
        self._MatrixFormat = MatrixFormat

        self._num_rows, self._num_errors = H.shape
        self._num_cols = self._num_errors + self._num_rows
        self._highs = Highs()
        self._configure_options(config)
        self._build_model(H, weights)

    def _configure_options(self, config: SolverConfig) -> None:
        self._set_option("output_flag", bool(config.verbose))
        if config.time_limit is not None:
            self._set_option("time_limit", float(config.time_limit))
        if config.gap is not None:
            self._set_option("mip_rel_gap", float(config.gap))
        if config.threads is not None:
            self._set_option("threads", int(config.threads))
        for key, value in config.options.items():
            self._set_option(key, value)

    def _set_option(self, key: str, value: Any) -> None:
        status = self._highs.setOptionValue(key, value)
        if status != self._HighsStatus.kOk:
            raise ValueError(f"HiGHS rejected option '{key}'")

    def _build_model(self, H: np.ndarray, weights: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.uint8)
        weights = np.asarray(weights, dtype=float)
        m, n = H.shape
        row_sums = H.sum(axis=1).astype(int)

        col_cost = [0.0] * self._num_cols
        col_lower = [0.0] * self._num_cols
        col_upper = [0.0] * self._num_cols
        integrality = [self._HighsVarType.kInteger] * self._num_cols

        for j in range(n):
            col_cost[j] = float(weights[j])
            col_upper[j] = 1.0

        for i in range(m):
            idx = n + i
            col_upper[idx] = float(row_sums[i] // 2)

        row_lower = [0.0] * m
        row_upper = [0.0] * m

        starts = [0]
        indices = []
        values = []

        for j in range(n):
            rows = np.flatnonzero(H[:, j])
            for r in rows:
                indices.append(int(r))
                values.append(1.0)
            starts.append(len(indices))

        for i in range(m):
            indices.append(int(i))
            values.append(-2.0)
            starts.append(len(indices))

        mat = self._HighsSparseMatrix()
        mat.num_row_ = m
        mat.num_col_ = self._num_cols
        mat.start_ = starts
        mat.index_ = indices
        mat.value_ = values
        mat.format_ = self._MatrixFormat.kColwise

        lp = self._HighsLp()
        lp.num_col_ = self._num_cols
        lp.num_row_ = m
        lp.col_cost_ = col_cost
        lp.col_lower_ = col_lower
        lp.col_upper_ = col_upper
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper
        lp.integrality_ = integrality
        lp.a_matrix_ = mat

        status = self._highs.passModel(lp)
        if status != self._HighsStatus.kOk:
            raise RuntimeError("Failed to initialize HiGHS model.")

    def solve(self, syndrome: np.ndarray) -> Tuple[np.ndarray, float, str]:
        if syndrome.shape[0] != self._num_rows:
            raise ValueError(
                f"Syndrome length {syndrome.shape[0]} does not match {self._num_rows}"
            )
        for i in range(self._num_rows):
            s = float(syndrome[i])
            status = self._highs.changeRowBounds(i, s, s)
            if status != self._HighsStatus.kOk:
                raise RuntimeError("Failed to update HiGHS row bounds.")

        status = self._highs.run()
        if status != self._HighsStatus.kOk:
            raise RuntimeError("HiGHS failed to solve the model.")

        model_status = self._highs.getModelStatus()
        if model_status not in (
            self._HighsModelStatus.kOptimal,
            self._HighsModelStatus.kTimeLimit,
            self._HighsModelStatus.kObjectiveBound,
            self._HighsModelStatus.kObjectiveTarget,
            self._HighsModelStatus.kSolutionLimit,
        ):
            raise RuntimeError(f"HiGHS terminated with status {model_status}")

        solution = self._highs.getSolution()
        values = np.asarray(solution.col_value[: self._num_errors], dtype=float)
        correction = (values > 0.5).astype(np.uint8)
        objective = float(self._highs.getObjectiveValue())
        return correction, objective, str(model_status)


class _DirectGurobiSolver:
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
