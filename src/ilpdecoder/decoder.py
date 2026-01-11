"""
Main Decoder class using Pyomo for ILP modeling.

This module provides the primary user-facing interface for ILP-based
quantum error correction decoding. Uses Pyomo for solver-agnostic
modeling.

Key Features:
- Solver-agnostic modeling via Pyomo
- Multiple construction methods (from_parity_check_matrix, from_stim_dem)
- Easy solver switching (scip, highs, gurobi, cplex, cbc, glpk)
- Maximum-likelihood and minimum-weight decoding

Example Usage:
    # From parity-check matrix (uses HiGHS by default)
    decoder = Decoder.from_parity_check_matrix(H)
    correction = decoder.decode(syndrome)
    
    # Use different solver
    decoder = Decoder.from_parity_check_matrix(H, solver="highs")
    
    # Change solver at runtime
    decoder.set_solver("gurobi", time_limit=60)
    
    # From Stim DEM
    decoder = Decoder.from_stim_dem(dem)
    correction, observables = decoder.decode(detector_outcomes)
"""

from typing import Union, List, Optional, Tuple, Dict, Any
from pathlib import Path
import math

import numpy as np
from scipy.sparse import spmatrix

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    Objective,
    Binary,
    NonNegativeIntegers,
    minimize,
    value,
    SolverFactory,
    TerminationCondition,
)

from ilpdecoder.solver import (
    SolverConfig,
    get_default_solver,
    get_pyomo_solver_name,
    get_available_solvers,
)


class Decoder:
    """
    ILP-based quantum error correction decoder using Pyomo.
    
    This class provides a PyMatching-like API for decoding using
    Integer Linear Programming. It uses Pyomo for solver-agnostic
    modeling.
    
    Solver switching is trivial - just call set_solver() with a different
    solver name. No need to rebuild the model.
    
    Supported Solvers:
        - highs: HiGHS solver (default)
        - scip: SCIP solver
        - gurobi: Gurobi (requires license)
        - cplex: IBM CPLEX (requires license)
        - cbc: COIN-OR CBC
        - glpk: GNU Linear Programming Kit
    
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
            solver: Solver name ("scip", "highs", "gurobi", etc.)
                Default is "highs" if available.
            **solver_options: Solver options (time_limit, gap, verbose, etc.)
            
        Returns:
            Configured Decoder instance
            
        Example:
            >>> H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
            >>> decoder = Decoder.from_parity_check_matrix(H)
            >>> correction = decoder.decode([1, 0, 1])
            
            >>> # With different solver
            >>> decoder = Decoder.from_parity_check_matrix(H, solver="highs")
        """
        decoder = cls()
        
        # Convert to numpy array
        if isinstance(parity_check_matrix, spmatrix):
            H = parity_check_matrix.toarray()
        else:
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
            solver: Solver name ("scip", "highs", etc.). Default is "highs".
            merge_parallel_edges: If True, merge parallel error mechanisms
            flatten_dem: If True, call dem.flattened() to inline repeats and
                apply detector shifts (may increase DEM size).
            **solver_options: Solver options (time_limit, gap, verbose, etc.)
        
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
        **options
    ):
        """
        Set or change the solver.
        
        You can switch solvers at any time without rebuilding the model.
        
        Args:
            solver: Solver name. Options:
                - "highs": HiGHS solver (default)
                - "scip": SCIP solver
                - "gurobi": Gurobi (requires license)
                - "cplex": IBM CPLEX (requires license)
                - "cbc": COIN-OR CBC
                - "glpk": GNU Linear Programming Kit
            time_limit: Maximum solving time in seconds
            gap: Relative ILP gap tolerance
            threads: Number of threads (solver-dependent)
            verbose: Print solver output
            **options: Additional solver-specific options
            
        Example:
            >>> decoder.set_solver("scip", time_limit=30)
            >>> decoder.set_solver("highs", threads=4)
            >>> decoder.set_solver("gurobi", gap=0.01, verbose=True)
        """
        if solver is None:
            solver = get_default_solver()
        
        self._solver_config = SolverConfig(
            name=solver.lower(),
            time_limit=time_limit,
            gap=gap,
            threads=threads,
            verbose=verbose,
            options=options
        )
    
    def get_solver_options(self) -> Dict[str, Any]:
        """Get current solver configuration as a dictionary."""
        return {
            "solver": self._solver_config.name,
            "time_limit": self._solver_config.time_limit,
            "gap": self._solver_config.gap,
            "threads": self._solver_config.threads,
            "verbose": self._solver_config.verbose,
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
        
        # Build and solve Pyomo model
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
        H = self._H
        weights = self._weights
        m, n = H.shape
        
        # Create Pyomo model
        model = ConcreteModel()
        
        # Error variables: e[j] ∈ {0,1}
        model.e = Var(range(n), within=Binary)
        
        # Auxiliary variables for mod-2 linearization: a[i] ∈ Z≥0
        # Upper bound: max value of sum H[i,:] is sum of row, so a[i] <= sum(H[i,:]) // 2
        def aux_bounds(model, i):
            max_sum = int(np.sum(H[i, :]))
            return (0, max(0, (max_sum - syndrome[i]) // 2))
        
        model.a = Var(range(m), within=NonNegativeIntegers, bounds=aux_bounds)
        
        # Objective: minimize weighted errors
        model.obj = Objective(
            expr=sum(weights[j] * model.e[j] for j in range(n)),
            sense=minimize
        )
        
        # Constraints: H @ e = syndrome + 2 * a (mod-2 linearization)
        def syndrome_constraint(model, i):
            lhs = sum(int(H[i, j]) * model.e[j] for j in range(n) if H[i, j] != 0)
            return lhs == int(syndrome[i]) + 2 * model.a[i]
        
        model.syndrome_cons = Constraint(range(m), rule=syndrome_constraint)
        
        # Get solver
        solver_name = get_pyomo_solver_name(self._solver_config.name)
        solver = SolverFactory(solver_name)
        
        if not solver.available():
            raise RuntimeError(
                f"Solver '{self._solver_config.name}' is not available. "
                f"Available solvers: {get_available_solvers()}"
            )
        
        # Set solver options
        options = self._solver_config.to_pyomo_options()
        for key, val in options.items():
            solver.options[key] = val
        
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
