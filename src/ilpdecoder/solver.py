"""
Solver configuration for ILPDecoder.

This module provides solver configuration for selecting and configuring backends.
The default backend is direct HiGHS; direct Gurobi is optional; Pyomo is used
for other solvers.

Supported Solvers:
    - highs: HiGHS solver (default, direct backend)
    - gurobi: Gurobi solver (direct backend, requires license)
    - scip: SCIP solver (via scip or scipampl)
    - cplex: IBM CPLEX (requires license)
    - cbc: COIN-OR CBC
    - glpk: GNU Linear Programming Kit

Example:
    # Use default solver (HiGHS)
    decoder = Decoder.from_parity_check_matrix(H)
    
    # Use specific solver
    decoder = Decoder.from_parity_check_matrix(H, solver="highs")
    
    # Change solver
    decoder.set_solver("gurobi", time_limit=60)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import shutil


@dataclass
class SolverConfig:
    """
    Configuration for ILP solver.
    
    Attributes:
        name: Solver name (highs, scip, gurobi, cplex, cbc, glpk)
        time_limit: Maximum solving time in seconds
        gap: Relative ILP gap tolerance
        threads: Number of threads (solver-dependent)
        verbose: Print solver output
        direct: Use a direct backend when available (e.g., HiGHS, Gurobi)
        options: Additional solver-specific options
    """
    name: str = "highs"  # HiGHS is the default (easy to pip install)
    time_limit: Optional[float] = None
    gap: Optional[float] = None
    threads: Optional[int] = None
    verbose: bool = False
    direct: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_pyomo_options(self) -> Dict[str, Any]:
        """Convert to Pyomo solver options."""
        opts = {}
        
        if self.name == "scip":
            if self.time_limit is not None:
                opts["limits/time"] = self.time_limit
            if self.gap is not None:
                opts["limits/gap"] = self.gap
        elif self.name == "highs":
            if self.time_limit is not None:
                opts["time_limit"] = self.time_limit
            if self.gap is not None:
                opts["mip_rel_gap"] = self.gap
            if self.threads is not None:
                opts["threads"] = self.threads
        elif self.name == "gurobi":
            if self.time_limit is not None:
                opts["TimeLimit"] = self.time_limit
            if self.gap is not None:
                opts["MIPGap"] = self.gap
            if self.threads is not None:
                opts["Threads"] = self.threads
        elif self.name == "cplex":
            if self.time_limit is not None:
                opts["timelimit"] = self.time_limit
            if self.gap is not None:
                opts["mip_tolerances_mipgap"] = self.gap
            if self.threads is not None:
                opts["threads"] = self.threads
        elif self.name in ("cbc", "glpk"):
            if self.time_limit is not None:
                opts["seconds"] = self.time_limit if self.name == "cbc" else None
                opts["tmlim"] = self.time_limit if self.name == "glpk" else None
            if self.gap is not None:
                opts["ratioGap"] = self.gap if self.name == "cbc" else None
        
        # Add any custom options
        opts.update(self.options)
        
        # Remove None values
        return {k: v for k, v in opts.items() if v is not None}


# Solver name mappings (Pyomo solver names)
SOLVER_EXECUTABLES = {
    "scip": ["scip", "scipampl"],
    "highs": ["highs", "highspy"],
    "gurobi": ["gurobi", "gurobi_direct"],
    "cplex": ["cplex", "cplex_direct"],
    "cbc": ["cbc"],
    "glpk": ["glpk"],
}

# Default solver preference order (HiGHS is easy to install via pip)
DEFAULT_SOLVER_ORDER = ["highs", "scip", "cbc", "glpk", "gurobi", "cplex"]


def is_pyomo_available() -> bool:
    """Return True if Pyomo is available."""
    try:
        import pyomo  # noqa: F401
    except Exception:
        return False
    return True


def _highs_available() -> bool:
    try:
        import highspy  # noqa: F401
        return True
    except Exception:
        return shutil.which("highs") is not None


def is_gurobi_available() -> bool:
    """Return True if gurobipy is importable for the direct Gurobi backend."""
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return False
    return True


def require_pyomo() -> None:
    """Raise if Pyomo is not installed."""
    if not is_pyomo_available():
        raise ImportError(
            "Pyomo is required for non-HiGHS solvers. "
            "Install with: pip install ilpdecoder[pyomo]"
        )


def get_available_solvers() -> List[str]:
    """
    Get list of available solvers.
    
    Returns:
        List of solver names that are installed and available.
    """
    import logging
    import warnings
    
    available = []
    if _highs_available():
        available.append("highs")
    if is_gurobi_available():
        available.append("gurobi")
    if not is_pyomo_available():
        return list(dict.fromkeys(available))
    
    # Suppress Pyomo warnings during solver detection
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    
    for solver_name, executables in SOLVER_EXECUTABLES.items():
        if solver_name == "highs":
            continue
        # First check for executable in PATH
        for exe in executables:
            if shutil.which(exe) is not None:
                available.append(solver_name)
                break
        else:
            # Check if Pyomo can find the solver
            try:
                from pyomo.environ import SolverFactory
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for exe in executables:
                        try:
                            solver = SolverFactory(exe)
                            if solver.available():
                                available.append(solver_name)
                                break
                        except Exception:
                            continue
            except Exception:
                pass
    
    return list(dict.fromkeys(available))


def get_default_solver() -> str:
    """
    Get the default solver.
    
    Returns HiGHS if available, otherwise the first available solver.
    
    Returns:
        Name of the default solver.
        
    Raises:
        RuntimeError: If no solver is available.
    """
    if _highs_available():
        return "highs"
    available = get_available_solvers()
    for solver in DEFAULT_SOLVER_ORDER:
        if solver in available:
            return solver
    if available:
        return available[0]
    
    raise RuntimeError(
        "No ILP solver available. Please install one of:\n"
        "  - HiGHS: pip install highspy\n"
        "  - Gurobi: pip install ilpdecoder[gurobi]\n"
        "  - Pyomo solvers: pip install ilpdecoder[pyomo]"
    )


def get_pyomo_solver_name(solver: str) -> str:
    """
    Get the Pyomo solver name for a given solver.
    
    Args:
        solver: User-facing solver name
        
    Returns:
        Pyomo-compatible solver name
    """
    import logging
    import warnings
    
    # Suppress Pyomo warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    
    # Try executables in order
    executables = SOLVER_EXECUTABLES.get(solver.lower(), [solver])
    
    try:
        from pyomo.environ import SolverFactory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for exe in executables:
                try:
                    s = SolverFactory(exe)
                    if s.available():
                        return exe
                except Exception:
                    continue
    except Exception:
        pass
    
    # Return first option as fallback
    return executables[0] if executables else solver
