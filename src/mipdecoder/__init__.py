"""
MIPDecoder: ILP-based Quantum Error Correction Decoder

A PyMatching-like decoder using Pyomo for solver-agnostic ILP modeling,
similar to JuMP in Julia.

Example:
    >>> from mipdecoder import Decoder
    >>> 
    >>> # Create decoder from parity-check matrix
    >>> decoder = Decoder.from_parity_check_matrix(H)
    >>> correction = decoder.decode(syndrome)
    >>> 
    >>> # Use different solver (like JuMP's set_optimizer)
    >>> decoder.set_solver("highs", time_limit=30)
    >>> 
    >>> # Create from Stim DEM
    >>> decoder = Decoder.from_stim_dem(dem)
    >>> correction, observables = decoder.decode(detector_outcomes)

Supported Solvers:
    - scip: SCIP solver (default)
    - highs: HiGHS solver
    - gurobi: Gurobi (requires license)
    - cplex: IBM CPLEX (requires license)
    - cbc: COIN-OR CBC
    - glpk: GNU Linear Programming Kit
"""

from mipdecoder.decoder import Decoder
from mipdecoder.solver import get_available_solvers, get_default_solver, SolverConfig

__version__ = "0.1.0"
__all__ = ["Decoder", "get_available_solvers", "get_default_solver", "SolverConfig"]
