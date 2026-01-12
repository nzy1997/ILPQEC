"""
ILPDecoder: ILP-based Quantum Error Correction Decoder

A PyMatching-like decoder using a direct HiGHS backend by default, with
an optional direct Gurobi backend and a Pyomo backend for other solvers.

Example:
    >>> from ilpdecoder import Decoder
    >>> 
    >>> # Create decoder from parity-check matrix
    >>> decoder = Decoder.from_parity_check_matrix(H)
    >>> correction = decoder.decode(syndrome)
    >>> 
    >>> # Use different solver (requires Pyomo extra)
    >>> decoder.set_solver("scip", time_limit=30, direct=False)
    >>> 
    >>> # Create from Stim DEM
    >>> decoder = Decoder.from_stim_dem(dem)
    >>> correction, observables = decoder.decode(detector_outcomes)

Supported Solvers:
    - highs: HiGHS solver (default, direct backend)
    - gurobi: Gurobi solver (direct backend, requires license)
    - scip: SCIP solver (Pyomo required)
    - cplex: IBM CPLEX (requires license, Pyomo required)
    - cbc: COIN-OR CBC (Pyomo required)
    - glpk: GNU Linear Programming Kit (Pyomo required)
"""

from ilpdecoder.decoder import Decoder
from ilpdecoder.solver import get_available_solvers, get_default_solver, SolverConfig

__version__ = "0.1.0"
__all__ = ["Decoder", "get_available_solvers", "get_default_solver", "SolverConfig"]
