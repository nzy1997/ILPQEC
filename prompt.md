You are an expert in quantum error correction (QEC), mathematical optimization, and optimization modeling frameworks (e.g. JuMP / MathOptInterface).
I want you to design and implement a Python package for quantum error correction decoding based on Integer Linear Programming (ILP / MILP).
This package should act as a PyMatching-like decoder, but with an ILP backend, and with a JuMP-style modular solver abstraction.
1. High-Level Goal
The package should:
Allow users to build a decoding model once
Allow users to choose or switch the solver backend at runtime, without changing the model
This design should explicitly mirror the separation of concerns in:
JuMP (modeling layer)
  +
MathOptInterface (solver abstraction)
  +
SCIP / HiGHS / Gurobi (solver backends)
2. Supported Decoding Frontends
The package must support two decoding entry points:
(A) Parity-check matrix decoding
Input: binary parity-check matrix H ∈ {0,1}^{m×n}
Input: syndrome s ∈ {0,1}^m
Output: correction vector e ∈ {0,1}^n
(B) Stim detector error model decoding
Input: stim.DetectorErrorModel
Automatically extract:
parity constraints between detectors and errors
logical observables
error probabilities / weights
3. Mathematical Model (ILP)
Formulate decoding as an integer linear program:
Binary decision variables represent physical errors (and auxiliary variables if needed).
Linear constraints enforce syndrome consistency:
H e = s (mod 2)
using explicit mod-2 linearization (no black-box tricks).
Objective:
minimum-weight decoding
or maximum-likelihood decoding using log-probability weights
4. Modular Solver Backend (CRITICAL)
The solver backend must be fully modular and user-configurable, inspired by JuMP / MOI.
Requirements:
Define a solver-agnostic intermediate model representation
Define a SolverInterface / Backend API, e.g.:
add_binary_var
add_linear_constraint
set_objective
solve()
Implement multiple backends, such as:
SCIP (via pyscipopt)
HiGHS (via Pyomo or highspy)
Optional: Gurobi / CPLEX
Users must be able to select the solver like:
decoder = Decoder.from_parity_check_matrix(H)
decoder.set_solver("scip", time_limit=10)
correction = decoder.decode(syndrome)
or:
decoder = Decoder.from_stim_dem(dem, solver="highs")
Switching solvers must not require rebuilding the decoding logic
5. Public API (PyMatching-like)
Design the public API to resemble PyMatching:
decoder = Decoder.from_parity_check_matrix(H, weights=...)
correction = decoder.decode(syndrome)
and:
decoder = Decoder.from_stim_dem(dem)
correction, logicals = decoder.decode(detector_outcomes)
6. Package Architecture
Please design a clean, extensible architecture including:
Frontend:
Parity-check matrix parser
Stim DEM parser
Core:
ILP model builder (solver-agnostic)
Mod-2 constraint linearization utilities
Backend:
Abstract solver interface
Concrete solver implementations (SCIP / HiGHS)
Public API layer
Tests and examples
Use:
Clear class hierarchies
Type hints
Docstrings
7. Scope and Assumptions
Prioritize correctness and clarity over performance
Target small-to-medium QEC codes (e.g. surface code distance ≤ 9)
Design should allow future extensions:
cutting planes
warm starts
heuristic / hybrid decoders
8. Deliverables
Please provide:
Overall architecture explanation
Mathematical formulation details
Core Python code (modular and runnable)
Example usage:
parity-check matrix decoding
Stim detector error model decoding
Think like a JuMP / MOI designer, not a one-off script author.
