# MIPDecoder

An ILP-based Quantum Error Correction (QEC) decoder built on **Pyomo** for solver-agnostic optimization modeling.

## Overview

MIPDecoder provides:

- **Solver-agnostic ILP modeling** via Pyomo — write once, use any supported solver
- **PyMatching-like API** for quantum error correction decoding
- **Easy solver switching**: HiGHS (default), SCIP, Gurobi, CPLEX, CBC, GLPK
- **Support for parity-check matrix** and **Stim DetectorErrorModel** inputs

## Installation

```bash
# Basic installation (requires a solver to be installed separately)
pip install mipdecoder

# With Stim support
pip install mipdecoder[stim]
```

### Installing Solvers

MIPDecoder uses Pyomo, which requires an external solver. Install at least one:

```bash
# HiGHS (default, open-source, easy to install)
pip install highspy

# SCIP (open-source)
# Download from: https://www.scipopt.org/
# Or via conda: conda install -c conda-forge scip

# CBC (open-source)
# Ubuntu/Debian: apt install coinor-cbc
# macOS: brew install cbc

# GLPK (open-source)
# Ubuntu/Debian: apt install glpk-utils
# macOS: brew install glpk

# Gurobi (commercial, free academic license)
# https://www.gurobi.com/

# CPLEX (commercial, free academic license)
# https://www.ibm.com/products/ilog-cplex-optimization-studio
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/mipdecoder/mipdecoder
cd mipdecoder

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv add pyomo numpy scipy stim
uv add pytest --dev

# Or using pip
pip install -e ".[dev]"
```

### Running Tests Locally

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_decoder.py -v

# Run a quick functionality check
python main.py
```

### Running Examples

```bash
python examples/basic_usage.py
python examples/surface_code_example.py
```

## Quick Start

### Parity-Check Matrix Decoding

```python
import numpy as np
from mipdecoder import Decoder

# Define a simple repetition code parity-check matrix
H = np.array([
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
])

# Create decoder (uses HiGHS by default)
decoder = Decoder.from_parity_check_matrix(H)

# Decode a syndrome
syndrome = [1, 0, 0, 1]
correction = decoder.decode(syndrome)
print(f"Correction: {correction}")
```

### Switching Solvers

```python
# Use SCIP instead of HiGHS
decoder = Decoder.from_parity_check_matrix(H, solver="scip")

# Or change solver later
decoder.set_solver("gurobi", time_limit=30)
decoder.set_solver("cplex", gap=0.01)
```

### Stim DetectorErrorModel Decoding

```python
import stim
from mipdecoder import Decoder

# Generate a surface code circuit
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01
)

# Get detector error model
dem = circuit.detector_error_model(decompose_errors=True)

# Create decoder
decoder = Decoder.from_stim_dem(dem, solver="scip")

# Sample and decode
sampler = circuit.compile_detector_sampler()
detection_events, observables = sampler.sample(shots=100, separate_observables=True)

for i in range(10):
    _, predicted_obs = decoder.decode(detection_events[i])
    print(f"Shot {i}: predicted={predicted_obs}, actual={observables[i]}")
```

#### Stim DEM Support Notes

- Only `error(p)` lines are parsed; tags in `error[...]` are ignored. `detector` and
  `logical_observable` metadata lines are ignored. `shift_detectors` offsets are applied.
  `repeat` blocks are flattened by default; this can expand large DEMs.
  `detector_separator` is unsupported and raises an error.
- The `^` separator is treated as whitespace and does not change parsing.
- If you want to fail fast instead of flattening, pass `flatten_dem=False`.

### Maximum-Likelihood Decoding with Weights

```python
import numpy as np
from mipdecoder import Decoder

H = np.array([[1, 1, 0], [0, 1, 1]])
error_probs = [0.1, 0.01, 0.1]

# Weights are computed automatically from probabilities
decoder = Decoder.from_parity_check_matrix(H, error_probabilities=error_probs)

syndrome = [1, 1]
correction, weight = decoder.decode(syndrome, return_weight=True)
print(f"ML correction: {correction}, weight: {weight}")
```

Note: `error_probabilities` must be in (0, 0.5]; pass explicit `weights` for p > 0.5.

## Mathematical Formulation

### ILP Model for QEC Decoding

Given:
- Binary parity-check matrix H ∈ {0,1}^{m×n}
- Syndrome s ∈ {0,1}^m
- Weights w ∈ ℝ^n (typically w_j = log((1-p_j)/p_j))

**Decision Variables:**
- e_j ∈ {0,1} for j = 0,...,n-1 (error indicators)
- a_i ∈ ℤ≥0 for i = 0,...,m-1 (auxiliary for mod-2 linearization)

**Objective:**
```
minimize Σ_j w_j · e_j
```

**Constraints (mod-2 linearization):**
```
Σ_j H[i,j] · e_j - s_i = 2·a_i    for i = 0,...,m-1
```

## Solver Options

```python
decoder.set_solver(
    "scip",           # Solver name
    time_limit=30,    # Max solving time (seconds)
    gap=0.01,         # Relative MIP gap tolerance
    threads=4,        # Number of threads (solver-dependent)
    verbose=True,     # Print solver output
)
```

| Option | Description | Supported Solvers |
|--------|-------------|-------------------|
| `time_limit` | Max solving time (seconds) | All |
| `gap` | Relative MIP gap tolerance | All |
| `threads` | Number of threads | HiGHS, Gurobi, CPLEX |
| `verbose` | Print solver output | All |

## API Reference

### `Decoder`

Main decoder class.

**Class Methods:**
- `from_parity_check_matrix(H, weights=None, error_probabilities=None, solver=None)` - Create from parity-check matrix
- `from_stim_dem(dem, solver=None, merge_parallel_edges=True, flatten_dem=True)` - Create from Stim DetectorErrorModel

**Instance Methods:**
- `decode(syndrome, return_weight=False)` - Decode a single syndrome
- `decode_batch(syndromes)` - Decode multiple syndromes
- `set_solver(name, **options)` - Switch solver

**Properties:**
- `num_detectors` - Number of parity checks/detectors
- `num_errors` - Number of error mechanisms
- `num_observables` - Number of logical observables (for DEM)
- `solver_name` - Current solver name

### `get_available_solvers()`

Returns a list of available solver names.

```python
from mipdecoder import get_available_solvers
print(get_available_solvers())  # e.g., ['scip', 'highs', 'cbc']
```

## License

MIT License
