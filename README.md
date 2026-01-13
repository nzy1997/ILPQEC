# ILPDecoder

[![CI](https://github.com/nzy1997/ILPDecoder/actions/workflows/ci.yml/badge.svg)](https://github.com/nzy1997/ILPDecoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nzy1997/ILPDecoder/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/ILPDecoder)

ILPDecoder is a Python package for maximum-likelihood quantum error correction decoding using integer linear programming (ILP). It turns parity-check matrices or Stim `DetectorErrorModel`s into an ILP and solves it with a **direct HiGHS** backend by default (no Pyomo required). An optional **direct Gurobi** backend is available for licensed users. It is aimed at correctness-focused baselines, solver comparisons, and small-to-medium code studies rather than high-throughput production decoding.

Documentation: https://nzy1997.github.io/ILPDecoder/

## Scope and Highlights

What it does well:
- **Direct HiGHS decoding** out of the box (no Pyomo dependency).
- **Optional direct Gurobi backend** when `gurobipy` is installed.
- **Optional solver switching** via Pyomo (SCIP, Gurobi, CPLEX, CBC, GLPK) when installed.
- **Inputs from parity-check matrices** or **Stim DetectorErrorModel**.
- **Maximum-likelihood decoding** via weights or error probabilities.
- **PyMatching-like API** for easy experimentation.

When it is not a fit:
- Large code distances or high-shot workloads where ILP scaling dominates; use MWPM/BPOSD for throughput.

## Installation

```bash
# Basic installation (direct HiGHS backend)
pip install ilpdecoder

# Optional: Pyomo backend for other solvers
pip install ilpdecoder[pyomo]

# Optional: direct Gurobi backend (licensed)
pip install ilpdecoder[gurobi]

# With Stim support
pip install ilpdecoder[stim]

# With SciPy sparse-matrix support
pip install ilpdecoder[scipy]
```

Note: the `gurobi` extra installs `gurobipy` wheels for Python 3.9-3.12. On
Python 3.13+ install `gurobipy` manually or use the Pyomo backend.

### Installing Solvers

The default backend uses HiGHS via `highspy` (installed with ILPDecoder). For
direct Gurobi, install `gurobipy`. For other solvers, install the Pyomo extra
and a solver binary:

```bash
# Pyomo backend for alternate solvers
pip install ilpdecoder[pyomo]

# Direct Gurobi backend (licensed)
pip install ilpdecoder[gurobi]

# Gurobi wheels are typically available for Python 3.9-3.12.
# For Python 3.13+ install gurobipy manually or use the Pyomo backend.

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
git clone https://github.com/nzy1997/ILPDecoder
cd ILPDecoder

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv add highspy numpy scipy stim
uv add pyomo --dev
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
benchmark/.venv/bin/python benchmark/benchmark_decoders.py --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

## Quick Start

### Parity-Check Matrix Decoding

```python
import numpy as np
from ilpdecoder import Decoder

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

Note: passing SciPy sparse matrices requires `scipy` to be installed (e.g., `pip install ilpdecoder[scipy]`).

### Switching Solvers

```python
# Use SCIP instead of HiGHS
decoder = Decoder.from_parity_check_matrix(H, solver="scip")

# Or change solver later
decoder.set_solver("gurobi", time_limit=30)
decoder.set_solver("cplex", gap=0.01)
```

Note: non-HiGHS solvers require the Pyomo extra (`pip install ilpdecoder[pyomo]`),
except for the direct Gurobi backend (`pip install ilpdecoder[gurobi]`).

### Stim DetectorErrorModel Decoding

```python
import stim
from ilpdecoder import Decoder

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
from ilpdecoder import Decoder

H = np.array([[1, 1, 0], [0, 1, 1]])
error_probs = [0.1, 0.01, 0.1]

# Weights are computed automatically from probabilities
decoder = Decoder.from_parity_check_matrix(H, error_probabilities=error_probs)

syndrome = [1, 1]
correction, weight = decoder.decode(syndrome, return_weight=True)
print(f"ML correction: {correction}, weight: {weight}")
```

Note: `error_probabilities` must be in (0, 0.5]; pass explicit `weights` for p > 0.5.

## Benchmark

Install optional deps for the benchmarks:

```bash
pip install stim pymatching ldpc
```

Non-HiGHS solvers require the Pyomo extra (`pip install ilpdecoder[pyomo]`),
except for the direct Gurobi backend (`pip install ilpdecoder[gurobi]`).

Notes:
- Direct backends: HiGHS, Gurobi.
- Pyomo backends: HiGHS, SCIP, CBC, GLPK, Gurobi, CPLEX.
- BPOSD runs with `max_iter=50`, `osd_order=0`, and `bp_method=minimum_sum`.

### Circuit-level rotated surface code memory

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 2.7469 | 1.610% |
| ILP[gurobi] (direct) | 0.5923 | 1.620% |
| ILP[scip] (Pyomo) | 27.1241 | 1.620% |
| ILP[cbc] (Pyomo) | 13.7808 | 1.620% |
| ILP[glpk] (Pyomo) | 7.8176 | 1.610% |
| MWPM (pymatching) | 0.0034 | 2.090% |
| BPOSD (ldpc) | 0.0308 | 7.740% |

### Code-capacity surface code (data errors only, perfect syndrome)

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py --noise-model code_capacity --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk --shots 10000 --distance 3 --rounds 1 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 3.1914 | 0.120% |
| ILP[gurobi] (direct) | 0.0826 | 0.120% |
| ILP[scip] (Pyomo) | 22.6194 | 0.120% |
| ILP[cbc] (Pyomo) | 9.8211 | 0.120% |
| ILP[glpk] (Pyomo) | 4.7919 | 0.120% |
| MWPM (pymatching) | 0.0033 | 0.120% |
| BPOSD (ldpc) | 0.0029 | 0.120% |

### Color code (`color_code:memory_xyz`)

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py --code-task color_code:memory_xyz --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 2.0008 | 4.510% |
| ILP[gurobi] (direct) | 0.3164 | 4.500% |
| ILP[scip] (Pyomo) | 24.0461 | 4.500% |
| ILP[cbc] (Pyomo) | 11.0780 | 4.510% |
| ILP[glpk] (Pyomo) | 5.8961 | 4.500% |
| MWPM (pymatching) | 0.0041 | 13.610% |
| BPOSD (ldpc) | 0.0124 | 9.970% |

## Solver Options

```python
decoder.set_solver(
    "scip",            # Solver name (Pyomo required)
    direct=False,      # Use Pyomo backend
    time_limit=30,     # Max solving time (seconds)
    gap=0.01,          # Relative ILP gap tolerance
    threads=4,         # Number of threads (solver-dependent)
    verbose=True,      # Print solver output
)
```

| Option | Description | Supported Solvers |
|--------|-------------|-------------------|
| `time_limit` | Max solving time (seconds) | All |
| `gap` | Relative ILP gap tolerance | All |
| `threads` | Number of threads | HiGHS, Gurobi, CPLEX |
| `verbose` | Print solver output | All |
| `direct` | Use direct backend (default for HiGHS) | HiGHS, Gurobi |

Note: `direct` defaults to True when `solver="highs"`. For Gurobi, it defaults
to True when `gurobipy` is installed; set `direct=False` to use Pyomo.

Backend map:
- Direct backends: HiGHS, Gurobi.
- Pyomo backends: HiGHS, SCIP, CBC, GLPK, Gurobi, CPLEX.

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
from ilpdecoder import get_available_solvers
print(get_available_solvers())  # e.g., ['scip', 'highs', 'cbc']
```

## License

MIT License
