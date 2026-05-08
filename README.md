# ILPQEC

[![CI](https://github.com/nzy1997/ILPQEC/actions/workflows/ci.yml/badge.svg)](https://github.com/nzy1997/ILPQEC/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nzy1997/ILPQEC/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/ILPQEC)

ILPQEC is a Python package for maximum-likelihood quantum error correction decoding using integer linear programming (ILP). It turns parity-check matrices or Stim `DetectorErrorModel`s into an ILP and solves it with a built-in backend out of the box. It is aimed at correctness-focused baselines, solver comparisons, and small-to-medium code studies rather than high-throughput production decoding.

Documentation: https://nzy1997.github.io/ILPQEC/

## Scope and Highlights

What it does well:
- **Out-of-the-box decoding** with minimal setup.
- **Inputs from parity-check matrices** or **Stim DetectorErrorModel**.
- **Maximum-likelihood decoding** via weights or error probabilities.
- **Exact CSS code analysis** from `Hx` and `Hz`, including code distance and logical operators.
- **PyMatching-like API** for easy experimentation.

When it is not a fit:
- Large code distances or high-shot workloads where ILP scaling dominates; use MWPM/BPOSD for throughput.

## Installation

### Python Environment Setup

ILPQEC supports Python 3.9+ (3.9–3.12 recommended). If you plan to use
Gurobi, install Python < 3.13 because `gurobipy` wheels are not available on
3.13 at the moment.

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install --upgrade pip
```

On Windows:

```bash
uv venv
.venv\\Scripts\\activate
uv pip install --upgrade pip
```

### Install the Package

PyPI package and import name: `ilpqec`.

```bash
# Basic installation
uv pip install ilpqec

# Direct HiGHS backend for decoding and CSS analysis
uv pip install highspy

# With Stim support
uv pip install ilpqec[stim]

# With sinter integration
uv pip install ilpqec[sinter]

# With SciPy sparse-matrix support
uv pip install ilpqec[scipy]
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/nzy1997/ILPQEC
cd ILPQEC

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Running Tests Locally

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_decoder.py -v

# Run a quick functionality check
uv run python main.py
```

### Running Examples

```bash
uv run python examples/basic_usage.py
uv run python examples/surface_code_example.py
uv run --python benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

### Serve Docs Locally

```bash
# From the repo root
uv pip install -e ".[docs]"
uv run mkdocs serve
```

## Quick Start

### Parity-Check Matrix Decoding

```python
import numpy as np
from ilpqec import Decoder

# Define a simple repetition code parity-check matrix
H = np.array([
    [1, 1, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1],
])

# Create decoder
decoder = Decoder.from_parity_check_matrix(H)

# Decode a syndrome
syndrome = [1, 0, 0, 1]
correction = decoder.decode(syndrome)
print(f"Correction: {correction}")
```

Note: passing SciPy sparse matrices requires `scipy` to be installed (e.g., `uv pip install ilpqec[scipy]`).

### CSS Code Distance and Logical Operators

```python
import numpy as np
from ilpqec import CSSCode

h = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

code = CSSCode.from_parity_check_matrices(h, h)

distance = code.distance(solver="highs")
print(distance.d, distance.dx, distance.dz)
print(distance.shortest_x)
print(distance.shortest_z)

basis = code.logical_basis(reduce=False)
reduced = code.logical_basis(reduce=True, solver="highs")
print(basis.x @ basis.z.T % 2)
print(reduced.x)
print(reduced.z)
```

`distance()` returns globally shortest nontrivial X/Z logical operators. `logical_basis(reduce=True)`
instead fixes the canonical logical cosets from Gaussian elimination and reduces
each generator to the exact minimum-weight representative of that coset.

Notes:
- CSS analysis currently supports binary CSS codes only.
- Exact `distance()` and `logical_basis(reduce=True)` require the direct HiGHS backend.
- Positive solver gaps are rejected; these APIs raise instead of returning approximate results.
- If `k == 0`, logical-operator APIs raise `ValueError`.
- SciPy sparse `Hx` / `Hz` inputs require `uv pip install ilpqec[scipy]`.

### Stim DetectorErrorModel Decoding

```python
import stim
from ilpqec import Decoder

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
decoder = Decoder.from_stim_dem(dem)

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

### Sinter Integration (optional)

ILPQEC includes a sinter decoder wrapper for benchmarking and sampling.
Install with:

```bash
uv pip install ilpqec[sinter]
```

Example usage:

```python
import sinter
import stim
from ilpqec.sinter_decoder import SinterIlpDecoder

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
)

tasks = [
    sinter.Task(
        circuit=circuit,
        decoder="ilpqec",
    )
]

stats = sinter.collect(
    tasks=tasks,
    custom_decoders={"ilpqec": SinterIlpDecoder()},
)
```

Notes:
- The sinter adapter currently uses the direct HiGHS backend only.
- `ilpqec[sinter]` includes `stim` and `sinter` dependencies.

### Maximum-Likelihood Decoding with Weights

```python
import numpy as np
from ilpqec import Decoder

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

Benchmarks use extra dependencies and optional solver backends. Use a dedicated
virtual environment under `benchmark/`:

```bash
uv venv benchmark/.venv
uv pip install --python benchmark/.venv/bin/python --upgrade pip

# ILPQEC + optional solver backends
uv pip install --python benchmark/.venv/bin/python -e ".[pyomo,gurobi]"

# Benchmark dependencies
uv pip install --python benchmark/.venv/bin/python stim pymatching ldpc tesseract-decoder
```

Notes:
- Gurobi requires a valid license and Python < 3.13.
- If you do not need Gurobi, drop `gurobi` from the extras.
- BPOSD runs with `max_iter=50`, `osd_order=0`, and `bp_method=minimum_sum`.
- Tesseract runs with `det_beam=50` by default (adjustable via `--tesseract-beam`).

### Circuit-level rotated surface code memory

```bash
uv run --python benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 2.7514 | 1.640% |
| ILP[gurobi] (direct) | 0.6403 | 1.650% |
| ILP[scip] | 28.2160 | 1.670% |
| ILP[cbc] | 14.9315 | 1.670% |
| ILP[glpk] | 8.6292 | 1.670% |
| MWPM (pymatching) | 0.0035 | 2.150% |
| BPOSD (ldpc) | 0.0308 | 7.680% |
| Tesseract | 0.1602 | 1.640% |

### Code-capacity surface code (data errors only, perfect syndrome)

```bash
uv run --python benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --noise-model code_capacity --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 1 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 3.2321 | 0.070% |
| ILP[gurobi] (direct) | 0.0838 | 0.070% |
| ILP[scip] | 23.4834 | 0.070% |
| ILP[cbc] | 10.4697 | 0.070% |
| ILP[glpk] | 5.0085 | 0.070% |
| MWPM (pymatching) | 0.0036 | 0.070% |
| BPOSD (ldpc) | 0.0028 | 0.070% |
| Tesseract | 0.0093 | 0.070% |

### Color code (`color_code:memory_xyz`)

```bash
uv run --python benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --code-task color_code:memory_xyz --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000, your numbers will vary):

| Decoder | Time (ms/shot) | Logical Error Rate |
|--------|---------------|--------------------|
| ILP[highs] (direct) | 2.0226 | 4.450% |
| ILP[gurobi] (direct) | 0.3184 | 4.420% |
| ILP[scip] | 24.9402 | 4.420% |
| ILP[cbc] | 11.6961 | 4.450% |
| ILP[glpk] | 6.0799 | 4.420% |
| MWPM (pymatching) | 0.0034 | 13.420% |
| BPOSD (ldpc) | 0.0114 | 9.830% |
| Tesseract | 0.0600 | 4.450% |

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
from ilpqec import get_available_solvers
print(get_available_solvers())  # e.g., ['scip', 'highs', 'cbc']
```

## License

MIT License
