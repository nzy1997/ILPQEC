# ILPDecoder

[![CI](https://github.com/nzy1997/ILPDecoder/actions/workflows/ci.yml/badge.svg)](https://github.com/nzy1997/ILPDecoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nzy1997/ILPDecoder/branch/main/graph/badge.svg)](https://codecov.io/gh/nzy1997/ILPDecoder)

ILPDecoder is a Python package for maximum-likelihood quantum error correction decoding using integer linear programming (ILP). It turns parity-check matrices or Stim `DetectorErrorModel`s into an ILP and solves it with a built-in backend out of the box. It is aimed at correctness-focused baselines, solver comparisons, and small-to-medium code studies rather than high-throughput production decoding.

Documentation: https://nzy1997.github.io/ILPDecoder/

## Scope and Highlights

What it does well:
- **Out-of-the-box decoding** with minimal setup.
- **Inputs from parity-check matrices** or **Stim DetectorErrorModel**.
- **Maximum-likelihood decoding** via weights or error probabilities.
- **PyMatching-like API** for easy experimentation.

When it is not a fit:
- Large code distances or high-shot workloads where ILP scaling dominates; use MWPM/BPOSD for throughput.

## Installation

### Python Environment Setup

ILPDecoder supports Python 3.9+ (3.9–3.12 recommended). If you plan to use
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

```bash
# Basic installation
uv pip install ilpdecoder

# With Stim support
uv pip install ilpdecoder[stim]

# With sinter integration
uv pip install ilpdecoder[sinter]

# With SciPy sparse-matrix support
uv pip install ilpdecoder[scipy]
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

### Benchmark Environment (recommended)

Benchmarks use extra dependencies (`stim`, `pymatching`, `ldpc`) and optional
solvers. Use a dedicated virtual environment under `benchmark/`:

```bash
uv venv benchmark/.venv
uv pip install --python benchmark/.venv/bin/python --upgrade pip

# ILPDecoder + optional solver backends
uv pip install --python benchmark/.venv/bin/python -e ".[pyomo,gurobi]"

# Benchmark dependencies
uv pip install --python benchmark/.venv/bin/python stim pymatching ldpc
```

Notes:
- Gurobi requires a valid license and Python < 3.13.
- If you do not need Gurobi, drop `gurobi` from the extras.

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

# Create decoder
decoder = Decoder.from_parity_check_matrix(H)

# Decode a syndrome
syndrome = [1, 0, 0, 1]
correction = decoder.decode(syndrome)
print(f"Correction: {correction}")
```

Note: passing SciPy sparse matrices requires `scipy` to be installed (e.g., `uv pip install ilpdecoder[scipy]`).

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

ILPDecoder includes a sinter decoder wrapper for benchmarking and sampling.
Install with:

```bash
uv pip install ilpdecoder[sinter]
```

Example usage:

```python
import sinter
import stim
from ilpdecoder.sinter_decoder import SinterIlpDecoder

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
)

tasks = [
    sinter.Task(
        circuit=circuit,
        decoder="ilpdecoder",
    )
]

stats = sinter.collect(
    tasks=tasks,
    custom_decoders={"ilpdecoder": SinterIlpDecoder()},
)
```

Notes:
- The sinter adapter currently uses the direct HiGHS backend only.
- `ilpdecoder[sinter]` includes `stim` and `sinter` dependencies.

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
uv pip install stim pymatching ldpc tesseract-decoder
```

Notes:
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
from ilpdecoder import get_available_solvers
print(get_available_solvers())  # e.g., ['scip', 'highs', 'cbc']
```

## License

MIT License
