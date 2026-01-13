# ILPDecoder

ILPDecoder is a Python package for maximum-likelihood quantum error correction decoding
using integer linear programming (ILP). It builds an ILP from parity-check matrices
or Stim DetectorErrorModels and solves it with a direct HiGHS backend by default
(no Pyomo required). Optional backends are available for direct Gurobi (licensed)
or Pyomo-based solvers.

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

## Quickstart

```python
import numpy as np
from ilpdecoder import Decoder

H = np.array([
    [1, 1, 0],
    [0, 1, 1],
], dtype=np.uint8)

# Build a decoder from a parity-check matrix.
decoder = Decoder.from_parity_check_matrix(H)

# Example syndrome and decode.
syndrome = np.array([1, 0], dtype=np.uint8)
error, _ = decoder.decode(syndrome)
print(error)
```

## Solver Backends

- Direct HiGHS: default backend, installed with the package.
- Direct Gurobi: optional licensed backend via `pip install ilpdecoder[gurobi]`.
- Pyomo backend: optional solver switching (SCIP, CBC, GLPK, Gurobi, CPLEX) via
  `pip install ilpdecoder[pyomo]`.

## Stim DEM Notes

- Only `error(p)` lines are parsed; tags in `error[...]` are ignored.
- `detector` and `logical_observable` metadata lines are ignored.
- `shift_detectors` offsets are applied.
- `repeat` blocks are flattened by default; set `flatten_dem=False` to disable.
- `detector_separator` is unsupported and raises an error.
- The `^` separator is treated as whitespace. For correlated mechanisms, prefer
  `decompose_errors=True` in Stim to avoid ambiguous alternatives.

## Build This Site

```bash
pip install .[docs]
mkdocs serve
```
