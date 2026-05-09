# ILPQEC

ILPQEC is a Python package for maximum-likelihood quantum error correction decoding
using integer linear programming (ILP). It builds an ILP from parity-check matrices
or Stim DetectorErrorModels and solves it with a direct HiGHS backend by default
(no Pyomo required). Optional backends are available for direct Gurobi (licensed)
or Pyomo-based solvers.

## Installation

PyPI package and import name: `ilpqec`.

```bash
# Basic installation (direct HiGHS backend)
pip install ilpqec

# Optional: Pyomo backend for other solvers
pip install ilpqec[pyomo]

# Optional: direct Gurobi backend (licensed)
pip install ilpqec[gurobi]

# Optional: sinter integration (benchmarking)
pip install ilpqec[sinter]

# With Stim support
pip install ilpqec[stim]

# With SciPy sparse-matrix support
pip install ilpqec[scipy]
```

## Quickstart

### Parity-check matrix decoding

```python
import numpy as np
from ilpqec import Decoder

H = np.array([
    [1, 1, 0],
    [0, 1, 1],
], dtype=np.uint8)

# Uses direct HiGHS by default.
decoder = Decoder.from_parity_check_matrix(H)

syndrome = np.array([1, 0], dtype=np.uint8)
error, _ = decoder.decode(syndrome)
print(error)
```

### CSS code analysis from `Hx` and `Hz`

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
reduced = code.logical_basis(reduce=True, solver="highs")

print(distance.d, distance.dx, distance.dz)
print(distance.shortest_x)
print(distance.shortest_z)
print(reduced.x)
print(reduced.z)
```

`distance()` returns the globally shortest X/Z logical operators. `logical_basis(reduce=True)`
keeps the canonical logical cosets and reduces each one to the exact
minimum-weight representative.

This exact-analysis path currently supports binary CSS codes only and requires
the direct HiGHS backend.

### Stim DetectorErrorModel decoding

```python
import stim
from ilpqec import Decoder

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
)

dem = circuit.detector_error_model(decompose_errors=True)
decoder = Decoder.from_stim_dem(dem)

sampler = circuit.compile_detector_sampler()
detections, observables = sampler.sample(shots=100, separate_observables=True)

for i in range(5):
    _, predicted = decoder.decode(detections[i])
    print(f"shot {i}: predicted={predicted}, actual={observables[i]}")
```

### DEM equivalent distance

```python
import stim
from ilpqec import dem_distance

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
)

dem = circuit.detector_error_model(decompose_errors=True)

result = dem_distance(dem, solver="highs")
targeted = dem_distance(dem, target_observables=[1], solver="highs")

print(result.distance)
print(targeted.observable_mask)
```

`dem_distance(...)` returns an exact minimum-weight logical fault for the DEM.
If `target_observables` is provided, it instead enforces that exact nonzero
logical mask. The objective ignores `error(p)` probabilities and minimizes the
number of selected DEM mechanisms.

## Documentation Map

- CSS parity-check analysis: `css_code.md`
- DEM equivalent distance: `dem_distance.md`
- Solver backends and configuration: `solvers.md`
- ILP formulation and assumptions: `math.md`
- Stim DEM support and caveats: `stim_dem.md`
- Sinter integration: `sinter.md`
- Examples walkthrough: `examples.md`
- Benchmarks and scripts: `benchmarks.md`
