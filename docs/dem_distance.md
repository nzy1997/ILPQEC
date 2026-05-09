# DEM Equivalent Distance

`dem_distance(...)` computes an exact minimum-weight logical fault for a Stim
`DetectorErrorModel` by solving an ILP over the DEM mechanisms.

## Quick Example

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
print(result.observable_mask)
print(targeted.distance)
print(targeted.observable_mask)
```

For this rotated surface-code DEM, both calls return distance `3`. The targeted
call constrains the returned fault to flip the requested logical observable mask.

## Semantics

- The input DEM must contain at least one logical observable.
- The optimization objective is Hamming weight over DEM mechanisms, not
  likelihood. `error(p)` probabilities are ignored by `dem_distance(...)`.
- `target_observables=None` searches for the minimum-weight fault whose logical
  effect is nonzero.
- `target_observables=[...]` switches to targeted mode and requires an exact
  binary observable mask. The mask must be one-dimensional, match the DEM
  observable count, and contain at least one `1`.
- `merge_parallel_edges` and `flatten_dem` are passed through to the DEM parser,
  matching `Decoder.from_stim_dem(...)`.
- `solver` and extra keyword arguments are forwarded to the underlying ILP
  solve.

## Result Fields

`dem_distance(...)` returns a `DEMDistanceResult` dataclass with:

- `distance`: minimum number of DEM mechanisms in the returned logical fault.
- `fault_vector`: binary NumPy vector over DEM mechanisms, with `1` entries for
  selected mechanisms.
- `fault_indices`: Python list of the selected mechanism indices.
- `observable_mask`: binary NumPy vector giving the logical observables flipped
  by `fault_vector`.

In targeted mode, `observable_mask` matches the requested
`target_observables`. In default mode, it is guaranteed to be nonzero.
