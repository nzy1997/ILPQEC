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
- Exact DEM distance currently has the same backend restriction as the CSS
  exact-distance helpers: it requires the direct HiGHS backend.
- The optimization objective is Hamming weight over DEM mechanisms, not
  likelihood. `error(p)` probabilities are ignored by `dem_distance(...)`.
- `target_observables=None` searches for the minimum-weight fault whose logical
  effect is nonzero.
- `target_observables=[...]` switches to targeted mode and requires an exact
  binary observable mask. The mask must be one-dimensional, match the DEM
  observable count, and contain at least one `1`.
- `merge_parallel_edges` and `flatten_dem` are passed through to the DEM parser,
  matching `Decoder.from_stim_dem(...)`.
- The `solver` argument is therefore currently expected to be `"highs"`, and
  extra keyword arguments are forwarded to that direct HiGHS solve.
- If a targeted observable mask is infeasible, or if the exact solve exits
  without proving optimality, `dem_distance(...)` raises `RuntimeError` from
  the exact solve path.

## Result Fields

`dem_distance(...)` returns a `DEMDistanceResult` dataclass with:

- `distance`: minimum number of DEM mechanisms in the returned logical fault.
- `fault_vector`: binary NumPy vector over the parser-produced DEM columns, with
  `1` entries for selected mechanisms after any `flatten_dem` /
  `merge_parallel_edges` preprocessing.
- `fault_indices`: Python list of indices into that same parser-produced DEM
  column ordering, not stable references to original DEM source lines.
- `observable_mask`: binary NumPy vector giving the logical observables flipped
  by `fault_vector`.

In targeted mode, `observable_mask` matches the requested
`target_observables`. In default mode, it is guaranteed to be nonzero.
