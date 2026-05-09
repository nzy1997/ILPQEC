# Stim DetectorErrorModel Support

ILPQEC parses Stim DetectorErrorModels into an ILP by extracting the
`error(p)` mechanisms and building parity-check and observable matrices.

## Supported Instructions

- `error(p) ...` lines are parsed. Tags in `error[...]` are ignored.
- `detector` and `logical_observable` metadata lines are ignored.
- `shift_detectors` offsets are applied.
- `repeat` blocks are flattened by default (`flatten_dem=True`).
- `detector_separator` is unsupported and raises an error.

## The `^` Separator

Stim uses `^` to describe alternative components of a correlated mechanism.
ILPQEC currently treats `^` as whitespace, which discards the alternative
structure. If your DEM contains `^`, prefer `decompose_errors=True` when
building the DEM so each error is a single unambiguous mechanism.

## Flattening and Size

Flattening expands `repeat` blocks and applies `shift_detectors`. This is
convenient for parsing but can increase the DEM size significantly. To fail
fast instead, pass `flatten_dem=False` when creating the decoder.

## Recommendations

- Use `decompose_errors=True` when constructing DEMs from Stim circuits.
- Use `flatten_dem=True` unless the DEM is very large, and then pre-flatten
  only when needed.

## Equivalent Distance Analysis

ILPQEC also exposes `dem_distance(...)` for exact minimum-weight logical-fault
search directly on a DEM.

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

This objective ignores the `error(p)` probabilities stored in the DEM and
minimizes the number of selected DEM mechanisms instead.
