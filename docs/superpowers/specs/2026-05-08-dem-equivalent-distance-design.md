# DEM Equivalent Distance Design

## Summary

Add an exact analysis API that computes the equivalent distance of a Stim
`DetectorErrorModel` (DEM). The distance is defined as the minimum number of
DEM error mechanisms whose combined detector syndrome is zero while their
combined logical-observable flip is nonzero, or matches a caller-specified
target observable mask.

This is an analysis feature, not a decoder feature. It should reuse the
existing exact ILP helpers that already support CSS code distance and fixed
coset minimization.

## Scope

The first implementation supports binary Stim DEM analysis only:

- Input is a `stim.DetectorErrorModel` or its string representation.
- The objective is unweighted Hamming weight over DEM columns.
- The feature ignores error probabilities when optimizing.
- The feature supports either:
  - the minimum weight over any nonzero logical-observable mask, or
  - the minimum weight for one exact target logical-observable mask.

Out of scope:

- Probability-weighted objectives
- Approximate or heuristic distance bounds
- Integration into `Decoder.decode(...)`
- A new long-lived DEM analysis object such as `DEMCode`
- General non-Stim code descriptions outside the current DEM parser

## User-Facing API

Expose a new standalone function from `ilpqec`:

```python
from ilpqec import dem_distance

result = dem_distance(dem)
print(result.distance)
print(result.fault_indices)
print(result.observable_mask)

target = [1, 0, 1]
targeted = dem_distance(dem, target_observables=target)
print(targeted.distance)
```

Function signature:

```python
dem_distance(
    dem,
    *,
    target_observables=None,
    merge_parallel_edges=True,
    flatten_dem=True,
    solver=None,
    **solver_options,
) -> DEMDistanceResult
```

Return type:

```python
DEMDistanceResult(
    distance: int,
    fault_vector: np.ndarray,
    fault_indices: tuple[int, ...],
    observable_mask: np.ndarray,
)
```

Field semantics:

- `distance`: minimum number of selected DEM error mechanisms
- `fault_vector`: binary vector over DEM columns; `1` means that mechanism is
  present in the minimum-weight undetected logical fault
- `fault_indices`: indices of nonzero entries in `fault_vector`
- `observable_mask`: logical-observable mask produced by `fault_vector`

## Exact Semantics

Let `H` be the detector parity-check matrix parsed from the DEM, and let `O`
be the observable matrix. Each DEM column is one binary fault mechanism.

Default mode:

- Solve for the smallest binary vector `x` such that:
  - `H x = 0 mod 2`
  - `O x != 0 mod 2`
- Return the minimum Hamming weight `|x|`.

Targeted mode:

- Given a binary target mask `t`, solve for the smallest binary vector `x`
  such that:
  - `H x = 0 mod 2`
  - `O x = t mod 2`
- Return the minimum Hamming weight `|x|`.

This makes the API suitable for both:

- single-logical surface-code DEMs, where the default result should match the
  targeted result for `[1]`
- multi-logical DEMs, where callers may either ask for the global minimum over
  all nonzero masks or force one specific mask

## Architecture

Add one focused module:

- `src/ilpqec/dem_distance.py`

This module owns:

- input normalization for target observable masks
- DEM parsing via the existing parser path
- routing to the correct exact ILP helper
- result packaging into `DEMDistanceResult`

No new solver backend should be introduced. The implementation should reuse the
existing direct HiGHS exact model construction in `distance_ilp.py`.

## Data Flow

The internal flow is:

1. Parse the DEM using the existing parser logic already used by
   `Decoder.from_stim_dem(...)`.
2. Extract:
   - `H`: detector matrix
   - `O`: observable matrix
   - `weights`: parsed but ignored for this feature
3. Choose one of two exact ILP reductions:
   - default mode uses `minimize_nonzero_logical_operator(H, O, ...)`
   - targeted mode uses `minimize_weight_with_fixed_syndrome(...)` on the
     augmented matrix `[[H], [O]]`
4. Convert the optimizer output vector into:
   - `distance`
   - `fault_vector`
   - `fault_indices`
   - `observable_mask = (O @ fault_vector) % 2`

The public function should not expose the internal matrices directly. It should
stay at the DEM-analysis level.

## Solver Behavior

The solver contract should match the existing exact-distance APIs:

- Default solver is the current exact backend choice already used by
  `distance_ilp.py`, which today means direct HiGHS.
- Positive relative gaps are rejected because the API promises an exact
  distance.
- If the solver stops without proving optimality, for example because of a time
  limit, the function raises `RuntimeError`.
- Existing exact-solver options such as `time_limit`, `threads`, and `verbose`
  remain available through `**solver_options`.

The feature intentionally ignores DEM probabilities in the objective, even
though the parser computes decoder weights.

## Validation and Errors

Validation rules:

- Reject DEMs with zero logical observables because there is no logical fault
  notion to optimize.
- Reject `target_observables` whose length does not match the number of DEM
  observables.
- Reject non-binary `target_observables`.
- Reject the all-zero target observable mask because it would ask for an
  undetected non-logical fault instead of a logical failure.
- Reuse the current DEM parser behavior for unsupported instructions and
  flattening rules.

Error semantics:

- `ValueError` for invalid input shape or invalid target masks
- `ImportError` if `stim` is required but unavailable
- `RuntimeError` if the exact solver cannot prove optimality

## Module Boundaries

Keep responsibilities separated:

- `Decoder` remains a syndrome-decoding API and should not gain a distance
  method for this feature.
- `distance_ilp.py` remains the only owner of exact binary-vector ILP model
  building.
- `dem_distance.py` should be thin glue code plus validation.

This keeps the new feature aligned with the current split between:

- `CSSCode.distance()` for exact code analysis
- `Decoder.decode(...)` for inference on observed syndromes

## Testing Strategy

Add three layers of tests.

### 1. Small DEM unit tests

Create tiny hand-written DEM examples that verify:

- default mode finds the minimum nonzero observable fault
- targeted mode hits the requested observable mask exactly
- returned `fault_vector` satisfies zero detector syndrome
- returned `observable_mask` matches the parsed observable action

Also cover:

- DEMs with no observables
- target mask length mismatch
- non-binary target mask
- zero target mask

### 2. Brute-force cross-checks

For small parsed DEMs with a manageable number of columns, enumerate all fault
subsets and compare the exact optimum weight against `dem_distance(...)`.

This validates the correctness of the ILP reduction independently of any
surface-code-specific expectation.

### 3. Stim surface-code integration tests

Use Stim-generated rotated surface-code memory circuits with
`decompose_errors=True`.

Required coverage:

- `surface_code:rotated_memory_x`
- `distance=3`
- `rounds=3`
- result distance equals `3`
- default mode matches targeted mode for `[1]`

Optional slower coverage can add `distance=5` if test runtime remains
acceptable.

## Benchmark Plan

Benchmarking should answer a practical question:

> On the current machine and exact backend, how large a Stim rotated surface
> code circuit can this equivalent-distance computation solve while proving
> optimality, and how long does it take?

The benchmark should sweep surface-code distances such as `3, 5, 7, 9, ...`
and report for each instance:

- circuit distance
- rounds
- number of detectors
- number of DEM mechanisms
- solved equivalent distance
- wall-clock solve time
- whether optimality was proven within the requested time limit

Documentation should present the results as rough local measurements, not a
guaranteed universal limit, because machine speed and solver versions matter.

## Documentation Changes

Document the new API in the user-facing docs alongside the existing CSS
distance section and Stim DEM support notes.

At minimum:

- add a short example to `README.md`
- add a dedicated docs page or section describing DEM equivalent distance
- mention that the objective ignores probabilities and counts mechanisms only
- mention that exact solving currently follows the same backend restrictions as
  CSS distance

## Implementation Notes

The implementation should prefer extracting or reusing existing helpers rather
than duplicating DEM parsing logic. If reaching into `Decoder()._parse_dem(...)`
is too awkward for a public analysis API, a small internal helper extraction is
acceptable, but only if it stays tightly scoped and does not broaden the
refactor.

The design goal is a minimal feature addition with exact semantics and strong
testability, not a new abstraction hierarchy.
