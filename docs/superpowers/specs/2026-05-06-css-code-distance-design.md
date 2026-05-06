# CSS Code Distance and Logical Operators Design

## Summary

Add code-level support for binary CSS codes whose input is a pair of parity-check
matrices `(Hx, Hz)`. The feature computes:

- The exact X distance `dx`, Z distance `dz`, and code distance `d = min(dx, dz)`.
- One minimum-weight X logical operator and one minimum-weight Z logical operator.
- A separate logical-basis API that returns paired X/Z logical generators, optionally
  reducing every fixed logical coset to its strict minimum-weight representative.

This is a code analysis feature, not a syndrome-decoding feature. It should live beside
the existing `Decoder` API and reuse ILPQEC's solver configuration style.

## Scope

The first implementation supports only binary CSS stabilizer codes:

- `Hx` and `Hz` are two-dimensional binary matrices.
- Both matrices have the same number of columns `n`.
- The default validation requires `Hx @ Hz.T == 0 mod 2`.
- The number of logical qubits is `k = n - rank(Hx) - rank(Hz)`.

General non-CSS stabilizer codes, qudit codes, subsystem codes, and approximate or
randomized distance bounds are out of scope for this feature.

## Public API

Expose a new `CSSCode` class from `ilpqec`:

```python
from ilpqec import CSSCode

code = CSSCode.from_parity_check_matrices(Hx, Hz)

distance = code.distance()
print(distance.d, distance.dx, distance.dz)
print(distance.shortest_x, distance.shortest_z)

basis = code.logical_basis(reduce=True)
print(basis.x, basis.z)
```

The constructor accepts dense Python lists, NumPy arrays, and SciPy sparse matrices when
SciPy is installed. Inputs are normalized to `np.uint8` matrices over GF(2).

Public method signatures:

```python
CSSCode.from_parity_check_matrices(
    Hx,
    Hz,
    *,
    validate_commutation: bool = True,
) -> CSSCode

CSSCode.distance(
    *,
    solver: str | None = None,
    **solver_options,
) -> CSSDistanceResult

CSSCode.logical_basis(
    *,
    reduce: bool = False,
    solver: str | None = None,
    **solver_options,
) -> CSSLogicalBasis
```

Expose read-only code properties:

```python
code.hx
code.hz
code.n
code.k
code.rank_x
code.rank_z
```

Use small dataclasses for structured returns:

```python
CSSDistanceResult(
    d: int,
    dx: int,
    dz: int,
    shortest_x: np.ndarray,
    shortest_z: np.ndarray,
)

CSSLogicalBasis(
    x: np.ndarray,  # shape (k, n)
    z: np.ndarray,  # shape (k, n)
)
```

If `k == 0`, `distance()` and `logical_basis()` raise `ValueError` because the code has
no nontrivial logical operators.

## Module Layout

Add three focused modules:

- `src/ilpqec/css_code.py`: user-facing `CSSCode`, result dataclasses, validation,
  caching, and orchestration.
- `src/ilpqec/gf2.py`: pure GF(2) linear algebra helpers such as row reduction, rank,
  nullspace, rowspace extension, and paired quotient-basis construction.
- `src/ilpqec/distance_ilp.py`: exact minimum-weight binary-vector ILP routines used by
  distance and logical-basis reduction.

Keep `Decoder` unchanged except for any shared helper extraction that is necessary to
avoid duplicating solver backend code.

## Reference Behavior

The design follows the same mathematical split as the reference packages in `refs/`:

- TensorQEC computes null spaces and logical operators via GF(2) Gaussian elimination,
  then uses integer programming to minimize logical operators.
- qLDPC separates code-level logical operators and distance from decoding, and reduces
  logical operators by solving parity constraints augmented with dual-logical constraints.

ILPQEC should use these references for behavior and test expectations, but the new code
must fit ILPQEC's Python API and solver configuration patterns.

## GF(2) Logical Basis

For a CSS code:

```text
X logical space = ker(Hz) / rowspace(Hx)
Z logical space = ker(Hx) / rowspace(Hz)
```

The Gaussian-elimination layer produces paired bases `Lx` and `Lz` satisfying:

```text
Hz @ Lx.T == 0 mod 2
Hx @ Lz.T == 0 mod 2
Lx @ Lz.T == I_k mod 2
```

The basis does not need to be minimum weight when `reduce=False`. It only needs to be
stable, valid, and paired. `logical_basis(reduce=True)` is responsible for strict
minimum-weight reduction of each fixed coset.

## Distance ILP

The distance calculation returns only the global shortest X and Z logicals. It does not
compute or reduce the full logical basis unless the caller asks for it separately.

For X distance, solve:

```text
minimize    sum_j x_j
subject to  Hz x = 0 mod 2
            Lz_i x = r_i mod 2 for each i
            sum_i r_i >= 1
            x_j in {0, 1}
            r_i in {0, 1}
```

Here `Lz` is the paired Z logical basis. The selector vector `r` forces the candidate
to have nonzero pairing with the Z logical space, which excludes X stabilizers and the
zero vector while allowing any nontrivial X logical coset.

For Z distance, swap X and Z:

```text
minimize    sum_j z_j
subject to  Hx z = 0 mod 2
            Lx_i z = r_i mod 2 for each i
            sum_i r_i >= 1
            z_j in {0, 1}
            r_i in {0, 1}
```

The objective is unweighted Hamming weight. The result stores one optimizer-selected
minimum operator for each direction.

## Logical Basis Reduction ILP

`logical_basis(reduce=True)` first obtains the canonical paired basis from GF(2)
elimination. It then reduces each fixed logical coset exactly.

For the `i`th X logical, solve:

```text
minimize    sum_j x_j
subject to  Hz x = 0 mod 2
            Lz_i x = 1 mod 2
            Lz_j x = 0 mod 2 for j != i
            x_j in {0, 1}
```

For the `i`th Z logical, solve:

```text
minimize    sum_j z_j
subject to  Hx z = 0 mod 2
            Lx_i z = 1 mod 2
            Lx_j z = 0 mod 2 for j != i
            z_j in {0, 1}
```

This makes every returned generator a strict minimum-weight representative of its
selected canonical coset. It does not search over all possible basis choices.

## Solver Integration

The default solver remains direct HiGHS, matching the current `Decoder` behavior.
Callers may pass the same solver options already used by `Decoder`, including
`solver`, `direct`, `time_limit`, `gap`, `threads`, `verbose`, and backend-specific
options.

The first implementation keeps distance-specific model building in `distance_ilp.py`.
It shares `SolverConfig` and solver availability helpers with `Decoder`, but does not
refactor the existing decoder backend unless implementation reveals a small, obvious
helper extraction.

Because these APIs promise exact distances and strict minimum-weight representatives,
positive relative MIP gaps are rejected. `gap=None` and `gap=0` are accepted. If a
solver stops before proving optimality, for example due to `time_limit`, the API raises
`RuntimeError` instead of returning a best-known feasible vector as an exact result.

## Validation and Errors

Validation rules:

- Reject non-two-dimensional inputs.
- Reject matrices with different numbers of columns.
- Reject non-binary inputs instead of silently reducing arbitrary integers modulo 2.
- Reject non-commuting CSS checks when `validate_commutation=True`.
- Reject `k < 0`, which indicates inconsistent rank assumptions or invalid input.
- Raise `ValueError` for `k == 0` in APIs that require nontrivial logical operators.

`validate_commutation=False` skips only the commutation check. It does not skip shape,
binary, or rank validation.

## Tests

Add focused test coverage:

- GF(2) row reduction, rank, nullspace, and paired logical basis identities.
- Distance on known small CSS codes, including a Steane-style `[[7,1,3]]` case.
- Distance returns operators that commute with the opposite stabilizers and pair
  nontrivially with the opposite logical space.
- `logical_basis(reduce=False)` returns paired valid bases.
- `logical_basis(reduce=True)` returns paired bases and each row is a strict
  minimum-weight representative of the fixed coset on small brute-force-checkable codes.
- Validation failures for shape mismatch, non-binary entries, non-commuting checks, and
  `k == 0`.

Tests that require a solver should follow the existing solver availability skip pattern.
Pure GF(2) and validation tests should not require a solver backend.

## Documentation

Add a docs page `docs/css_code.md` covering:

- Binary CSS input convention: `Hx` is X-type checks and `Hz` is Z-type checks.
- `distance()` example and result fields.
- `logical_basis(reduce=False)` versus `logical_basis(reduce=True)`.
- Exactness and scaling caveats for ILP-based distance calculations.

Update `README.md`, `docs/index.md`, and `mkdocs.yml` to link the new page.

## Acceptance Criteria

- `from ilpqec import CSSCode` works.
- Users can compute `d`, `dx`, `dz`, `shortest_x`, and `shortest_z` from binary `Hx` and
  `Hz`.
- Users can separately compute paired logical bases, with optional strict per-coset
  minimum-weight reduction.
- The existing `Decoder` tests continue to pass.
- New tests cover validation, GF(2) helpers, distance, and logical basis reduction.
- Documentation explains the input convention and exactness limitations.
