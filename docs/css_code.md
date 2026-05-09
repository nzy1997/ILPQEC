# CSS Code Analysis

ILPQEC can analyze binary CSS stabilizer codes directly from parity-check
matrices `Hx` and `Hz`.

Use [`CSSCode`](/Users/nzy/pycode/ILPQEC/.worktrees/css-code-distance/src/ilpqec/css_code.py:49) when you want exact logical-operator analysis instead of syndrome decoding.

## Scope

- `CSSCode.from_parity_check_matrices(Hx, Hz)` builds a binary CSS code.
- `code.distance()` returns exact `d`, `dx`, `dz`, and one minimum-weight X/Z logical.
- `code.logical_basis(reduce=False)` returns a paired logical basis from GF(2) elimination.
- `code.logical_basis(reduce=True)` keeps the same canonical logical cosets and reduces each one to its exact minimum-weight representative with ILP.

The current exact-analysis path supports binary CSS codes only and uses the
direct HiGHS backend.

## Requirements

Install the core package plus any optional dependencies you need:

```bash
uv pip install ilpqec
uv pip install highspy
```

Optional extras:

- install `ilpqec[scipy]` if you want to pass SciPy sparse parity-check matrices
- install `ilpqec[stim]` only if you also use Stim-based decoding APIs elsewhere

The CSS analysis APIs require an exact direct HiGHS solve. They do not support
Pyomo-backed solvers or approximate MIP gaps.

## Build a CSS code

```python
import numpy as np
from ilpqec import CSSCode

hx = np.array([
    [1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=np.uint8)

code = CSSCode.from_parity_check_matrices(hx, hx)

print(code.n)       # 7
print(code.k)       # 1
print(code.rank_x)  # 3
print(code.rank_z)  # 3
```

By default, construction validates CSS commutation:

```python
code = CSSCode.from_parity_check_matrices(Hx, Hz, validate_commutation=True)
```

This checks `Hx @ Hz.T == 0 mod 2`.

Accepted inputs:

- dense NumPy arrays
- Python nested lists that convert to binary matrices
- SciPy sparse matrices when `scipy` is installed

Rejected inputs:

- non-binary entries
- non-2D matrices
- mismatched column counts
- non-commuting `Hx, Hz` pairs unless `validate_commutation=False`

## Exact distance

```python
distance = code.distance(solver="highs")

print(distance.d)
print(distance.dx, distance.dz)
print(distance.shortest_x)
print(distance.shortest_z)
```

`distance()` searches globally over all nontrivial X and Z logical cosets. It
returns:

- `d`: `min(dx, dz)`
- `dx`: exact X distance
- `dz`: exact Z distance
- `shortest_x`: one minimum-weight X logical operator
- `shortest_z`: one minimum-weight Z logical operator

## Logical bases

### Canonical paired basis

```python
basis = code.logical_basis(reduce=False)
print(basis.x)
print(basis.z)
```

This basis is constructed by GF(2) Gaussian elimination and paired so that
`basis.x @ basis.z.T == I mod 2`.

### Per-coset exact reduction

```python
reduced = code.logical_basis(reduce=True, solver="highs")
print(reduced.x)
print(reduced.z)
```

This does **not** search over all equivalent logical bases. Instead, it fixes
the canonical logical cosets chosen by Gaussian elimination and solves one exact
minimum-weight problem for each generator.

That distinction matters:

- `distance()` gives the globally shortest nontrivial X/Z logicals.
- `logical_basis(reduce=True)` gives a paired basis where each returned row is
  the strict minimum-weight representative of its fixed canonical coset.

## Choosing the API

Use `distance()` when you want code parameters:

- exact `dx`, `dz`, and `d`
- one shortest X logical
- one shortest Z logical

Use `logical_basis(reduce=False)` when you want a fast canonical paired basis
from GF(2) elimination.

Use `logical_basis(reduce=True)` when you want a paired basis but also want each
canonical logical generator reduced exactly inside its own fixed coset.

`logical_basis(reduce=True)` is not a substitute for `distance()`. A reduced
basis generator can be heavier than the globally shortest logical operator,
because it is constrained to remain in one selected canonical coset.

## Exactness and errors

- Positive optimality gaps are rejected.
- If HiGHS does not prove optimality, ILPQEC raises instead of returning an
  approximate answer.
- If `k == 0`, `distance()` and `logical_basis()` raise `ValueError` because
  logical operators are undefined.

## Common errors

`ImportError: Direct HiGHS backend requires highspy`

- install `highspy`
- pass `solver="highs"` or rely on the default exact backend

`ImportError: Sparse CSS parity-check matrices require SciPy`

- install `ilpqec[scipy]`
- or convert sparse matrices to dense arrays before calling `CSSCode`

`ValueError: CSS parity checks must commute`

- ensure `Hx @ Hz.T == 0 mod 2`
- only disable commutation validation if you explicitly want to inspect an
  invalid pair

`ValueError: CSS code has no logical qubits`

- this means `k == 0`
- distance and logical-operator APIs are undefined for such a code

`ValueError: Exact distance APIs require exact optimization`

- do not pass a positive `gap`
- these APIs intentionally refuse approximate MIP solves
