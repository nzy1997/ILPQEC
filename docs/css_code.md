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

## Exactness and errors

- Positive optimality gaps are rejected.
- If HiGHS does not prove optimality, ILPQEC raises instead of returning an
  approximate answer.
- If `k == 0`, `distance()` and `logical_basis()` raise `ValueError` because
  logical operators are undefined.
