# DEM Equivalent Distance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact `dem_distance(...)` API that computes the minimum number of Stim DEM mechanisms needed to cause an undetected logical fault, plus a benchmark path for rotated surface-code circuits.

**Architecture:** Keep DEM analysis separate from syndrome decoding by adding one focused `dem_distance.py` module that reuses the existing DEM parser and exact ILP helpers in `distance_ilp.py`. Validate target observable masks in the analysis layer, expose a structured result dataclass, document the new API in the existing Stim/CSS docs, and add a dedicated benchmark script instead of overloading the per-shot decoder benchmark.

**Tech Stack:** Python 3.9+, NumPy, optional Stim, direct HiGHS via `highspy`, pytest, mkdocs-material.

---

## File Structure

- Create `src/ilpqec/dem_distance.py`: `DEMDistanceResult`, `dem_distance(...)`, target-mask validation, DEM parsing orchestration, and result packaging.
- Modify `src/ilpqec/__init__.py`: export `dem_distance` and `DEMDistanceResult`.
- Create `tests/test_dem_distance.py`: exact solver-backed behavior tests, brute-force cross-checks, and a Stim rotated-surface-code regression.
- Create `tests/test_dem_distance_coverage.py`: validation, parser-flag forwarding, result-copy behavior, and helper-path coverage with monkeypatching.
- Create `docs/dem_distance.md`: dedicated user docs for DEM equivalent distance.
- Modify `README.md`: add a short public example in the quickstart and mention the new docs page.
- Modify `docs/index.md`: add a matching quickstart example and documentation-map entry.
- Modify `docs/stim_dem.md`: document the analysis API next to the existing DEM parsing notes.
- Modify `mkdocs.yml`: add the DEM equivalent distance page to nav.
- Create `benchmark/benchmark_dem_distance.py`: exact equivalent-distance benchmark script for Stim circuits.
- Modify `docs/benchmarks.md`: document how to run the new benchmark and how to interpret its output.

---

### Task 1: Core DEM Distance API

**Files:**
- Create: `src/ilpqec/dem_distance.py`
- Modify: `src/ilpqec/__init__.py`
- Create: `tests/test_dem_distance.py`

- [ ] **Step 1: Write the failing solver-backed behavior tests**

Create `tests/test_dem_distance.py` with this initial content:

```python
"""Tests for exact DEM equivalent-distance analysis."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from ilpqec import DEMDistanceResult, dem_distance
from ilpqec.decoder import Decoder
from ilpqec.solver import get_available_solvers


pytestmark = pytest.mark.skipif(
    "highs" not in get_available_solvers(),
    reason="DEM distance tests require the HiGHS backend",
)


class FakeDem:
    """Minimal DEM-like object for exact-distance tests."""

    def __init__(self, text: str, *, num_detectors: int, num_observables: int):
        self._text = text
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def __str__(self) -> str:
        return self._text

    def flattened(self) -> "FakeDem":
        return self


def brute_force_dem_distance(dem, target=None):
    """Brute-force the minimum undetected logical fault for a small DEM."""
    H, obs_matrix, _ = Decoder()._parse_dem(dem, merge_parallel=True, flatten_dem=True)
    num_faults = H.shape[1]
    target = None if target is None else np.asarray(target, dtype=np.uint8)

    for weight in range(num_faults + 1):
        for bits in product([0, 1], repeat=num_faults):
            vector = np.array(bits, dtype=np.uint8)
            if int(vector.sum()) != weight:
                continue
            if H.size and np.any((H @ vector) % 2):
                continue

            mask = (obs_matrix @ vector) % 2
            if target is None:
                if not np.any(mask):
                    continue
            elif not np.array_equal(mask, target):
                continue

            return vector, mask

    raise AssertionError("No feasible logical fault found")


def test_dem_distance_default_mode_matches_bruteforce():
    dem = FakeDem(
        "error(0.1) D0 L0\n"
        "error(0.1) D0\n"
        "error(0.1) L1\n",
        num_detectors=1,
        num_observables=2,
    )

    result = dem_distance(dem, solver="highs")
    expected_vector, expected_mask = brute_force_dem_distance(dem)

    assert isinstance(result, DEMDistanceResult)
    assert result.distance == int(expected_vector.sum()) == 1
    np.testing.assert_array_equal(result.fault_vector, expected_vector)
    assert result.fault_indices == (2,)
    np.testing.assert_array_equal(result.observable_mask, expected_mask)


def test_dem_distance_target_mask_matches_bruteforce():
    dem = FakeDem(
        "error(0.1) D0 L0\n"
        "error(0.1) D0\n"
        "error(0.1) L1\n",
        num_detectors=1,
        num_observables=2,
    )

    target = np.array([1, 0], dtype=np.uint8)
    result = dem_distance(dem, target_observables=target, solver="highs")
    expected_vector, expected_mask = brute_force_dem_distance(dem, target=target)

    assert result.distance == int(expected_vector.sum()) == 2
    np.testing.assert_array_equal(result.fault_vector, expected_vector)
    assert result.fault_indices == (0, 1)
    np.testing.assert_array_equal(result.observable_mask, expected_mask)


def test_rotated_surface_code_distance_three_matches_targeted_result():
    stim = pytest.importorskip("stim")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model(decompose_errors=True)

    default_result = dem_distance(dem, solver="highs")
    targeted_result = dem_distance(dem, target_observables=[1], solver="highs")

    assert default_result.distance == 3
    assert targeted_result.distance == 3
    np.testing.assert_array_equal(default_result.observable_mask, np.array([1], dtype=np.uint8))
    np.testing.assert_array_equal(targeted_result.observable_mask, np.array([1], dtype=np.uint8))
```

- [ ] **Step 2: Run the tests to verify the new API is missing**

Run:

```bash
uv run pytest tests/test_dem_distance.py -v
```

Expected: FAIL during collection with an import error because `ilpqec` does not yet export `dem_distance` or `DEMDistanceResult`.

- [ ] **Step 3: Implement the minimal public API and export it**

Create `src/ilpqec/dem_distance.py`:

```python
"""Exact equivalent-distance analysis for Stim detector error models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ilpqec.decoder import Decoder
from ilpqec.distance_ilp import (
    minimize_nonzero_logical_operator,
    minimize_weight_with_fixed_syndrome,
)


@dataclass(frozen=True)
class DEMDistanceResult:
    """Exact equivalent-distance result for a Stim DEM."""

    distance: int
    fault_vector: np.ndarray
    fault_indices: tuple[int, ...]
    observable_mask: np.ndarray


def dem_distance(
    dem,
    *,
    target_observables=None,
    merge_parallel_edges: bool = True,
    flatten_dem: bool = True,
    solver: Optional[str] = None,
    **solver_options: Any,
) -> DEMDistanceResult:
    """Return the minimum undetected logical fault size for a DEM."""
    H, obs_matrix, _ = Decoder()._parse_dem(dem, merge_parallel_edges, flatten_dem)

    if target_observables is None:
        result = minimize_nonzero_logical_operator(
            H,
            obs_matrix,
            solver=solver,
            **solver_options,
        )
    else:
        target = np.asarray(target_observables, dtype=np.uint8)
        matrix = np.vstack([H, obs_matrix])
        syndrome = np.concatenate([np.zeros(H.shape[0], dtype=np.uint8), target])
        result = minimize_weight_with_fixed_syndrome(
            matrix,
            syndrome,
            solver=solver,
            **solver_options,
        )

    observable_mask = (obs_matrix @ result.vector) % 2
    fault_indices = tuple(int(i) for i in np.flatnonzero(result.vector))
    return DEMDistanceResult(
        distance=result.weight,
        fault_vector=result.vector,
        fault_indices=fault_indices,
        observable_mask=observable_mask,
    )
```

Modify `src/ilpqec/__init__.py` to export the new symbols:

```python
from ilpqec.css_code import CSSCode, CSSDistanceResult, CSSLogicalBasis
from ilpqec.decoder import Decoder
from ilpqec.dem_distance import DEMDistanceResult, dem_distance
from ilpqec.solver import SolverConfig, get_available_solvers, get_default_solver

__version__ = "0.1.0"
__all__ = [
    "CSSCode",
    "CSSDistanceResult",
    "CSSLogicalBasis",
    "DEMDistanceResult",
    "Decoder",
    "dem_distance",
    "get_available_solvers",
    "get_default_solver",
    "SolverConfig",
]
```

- [ ] **Step 4: Run the behavior tests again**

Run:

```bash
uv run pytest tests/test_dem_distance.py -v
```

Expected: PASS for the two fake-DEM tests. The rotated surface-code test should PASS when `stim` is installed or SKIP when `stim` is unavailable.

- [ ] **Step 5: Commit the core API**

Run:

```bash
git add src/ilpqec/dem_distance.py src/ilpqec/__init__.py tests/test_dem_distance.py
git commit -m "feat: add DEM equivalent distance API"
```

Expected: a commit containing the new module, exports, and the core solver-backed regression tests.

---

### Task 2: Validation and Coverage Hardening

**Files:**
- Modify: `src/ilpqec/dem_distance.py`
- Create: `tests/test_dem_distance_coverage.py`

- [ ] **Step 1: Write the failing validation and helper-path tests**

Create `tests/test_dem_distance_coverage.py` with this content:

```python
"""Extra coverage tests for DEM equivalent-distance helpers and edge cases."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ilpqec.dem_distance as dem_distance_module
from ilpqec import dem_distance


class FakeDem:
    """Minimal DEM-like object for validation tests."""

    def __init__(self, text: str, *, num_detectors: int, num_observables: int):
        self._text = text
        self.num_detectors = num_detectors
        self.num_observables = num_observables

    def __str__(self) -> str:
        return self._text

    def flattened(self) -> "FakeDem":
        return self


def test_dem_distance_rejects_dem_without_observables():
    dem = FakeDem("error(0.1) D0\n", num_detectors=1, num_observables=0)

    with pytest.raises(ValueError, match="at least one logical observable"):
        dem_distance(dem)


def test_dem_distance_rejects_target_length_mismatch():
    dem = FakeDem("error(0.1) L0\n", num_detectors=0, num_observables=1)

    with pytest.raises(ValueError, match="same length as the number of observables"):
        dem_distance(dem, target_observables=[1, 0])


def test_dem_distance_rejects_non_binary_target():
    dem = FakeDem("error(0.1) L0 L1\n", num_detectors=0, num_observables=2)

    with pytest.raises(ValueError, match="must be binary"):
        dem_distance(dem, target_observables=[1, 2])


def test_dem_distance_rejects_zero_target():
    dem = FakeDem("error(0.1) L0\n", num_detectors=0, num_observables=1)

    with pytest.raises(ValueError, match="must be nonzero"):
        dem_distance(dem, target_observables=[0])


def test_dem_distance_targeted_mode_builds_augmented_matrix(monkeypatch):
    H = np.array([[1, 1, 0]], dtype=np.uint8)
    O = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    seen = {}

    def fake_parse(self, dem, merge_parallel, flatten_dem):
        seen["parse"] = (merge_parallel, flatten_dem)
        return H, O, np.zeros(H.shape[1], dtype=float)

    def fake_minimize(matrix, syndrome, *, solver=None, **solver_options):
        seen["matrix"] = matrix.copy()
        seen["syndrome"] = syndrome.copy()
        seen["solver"] = solver
        seen["solver_options"] = dict(solver_options)
        return SimpleNamespace(vector=np.array([1, 1, 0], dtype=np.uint8), weight=2)

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse, raising=False)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_weight_with_fixed_syndrome",
        fake_minimize,
    )

    result = dem_distance(
        object(),
        target_observables=[1, 0],
        merge_parallel_edges=False,
        flatten_dem=False,
        solver="highs",
        time_limit=7.0,
    )

    assert seen["parse"] == (False, False)
    np.testing.assert_array_equal(seen["matrix"], np.vstack([H, O]))
    np.testing.assert_array_equal(seen["syndrome"], np.array([0, 1, 0], dtype=np.uint8))
    assert seen["solver"] == "highs"
    assert seen["solver_options"] == {"time_limit": 7.0}
    assert result.distance == 2
    assert result.fault_indices == (0, 1)
    np.testing.assert_array_equal(result.observable_mask, np.array([1, 1], dtype=np.uint8))


def test_dem_distance_returns_copies_of_result_arrays(monkeypatch):
    H = np.zeros((0, 3), dtype=np.uint8)
    O = np.array([[1, 0, 1]], dtype=np.uint8)
    source_vector = np.array([1, 0, 0], dtype=np.uint8)

    def fake_parse(self, dem, merge_parallel, flatten_dem):
        return H, O, np.zeros(3, dtype=float)

    def fake_minimize(check_matrix, dual_logicals, *, solver=None, **solver_options):
        return SimpleNamespace(vector=source_vector, weight=1)

    monkeypatch.setattr(dem_distance_module.Decoder, "_parse_dem", fake_parse, raising=False)
    monkeypatch.setattr(
        dem_distance_module,
        "minimize_nonzero_logical_operator",
        fake_minimize,
    )

    result = dem_distance(object(), solver="highs")
    result.fault_vector[0] = 0
    result.observable_mask[0] = 0

    again = dem_distance(object(), solver="highs")
    np.testing.assert_array_equal(again.fault_vector, np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(again.observable_mask, np.array([1], dtype=np.uint8))
```

- [ ] **Step 2: Run the new coverage tests to see the missing validation behavior**

Run:

```bash
uv run pytest tests/test_dem_distance_coverage.py -v
```

Expected: FAIL on the validation-message assertions and the copy-behavior assertion, because the initial implementation does not yet normalize target masks or return defensive copies.

- [ ] **Step 3: Harden `dem_distance.py` with validation and defensive copies**

Replace `src/ilpqec/dem_distance.py` with:

```python
"""Exact equivalent-distance analysis for Stim detector error models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ilpqec.decoder import Decoder
from ilpqec.distance_ilp import (
    minimize_nonzero_logical_operator,
    minimize_weight_with_fixed_syndrome,
)


@dataclass(frozen=True)
class DEMDistanceResult:
    """Exact equivalent-distance result for a Stim DEM."""

    distance: int
    fault_vector: np.ndarray
    fault_indices: tuple[int, ...]
    observable_mask: np.ndarray


def _as_binary_target_observables(target_observables, num_observables: int) -> np.ndarray:
    target = np.asarray(target_observables)
    if target.ndim != 1:
        raise ValueError("target_observables must be one-dimensional")
    if target.shape[0] != num_observables:
        raise ValueError(
            "target_observables must have the same length as the number of observables"
        )
    if np.any((target != 0) & (target != 1)):
        raise ValueError("target_observables must be binary")

    target = np.asarray(target, dtype=np.uint8)
    if not np.any(target):
        raise ValueError("target_observables must be nonzero")
    return target.copy()


def dem_distance(
    dem,
    *,
    target_observables=None,
    merge_parallel_edges: bool = True,
    flatten_dem: bool = True,
    solver: Optional[str] = None,
    **solver_options: Any,
) -> DEMDistanceResult:
    """Return the minimum undetected logical fault size for a DEM."""
    H, obs_matrix, _ = Decoder()._parse_dem(dem, merge_parallel_edges, flatten_dem)
    if obs_matrix.shape[0] == 0:
        raise ValueError("DEM must contain at least one logical observable")

    if target_observables is None:
        result = minimize_nonzero_logical_operator(
            H,
            obs_matrix,
            solver=solver,
            **solver_options,
        )
    else:
        target = _as_binary_target_observables(
            target_observables,
            obs_matrix.shape[0],
        )
        matrix = np.vstack([H, obs_matrix])
        syndrome = np.concatenate([np.zeros(H.shape[0], dtype=np.uint8), target])
        result = minimize_weight_with_fixed_syndrome(
            matrix,
            syndrome,
            solver=solver,
            **solver_options,
        )

    fault_vector = result.vector.copy()
    observable_mask = np.asarray((obs_matrix @ fault_vector) % 2, dtype=np.uint8).copy()
    fault_indices = tuple(int(i) for i in np.flatnonzero(fault_vector))
    return DEMDistanceResult(
        distance=result.weight,
        fault_vector=fault_vector,
        fault_indices=fault_indices,
        observable_mask=observable_mask,
    )
```

- [ ] **Step 4: Run the full DEM-distance test set**

Run:

```bash
uv run pytest tests/test_dem_distance.py tests/test_dem_distance_coverage.py -v
```

Expected: PASS for all fake-DEM and helper-path tests, with the rotated surface-code regression still either PASS or SKIP depending on Stim availability.

- [ ] **Step 5: Commit the validation hardening**

Run:

```bash
git add src/ilpqec/dem_distance.py tests/test_dem_distance_coverage.py
git commit -m "test: harden DEM distance validation coverage"
```

Expected: a commit that adds explicit input validation, defensive copies, and monkeypatch coverage for the analysis helper paths.

---

### Task 3: Public Documentation

**Files:**
- Create: `docs/dem_distance.md`
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/stim_dem.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Add the dedicated docs page and link it from the existing docs**

Create `docs/dem_distance.md`:

```markdown
# DEM Equivalent Distance

ILPQEC can compute the exact equivalent distance of a Stim
`DetectorErrorModel` by treating each DEM mechanism as one binary fault
variable and solving for the smallest zero-syndrome fault set that flips a
logical observable.

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
print(result.distance)
print(result.fault_indices)
print(result.observable_mask)

targeted = dem_distance(dem, target_observables=[1], solver="highs")
print(targeted.distance)
```

## Semantics

- The objective ignores error probabilities and counts DEM mechanisms only.
- Default mode minimizes over any nonzero logical-observable mask.
- `target_observables=[...]` forces one exact logical mask.
- Exact solving currently follows the same direct HiGHS restriction as
  `CSSCode.distance()`.

## Result Fields

- `distance`: minimum number of selected DEM mechanisms
- `fault_vector`: binary indicator over DEM columns
- `fault_indices`: tuple of selected DEM column indices
- `observable_mask`: logical mask produced by `fault_vector`
```

In `README.md`, insert this section immediately after the existing “Stim DetectorErrorModel decoding” section:

```markdown
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
print(result.fault_indices)
print(targeted.observable_mask)
```

`dem_distance(...)` ignores probabilities and minimizes the number of DEM
mechanisms needed to produce an undetected logical fault.
```

Also add one bullet to the `README.md` “Documentation Map” list:

```markdown
- DEM equivalent distance analysis: `dem_distance.md`
```

In `docs/index.md`, insert this new quickstart section immediately after the existing “Stim DetectorErrorModel decoding” section:

```markdown
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
print(result.distance)
print(result.observable_mask)
```

This exact-analysis path ignores probabilities and counts DEM mechanisms only.
```

Also add one bullet to the `docs/index.md` “Documentation Map” list:

```markdown
- DEM equivalent distance analysis: `dem_distance.md`
```

In `docs/stim_dem.md`, append this section after “Recommendations”:

```markdown
## Equivalent Distance Analysis

If you want exact logical-fault analysis instead of decoding, use
`dem_distance(...)` on the parsed DEM:

```python
import stim
from ilpqec import dem_distance

dem = stim.Circuit.generated(
    "surface_code:rotated_memory_x",
    distance=3,
    rounds=3,
    after_clifford_depolarization=0.01,
).detector_error_model(decompose_errors=True)

result = dem_distance(dem, solver="highs")
print(result.distance)
```

This objective ignores probabilities and minimizes the number of DEM
mechanisms in an undetected logical fault.
```

In `mkdocs.yml`, add the new page to nav immediately after `CSS Code Analysis`:

```yaml
nav:
  - Home: index.md
  - CSS Code Analysis: css_code.md
  - DEM Equivalent Distance: dem_distance.md
  - Solver Backends: solvers.md
  - Mathematical Formulation: math.md
  - Stim DEM Support: stim_dem.md
  - Sinter Integration: sinter.md
  - Examples: examples.md
  - Benchmarks: benchmarks.md
```

- [ ] **Step 2: Build the docs site strictly**

Run:

```bash
uv run mkdocs build --strict
```

Expected: PASS with a successful site build and no missing-page errors.

- [ ] **Step 3: Commit the docs update**

Run:

```bash
git add docs/dem_distance.md README.md docs/index.md docs/stim_dem.md mkdocs.yml
git commit -m "docs: document DEM equivalent distance"
```

Expected: a commit that adds the new public docs page and threads the feature through the existing README/docs navigation.

---

### Task 4: Benchmark Script and Benchmark Docs

**Files:**
- Create: `benchmark/benchmark_dem_distance.py`
- Modify: `docs/benchmarks.md`

- [ ] **Step 1: Add a dedicated benchmark script for exact DEM distance**

Create `benchmark/benchmark_dem_distance.py`:

```python
"""
Benchmark exact DEM equivalent-distance solves on Stim circuits.

Requirements:
    pip install stim highspy
"""

from __future__ import annotations

import argparse
import time

from ilpqec import dem_distance
from ilpqec.decoder import Decoder


def parse_dem(dem):
    """Parse a Stim DEM without configuring a solver."""
    return Decoder()._parse_dem(dem, merge_parallel=True, flatten_dem=True)


def build_circuit(stim, code_task: str, noise_model: str, distance: int, rounds: int, noise: float):
    if noise_model == "code_capacity":
        return stim.Circuit.generated(
            code_task,
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=0.0,
            before_round_data_depolarization=noise,
            before_measure_flip_probability=0.0,
            after_reset_flip_probability=0.0,
        )
    return stim.Circuit.generated(
        code_task,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact DEM equivalent distance on Stim circuits."
    )
    parser.add_argument(
        "--code-task",
        type=str,
        default="surface_code:rotated_memory_x",
        help="Stim code task to generate.",
    )
    parser.add_argument(
        "--noise-model",
        choices=("circuit", "code_capacity"),
        default="circuit",
        help="Noise model: circuit (default) or code_capacity (data-only).",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default="3,5,7",
        help="Comma-separated circuit distances to benchmark.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of rounds. Defaults to the same value as each circuit distance.",
    )
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--solver", type=str, default="highs")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Optional exact-solver time limit in seconds.",
    )
    args = parser.parse_args()

    try:
        import stim
    except Exception as exc:
        raise SystemExit("stim is required (pip install stim)") from exc

    distances = [int(token.strip()) for token in args.distances.split(",") if token.strip()]

    print("Benchmarking DEM equivalent distance")
    print(
        "code_task={code_task} noise_model={noise_model} distances={distances} noise={noise}".format(
            code_task=args.code_task,
            noise_model=args.noise_model,
            distances=",".join(str(d) for d in distances),
            noise=args.noise,
        )
    )

    for distance in distances:
        rounds = args.rounds if args.rounds is not None else distance
        circuit = build_circuit(
            stim,
            args.code_task,
            args.noise_model,
            distance,
            rounds,
            args.noise,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        H, _, _ = parse_dem(dem)

        start = time.perf_counter()
        equivalent_distance = "-"
        status = "optimal"
        try:
            result = dem_distance(
                dem,
                solver=args.solver,
                time_limit=args.time_limit,
            )
            equivalent_distance = str(result.distance)
        except Exception as exc:
            status = f"failed ({exc})"
        elapsed = time.perf_counter() - start

        print(
            "distance={distance:<2d} rounds={rounds:<2d} detectors={detectors:<4d} "
            "mechanisms={mechanisms:<4d} equivalent_distance={equivalent_distance:<3s} "
            "time={elapsed:8.3f}s status={status}".format(
                distance=distance,
                rounds=rounds,
                detectors=dem.num_detectors,
                mechanisms=H.shape[1],
                equivalent_distance=equivalent_distance,
                elapsed=elapsed,
                status=status,
            )
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a surface-code sweep to estimate the largest solvable distance**

Run:

```bash
uv run python benchmark/benchmark_dem_distance.py --distances 3,5,7,9 --solver highs --time-limit 300
```

Expected: one output row per requested distance. In the final implementation notes, report the largest row whose `status=optimal` as the rough practical exact-solve limit on the current machine.

- [ ] **Step 3: Document how to run and read the new benchmark**

In `docs/benchmarks.md`, append this section after the existing benchmark sections:

```markdown
## DEM equivalent distance on rotated surface-code memory

Use the dedicated exact-analysis benchmark when you want to know how large a
Stim DEM instance can be solved to proven optimality:

```bash
benchmark/.venv/bin/python benchmark/benchmark_dem_distance.py \
  --distances 3,5,7,9 --solver highs --time-limit 300
```

The script prints one line per circuit distance with:

- the circuit distance and rounds
- the number of detectors
- the number of parsed DEM mechanisms
- the solved equivalent distance
- wall-clock solve time
- whether optimality was proven before the time limit

This is a local exact benchmark, so the largest solvable distance depends on
your machine and solver version.
```

- [ ] **Step 4: Rebuild the docs after the benchmark docs change**

Run:

```bash
uv run mkdocs build --strict
```

Expected: PASS with the new benchmark section included in the generated site.

- [ ] **Step 5: Commit the benchmark tooling**

Run:

```bash
git add benchmark/benchmark_dem_distance.py docs/benchmarks.md
git commit -m "bench: add DEM equivalent distance benchmark"
```

Expected: a commit that adds the dedicated benchmark script and documents how to use it to estimate the practical exact-solve limit.
