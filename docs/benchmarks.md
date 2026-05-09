# Benchmarks

This page summarizes the benchmark scripts and recent local results. Numbers
will vary by machine and solver versions.

## Benchmark Environment

Use a dedicated virtual environment under `benchmark/` to avoid polluting your
global Python:

```bash
python3 -m venv benchmark/.venv
source benchmark/.venv/bin/activate
python -m pip install --upgrade pip

# ILPQEC + optional solver backends
python -m pip install -e ".[pyomo,gurobi]"

# Benchmark dependencies
python -m pip install stim pymatching ldpc
```

Notes:
- Gurobi requires a valid license and Python < 3.13.
- If you do not need Gurobi, drop `gurobi` from the extras.

## Requirements

```bash
pip install stim pymatching ldpc tesseract-decoder
```

Non-HiGHS solvers require the Pyomo extra (`pip install ilpqec[pyomo]`),
except for the direct Gurobi backend (`pip install ilpqec[gurobi]`).

Notes:

- Direct backends: HiGHS, Gurobi.
- Pyomo backends: HiGHS, SCIP, CBC, GLPK, Gurobi, CPLEX.
- BPOSD runs with `max_iter=50`, `osd_order=0`, and `bp_method=minimum_sum`.
- Tesseract runs with `det_beam=50` by default (adjustable via `--tesseract-beam`).

## Circuit-level rotated surface code memory

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --compare-ilp-solvers --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000):

| Decoder | Time (ms/shot) | Logical Error Rate |
| --- | --- | --- |
| ILP[highs] (direct) | 2.7514 | 1.640% |
| ILP[gurobi] (direct) | 0.6403 | 1.650% |
| ILP[scip] (Pyomo) | 28.2160 | 1.670% |
| ILP[cbc] (Pyomo) | 14.9315 | 1.670% |
| ILP[glpk] (Pyomo) | 8.6292 | 1.670% |
| MWPM (pymatching) | 0.0035 | 2.150% |
| BPOSD (ldpc) | 0.0308 | 7.680% |
| Tesseract | 0.1602 | 1.640% |

## Code-capacity surface code (data errors only, perfect syndrome)

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --noise-model code_capacity --compare-ilp-solvers \
  --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 1 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000):

| Decoder | Time (ms/shot) | Logical Error Rate |
| --- | --- | --- |
| ILP[highs] (direct) | 3.2321 | 0.070% |
| ILP[gurobi] (direct) | 0.0838 | 0.070% |
| ILP[scip] (Pyomo) | 23.4834 | 0.070% |
| ILP[cbc] (Pyomo) | 10.4697 | 0.070% |
| ILP[glpk] (Pyomo) | 5.0085 | 0.070% |
| MWPM (pymatching) | 0.0036 | 0.070% |
| BPOSD (ldpc) | 0.0028 | 0.070% |
| Tesseract | 0.0093 | 0.070% |

## Color code (`color_code:memory_xyz`)

```bash
benchmark/.venv/bin/python benchmark/benchmark_decoders.py \
  --code-task color_code:memory_xyz --compare-ilp-solvers \
  --ilp-solvers highs,scip,gurobi,cbc,glpk \
  --shots 10000 --distance 3 --rounds 3 --noise 0.01
```

Results from a local macOS arm64 run (shots=10000):

| Decoder | Time (ms/shot) | Logical Error Rate |
| --- | --- | --- |
| ILP[highs] (direct) | 2.0226 | 4.450% |
| ILP[gurobi] (direct) | 0.3184 | 4.420% |
| ILP[scip] (Pyomo) | 24.9402 | 4.420% |
| ILP[cbc] (Pyomo) | 11.6961 | 4.450% |
| ILP[glpk] (Pyomo) | 6.0799 | 4.420% |
| MWPM (pymatching) | 0.0034 | 13.420% |
| BPOSD (ldpc) | 0.0114 | 9.830% |
| Tesseract | 0.0600 | 4.450% |

## DEM equivalent distance on rotated surface-code memory

Use the dedicated exact-analysis benchmark when you want to measure how large a
Stim DEM instance can be solved to proven optimality:

```bash
benchmark/.venv/bin/python benchmark/benchmark_dem_distance.py \
  --distances 3,5,7,9 --solver highs --time-limit 300
```

This benchmark currently targets only the exact direct-HiGHS path. Keep
`--solver highs`; other solver names are rejected at the CLI layer.

The script generates one circuit per requested distance, converts it to a
decomposed DEM, parses the DEM with `Decoder()._parse_dem(...)`, and then runs
`dem_distance(...)` on that exact model using the same
`merge_parallel_edges=True` and `flatten_dem=True` settings as the mechanism
count path.

Each output row reports:

- `distance`: the generated circuit distance
- `rounds`: the number of syndrome-extraction rounds used for that instance
- `detectors`: the DEM detector count
- `mechanisms`: the number of parsed DEM error mechanisms after flattening and
  parallel-edge merging
- `equivalent_distance`: the solved exact minimum logical-fault weight when the
  solve succeeds
- `time`: wall-clock runtime for the `dem_distance(...)` call
- `status`: `optimal` when HiGHS proves optimality, otherwise the solver error
  text for failures or time-limited runs

This is a local exact benchmark, so the largest solvable distance depends on
your machine, solver version, noise model, and time limit.
