"""
Benchmark exact DEM equivalent-distance solves on Stim circuits.

Requirements:
    pip install stim highspy
"""

from __future__ import annotations

import argparse
import time

from ilpqec import Decoder, dem_distance


def parse_dem(dem):
    """Parse a Stim DEM without requiring solver configuration."""
    return Decoder()._parse_dem(dem, merge_parallel=True, flatten_dem=True)


def parse_distances(raw: str) -> list[int]:
    """Parse a comma-separated distance list."""
    distances = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not distances:
        raise ValueError("at least one circuit distance is required")
    if any(distance <= 0 for distance in distances):
        raise ValueError("circuit distances must be positive integers")
    return distances


def build_circuit(stim, code_task: str, noise_model: str, distance: int, rounds: int, noise: float):
    """Build a generated Stim circuit for the selected code family and noise model."""
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


def benchmark_one(
    *,
    stim,
    code_task: str,
    noise_model: str,
    distance: int,
    rounds: int,
    noise: float,
    solver: str,
    time_limit: float | None,
) -> str:
    """Benchmark one circuit instance and return the formatted output row."""
    circuit = build_circuit(stim, code_task, noise_model, distance, rounds, noise)
    dem = circuit.detector_error_model(decompose_errors=True)
    h_matrix, _, _ = parse_dem(dem)

    equivalent_distance = "-"
    status = "optimal"
    start = time.perf_counter()
    try:
        result = dem_distance(
            dem,
            solver=solver,
            time_limit=time_limit,
        )
        equivalent_distance = str(result.distance)
    except Exception as exc:
        status = str(exc).strip().replace("\n", " ")
    elapsed = time.perf_counter() - start

    return (
        "distance={distance:<2d} rounds={rounds:<2d} detectors={detectors:<4d} "
        "mechanisms={mechanisms:<5d} equivalent_distance={equivalent_distance:<3s} "
        "time={elapsed:8.3f}s status={status}"
    ).format(
        distance=distance,
        rounds=rounds,
        detectors=dem.num_detectors,
        mechanisms=h_matrix.shape[1],
        equivalent_distance=equivalent_distance,
        elapsed=elapsed,
        status=status,
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
        help="Number of rounds. Defaults to each circuit distance.",
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

    try:
        distances = parse_distances(args.distances)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for distance in distances:
        rounds = args.rounds if args.rounds is not None else distance
        print(
            benchmark_one(
                stim=stim,
                code_task=args.code_task,
                noise_model=args.noise_model,
                distance=distance,
                rounds=rounds,
                noise=args.noise,
                solver=args.solver,
                time_limit=args.time_limit,
            )
        )


if __name__ == "__main__":
    main()
