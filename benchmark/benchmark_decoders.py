"""
Benchmark ILPDecoder vs MWPM (pymatching) and BPOSD (ldpc) on circuit-level decoding.

Requirements:
    pip install stim pymatching ldpc
    Plus an ILP solver (HiGHS recommended: pip install highspy)
"""

import argparse
import inspect
import time

import numpy as np

from ilpdecoder import Decoder, get_available_solvers


def parse_dem(dem):
    """Parse a Stim DEM without requiring a solver."""
    decoder = Decoder()
    return decoder._parse_dem(dem, merge_parallel=True, flatten_dem=True)


def build_bposd_decoder(H, error_probs):
    """Construct a BPOSD decoder with best-effort signature matching."""
    try:
        from ldpc import BpOsdDecoder
    except Exception as exc:
        raise ImportError("ldpc with BpOsdDecoder is required (pip install ldpc)") from exc

    mean_p = float(np.mean(error_probs))
    default_opts = {"max_iter": 50, "osd_order": 0, "bp_method": "minimum_sum"}

    try:
        sig = inspect.signature(BpOsdDecoder)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}

    if params:
        kwargs = {}
        if "pcm" in params:
            kwargs["pcm"] = H
        elif "h" in params:
            kwargs["h"] = H
        elif "H" in params:
            kwargs["H"] = H
        elif "parity_check_matrix" in params:
            kwargs["parity_check_matrix"] = H
        else:
            kwargs[next(iter(params))] = H

        if "error_channel" in params:
            kwargs["error_channel"] = error_probs
        elif "channel_probs" in params:
            kwargs["channel_probs"] = error_probs
        elif "error_rate" in params:
            kwargs["error_rate"] = error_probs
        elif "p" in params:
            kwargs["p"] = error_probs

        for key, val in default_opts.items():
            if key in params:
                kwargs[key] = val

        try:
            return BpOsdDecoder(**kwargs)
        except TypeError:
            # Fall back to a scalar error rate if a vector is not accepted.
            for key in ("error_rate", "channel_probs", "error_channel", "p"):
                if key in kwargs and isinstance(kwargs[key], np.ndarray):
                    kwargs[key] = mean_p
            return BpOsdDecoder(**kwargs)

    for kwargs in (
        {"pcm": H, "error_channel": error_probs, **default_opts},
        {"pcm": H, "error_rate": mean_p, **default_opts},
        {"parity_check_matrix": H, "channel_probs": error_probs, **default_opts},
        {"parity_check_matrix": H, "error_rate": mean_p, **default_opts},
        {"H": H, "channel_probs": error_probs, **default_opts},
        {"H": H, "error_rate": mean_p, **default_opts},
        {"h": H, "error_rate": mean_p, **default_opts},
        {"pcm": H},
    ):
        try:
            return BpOsdDecoder(**kwargs)
        except TypeError:
            continue

    return BpOsdDecoder(H)


def build_pymatching(dem):
    """Construct a pymatching decoder from a Stim DEM."""
    try:
        import pymatching
    except Exception as exc:
        raise ImportError("pymatching is required (pip install pymatching)") from exc
    return pymatching.Matching.from_detector_error_model(dem)


def predict_observables(pred, obs_matrix):
    """Convert a decoder output into observable predictions."""
    pred = np.asarray(pred, dtype=np.uint8)
    if pred.ndim == 0:
        pred = pred.reshape(1)
    if pred.shape[0] == obs_matrix.shape[0]:
        return pred
    return (obs_matrix @ pred) % 2


def benchmark(name, detections, observables, decode_fn):
    """Run a decoder and report time per shot and logical error rate."""
    start = time.perf_counter()
    correct = 0
    for i in range(detections.shape[0]):
        predicted = decode_fn(detections[i])
        if observables.size and np.array_equal(predicted, observables[i]):
            correct += 1
    elapsed = time.perf_counter() - start
    shots = detections.shape[0]
    ler = None
    if observables.size:
        ler = 1.0 - (correct / shots)
    ms_per_shot = (elapsed / shots) * 1000.0
    print(f"{name:12s}  {ms_per_shot:9.4f} ms/shot", end="")
    if ler is not None:
        print(f"  logical error rate: {ler:.3%}")
    else:
        print("  logical error rate: n/a")


def main():
    parser = argparse.ArgumentParser(description="Benchmark decoders on a surface code circuit.")
    parser.add_argument(
        "--code-task",
        type=str,
        default="surface_code:rotated_memory_x",
        help="Stim code task (e.g., surface_code:rotated_memory_x, color_code:memory_xyz)",
    )
    parser.add_argument(
        "--noise-model",
        choices=("circuit", "code_capacity"),
        default="circuit",
        help="Noise model: circuit (default) or code_capacity (data-only).",
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument(
        "--compare-ilp-solvers",
        action="store_true",
        help="Benchmark all available ILP solvers (excluding CPLEX).",
    )
    parser.add_argument(
        "--ilp-solvers",
        type=str,
        default="auto",
        help="Comma-separated ILP solvers to compare (auto=available).",
    )
    args = parser.parse_args()

    try:
        import stim
    except Exception as exc:
        raise SystemExit("stim is required (pip install stim)") from exc

    if args.noise_model == "code_capacity":
        circuit = stim.Circuit.generated(
            args.code_task,
            distance=args.distance,
            rounds=args.rounds,
            after_clifford_depolarization=0.0,
            before_round_data_depolarization=args.noise,
            before_measure_flip_probability=0.0,
            after_reset_flip_probability=0.0,
        )
    else:
        circuit = stim.Circuit.generated(
            args.code_task,
            distance=args.distance,
            rounds=args.rounds,
            after_clifford_depolarization=args.noise,
        )
    dem = circuit.detector_error_model(decompose_errors=True)
    try:
        H, obs_matrix, weights = parse_dem(dem)
    except Exception as exc:
        raise SystemExit(f"Failed to parse DEM: {exc}") from exc

    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(
        shots=args.shots, separate_observables=True
    )
    detections = detections.astype(np.uint8)
    observables = observables.astype(np.uint8)

    print("Benchmarking decoders")
    print(
        "code_task={code_task} noise_model={noise_model} distance={distance} rounds={rounds} shots={shots} noise={noise}".format(
            code_task=args.code_task,
            noise_model=args.noise_model,
            distance=args.distance,
            rounds=args.rounds,
            shots=args.shots,
            noise=args.noise,
        )
    )
    if observables.size == 0:
        print("Warning: no observables in this DEM; logical error rate will be n/a.")

    ilp_decoder = None
    available_solvers = [s.lower() for s in get_available_solvers()]
    ilp_solver_order = ["highs", "scip", "gurobi", "cbc", "glpk"]
    if args.compare_ilp_solvers:
        if args.ilp_solvers.strip().lower() == "auto":
            selected_solvers = [s for s in ilp_solver_order if s in available_solvers]
        else:
            requested = [s.strip().lower() for s in args.ilp_solvers.split(",") if s.strip()]
            selected_solvers = [s for s in requested if s in available_solvers]
            missing = [s for s in requested if s not in available_solvers]
            if missing:
                print(f"ILP solvers unavailable: {', '.join(missing)}")
        if not selected_solvers:
            print("ILP solvers skipped (no solver available)")
        else:
            for solver in selected_solvers:
                try:
                    ilp_decoder = Decoder.from_stim_dem(dem, solver=solver)

                    def ilp_decode(det, _decoder=ilp_decoder):
                        _, pred = _decoder.decode(det)
                        return pred

                    benchmark(f"ILP[{solver}]", detections, observables, ilp_decode)
                except Exception as exc:
                    print(f"ILP[{solver}]   skipped ({exc})")
    else:
        if not available_solvers:
            print("ILPDecoder   skipped (no solver available)")
        elif args.solver and args.solver.lower() not in available_solvers:
            print(f"ILPDecoder   skipped (solver '{args.solver}' not available)")
        else:
            try:
                ilp_decoder = Decoder.from_stim_dem(dem, solver=args.solver)

                def ilp_decode(det):
                    _, pred = ilp_decoder.decode(det)
                    return pred

                benchmark("ILPDecoder", detections, observables, ilp_decode)
            except Exception as exc:
                print(f"ILPDecoder   skipped ({exc})")

    try:
        mwpm = build_pymatching(dem)

        def mwpm_decode(det):
            pred = mwpm.decode(det)
            return predict_observables(pred, obs_matrix)

        benchmark("MWPM", detections, observables, mwpm_decode)
    except Exception as exc:
        print(f"MWPM         skipped ({exc})")

    try:
        error_probs = 1.0 / (1.0 + np.exp(np.clip(weights, -50, 50)))
        bposd = build_bposd_decoder(H, error_probs)

        def bposd_decode(det):
            if hasattr(bposd, "decode"):
                pred = bposd.decode(det)
            else:
                pred = bposd(det)
            return predict_observables(pred, obs_matrix)

        benchmark("BPOSD", detections, observables, bposd_decode)
    except Exception as exc:
        print(f"BPOSD        skipped ({exc})")


if __name__ == "__main__":
    main()
