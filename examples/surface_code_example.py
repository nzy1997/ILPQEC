"""
Surface code decoding example using Stim integration.

Requirements:
    pip install stim mipdecoder
    Plus a solver (SCIP, HiGHS, etc.)
"""

import numpy as np

try:
    import stim
    STIM_AVAILABLE = True
except ImportError:
    STIM_AVAILABLE = False

from mipdecoder import Decoder, get_available_solvers


def surface_code_example():
    """Decode a rotated surface code using ILP."""
    print("=" * 60)
    print("Surface Code Decoding with MIPDecoder")
    print("=" * 60)
    
    if not STIM_AVAILABLE:
        print("Stim is not installed. Install with: pip install stim")
        return
    
    if not get_available_solvers():
        print("No solver available. Install SCIP, HiGHS, CBC, or GLPK.")
        return
    
    distance = 3
    rounds = 3
    noise = 0.01
    
    print(f"\nGenerating surface code circuit:")
    print(f"  Distance: {distance}")
    print(f"  Rounds: {rounds}")
    print(f"  Noise: {noise}")
    
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise
    )
    
    dem = circuit.detector_error_model(decompose_errors=True)
    print(f"\nDetector error model:")
    print(f"  Detectors: {dem.num_detectors}")
    print(f"  Observables: {dem.num_observables}")
    print(f"  Error mechanisms: {dem.num_errors}")
    
    decoder = Decoder.from_stim_dem(dem)
    print(f"\nDecoder: {decoder}")
    
    num_shots = 100
    print(f"\nSampling {num_shots} shots...")
    
    sampler = circuit.compile_detector_sampler()
    detection_events, actual_observables = sampler.sample(
        shots=num_shots,
        separate_observables=True
    )
    
    print("Decoding...")
    
    num_correct = 0
    num_detected = 0
    
    for i in range(num_shots):
        if np.any(detection_events[i]):
            num_detected += 1
        
        _, predicted = decoder.decode(detection_events[i])
        
        if np.array_equal(predicted, actual_observables[i]):
            num_correct += 1
    
    print(f"\nResults:")
    print(f"  Shots with detections: {num_detected}/{num_shots}")
    print(f"  Correct predictions: {num_correct}/{num_shots}")
    print(f"  Logical error rate: {(num_shots - num_correct) / num_shots:.2%}")


def compare_solvers():
    """Compare solve times between different solvers."""
    print("\n" + "=" * 60)
    print("Solver Comparison")
    print("=" * 60)
    
    if not STIM_AVAILABLE or not get_available_solvers():
        print("Stim or solver not available. Skipping.")
        return
    
    import time
    
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01
    )
    
    dem = circuit.detector_error_model(decompose_errors=True)
    
    sampler = circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(shots=20, separate_observables=True)
    
    available = get_available_solvers()
    print(f"\nAvailable solvers: {available}")
    print(f"Testing with {len(detection_events)} shots...")
    
    for solver in available:
        decoder = Decoder.from_stim_dem(dem, solver=solver)
        
        start = time.time()
        for i in range(len(detection_events)):
            decoder.decode(detection_events[i])
        elapsed = time.time() - start
        
        avg_time = elapsed / len(detection_events)
        print(f"  {solver.upper():8s}: {avg_time*1000:6.1f} ms/shot")


def main():
    print("MIPDecoder Surface Code Example")
    print("=" * 60)
    
    surface_code_example()
    compare_solvers()
    
    print("\n" + "=" * 60)
    print("Example completed!")


if __name__ == "__main__":
    main()
