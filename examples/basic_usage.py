"""
Basic usage examples for MIPDecoder.

This script demonstrates the core functionality:
1. Decoding with parity-check matrices
2. Decoding with Stim detector error models
3. Switching solvers at runtime
4. Maximum-likelihood decoding with weights
"""

import numpy as np

from mipdecoder import Decoder, get_available_solvers


def example_repetition_code():
    """Example: Decode a 5-qubit repetition code."""
    print("=" * 60)
    print("Example: 5-Qubit Repetition Code")
    print("=" * 60)
    
    H = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])
    
    # Create decoder (uses default solver)
    decoder = Decoder.from_parity_check_matrix(H)
    print(f"Decoder: {decoder}")
    
    # Test case: single error on qubit 2
    true_error = np.array([0, 0, 1, 0, 0])
    syndrome = (H @ true_error) % 2
    print(f"\nTrue error: {true_error}")
    print(f"Syndrome: {syndrome}")
    
    correction = decoder.decode(syndrome)
    print(f"Correction: {correction}")
    
    is_valid = np.array_equal((H @ correction) % 2, syndrome)
    print(f"Valid correction: {is_valid}")
    
    correction, weight = decoder.decode(syndrome, return_weight=True)
    print(f"Solution weight: {weight}")


def example_weighted_decoding():
    """Example: Maximum-likelihood decoding with weighted errors."""
    print("\n" + "=" * 60)
    print("Example: Weighted Decoding (Maximum Likelihood)")
    print("=" * 60)
    
    H = np.array([[1, 1, 0], [0, 1, 1]])
    error_probs = [0.1, 0.01, 0.1]  # Middle qubit more reliable
    
    decoder = Decoder.from_parity_check_matrix(H, error_probabilities=error_probs)
    
    print(f"Error probabilities: {error_probs}")
    print(f"Computed weights: {decoder.get_weights()}")
    
    syndrome = [1, 1]
    print(f"\nSyndrome: {syndrome}")
    
    correction = decoder.decode(syndrome)
    print(f"Correction: {correction}")
    print("(Should prefer [1,0,1] since middle qubit is more reliable)")


def example_stim_dem():
    """Example: Decoding with Stim detector error model."""
    print("\n" + "=" * 60)
    print("Example: Stim Detector Error Model")
    print("=" * 60)
    
    dem_str = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 L1
"""
    
    decoder = Decoder.from_stim_dem(dem_str)
    print(f"Decoder: {decoder}")
    print(f"Detectors: {decoder.num_detectors}")
    print(f"Observables: {decoder.num_observables}")
    print(f"Error mechanisms: {decoder.num_errors}")
    
    detector_outcomes = [1, 0]
    print(f"\nDetector outcomes: {detector_outcomes}")
    
    correction, observables = decoder.decode(detector_outcomes)
    print(f"Predicted observables: {observables}")


def example_solver_switching():
    """
    Example: Switching between solvers.
    
    You can switch solvers without rebuilding the model.
    """
    print("\n" + "=" * 60)
    print("Example: Solver Switching")
    print("=" * 60)
    
    available = get_available_solvers()
    print(f"Available solvers: {available}")
    
    if not available:
        print("No solvers available. Install SCIP, HiGHS, CBC, or GLPK.")
        return
    
    H = np.array([[1, 1, 0], [0, 1, 1]])
    decoder = Decoder.from_parity_check_matrix(H)
    
    for solver in available:
        print(f"\n--- Using {solver.upper()} ---")
        
        # Switch solver
        decoder.set_solver(solver, verbose=False)
        
        syndrome = [1, 0]
        correction, weight = decoder.decode(syndrome, return_weight=True)
        
        print(f"Correction: {correction}")
        print(f"Weight: {weight}")


def example_batch_decoding():
    """Example: Batch decoding multiple syndromes."""
    print("\n" + "=" * 60)
    print("Example: Batch Decoding")
    print("=" * 60)
    
    H = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])
    
    decoder = Decoder.from_parity_check_matrix(H)
    
    np.random.seed(42)
    syndromes = np.random.randint(0, 2, size=(5, 4))
    
    print("Syndromes:")
    for i, s in enumerate(syndromes):
        print(f"  {i}: {s}")
    
    corrections = decoder.decode_batch(syndromes)
    
    print("\nCorrections:")
    for i, c in enumerate(corrections):
        valid = np.array_equal((H @ c) % 2, syndromes[i])
        print(f"  {i}: {c} (valid: {valid})")


def main():
    """Run all examples."""
    print("MIPDecoder Examples")
    print("=" * 60)
    print(f"Available solvers: {get_available_solvers()}")
    
    if not get_available_solvers():
        print("\nNo solver available!")
        print("Install one of: SCIP, HiGHS, CBC, GLPK")
        return
    
    example_repetition_code()
    example_weighted_decoding()
    example_stim_dem()
    example_solver_switching()
    example_batch_decoding()
    
    print("\n" + "=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()
