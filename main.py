"""
ILPDecoder - ILP-based Quantum Error Correction Decoder

Run basic functionality test.
"""

import numpy as np
from ilpdecoder import Decoder, get_available_solvers


def main():
    """Run basic functionality test."""
    print("ILPDecoder - ILP-based QEC Decoder (direct HiGHS backend)")
    print("=" * 50)
    
    available = get_available_solvers()
    print(f"\nAvailable solvers: {available}")
    
    if not available:
        print("\nNo solver available!")
        print("Please install one of:")
        print("  - HiGHS: pip install highspy")
        print("  - Gurobi: pip install ilpdecoder[gurobi]")
        print("  - Pyomo solvers: pip install ilpdecoder[pyomo]")
        return 1
    
    # Test with simple repetition code
    print("\n--- Test: 5-qubit Repetition Code ---")
    
    H = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])
    
    decoder = Decoder.from_parity_check_matrix(H)
    print(f"Created decoder: {decoder}")
    
    syndrome = [0, 1, 1, 0]
    print(f"\nSyndrome: {syndrome}")
    
    correction = decoder.decode(syndrome)
    print(f"Correction: {correction}")
    
    computed_syndrome = (H @ correction) % 2
    is_valid = np.array_equal(computed_syndrome, syndrome)
    print(f"Valid: {is_valid}")
    
    # Test with DEM
    print("\n--- Test: Simple Detector Error Model ---")
    
    dem_str = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 L1
"""
    
    decoder = Decoder.from_stim_dem(dem_str)
    print(f"Created decoder: {decoder}")
    
    detector_outcomes = [1, 0]
    print(f"\nDetector outcomes: {detector_outcomes}")
    
    _, observables = decoder.decode(detector_outcomes)
    print(f"Predicted observables: {observables}")
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    return 0


if __name__ == "__main__":
    exit(main())
