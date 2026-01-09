"""Tests for the main Decoder class."""

import pytest
import numpy as np

from ilpdecoder import Decoder, get_available_solvers


# Skip tests if no solver is available
pytestmark = pytest.mark.skipif(
    len(get_available_solvers()) == 0,
    reason="No solver backend available"
)


class TestDecoderFromParityCheck:
    """Test Decoder with parity-check matrix input."""
    
    def test_repetition_code_decode(self):
        """Test decoding a repetition code."""
        H = np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ])
        
        decoder = Decoder.from_parity_check_matrix(H)
        
        # Test with single error on qubit 2
        syndrome = [0, 1, 1, 0]
        correction = decoder.decode(syndrome)
        
        # Verify correction satisfies syndrome
        assert np.array_equal((H @ correction) % 2, syndrome)
    
    def test_decode_with_weight(self):
        """Test decoding with weight return."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        syndrome = [1, 1]
        correction, weight = decoder.decode(syndrome, return_weight=True)
        
        assert isinstance(weight, float)
        assert np.array_equal((H @ correction) % 2, syndrome)
    
    def test_decode_weighted(self):
        """Test that weights affect decoding."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        
        # High weight on middle qubit
        decoder = Decoder.from_parity_check_matrix(H, weights=[1.0, 100.0, 1.0])
        
        syndrome = [1, 1]
        correction = decoder.decode(syndrome)
        
        # Should prefer correction without middle qubit
        assert correction[1] == 0
    
    def test_no_error_syndrome(self):
        """Test decoding with no errors (zero syndrome)."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        syndrome = [0, 0]
        correction = decoder.decode(syndrome)
        
        np.testing.assert_array_equal(correction, [0, 0, 0])


class TestDecoderFromDEM:
    """Test Decoder with Stim DEM input."""
    
    def test_simple_dem_decode(self):
        """Test decoding with a simple DEM."""
        dem_str = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 L1
"""
        decoder = Decoder.from_stim_dem(dem_str)
        
        assert decoder.num_detectors == 2
        assert decoder.num_observables == 2
        
        detector_outcomes = [1, 0]
        correction, observables = decoder.decode(detector_outcomes)
        
        assert len(observables) == 2
    
    def test_dem_observable_prediction(self):
        """Test that observable predictions are correct."""
        dem_str = """
error(0.1) D0 L0
error(0.1) D1 L1
"""
        decoder = Decoder.from_stim_dem(dem_str)
        
        detector_outcomes = [1, 0]
        _, observables = decoder.decode(detector_outcomes)
        
        assert observables[0] == 1
        assert observables[1] == 0


class TestDecoderBatch:
    """Test batch decoding."""
    
    def test_batch_decode_parity(self):
        """Test batch decoding with parity-check matrix."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        syndromes = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ])
        
        corrections = decoder.decode_batch(syndromes)
        
        assert corrections.shape == (4, 3)
        
        for i in range(4):
            computed = (H @ corrections[i]) % 2
            np.testing.assert_array_equal(computed, syndromes[i])


class TestDecoderSolverSwitch:
    """Test switching solvers."""
    
    def test_solver_switch(self):
        """Test that solver can be switched."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        available = get_available_solvers()
        if len(available) >= 1:
            decoder.set_solver(available[0])
            assert decoder.solver_name == available[0]
    
    def test_solver_options(self):
        """Test setting solver options."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H, time_limit=10, verbose=False)
        
        options = decoder.get_solver_options()
        assert options["time_limit"] == 10
        assert options["verbose"] == False


class TestDecoderProperties:
    """Test decoder properties."""
    
    def test_parity_properties(self):
        """Test properties for parity-check decoder."""
        H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        assert decoder.num_detectors == 3
        assert decoder.num_errors == 4
        assert decoder.num_observables == 0
    
    def test_dem_properties(self):
        """Test properties for DEM decoder."""
        dem_str = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 D2 L1
"""
        decoder = Decoder.from_stim_dem(dem_str)
        
        assert decoder.num_detectors == 3
        assert decoder.num_observables == 2
        assert decoder.num_errors == 3
    
    def test_get_parity_check_matrix(self):
        """Test getting parity check matrix."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        decoder = Decoder.from_parity_check_matrix(H)
        
        H_out = decoder.get_parity_check_matrix()
        np.testing.assert_array_equal(H_out, H)
    
    def test_get_weights(self):
        """Test getting weights."""
        H = np.array([[1, 1, 0], [0, 1, 1]])
        weights = [1.0, 2.0, 3.0]
        decoder = Decoder.from_parity_check_matrix(H, weights=weights)
        
        w_out = decoder.get_weights()
        np.testing.assert_array_equal(w_out, weights)
