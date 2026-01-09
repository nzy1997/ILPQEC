"""Validation and parsing tests that do not require a solver backend."""

import numpy as np
import pytest

from mipdecoder import Decoder


def test_weights_length_mismatch():
    """Weights must match the number of error mechanisms."""
    H = np.array([[1, 0, 1]])
    with pytest.raises(ValueError, match="weights must have length"):
        Decoder.from_parity_check_matrix(H, weights=[1.0, 2.0], solver="highs")


def test_error_probabilities_length_mismatch():
    """Error probabilities must match the number of error mechanisms."""
    H = np.array([[1, 0, 1]])
    with pytest.raises(ValueError, match="error_probabilities must have length"):
        Decoder.from_parity_check_matrix(H, error_probabilities=[0.1, 0.2], solver="highs")


def test_error_probabilities_above_half():
    """Probabilities above 0.5 are rejected to avoid negative weights."""
    H = np.array([[1, 0, 1]])
    with pytest.raises(ValueError, match="<= 0.5"):
        Decoder.from_parity_check_matrix(
            H, error_probabilities=[0.6, 0.1, 0.1], solver="highs"
        )


def test_dem_parses_caret_separator():
    """Caret separators are treated as whitespace in DEM parsing."""
    pytest.importorskip("stim")
    dem_str = """
error(0.1) D0 ^ D1 L0
error(0.2) D1 ^ D2
"""
    decoder = Decoder.from_stim_dem(dem_str, solver="highs")

    H = decoder.get_parity_check_matrix()
    expected = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(H, expected)
    assert decoder.num_observables == 1


def test_dem_parses_tagged_error_lines():
    """Tagged error lines are accepted and tags are ignored."""
    pytest.importorskip("stim")
    dem_str = """
error[foo](0.1) D0
error[bar,baz](0.1) D1 L0
"""
    decoder = Decoder.from_stim_dem(dem_str, solver="highs")
    H = decoder.get_parity_check_matrix()
    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(H, expected)
    assert decoder.num_observables == 1


def test_dem_applies_shift_detectors():
    """shift_detectors offsets are applied to subsequent errors."""
    dem_str = """
error(0.1) D0
shift_detectors 2
error(0.2) D0 D1
"""
    decoder = Decoder.from_stim_dem(dem_str, solver="highs")
    H = decoder.get_parity_check_matrix()
    expected = np.array(
        [
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 1],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(H, expected)


def test_dem_rejects_repeat_blocks_when_flatten_disabled():
    """repeat blocks should fail loudly when flattening is disabled."""
    dem_str = """
repeat 2 {
error(0.1) D0
}
"""
    with pytest.raises(ValueError, match="repeat"):
        Decoder.from_stim_dem(dem_str, solver="highs", flatten_dem=False)


def test_dem_ignores_detector_metadata():
    """detector/logical_observable metadata lines are ignored."""
    dem_str = """
detector D0
logical_observable L0
error(0.1) D0 L0
"""
    decoder = Decoder.from_stim_dem(dem_str, solver="highs")
    H = decoder.get_parity_check_matrix()
    expected = np.array([[1]], dtype=np.uint8)
    np.testing.assert_array_equal(H, expected)
    assert decoder.num_observables == 1
