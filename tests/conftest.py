"""Pytest configuration and fixtures."""

import pytest
import numpy as np


@pytest.fixture
def repetition_code_5():
    """5-qubit repetition code parity-check matrix."""
    return np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ])


@pytest.fixture
def simple_dem():
    """Simple detector error model string."""
    return """
error(0.1) D0 L0
error(0.1) D0 D1 L1
error(0.1) D1 L2
"""


@pytest.fixture
def surface_code_dem():
    """Surface code detector error model string (distance 3)."""
    return """
error(0.01) D0
error(0.01) D0 D1
error(0.01) D0 D2 L0
error(0.01) D1
error(0.01) D1 D3
error(0.01) D2 D3
error(0.01) D2
error(0.01) D3 L0
"""
