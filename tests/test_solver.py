"""Tests for solver configuration."""

import pytest

from mipdecoder.solver import (
    SolverConfig,
    get_available_solvers,
    get_default_solver,
    get_pyomo_solver_name,
)


class TestSolverConfig:
    """Test SolverConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SolverConfig()
        assert config.name == "highs"  # HiGHS is default (easy to pip install)
        assert config.time_limit is None
        assert config.verbose == False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SolverConfig(
            name="highs",
            time_limit=30,
            gap=0.01,
            verbose=True
        )
        assert config.name == "highs"
        assert config.time_limit == 30
        assert config.gap == 0.01
        assert config.verbose == True
    
    def test_pyomo_options_scip(self):
        """Test Pyomo options for SCIP."""
        config = SolverConfig(name="scip", time_limit=60, gap=0.05)
        opts = config.to_pyomo_options()
        
        assert opts.get("limits/time") == 60
        assert opts.get("limits/gap") == 0.05
    
    def test_pyomo_options_highs(self):
        """Test Pyomo options for HiGHS."""
        config = SolverConfig(name="highs", time_limit=60, gap=0.05, threads=4)
        opts = config.to_pyomo_options()
        
        assert opts.get("time_limit") == 60
        assert opts.get("mip_rel_gap") == 0.05
        assert opts.get("threads") == 4


class TestSolverAvailability:
    """Test solver availability functions."""
    
    def test_get_available_solvers(self):
        """Test that get_available_solvers returns a list."""
        available = get_available_solvers()
        assert isinstance(available, list)
    
    def test_get_default_solver(self):
        """Test that get_default_solver returns a string or raises."""
        available = get_available_solvers()
        if available:
            default = get_default_solver()
            assert isinstance(default, str)
            assert default in available
