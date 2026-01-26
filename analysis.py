"""
Analysis functions for computing metrics and parameter scanning.

This module provides functions to compute translocation statistics,
pull-back fractions, and perform parameter sweeps.
"""

import numpy as np
from typing import List, Dict
from .simulation import SimulationResult


def compute_translocation_stats(result: SimulationResult) -> Dict:
    """
    Compute translocation time statistics.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation results
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean': mean translocation time (s)
        - 'median': median translocation time (s)
        - 'std': standard deviation (s)
        - 'fraction_translocated': fraction of particles that translocated
        - 'times': array of translocation times (s)
    """
    translocated_times = result.translocation_time[result.translocated_mask]
    
    if len(translocated_times) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "fraction_translocated": 0.0,
            "times": np.array([]),
        }
    
    return {
        "mean": np.mean(translocated_times),
        "median": np.median(translocated_times),
        "std": np.std(translocated_times),
        "fraction_translocated": np.sum(result.translocated_mask) / len(result.translocated_mask),
        "times": translocated_times,
    }


def compute_pullback_fraction(result: SimulationResult) -> float:
    """
    Compute fraction of particles that were pulled back and recaptured.
    
    A particle is considered "pulled back" if it:
    1. Reached bottom reservoir (translocated)
    2. Had a pore association change after translocation
    
    This is a simplified version. A more sophisticated version would
    track the full trajectory to detect re-entry into membrane.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation results
        
    Returns
    -------
    float
        Fraction of translocated particles that showed pull-back behavior
    """
    if not np.any(result.translocated_mask):
        return 0.0
    
    # For tracked particles, check if pore_index changed after translocation
    # This is a simplified metric - full implementation would track
    # whether particles re-entered membrane after reaching bottom
    
    n_translocated = np.sum(result.translocated_mask)
    if n_translocated == 0:
        return 0.0
    
    # Count particles that had pore association changes
    # (This is a placeholder - full implementation would track re-entry)
    pullback_count = 0
    
    # For now, return 0.0 as a placeholder
    # Full implementation would:
    # 1. Track when each particle first reaches bottom
    # 2. Check if it later re-enters membrane region
    # 3. Check if it gets associated with a different pore
    
    return float(pullback_count) / n_translocated


def scan_parameter(
    config_base: Dict,
    param_name: str,
    param_values: List,
    n_repeats: int = 1,
) -> List[SimulationResult]:
    """
    Run multiple simulations while varying one parameter.
    
    Parameters
    ----------
    config_base : dict
        Base configuration dictionary
    param_name : str
        Parameter name to vary (e.g., 'geometry.pore_spacing_nm')
    param_values : List
        List of parameter values to test
    n_repeats : int
        Number of repeats per parameter value (for statistics)
        
    Returns
    -------
    List[SimulationResult]
        List of simulation results (one per parameter value per repeat)
    """
    from .simulation import run_simulation
    
    results = []
    
    for param_val in param_values:
        for repeat in range(n_repeats):
            # Create config copy
            config = _deep_update(config_base.copy(), param_name, param_val)
            
            # Update seed for different repeats
            if "simulation" not in config:
                config["simulation"] = {}
            config["simulation"]["seed"] = config_base.get("simulation", {}).get("seed", 42) + repeat
            
            print(f"Running: {param_name}={param_val}, repeat={repeat+1}/{n_repeats}")
            result = run_simulation(config)
            results.append(result)
    
    return results


def _deep_update(config: Dict, param_path: str, value) -> Dict:
    """
    Update nested dictionary using dot-separated path.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    param_path : str
        Dot-separated path (e.g., 'geometry.pore_spacing_nm')
    value
        Value to set
        
    Returns
    -------
    dict
        Updated configuration
    """
    keys = param_path.split(".")
    d = config
    
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    
    d[keys[-1]] = value
    return config
