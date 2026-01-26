"""
PNPS-Based Nanopore Array Simulator

A Python package for simulating PNPS-driven transport of spherical particles
through an array of nanopores using Langevin dynamics.
"""

__version__ = "0.1.0"

from .simulation import run_simulation, SimulationResult, load_config
from .geometry import Geometry, Pore, build_simple_array
from .particles import Particle, ParticleSpecies
from .analysis import compute_translocation_stats, compute_pullback_fraction

__all__ = [
    "run_simulation",
    "SimulationResult",
    "load_config",
    "Geometry",
    "Pore",
    "build_simple_array",
    "Particle",
    "ParticleSpecies",
    "compute_translocation_stats",
    "compute_pullback_fraction",
]
