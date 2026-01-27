"""
Particle species and individual particle state.

This module defines particle data structures and initialization functions.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from .geometry import Geometry, nearest_pore_index
from .constants import kB


@dataclass
class ParticleSpecies:
    """Represents a species of particles with uniform properties."""
    radius: float  # particle radius (meters)
    density: float  # particle density (kg/m³)
    z_bare: int  # bare charge in elementary charge units


@dataclass
class Particle:
    """Represents a single particle in the simulation."""
    r: np.ndarray  # position [y, z] in meters, shape (2,)
    v: np.ndarray  # velocity [v_y, v_z] in m/s, shape (2,)
    active: bool = True  # still being tracked
    pore_index: int | None = None  # current pore association
    history_index: int = -1  # index for trajectory storage (-1 if not tracked)


def initialize_particles(
    N: int,
    geom: Geometry,
    species: ParticleSpecies,
    config: dict,
    rng: np.random.Generator,
) -> List[Particle]:
    """
    Initialize an ensemble of particles.
    
    Places particles in a lateral band near the top of the domain,
    optionally distributed around pore centers.
    
    Parameters
    ----------
    N : int
        Number of particles to create
    geom : Geometry
        Domain geometry
    species : ParticleSpecies
        Particle species properties
    config : dict
        Configuration dictionary. May contain:
        - 'particles.initial_distribution': 'uniform' or 'pore_centered'
        - 'particles.initial_velocity': 'zero' or 'maxwell_boltzmann'
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    List[Particle]
        List of initialized Particle objects
    """
    particles = []
    
    # Get initialization options
    init_dist = config.get("particles", {}).get("initial_distribution", "uniform")
    init_vel = config.get("particles", {}).get("initial_velocity", "zero")
    
    # Initial z position: near top of domain (2 radii margin)
    z_init = geom.z_top - 2.0 * species.radius
    
    # Initialize positions
    if init_dist == "pore_centered" and len(geom.pores) > 0:
        # Distribute around pore centers
        particles_per_pore = N // len(geom.pores)
        remainder = N % len(geom.pores)
        
        particle_idx = 0
        for pore_idx, pore in enumerate(geom.pores):
            n_here = particles_per_pore + (1 if pore_idx < remainder else 0)
            
            for _ in range(n_here):
                # Random offset around pore center
                offset_y = rng.normal(0, 0.1 * geom.pores[0].r_top)
                y_init = pore.center_y + offset_y
                
                # Wrap to lateral period
                y_init = (y_init + geom.lateral_period / 2) % geom.lateral_period - geom.lateral_period / 2
                
                # Find nearest pore
                pore_idx_init = nearest_pore_index(geom, y_init, z_init)
                
                particles.append(
                    Particle(
                        r=np.array([y_init, z_init]),
                        v=np.array([0.0, 0.0]),
                        active=True,
                        pore_index=pore_idx_init,
                        history_index=-1,
                    )
                )
                particle_idx += 1
    else:
        # Uniform distribution across lateral extent
        y_min = -geom.lateral_period / 2
        y_max = geom.lateral_period / 2
        
        for i in range(N):
            y_init = rng.uniform(y_min, y_max)
            
            # Find nearest pore
            pore_idx_init = nearest_pore_index(geom, y_init, z_init)
            
            particles.append(
                Particle(
                    r=np.array([y_init, z_init]),
                    v=np.array([0.0, 0.0]),
                    active=True,
                    pore_index=pore_idx_init,
                    history_index=-1,
                )
            )
    
    # Initialize velocities
    if init_vel == "maxwell_boltzmann":
        # Compute thermal velocity scale
        m = (4.0 / 3.0) * np.pi * (species.radius ** 3) * species.density
        T_K = config.get("solution", {}).get("T_K", 298.15)
        kT = kB * T_K
        v_thermal = np.sqrt(kT / m)
        
        for p in particles:
            # Maxwell-Boltzmann distribution (2D)
            p.v = rng.normal(0, v_thermal, size=2)
    # else: velocities remain zero
    
    return particles
