"""
Langevin dynamics integrator.

This module implements vectorized Euler-Maruyama integration for
Langevin dynamics with electric fields and electro-osmotic flow.
"""

import numpy as np
from typing import List
from .particles import Particle, ParticleSpecies
from .geometry import Geometry
from .electrostatics import Ez_profile, eo_velocity
from .constants import kB


def langevin_step(
    particles: List[Particle],
    geom: Geometry,
    species: ParticleSpecies,
    phys_params: dict,
    dt: float,
    rng: np.random.Generator,
    config: dict,
) -> None:
    """
    Perform one time step of Langevin dynamics for all active particles.
    
    Updates particle positions and velocities in-place.
    
    Implements Euler-Maruyama discretization:
    v_{n+1} = v_n + [-ζ/m (v_n - u_fluid) + q_eff/m E] Δt + σ η_n
    r_{n+1} = r_n + v_{n+1} Δt
    
    where σ = sqrt(2 ζ k_B T / m² * Δt)
    
    Parameters
    ----------
    particles : List[Particle]
        List of all particles
    geom : Geometry
        Domain geometry
    species : ParticleSpecies
        Particle species properties
    phys_params : dict
        Physical parameters with keys:
        - 'm': particle mass (kg)
        - 'zeta_drag': Stokes drag coefficient (kg/s)
        - 'q_eff': effective charge (C)
        - 'T_K': temperature (K)
        - 'solvent_name': solvent name
    dt : float
        Time step (seconds)
    rng : np.random.Generator
        Random number generator
    config : dict
        Configuration dictionary for electric field and EO flow
    """
    # Extract active particles
    active_particles = [p for p in particles if p.active]
    if len(active_particles) == 0:
        return
    
    n_active = len(active_particles)
    
    # Pack positions and velocities into arrays
    r_array = np.array([p.r for p in active_particles])  # shape (n_active, 2)
    v_array = np.array([p.v for p in active_particles])  # shape (n_active, 2)
    
    # Extract z positions for field computation
    z_positions = r_array[:, 1]  # shape (n_active,)
    
    # Compute electric field at each particle position
    Ez = Ez_profile(z_positions, config)  # shape (n_active,)
    
    # Determine region-specific zeta potential
    # Get zeta potentials from config
    zeta_top = config.get("electrostatics", {}).get("zeta_top_mV", -50.0) * 1e-3
    zeta_pore = config.get("electrostatics", {}).get("zeta_pore_mV", -100.0) * 1e-3
    zeta_bottom = config.get("electrostatics", {}).get("zeta_bottom_mV", -50.0) * 1e-3
    
    # Select zeta based on region
    zeta_array = np.zeros_like(z_positions)
    
    # Get max transition height
    max_trans = 0.0
    if hasattr(geom, "pores") and len(geom.pores) > 0:
        max_trans = max(p.transition_height for p in geom.pores)
        
    z_trans_bottom = geom.membrane_bottom - max_trans
    
    mask_top = z_positions >= geom.membrane_top
    mask_mem = (z_positions >= geom.membrane_bottom) & (z_positions < geom.membrane_top)
    mask_trans = (z_positions >= z_trans_bottom) & (z_positions < geom.membrane_bottom)
    mask_bot = z_positions < z_trans_bottom
    
    zeta_array[mask_top] = zeta_top
    zeta_array[mask_mem] = zeta_pore
    zeta_array[mask_trans] = 0.0  # Transition region carries no charge
    zeta_array[mask_bot] = zeta_bottom
    
    # Compute electro-osmotic velocity at each position
    v_eo_array = np.zeros_like(z_positions)
    for i in range(n_active):
        v_eo_array[i] = eo_velocity(Ez[i], zeta_array[i], phys_params["solvent_name"])
    
    # Physical parameters
    m = phys_params["m"]
    zeta = phys_params["zeta_drag"]
    q_eff = phys_params["q_eff"]
    T_K = phys_params["T_K"]
    
    # Noise amplitude
    sigma = np.sqrt(2 * zeta * kB * T_K / (m ** 2) * dt)
    
    # Electric field force (only in z-direction)
    E_force = np.zeros_like(r_array)
    E_force[:, 1] = (q_eff / m) * Ez  # z-component only
    
    # Drag force: -ζ/m * (v - u_fluid)
    u_fluid = np.zeros_like(r_array)
    u_fluid[:, 1] = v_eo_array  # EO flow only in z-direction
    
    drag_force = -(zeta / m) * (v_array - u_fluid)
    
    # Deterministic acceleration
    a_det = drag_force + E_force
    
    # Random noise (2D Gaussian)
    eta = rng.normal(0, 1, size=(n_active, 2))
    noise = sigma * eta
    
    # Update velocity: v_{n+1} = v_n + a_det * dt + noise
    v_new = v_array + a_det * dt + noise
    
    # Update position: r_{n+1} = r_n + v_{n+1} * dt
    r_new = r_array + v_new * dt
    
    # Unpack back to Particle objects
    for i, p in enumerate(active_particles):
        p.v = v_new[i]
        p.r = r_new[i]
