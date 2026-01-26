"""
Main simulation orchestration.

This module coordinates the complete simulation run, including
geometry setup, particle initialization, time stepping, boundary
conditions, and result collection.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import yaml
from pathlib import Path

from .geometry import Geometry, build_simple_array, is_inside_pore, nearest_pore_index, apply_lateral_bc, pore_radius
from .particles import Particle, ParticleSpecies, initialize_particles
from .integrator import langevin_step
from .electrostatics import debye_length, effective_charge
from .constants import e_charge, nm, SOLVENTS


@dataclass
class SimulationResult:
    """Container for simulation results."""
    times: np.ndarray  # shape (n_steps,)
    Y: np.ndarray  # shape (n_steps, n_tracked)
    Z: np.ndarray  # shape (n_steps, n_tracked)
    pore_index: np.ndarray  # shape (n_steps, n_tracked)
    translocated_mask: np.ndarray  # shape (n_particles,) bool
    translocation_time: np.ndarray  # shape (n_particles,) float
    geometry: Geometry
    config: dict


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file or return default.
    
    Parameters
    ----------
    config_path : str or None
        Path to YAML config file. If None, uses default.
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        default_path = Path(__file__).parent / "configs" / "default_config.yaml"
        config_path = str(default_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def run_simulation(config: Optional[dict] = None, config_path: Optional[str] = None) -> SimulationResult:
    """
    Run a complete simulation.
    
    Parameters
    ----------
    config : dict or None
        Configuration dictionary. If None, loads from config_path.
    config_path : str or None
        Path to YAML config file. Used if config is None.
        
    Returns
    -------
    SimulationResult
        Complete simulation results
    """
    # Load config if needed
    if config is None:
        config = load_config(config_path)
    
    # Extract parameters
    geom_config = config.get("geometry", {})
    particle_config = config.get("particles", {})
    solution_config = config.get("solution", {})
    sim_config = config.get("simulation", {})
    electrostatics_config = config.get("electrostatics", {})
    
    # Build geometry
    geom = build_simple_array(
        n_pores=geom_config.get("n_pores", 3),
        spacing_nm=geom_config.get("pore_spacing_nm", 10.0),
        r_pore_nm=geom_config.get("pore_radius_nm", 5.0),
        L_pore_nm=geom_config.get("pore_length_nm", 10.0),
        z_top_nm=geom_config.get("z_top_nm", 100.0),
        z_bottom_nm=geom_config.get("z_bottom_nm", -50.0),
        r_pore_bottom_nm=geom_config.get("pore_radius_bottom_nm"),
        taper_type=geom_config.get("taper_type", "constant"),
    )
    
    # Add geometry to config for field computations
    config["geometry"] = {
        **geom_config,
        "z_top": geom.z_top,
        "z_bottom": geom.z_bottom,
        "membrane_top": geom.membrane_top,
        "membrane_bottom": geom.membrane_bottom,
    }
    
    # Create particle species
    a_nm = particle_config.get("radius_nm", 2.0)
    a = a_nm * nm
    species = ParticleSpecies(
        radius=a,
        density=particle_config.get("density_kg_per_m3", 1000.0),
        z_bare=particle_config.get("z_bare", -10),
    )
    
    # Compute physical parameters
    solvent_name = solution_config.get("solvent", "water")
    if solvent_name not in SOLVENTS:
        raise ValueError(f"Solvent {solvent_name} not found")
    
    sol_props = SOLVENTS[solvent_name]
    eta = sol_props["eta"]
    T_K = solution_config.get("T_K", 298.15)
    
    # Mass and drag
    m = (4.0 / 3.0) * np.pi * (a ** 3) * species.density
    zeta_drag = 6 * np.pi * eta * a
    
    # Debye length and effective charge
    salt_name = solution_config.get("salt", "NaCl")
    concentration_M = solution_config.get("concentration_M", 0.1)
    
    lambda_D = debye_length(solvent_name, salt_name, concentration_M, T_K)
    q_bare = species.z_bare * e_charge
    q_eff = effective_charge(q_bare, a, lambda_D)
    
    # Time step
    if sim_config.get("dt_s") is not None:
        dt = sim_config["dt_s"]
    else:
        tau = m / zeta_drag
        dt = 0.01 * tau
    
    # Physical parameters dict
    phys_params = {
        "m": m,
        "zeta_drag": zeta_drag,
        "q_eff": q_eff,
        "T_K": T_K,
        "solvent_name": solvent_name,
    }
    
    # Initialize particles
    seed = sim_config.get("seed", 42)
    rng = np.random.default_rng(seed)
    
    n_particles = particle_config.get("n_particles", 100)
    particles = initialize_particles(n_particles, geom, species, config, rng)
    
    # Assign history indices to tracked subset
    n_tracked = sim_config.get("n_tracked", 20)
    tracked_indices = rng.choice(len(particles), size=min(n_tracked, len(particles)), replace=False)
    for idx, p in enumerate(particles):
        if idx in tracked_indices:
            p.history_index = len([i for i in tracked_indices if i < idx])
    
    # Simulation parameters
    t_max = sim_config.get("t_max_s", 1e-6)
    max_steps = int(t_max / dt)
    save_every = max(1, max_steps // 1000)  # Save ~1000 time points
    
    # Initialize result arrays
    n_saved = (max_steps // save_every) + 1
    times = np.zeros(n_saved)
    Y = np.full((n_saved, n_tracked), np.nan)
    Z = np.full((n_saved, n_tracked), np.nan)
    pore_index_array = np.full((n_saved, n_tracked), -1, dtype=int)
    
    # Translocation tracking
    translocated_mask = np.zeros(n_particles, dtype=bool)
    translocation_time = np.full(n_particles, np.nan)
    
    # Main simulation loop
    t = 0.0
    step = 0
    save_idx = 0
    
    print(f"Starting simulation: {n_particles} particles, dt={dt:.2e} s, max_steps={max_steps}")
    
    while step < max_steps and any(p.active for p in particles):
        # Save state
        if step % save_every == 0:
            times[save_idx] = t
            for p in particles:
                if p.history_index >= 0:
                    Y[save_idx, p.history_index] = p.r[0]
                    Z[save_idx, p.history_index] = p.r[1]
                    pore_index_array[save_idx, p.history_index] = (
                        p.pore_index if p.pore_index is not None else -1
                    )
            save_idx += 1
        
        # Integration step
        langevin_step(particles, geom, species, phys_params, dt, rng, config)
        
        # Apply boundary conditions and constraints
        for p in particles:
            if not p.active:
                continue
            
            # Lateral periodic boundary
            p.r[0] = apply_lateral_bc(np.array([p.r[0]]), geom)[0]
            
            # Top boundary: reflect if above domain
            if p.r[1] > geom.z_top:
                p.r[1] = geom.z_top
                p.v[1] = -abs(p.v[1])  # Reflect downward
            
            # Bottom boundary: mark as translocated
            if p.r[1] < geom.z_bottom:
                if not translocated_mask[particles.index(p)]:
                    translocated_mask[particles.index(p)] = True
                    translocation_time[particles.index(p)] = t
                p.active = False
                continue
            
            # Membrane constraints
            if geom.membrane_bottom <= p.r[1] <= geom.membrane_top:
                # Check if inside any pore
                inside = is_inside_pore(geom, p.r[0], p.r[1])
                
                if not inside:
                    # Reflect from membrane wall
                    # Simple reflection: reverse z-velocity if moving into wall
                    if p.v[1] < 0:  # Moving down into membrane
                        p.v[1] = abs(p.v[1])  # Reflect upward
                        p.r[1] = max(p.r[1], geom.membrane_top)
                    elif p.v[1] > 0:  # Moving up into membrane
                        p.v[1] = -abs(p.v[1])  # Reflect downward
                        p.r[1] = min(p.r[1], geom.membrane_bottom)
                else:
                    # Inside pore: check radius constraint
                    pore_idx = nearest_pore_index(geom, p.r[0], p.r[1])
                    if pore_idx is not None:
                        pore = geom.pores[pore_idx]
                        r_at_z = pore_radius(pore, p.r[1])
                        dy = p.r[0] - pore.center_y
                        r_particle = abs(dy)
                        
                        if r_particle > r_at_z:
                            # Reflect radially inward
                            normal_y = dy / r_particle if r_particle > 0 else 1.0
                            v_dot_n = p.v[0] * normal_y
                            p.v[0] -= 2 * v_dot_n * normal_y
                            p.r[0] = pore.center_y + normal_y * r_at_z
            
            # Update pore association
            p.pore_index = nearest_pore_index(geom, p.r[0], p.r[1])
        
        t += dt
        step += 1
        
        if step % 10000 == 0:
            n_active = sum(1 for p in particles if p.active)
            print(f"Step {step}/{max_steps}, t={t:.2e} s, active={n_active}", end="\r")
    
    print(f"\nSimulation complete: {step} steps, {np.sum(translocated_mask)} translocations")
    
    # Trim result arrays to actual size
    times = times[:save_idx]
    Y = Y[:save_idx, :]
    Z = Z[:save_idx, :]
    pore_index_array = pore_index_array[:save_idx, :]
    
    return SimulationResult(
        times=times,
        Y=Y,
        Z=Z,
        pore_index=pore_index_array,
        translocated_mask=translocated_mask,
        translocation_time=translocation_time,
        geometry=geom,
        config=config,
    )
