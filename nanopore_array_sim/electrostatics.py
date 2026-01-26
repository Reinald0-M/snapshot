"""
Electric fields, potentials, and electro-osmotic velocities.

This module computes the electric potential profile, electric fields,
Debye screening, effective charges, and electro-osmotic flow velocities.
"""

import numpy as np
from .constants import eps0, kB, e_charge, N_A, nm
from .constants import ionic_strength, SOLVENTS


def phi_profile(z: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute piecewise-linear potential profile along z.
    
    Potential values:
    - Top reservoir: Phi_top_mV
    - Membrane/pore: Phi_mid_mV  
    - Bottom reservoir: Phi_bottom_mV
    
    Parameters
    ----------
    z : np.ndarray
        z positions (meters)
    config : dict
        Configuration dictionary with keys:
        - 'electrostatics.Phi_top_mV': float
        - 'electrostatics.Phi_mid_mV': float
        - 'electrostatics.Phi_bottom_mV': float
        - 'geometry.membrane_top': float (meters)
        - 'geometry.membrane_bottom': float (meters)
        
    Returns
    -------
    np.ndarray
        Potential in Volts (same shape as z)
    """
    z = np.asarray(z)
    shape = z.shape
    z_flat = z.flatten()
    
    # Get potential values (convert mV to V)
    phi_top = config.get("electrostatics", {}).get("Phi_top_mV", 200.0) * 1e-3
    phi_mid = config.get("electrostatics", {}).get("Phi_mid_mV", -200.0) * 1e-3
    phi_bot = config.get("electrostatics", {}).get("Phi_bottom_mV", -1000.0) * 1e-3
    
    # Get geometry boundaries
    z_top = config.get("geometry", {}).get("z_top", 100.0 * nm)
    z_mem_top = config.get("geometry", {}).get("membrane_top", 0.0)
    z_mem_bot = config.get("geometry", {}).get("membrane_bottom", -10.0 * nm)
    z_bot = config.get("geometry", {}).get("z_bottom", -50.0 * nm)
    
    # Piecewise linear interpolation
    phi_flat = np.zeros_like(z_flat)
    
    # Top reservoir: linear from phi_top to phi_mid
    mask_top = z_flat >= z_mem_top
    if np.any(mask_top):
        if z_top > z_mem_top:
            alpha = (z_flat[mask_top] - z_mem_top) / (z_top - z_mem_top)
            phi_flat[mask_top] = phi_mid + alpha * (phi_top - phi_mid)
        else:
            phi_flat[mask_top] = phi_top
    
    # Membrane/pore: constant phi_mid
    mask_mem = (z_flat >= z_mem_bot) & (z_flat < z_mem_top)
    phi_flat[mask_mem] = phi_mid
    
    # Bottom reservoir: linear from phi_mid to phi_bot
    mask_bot = z_flat < z_mem_bot
    if np.any(mask_bot):
        if z_bot < z_mem_bot:
            alpha = (z_flat[mask_bot] - z_mem_bot) / (z_bot - z_mem_bot)
            phi_flat[mask_bot] = phi_mid + alpha * (phi_bot - phi_mid)
        else:
            phi_flat[mask_bot] = phi_bot
    
    return phi_flat.reshape(shape)


def Ez_profile(z: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute electric field E_z = -d(phi)/dz.
    
    Since phi is piecewise linear, E_z is piecewise constant.
    
    Parameters
    ----------
    z : np.ndarray
        z positions (meters)
    config : dict
        Configuration dictionary (same as phi_profile)
        
    Returns
    -------
    np.ndarray
        Electric field in V/m (same shape as z)
    """
    z = np.asarray(z)
    shape = z.shape
    z_flat = z.flatten()
    
    # Get potential values (convert mV to V)
    phi_top = config.get("electrostatics", {}).get("Phi_top_mV", 200.0) * 1e-3
    phi_mid = config.get("electrostatics", {}).get("Phi_mid_mV", -200.0) * 1e-3
    phi_bot = config.get("electrostatics", {}).get("Phi_bottom_mV", -1000.0) * 1e-3
    
    # Get geometry boundaries
    z_top = config.get("geometry", {}).get("z_top", 100.0 * nm)
    z_mem_top = config.get("geometry", {}).get("membrane_top", 0.0)
    z_mem_bot = config.get("geometry", {}).get("membrane_bottom", -10.0 * nm)
    z_bot = config.get("geometry", {}).get("z_bottom", -50.0 * nm)
    
    # Piecewise constant E_z = -dphi/dz
    Ez_flat = np.zeros_like(z_flat)
    
    # Top reservoir
    mask_top = z_flat >= z_mem_top
    if np.any(mask_top) and z_top > z_mem_top:
        Ez_flat[mask_top] = -(phi_top - phi_mid) / (z_top - z_mem_top)
    
    # Membrane/pore: zero field (constant potential)
    mask_mem = (z_flat >= z_mem_bot) & (z_flat < z_mem_top)
    Ez_flat[mask_mem] = 0.0
    
    # Bottom reservoir
    mask_bot = z_flat < z_mem_bot
    if np.any(mask_bot) and z_bot < z_mem_bot:
        Ez_flat[mask_bot] = -(phi_bot - phi_mid) / (z_bot - z_mem_bot)
    
    return Ez_flat.reshape(shape)


def debye_length(
    solvent_name: str, salt_name: str, ion_concentration_M: float, T_K: float
) -> float:
    """
    Compute Debye screening length.
    
    λ_D = sqrt(ε k_B T / (2 I e² N_A))
    
    Parameters
    ----------
    solvent_name : str
        Solvent name (must be in SOLVENTS)
    salt_name : str
        Salt name (must be in SALTS)
    ion_concentration_M : float
        Salt concentration in mol/L
    T_K : float
        Temperature in Kelvin
        
    Returns
    -------
    float
        Debye length in meters
    """
    if solvent_name not in SOLVENTS:
        raise ValueError(f"Solvent {solvent_name} not found. Available: {list(SOLVENTS.keys())}")
    
    props = SOLVENTS[solvent_name]
    eps = props["eps_r"] * eps0
    kT = kB * T_K
    
    I = ionic_strength(ion_concentration_M, salt_name)
    
    if I <= 0:
        return np.inf
    
    # Convert I from mol/L to m^-3
    ion_density_m3 = I * 1000 * N_A
    
    lambda_D = np.sqrt((eps * kT) / (2 * ion_density_m3 * e_charge ** 2))
    
    return lambda_D


def effective_charge(q_bare: float, a: float, lambda_D: float) -> float:
    """
    Compute effective charge with Debye screening.
    
    q_eff = q_bare * (1 + a/λ_D) * exp(-a/λ_D)
    
    Parameters
    ----------
    q_bare : float
        Bare charge in Coulombs
    a : float
        Particle radius in meters
    lambda_D : float
        Debye length in meters
        
    Returns
    -------
    float
        Effective charge in Coulombs
    """
    if not np.isfinite(lambda_D) or lambda_D <= 0:
        return q_bare
    
    return q_bare * (1 + a / lambda_D) * np.exp(-a / lambda_D)


def eo_velocity(Ez: np.ndarray, zeta_potential: float, solvent_name: str) -> np.ndarray:
    """
    Compute electro-osmotic slip velocity.
    
    v_EO = -(ε ζ / η) * E_z  (Smoluchowski)
    
    Parameters
    ----------
    Ez : np.ndarray
        Electric field in z-direction (V/m)
    zeta_potential : float
        Wall zeta potential in Volts
    solvent_name : str
        Solvent name (must be in SOLVENTS)
        
    Returns
    -------
    np.ndarray
        Electro-osmotic velocity in m/s (same shape as Ez)
    """
    if solvent_name not in SOLVENTS:
        raise ValueError(f"Solvent {solvent_name} not found. Available: {list(SOLVENTS.keys())}")
    
    props = SOLVENTS[solvent_name]
    eps = props["eps_r"] * eps0
    eta = props["eta"]
    
    v_eo = -(eps * zeta_potential / eta) * Ez
    
    return v_eo
