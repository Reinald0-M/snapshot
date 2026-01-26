"""
Domain geometry and pore array definitions.

This module defines the 2D (y-z) geometry of the nanopore array,
including pore positions, membrane boundaries, and geometric queries.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .constants import nm


@dataclass
class Pore:
    """Represents a single nanopore in the array."""
    center_y: float  # lateral position of pore center (meters)
    z_top: float  # top of pore opening (meters)
    z_bottom: float  # bottom of pore opening (meters)
    r_top: float  # radius at top (meters)
    r_bottom: float  # radius at bottom (meters)


@dataclass
class Geometry:
    """Represents the complete 2D domain geometry."""
    z_top: float  # top of domain (meters)
    z_bottom: float  # bottom of domain (meters)
    membrane_top: float  # top of membrane layer (meters)
    membrane_bottom: float  # bottom of membrane layer (meters)
    pores: list[Pore]  # list of pores in the array
    lateral_period: float  # periodicity for lateral boundary conditions (meters)


def build_simple_array(
    n_pores: int,
    spacing_nm: float,
    r_pore_nm: float,
    L_pore_nm: float,
    z_top_nm: float,
    z_bottom_nm: float,
) -> Geometry:
    """
    Build a simple array of identical cylindrical nanopores.
    
    Parameters
    ----------
    n_pores : int
        Number of pores in the array
    spacing_nm : float
        Center-to-center spacing between pores (nm)
    r_pore_nm : float
        Pore radius (nm)
    L_pore_nm : float
        Pore length / membrane thickness (nm)
    z_top_nm : float
        Top of domain (nm)
    z_bottom_nm : float
        Bottom of domain (nm)
        
    Returns
    -------
    Geometry
        Complete geometry object with pore array
    """
    # Convert to meters
    spacing = spacing_nm * nm
    r_pore = r_pore_nm * nm
    L_pore = L_pore_nm * nm
    z_top = z_top_nm * nm
    z_bottom = z_bottom_nm * nm
    
    # Define membrane boundaries (centered in domain)
    # Place membrane roughly in middle of domain
    domain_height = z_top - z_bottom
    membrane_center = (z_top + z_bottom) / 2
    membrane_top = membrane_center + L_pore / 2
    membrane_bottom = membrane_center - L_pore / 2
    
    # Create pores
    pores = []
    for k in range(n_pores):
        center_y = (k - (n_pores - 1) / 2) * spacing
        pore = Pore(
            center_y=center_y,
            z_top=membrane_top,
            z_bottom=membrane_bottom,
            r_top=r_pore,
            r_bottom=r_pore,
        )
        pores.append(pore)
    
    # Lateral periodicity: enough to cover all pores plus some margin
    lateral_period = n_pores * spacing * 1.5
    
    return Geometry(
        z_top=z_top,
        z_bottom=z_bottom,
        membrane_top=membrane_top,
        membrane_bottom=membrane_bottom,
        pores=pores,
        lateral_period=lateral_period,
    )


def pore_radius(pore: Pore, z: float) -> float:
    """
    Get pore radius at a given z position.
    
    For tapered pores, linearly interpolates between r_top and r_bottom.
    For constant-radius pores, returns the constant value.
    
    Parameters
    ----------
    pore : Pore
        The pore object
    z : float
        z position (meters)
        
    Returns
    -------
    float
        Pore radius at z (meters)
    """
    if z < pore.z_bottom or z > pore.z_top:
        return 0.0
    
    if pore.z_top == pore.z_bottom:
        return pore.r_top
    
    # Linear interpolation
    alpha = (z - pore.z_bottom) / (pore.z_top - pore.z_bottom)
    return pore.r_top * (1 - alpha) + pore.r_bottom * alpha


def is_inside_pore(geom: Geometry, y: float, z: float) -> bool:
    """
    Check if a point (y, z) lies inside any pore opening.
    
    Parameters
    ----------
    geom : Geometry
        The geometry object
    y : float
        Lateral position (meters)
    z : float
        Vertical position (meters)
        
    Returns
    -------
    bool
        True if point is inside any pore
    """
    # Must be within membrane z-range
    if z < geom.membrane_bottom or z > geom.membrane_top:
        return False
    
    # Check each pore
    for pore in geom.pores:
        r_at_z = pore_radius(pore, z)
        if r_at_z <= 0:
            continue
        
        dy = y - pore.center_y
        r_sq = dy ** 2
        
        if r_sq <= r_at_z ** 2:
            return True
    
    return False


def nearest_pore_index(geom: Geometry, y: float, z: float) -> Optional[int]:
    """
    Find the index of the nearest pore center.
    
    Only considers pores if the point is within the membrane z-range.
    Returns None if outside membrane or no pores nearby.
    
    Parameters
    ----------
    geom : Geometry
        The geometry object
    y : float
        Lateral position (meters)
    z : float
        Vertical position (meters)
        
    Returns
    -------
    int or None
        Index of nearest pore, or None if outside membrane
    """
    # Must be within membrane z-range
    if z < geom.membrane_bottom or z > geom.membrane_top:
        return None
    
    if len(geom.pores) == 0:
        return None
    
    # Find nearest pore center
    min_dist_sq = float("inf")
    nearest_idx = None
    
    for idx, pore in enumerate(geom.pores):
        dy = y - pore.center_y
        dist_sq = dy ** 2
        
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_idx = idx
    
    return nearest_idx


def apply_lateral_bc(y_array: np.ndarray, geom: Geometry) -> np.ndarray:
    """
    Apply periodic boundary conditions in the y-direction.
    
    Parameters
    ----------
    y_array : np.ndarray
        Array of y positions (meters)
    geom : Geometry
        The geometry object (contains lateral_period)
        
    Returns
    -------
    np.ndarray
        y positions wrapped to [0, lateral_period)
    """
    # Shift to [0, period) range
    y_wrapped = y_array % geom.lateral_period
    
    # Center around zero for symmetric array
    y_wrapped = y_wrapped - geom.lateral_period / 2
    
    return y_wrapped
