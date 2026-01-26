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
    r_pore_bottom_nm: Optional[float] = None,
    taper_type: str = "constant",
) -> Geometry:
    """
    Build a simple array of identical nanopores.
    
    Parameters
    ----------
    n_pores : int
        Number of pores in the array
    spacing_nm : float
        Center-to-center spacing between pores (nm)
    r_pore_nm : float
        Pore radius at top (nm), or constriction radius if tapered
    L_pore_nm : float
        Pore length / membrane thickness (nm)
    z_top_nm : float
        Top of domain (nm)
    z_bottom_nm : float
        Bottom of domain (nm)
    r_pore_bottom_nm : float or None
        Pore radius at bottom (nm). If None, uses r_pore_nm (constant radius)
    taper_type : str
        Type of taper: "constant" (cylindrical), "linear" (linear taper),
        "hourglass" (wider at top/bottom, narrow in middle)
        
    Returns
    -------
    Geometry
        Complete geometry object with pore array
    """
    # Convert to meters
    spacing = spacing_nm * nm
    r_pore_top = r_pore_nm * nm
    r_pore_bot = (r_pore_bottom_nm * nm) if r_pore_bottom_nm is not None else r_pore_top
    L_pore = L_pore_nm * nm
    z_top = z_top_nm * nm
    z_bottom = z_bottom_nm * nm
    
    # Define membrane boundaries (centered in domain)
    # Place membrane roughly in middle of domain
    domain_height = z_top - z_bottom
    membrane_center = (z_top + z_bottom) / 2
    membrane_top = membrane_center + L_pore / 2
    membrane_bottom = membrane_center - L_pore / 2
    
    # For hourglass shape, use narrower radius at middle
    if taper_type == "hourglass":
        # Hourglass: wider at top/bottom, narrowest in middle
        r_middle = min(r_pore_top, r_pore_bot) * 0.6  # 60% of minimum
        # Store as r_bottom for now, but we'll need to update pore_radius function
        # For simplicity, use linear interpolation but with r_middle at center
        r_pore_bot = r_pore_top  # Keep same at top and bottom
        # Note: Full hourglass support would require updating pore_radius function
    
    # Create pores
    pores = []
    for k in range(n_pores):
        center_y = (k - (n_pores - 1) / 2) * spacing
        pore = Pore(
            center_y=center_y,
            z_top=membrane_top,
            z_bottom=membrane_bottom,
            r_top=r_pore_top,
            r_bottom=r_pore_bot,
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


def pore_radius(pore: Pore, z: float, taper_type: str = "linear") -> float:
    """
    Get pore radius at a given z position.
    
    Supports different taper types:
    - "linear": Linear interpolation between r_top and r_bottom
    - "hourglass": Wider at top/bottom, narrowest in middle
    - "constant": Constant radius (if r_top == r_bottom)
    
    Parameters
    ----------
    pore : Pore
        The pore object
    z : float
        z position (meters)
    taper_type : str
        Type of taper to apply
        
    Returns
    -------
    float
        Pore radius at z (meters)
    """
    if z < pore.z_bottom or z > pore.z_top:
        return 0.0
    
    if pore.z_top == pore.z_bottom:
        return pore.r_top
    
    if taper_type == "hourglass":
        # Hourglass shape: wider at top/bottom, narrowest at middle
        z_center = (pore.z_top + pore.z_bottom) / 2
        z_mid = (pore.z_top - pore.z_bottom) / 2
        r_max = max(pore.r_top, pore.r_bottom)
        r_min = min(pore.r_top, pore.r_bottom) * 0.6  # Constriction to 60% of smaller radius
        
        # Distance from center (normalized)
        dist_from_center = abs(z - z_center) / z_mid
        
        # Parabolic shape: r = r_min + (r_max - r_min) * (1 - dist^2)
        r = r_min + (r_max - r_min) * (1 - dist_from_center ** 2)
        return max(r, r_min)  # Ensure minimum radius
    else:
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
