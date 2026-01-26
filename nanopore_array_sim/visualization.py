"""
Visualization functions for trajectories and analysis.

This module provides plotting functions for particle trajectories,
landing distributions, and parameter scans.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, List
from .simulation import SimulationResult
from .geometry import Geometry, pore_radius
from .constants import nm


def plot_paths(
    result: SimulationResult,
    geom: Optional[Geometry] = None,
    config: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D side view (y-z) showing trajectories and geometry.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation results
    geom : Geometry or None
        Geometry object (from result if None)
    config : dict or None
        Configuration (from result if None)
    save_path : str or None
        Path to save figure (if None, returns figure)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if geom is None:
        geom = result.geometry
    if config is None:
        config = result.config
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw membrane as rectangle
    mem_width = geom.lateral_period
    mem_height = geom.membrane_top - geom.membrane_bottom
    mem_rect = patches.Rectangle(
        (-mem_width / 2, geom.membrane_bottom),
        mem_width,
        mem_height,
        facecolor="gray",
        edgecolor="black",
        alpha=0.3,
        label="Membrane",
    )
    ax.add_patch(mem_rect)
    
    # Draw pore openings
    for idx, pore in enumerate(geom.pores):
        # Draw circles at top and bottom of pore
        circle_top = patches.Circle(
            (pore.center_y, pore.z_top),
            pore.r_top,
            facecolor="white",
            edgecolor="blue",
            linewidth=2,
            label="Pore" if idx == 0 else "",
        )
        circle_bot = patches.Circle(
            (pore.center_y, pore.z_bottom),
            pore.r_bottom,
            facecolor="white",
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(circle_top)
        ax.add_patch(circle_bot)
        
        # Draw pore centerline
        ax.plot([pore.center_y, pore.center_y], [pore.z_bottom, pore.z_top], "b--", alpha=0.3)
    
    # Plot trajectories
    n_tracked = result.Y.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_tracked))
    
    for j in range(n_tracked):
        y_traj = result.Y[:, j]
        z_traj = result.Z[:, j]
        
        # Remove NaN values
        valid = ~(np.isnan(y_traj) | np.isnan(z_traj))
        if np.any(valid):
            ax.plot(
                y_traj[valid] / nm,
                z_traj[valid] / nm,
                color=colors[j],
                linewidth=0.8,
                alpha=0.7,
            )
    
    # Formatting
    ax.set_xlabel("y (nm)", fontsize=12)
    ax.set_ylabel("z (nm)", fontsize=12)
    ax.set_title("Particle Trajectories (Side View)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    
    return fig


def plot_landing_histogram(
    result: SimulationResult,
    geom: Optional[Geometry] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot histogram of y-positions when particles first enter bottom reservoir.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation results
    geom : Geometry or None
        Geometry object (from result if None)
    save_path : str or None
        Path to save figure (if None, returns figure)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if geom is None:
        geom = result.geometry
    
    # Extract landing positions from trajectories
    # Find first time each tracked particle crosses z_bottom
    landing_y = []
    
    for j in range(result.Y.shape[1]):
        z_traj = result.Z[:, j]
        y_traj = result.Y[:, j]
        
        # Find first crossing of z_bottom
        below_mask = z_traj < geom.z_bottom
        if np.any(below_mask):
            first_below = np.where(below_mask)[0][0]
            if first_below > 0:
                landing_y.append(y_traj[first_below] / nm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if len(landing_y) > 0:
        ax.hist(landing_y, bins=30, alpha=0.7, edgecolor="black")
        
        # Overlay pore center positions
        for pore in geom.pores:
            ax.axvline(pore.center_y / nm, color="red", linestyle="--", linewidth=2, alpha=0.7)
    else:
        ax.text(0.5, 0.5, "No translocations", ha="center", va="center", transform=ax.transAxes)
    
    ax.set_xlabel("Landing y position (nm)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Landing Position Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    
    return fig


def plot_z_vs_time(
    result: SimulationResult,
    particle_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot time series of z(t) for selected particles.
    
    Parameters
    ----------
    result : SimulationResult
        Simulation results
    particle_indices : List[int] or None
        Indices of particles to plot (if None, plots all tracked)
    save_path : str or None
        Path to save figure (if None, returns figure)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if particle_indices is None:
        particle_indices = list(range(result.Z.shape[1]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(particle_indices)))
    geom = result.geometry
    
    for idx, j in enumerate(particle_indices):
        if j >= result.Z.shape[1]:
            continue
        
        z_traj = result.Z[:, j]
        valid = ~np.isnan(z_traj)
        
        if np.any(valid):
            ax.plot(
                result.times[valid] * 1e9,  # Convert to ns
                z_traj[valid] / nm,
                color=colors[idx],
                linewidth=1.5,
                alpha=0.8,
                label=f"Particle {j}",
            )
    
    # Mark membrane boundaries
    ax.axhline(geom.membrane_top / nm, color="black", linestyle="--", alpha=0.5, label="Membrane")
    ax.axhline(geom.membrane_bottom / nm, color="black", linestyle="--", alpha=0.5)
    ax.axhline(geom.z_bottom / nm, color="red", linestyle=":", alpha=0.7, label="Bottom")
    
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("z position (nm)", fontsize=12)
    ax.set_title("Vertical Displacement vs Time", fontsize=14, fontweight="bold")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    
    return fig


def plot_cross_capture_map(
    results_list: List[SimulationResult],
    param_values: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot pull-back fraction vs parameter value for parameter scans.
    
    Parameters
    ----------
    results_list : List[SimulationResult]
        List of simulation results for different parameter values
    param_values : np.ndarray
        Parameter values corresponding to each result
    save_path : str or None
        Path to save figure (if None, returns figure)
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    # Import here to avoid circular dependency
    from .analysis import compute_pullback_fraction
    
    pullback_fractions = []
    for result in results_list:
        frac = compute_pullback_fraction(result)
        pullback_fractions.append(frac)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(param_values, pullback_fractions, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Parameter Value", fontsize=12)
    ax.set_ylabel("Pull-back Fraction", fontsize=12)
    ax.set_title("Pull-back Fraction vs Parameter", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return None
    
    return fig
