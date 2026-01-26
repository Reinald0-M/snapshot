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
    show_potential: bool = True,
) -> plt.Figure:
    """
    Plot 2D side view (y-z) showing trajectories and geometry.
    
    Matches the sketch design with tapered pores, membrane structure,
    and green particle trajectories.
    
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
    show_potential : bool
        Whether to annotate potential values on the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if geom is None:
        geom = result.geometry
    if config is None:
        config = result.config
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set axis limits FIRST so patches are drawn in correct coordinate space
    # We'll refine these later, but need initial limits
    
    # Determine lateral extent for drawing
    if len(geom.pores) == 0:
        y_min, y_max = -50 * nm, 50 * nm
    elif len(geom.pores) == 1:
        # Single pore: show wider view with reservoirs
        pore = geom.pores[0]
        # Show wider view to see reservoirs
        y_min = -5 * pore.r_top
        y_max = 5 * pore.r_top
    else:
        # Multiple pores: show all pores with some margin
        pore_centers = [p.center_y for p in geom.pores]
        max_radius = max(p.r_top for p in geom.pores)
        y_min = min(pore_centers) - 2 * max_radius
        y_max = max(pore_centers) + 2 * max_radius
    
    # Draw top reservoir (above membrane) - make it clearly visible
    # Convert coordinates to nm for patches
    top_reservoir_height = geom.z_top - geom.membrane_top
    if top_reservoir_height > 0:
        top_reservoir = patches.Rectangle(
            (y_min / nm, geom.membrane_top / nm),
            (y_max - y_min) / nm,
            top_reservoir_height / nm,
            facecolor="lightblue",
            edgecolor="navy",
            linewidth=2.5,
            alpha=0.5,  # Increased alpha for visibility
            zorder=1,
            label="Top Reservoir",
        )
        ax.add_patch(top_reservoir)
    
    # Draw bottom reservoir (below membrane) - wider opening, clearly visible
    bottom_reservoir_height = geom.membrane_bottom - geom.z_bottom
    if bottom_reservoir_height > 0:
        # Bottom reservoir opens wider
        if len(geom.pores) == 1:
            # For single pore, make bottom reservoir much wider
            bottom_y_min = y_min * 2.5
            bottom_y_max = y_max * 2.5
        else:
            bottom_y_min = y_min * 1.5
            bottom_y_max = y_max * 1.5
        bottom_reservoir = patches.Rectangle(
            (bottom_y_min / nm, geom.z_bottom / nm),
            (bottom_y_max - bottom_y_min) / nm,
            bottom_reservoir_height / nm,
            facecolor="lightcoral",
            edgecolor="darkred",
            linewidth=2.5,
            alpha=0.5,  # Increased alpha for visibility
            zorder=1,
            label="Bottom Reservoir",
        )
        ax.add_patch(bottom_reservoir)
    
    # Draw membrane as solid block - make it clearly visible
    mem_height = geom.membrane_top - geom.membrane_bottom
    mem_rect = patches.Rectangle(
        (y_min / nm, geom.membrane_bottom / nm),
        (y_max - y_min) / nm,
        mem_height / nm,
        facecolor="gray",
        edgecolor="black",
        linewidth=3,
        alpha=0.7,  # Increased alpha for visibility
        zorder=2,
        label="Membrane",
    )
    ax.add_patch(mem_rect)
    
    # Draw tapered pores (hourglass shape)
    from .geometry import pore_radius
    
    # Determine taper type from pore geometry
    # If r_top == r_bottom (constant radius), it should be straight
    use_hourglass = False
    # Only use hourglass if explicitly requested via some other mechanism, 
    # but for now let's trust the geometry defined in the simulation
    
    for idx, pore in enumerate(geom.pores):
        # Create tapered pore shape by sampling radius at multiple z positions
        z_samples = np.linspace(pore.z_bottom, pore.z_top, 100)  # More samples for smoother curve
        
        # Use linear interpolation by default to match actual physics
        # The pore_radius function handles the shape based on taper_type passed here.
        # However, we don't know the original taper_type used during construction easily
        # unless we guess or look at r_top vs r_bottom.
        # But for constant radius, we definitely want linear (cylinder).
        
        taper_type = "linear" # Default to linear for visualization unless we have info otherwise
        r_samples = [pore_radius(pore, z, taper_type=taper_type) for z in z_samples]
        
        # Create pore outline (left and right boundaries)
        y_left = np.array([pore.center_y - r for r in r_samples])
        y_right = np.array([pore.center_y + r for r in r_samples])
        
        # Combine into closed polygon (convert to nm for plotting)
        pore_y_nm = np.concatenate([y_left, y_right[::-1], [y_left[0]]]) / nm
        pore_z_nm = np.concatenate([z_samples, z_samples[::-1], [z_samples[0]]]) / nm
        
        # Draw pore opening - make it clearly visible with white fill and blue border
        pore_patch = patches.Polygon(
            list(zip(pore_y_nm, pore_z_nm)),
            facecolor="white",
            edgecolor="blue",
            linewidth=3,  # Thicker line for visibility
            zorder=4,  # Above membrane
            label="Pore" if idx == 0 else "",
        )
        ax.add_patch(pore_patch)
        
        # Also draw the outline explicitly for better visibility
        ax.plot(
            pore_y_nm,
            pore_z_nm,
            color="blue",
            linewidth=3,
            zorder=5,
            alpha=0.9,
        )
        
        # Draw pore centerline (dashed) - convert to nm
        ax.plot(
            [pore.center_y / nm, pore.center_y / nm],
            [pore.z_bottom / nm, pore.z_top / nm],
            "b--",
            alpha=0.5,
            linewidth=1.5,
            zorder=3,
        )
    
    # Plot trajectories in green (matching sketch)
    n_tracked = result.Y.shape[1]
    
    for j in range(n_tracked):
        y_traj = result.Y[:, j]
        z_traj = result.Z[:, j]
        
        # Remove NaN values
        valid = ~(np.isnan(y_traj) | np.isnan(z_traj))
        if np.any(valid):
            ax.plot(
                y_traj[valid] / nm,
                z_traj[valid] / nm,
                color="green",
                linewidth=1.2,
                alpha=0.6,
                zorder=4,
            )
    
    # Add potential annotations if requested
    if show_potential and config:
        electrostatics = config.get("electrostatics", {})
        phi_top = electrostatics.get("Phi_top_mV", 200.0)
        phi_mid = electrostatics.get("Phi_mid_mV", -200.0)
        phi_bot = electrostatics.get("Phi_bottom_mV", -1000.0)
        
        # Determine annotation x position
        if len(geom.pores) == 1:
            annot_x = y_max * 0.85
            pore = geom.pores[0]  # Get pore for single pore case
        else:
            annot_x = y_max * 0.9
            pore = geom.pores[0] if len(geom.pores) > 0 else None
        
        # Top reservoir: positive voltage above nanopore opening
        top_z = geom.membrane_top + (geom.z_top - geom.membrane_top) * 0.3
        ax.text(
            annot_x,
            top_z / nm,
            f"+{phi_top:.0f} mV",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="orange", linewidth=2, alpha=0.8),
            ha="right",
            zorder=6,
        )
        
        # Pore/membrane region: negative voltage at bottom of opening
        if pore is not None:
            pore_bottom_z = pore.z_bottom
            ax.text(
                annot_x,
                pore_bottom_z / nm,
                f"{phi_mid:.0f} mV",
                fontsize=11,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="orange", linewidth=2, alpha=0.8),
                ha="right",
                zorder=6,
            )
        
        # Bottom reservoir: very large negative voltage
        bottom_z = geom.z_bottom + (geom.membrane_bottom - geom.z_bottom) * 0.7
        ax.text(
            annot_x,
            bottom_z / nm,
            f"{phi_bot:.0f} mV",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="orange", linewidth=2, alpha=0.8),
            ha="right",
            zorder=6,
        )
        
        # Add arrows showing potential gradient (for single pore)
        if len(geom.pores) == 1 and pore is not None:
            # Arrow from top reservoir to pore opening
            ax.annotate(
                "",
                xy=(pore.center_y / nm, pore.z_top / nm),
                xytext=(pore.center_y / nm, top_z / nm),
                arrowprops=dict(arrowstyle="->", color="red", lw=2, alpha=0.6),
                zorder=5,
            )
            # Arrow from pore bottom to bottom reservoir
            ax.annotate(
                "",
                xy=(pore.center_y / nm, bottom_z / nm),
                xytext=(pore.center_y / nm, pore.z_bottom / nm),
                arrowprops=dict(arrowstyle="->", color="red", lw=2, alpha=0.6),
                zorder=5,
            )
    
    # Formatting
    ax.set_xlabel("y (nm)", fontsize=12)
    ax.set_ylabel("z (nm)", fontsize=12)
    
    # Title based on number of pores
    if len(geom.pores) == 1:
        ax.set_title("Single Pore: Particle Trajectories (Side View)", fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Nanopore Array ({len(geom.pores)} pores): Particle Trajectories", fontsize=14, fontweight="bold")
    
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    
    # Set reasonable view limits AFTER drawing patches
    # For single pore, show wider view to see reservoirs
    if len(geom.pores) == 1:
        # Show wider view including bottom reservoir expansion
        bottom_y_min = y_min * 2.5
        bottom_y_max = y_max * 2.5
        x_margin = abs(bottom_y_max - bottom_y_min) * 0.1
        ax.set_xlim((bottom_y_min - x_margin) / nm, (bottom_y_max + x_margin) / nm)
    else:
        x_margin = abs(y_max - y_min) * 0.1
        ax.set_xlim((y_min - x_margin) / nm, (y_max + x_margin) / nm)
    
    z_margin = (geom.z_top - geom.z_bottom) * 0.05
    ax.set_ylim((geom.z_bottom - z_margin) / nm, (geom.z_top + z_margin) / nm)
    
    # Force matplotlib to update the plot with all patches
    ax.autoscale_view()
    
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
