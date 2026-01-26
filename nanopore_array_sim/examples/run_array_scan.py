"""
Parameter sweep for pull-back analysis.

This script runs simulations for different pore spacings and
analyzes the pull-back fraction to optimize array design.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nanopore_array_sim import run_simulation
from nanopore_array_sim.analysis import scan_parameter, compute_pullback_fraction, compute_translocation_stats
from nanopore_array_sim.visualization import plot_cross_capture_map
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Run parameter sweep for pore spacing."""
    print("=" * 60)
    print("Pore Spacing Parameter Scan")
    print("=" * 60)
    
    # Base configuration
    config_base = {
        "geometry": {
            "n_pores": 3,
            "pore_spacing_nm": 50.0,  # Will be varied
            "pore_radius_nm": 5.0,
            "pore_length_nm": 10.0,
            "z_top_nm": 100.0,
            "z_bottom_nm": -50.0,
        },
        "electrostatics": {
            "Phi_top_mV": 200.0,
            "Phi_mid_mV": -200.0,
            "Phi_bottom_mV": -1000.0,
            "zeta_top_mV": -50.0,
            "zeta_pore_mV": -100.0,
            "zeta_bottom_mV": -50.0,
        },
        "particles": {
            "n_particles": 50,
            "radius_nm": 2.0,
            "density_kg_per_m3": 1000.0,
            "z_bare": -10,
            "initial_distribution": "pore_centered",
            "initial_velocity": "zero",
        },
        "solution": {
            "solvent": "water",
            "salt": "NaCl",
            "concentration_M": 0.1,
            "T_K": 298.15,
        },
        "simulation": {
            "t_max_s": 1e-6,
            "dt_s": None,
            "n_tracked": 20,
            "seed": 42,
        },
    }
    
    # Parameter values to scan
    pore_spacings = [30.0, 40.0, 50.0, 60.0, 70.0]  # nm
    
    print(f"\nScanning pore spacing: {pore_spacings} nm")
    print(f"Number of repeats per value: 1")
    
    # Run parameter scan
    results = scan_parameter(
        config_base,
        "geometry.pore_spacing_nm",
        pore_spacings,
        n_repeats=1,
    )
    
    # Analyze results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    pullback_fractions = []
    translocation_fractions = []
    
    for i, (spacing, result) in enumerate(zip(pore_spacings, results)):
        pullback_frac = compute_pullback_fraction(result)
        stats = compute_translocation_stats(result)
        
        pullback_fractions.append(pullback_frac)
        translocation_fractions.append(stats["fraction_translocated"])
        
        print(f"\nSpacing: {spacing:.1f} nm")
        print(f"  Translocation fraction: {stats['fraction_translocated']:.2%}")
        print(f"  Pull-back fraction: {pullback_frac:.2%}")
        if stats["fraction_translocated"] > 0:
            print(f"  Mean translocation time: {stats['mean']*1e9:.2f} ns")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "figs"
    output_dir.mkdir(exist_ok=True)
    
    # Plot pull-back fraction vs spacing
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(pore_spacings, pullback_fractions, "o-", linewidth=2, markersize=8, color="red")
    ax1.set_xlabel("Pore Spacing (nm)", fontsize=12)
    ax1.set_ylabel("Pull-back Fraction", fontsize=12)
    ax1.set_title("Pull-back vs Pore Spacing", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pore_spacings, translocation_fractions, "o-", linewidth=2, markersize=8, color="blue")
    ax2.set_xlabel("Pore Spacing (nm)", fontsize=12)
    ax2.set_ylabel("Translocation Fraction", fontsize=12)
    ax2.set_title("Translocation vs Pore Spacing", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "array_scan_results.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'array_scan_results.png'}")
    
    # Also use the cross-capture map function
    fig2 = plot_cross_capture_map(results, np.array(pore_spacings))
    fig2.savefig(output_dir / "cross_capture_map.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'cross_capture_map.png'}")
    
    print("\n" + "=" * 60)
    print("Parameter scan complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
