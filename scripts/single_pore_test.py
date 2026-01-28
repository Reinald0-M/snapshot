import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snapshot import run_simulation
from snapshot.visualization import plot_paths, plot_landing_histogram, plot_z_vs_time
from snapshot.analysis import compute_translocation_stats
import numpy as np


def main():
    config = {
        'geometry': {
            'n_pores': 1,
            'pore_radius_nm': 5.0,
            'pore_length_nm': 10.0,
            'z_top_nm': 20.0,
            'z_bottom_nm': -20.0,
            'taper_type': 'constant',
            'transition_height_nm': 5.0,
            'transition_angle_deg': 45.0
        },
        'electrostatics': {
            'Phi_top_mV': 200.0,
            'Phi_mid_mV': -200.0,
            'Phi_bottom_mV': -1000.0,
            'zeta_top_mV': -20.0,
            'zeta_pore_mV': -20.0,
            'zeta_bottom_mV': -20.0
        },
        'particles': {
            'n_particles' : 5,
            'radius_nm' : 2.0,
            'density_kg_per_m3' : 1350.0,
            'z_bare' : 20,
            'initial_distribution' : 'uniform',
            'initial_velocity' : 'zero'
        },
        'solution': {
            'solvent': 'water', 
            'salt': 'None',
            'T_K': 298.15},
        'simulation': {'t_max_s': 10e-8,
            'dt_s': None,
            'n_tracked': 10,
            'seed': 12345}
    }
    # Run simulation
    print("\nRunning simulation...")
    result = run_simulation(config)
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_translocation_stats(result)
    
    print(f"\nTranslocation Statistics:")
    print(f"  Fraction translocated: {stats['fraction_translocated']:.2%}")
    if stats['fraction_translocated'] > 0:
        print(f"  Mean time: {stats['mean']*1e9:.2f} ns")
        print(f"  Median time: {stats['median']*1e9:.2f} ns")
        print(f"  Std dev: {stats['std']*1e9:.2f} ns")
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "figs/test/"
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Trajectory plot (with potential annotations)
    fig1 = plot_paths(result, show_potential=False)
    fig1.savefig(output_dir / "single_pore_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'single_pore_trajectories.png'}")
    
    # Landing histogram
    fig2 = plot_landing_histogram(result)
    fig2.savefig(output_dir / "single_pore_landing.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'single_pore_landing.png'}")
    
    # z vs time
    fig3 = plot_z_vs_time(result, particle_indices=list(range(min(5, result.Z.shape[1]))))
    fig3.savefig(output_dir / "single_pore_z_vs_time.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'single_pore_z_vs_time.png'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()