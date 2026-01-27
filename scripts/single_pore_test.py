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
            'pre_radius_nm': 5.0,
            'pore_length_nm': 10.0,
        }
        'electrtostatics': {}
        'particles': {}
        'solution': {}
        'simulation': {}
    }


if __name__ == '__main__':
    main()