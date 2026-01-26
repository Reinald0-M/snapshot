# PNPS-Based Nanopore Array Simulator

A Python package for simulating PNPS-driven transport of spherical particles through an array of nanopores using Langevin dynamics.

## Overview

This simulator implements a 2D (y-z) Langevin dynamics model for particles moving through a nanopore array under applied electric fields and electro-osmotic flow. The system is designed to study cross-capture and pull-back effects in multi-pore configurations.

## Features

- **Multi-pore array geometry**: Configurable arrays of identical nanopores
- **PNPS-inspired electrostatics**: Piecewise-linear potential profiles with Debye screening
- **Langevin dynamics**: Vectorized Euler-Maruyama integration for efficient simulation
- **Electro-osmotic flow**: Smoluchowski slip velocity with region-specific zeta potentials
- **Visualization**: Publication-quality plots of trajectories and analysis
- **Parameter scanning**: Tools for design optimization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single-Pore Validation

```bash
python nanopore_array_sim/examples/run_single_pore_demo.py
```

### Parameter Scan

```bash
python nanopore_array_sim/examples/run_array_scan.py
```

### Using the Package

```python
from nanopore_array_sim import run_simulation, load_config

# Load default configuration
config = load_config()

# Run simulation
result = run_simulation(config)

# Access results
print(f"Translocation fraction: {result.translocated_mask.mean():.2%}")
```

## Package Structure

```
nanopore_array_sim/
  __init__.py              # Package initialization
  constants.py              # Physical constants, solvents, salts
  geometry.py               # Domain geometry and pore array definitions
  electrostatics.py         # Electric fields, potentials, EO flow
  particles.py              # Particle classes and initialization
  integrator.py             # Langevin dynamics integrator
  simulation.py             # Main simulation orchestration
  visualization.py          # Trajectory plotting and analysis plots
  analysis.py               # Metrics computation and parameter scanning
  configs/
    default_config.yaml     # Default simulation parameters
  examples/
    run_single_pore_demo.py # Single-pore validation case
    run_array_scan.py       # Parameter sweep for pull-back analysis
```

## Configuration

Simulation parameters are specified in YAML format. See `configs/default_config.yaml` for the default configuration and parameter descriptions.

Key configuration sections:
- `geometry`: Pore array dimensions and spacing
- `electrostatics`: Applied potentials and zeta potentials
- `particles`: Particle properties and initial conditions
- `solution`: Solvent and salt properties
- `simulation`: Time stepping and tracking parameters

## Physics

The simulator implements:

1. **Langevin dynamics**: Underdamped Brownian motion with Stokes drag
2. **Electric fields**: Piecewise-linear potential profiles
3. **Debye screening**: Effective charge reduction with ionic strength
4. **Electro-osmotic flow**: Smoluchowski slip velocity
5. **Boundary conditions**: Periodic lateral boundaries, reflecting membrane walls

## Dependencies

- `numpy >= 1.20` (vectorized operations, random number generation)
- `matplotlib >= 3.3` (visualization)
- `pyyaml >= 5.4` (config file parsing)

## References

The physics implementation is aligned with PNPS/Langevin approaches in Baude et al. (2021) and related PNPS literature.

## License

See LICENSE file for details.
