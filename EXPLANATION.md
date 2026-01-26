# Complete Explanation: PNPS-Based Nanopore Array Simulator


## What This Repository Does

This simulator models how **charged spherical particles** (like DNA, proteins, or nanoparticles) move through an **array of nanopores** in a membrane under the influence of:
- **Electric fields** (applied voltages)
- **Electro-osmotic flow** (fluid flow driven by charged walls)
- **Brownian motion** (random thermal fluctuations)
- **Electrostatic screening** (Debye-Hückel theory)

The goal is to understand phenomena like:
- How particles translocate through pores
- Cross-capture between neighboring pores
- Pull-back effects (particles being recaptured)
- Optimal pore spacing for array designs

---

## The Physics: Mathematical Foundation

### 1. **Langevin Dynamics Equation**

The core physics is described by the **underdamped Langevin equation**:

$$
m \frac{d\mathbf{v}}{dt} = -\zeta (\mathbf{v} - \mathbf{u}_\text{fluid}) + q_\text{eff} \mathbf{E} + \sqrt{2 \zeta k_B T} \, \boldsymbol{\xi}(t)
$$

Where:
- **m**: particle mass (kg)
- **v**: particle velocity (m/s)
- **ζ = 6π η a**: Stokes drag coefficient
  - η: fluid viscosity (Pa·s)
  - a: particle radius (m)
- **u_fluid**: electro-osmotic flow velocity (m/s)
- **q_eff**: effective charge (C) - accounts for Debye screening
- **E**: electric field (V/m)
- **k_B T**: thermal energy (J)
- **ξ(t)**: Gaussian white noise (random force)

**Physical interpretation:**
- Left side: mass × acceleration
- First term: drag force (proportional to velocity relative to fluid)
- Second term: electric force (Coulomb's law)
- Third term: random thermal force (Brownian motion)

### 2. **Electrostatics: Potential Profile**

The electric potential is **piecewise-linear**:

$$
\Phi(z) = \begin{cases}
\Phi_\text{top} & z \geq z_\text{membrane,top} \\
\Phi_\text{mid} & z_\text{membrane,bot} \leq z < z_\text{membrane,top} \\
\Phi_\text{bottom} & z < z_\text{membrane,bot}
\end{cases}
$$

Electric field (1D, z-direction):
$$
E_z = -\frac{d\Phi}{dz}
$$

Since Φ is piecewise linear, **E_z is piecewise constant**.

**Typical values:**
- Top reservoir: **+200 mV** (positive, repels negative particles)
- Pore/membrane: **-200 mV** (negative, attracts particles)
- Bottom reservoir: **-1000 mV** (very negative, strong pull)

### 3. **Debye Screening: Effective Charge**

In electrolyte solutions, ions screen charges. The **Debye length** λ_D is:

$$
\lambda_D = \sqrt{\frac{\varepsilon k_B T}{2 I e^2 N_A}}
$$

Where:
- **ε = ε_r ε_0**: permittivity (F/m)
- **I**: ionic strength (mol/L)
- **e**: elementary charge (C)
- **N_A**: Avogadro's number

**Ionic strength** for salt with ions of charge z_i:
$$
I = \frac{1}{2} \sum_i c_i z_i^2
$$

**Effective charge** (reduced by screening):
$$
q_\text{eff} = q_\text{bare} \left(1 + \frac{a}{\lambda_D}\right) e^{-a/\lambda_D}
$$

- If λ_D → ∞ (no salt): q_eff = q_bare (no screening)
- If λ_D << a: q_eff << q_bare (strong screening)

### 4. **Electro-Osmotic Flow**

Charged walls create a **slip velocity** (Smoluchowski):

$$
v_\text{EO} = -\frac{\varepsilon \zeta}{\eta} E_z
$$

Where:
- **ζ**: zeta potential (V) - wall surface charge
- **ε**: permittivity
- **η**: viscosity

This creates a **fluid flow** that drags particles along.

### 5. **Numerical Integration: Euler-Maruyama**

The continuous Langevin equation is discretized:

**Velocity update:**
$$
\mathbf{v}_{n+1} = \mathbf{v}_n + \left[-\frac{\zeta}{m}(\mathbf{v}_n - \mathbf{u}_\text{fluid}) + \frac{q_\text{eff}}{m}\mathbf{E}\right]\Delta t + \sigma \boldsymbol{\eta}_n
$$

**Position update:**
$$
\mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_{n+1} \Delta t
$$

Where:
- **σ = √(2 ζ k_B T / m² × Δt)**: noise amplitude
- **η_n**: vector of independent standard normals
- **Δt**: time step (chosen as 0.01 × m/ζ for stability)

---

## How Everything Works: Step-by-Step

### **Phase 1: Initialization** (`simulation.py`)

1. **Load Configuration** (YAML file):
   ```python
   config = {
       "geometry": {"n_pores": 3, "pore_spacing_nm": 50.0, ...},
       "electrostatics": {"Phi_top_mV": 200.0, ...},
       "particles": {"n_particles": 100, "radius_nm": 2.0, ...},
       "solution": {"solvent": "water", "salt": "NaCl", ...}
   }
   ```

2. **Build Geometry** (`geometry.py`):
   - Create `Geometry` object with:
     - Domain boundaries (z_top, z_bottom)
     - Membrane layer (membrane_top, membrane_bottom)
     - Array of `Pore` objects (center_y, z_top, z_bottom, r_top, r_bottom)
   - For single pore: center at y=0
   - For array: pores spaced by `pore_spacing_nm`

3. **Compute Physical Parameters**:
   - **Mass**: m = (4/3)π a³ ρ
   - **Drag**: ζ = 6π η a (Stokes law)
   - **Debye length**: λ_D = √(ε k_B T / (2 I e² N_A))
   - **Effective charge**: q_eff = q_bare × (1 + a/λ_D) × exp(-a/λ_D)
   - **Time step**: dt = 0.01 × m/ζ (or from config)

4. **Initialize Particles** (`particles.py`):
   - Create N particles at top reservoir
   - Initial positions: random y, z near z_top
   - Initial velocities: zero or Maxwell-Boltzmann distribution
   - Assign `history_index` to track subset for full trajectories

### **Phase 2: Main Simulation Loop** (`simulation.py`)

For each time step:

1. **Langevin Integration** (`integrator.py`):
   ```python
   langevin_step(particles, geom, species, phys_params, dt, rng, config)
   ```
   
   **Inside the integrator:**
   - Pack active particles into arrays (vectorized)
   - Compute E_z at each particle's z position (`Ez_profile`)
   - Determine region (top/pore/bottom) for zeta potential
   - Compute v_EO at each position (`eo_velocity`)
   - Calculate forces:
     - Drag: -ζ/m × (v - v_EO)
     - Electric: q_eff/m × E_z
   - Generate random noise: σ × η (Gaussian)
   - Update velocity: v_new = v + (forces) × dt + noise
   - Update position: r_new = r + v_new × dt

2. **Boundary Conditions**:
   - **Lateral (y)**: Periodic boundary (wrap around)
   - **Top (z)**: Reflect if above domain
   - **Bottom (z)**: Mark as translocated if below z_bottom
   - **Membrane walls**: 
     - If outside pore: reflect velocity
     - If inside pore: check radius constraint, reflect if outside

3. **Update Pore Association**:
   - For each particle: find nearest pore center
   - Track which pore particle is associated with

4. **Store Trajectories**:
   - Save (y, z) positions for tracked particles (decimated in time)
   - Log translocation events immediately

### **Phase 3: Post-Processing** (`analysis.py`, `visualization.py`)

1. **Compute Statistics**:
   - Translocation fraction
   - Mean/median translocation time
   - Pull-back fraction (recaptured particles)

2. **Visualization**:
   - **plot_paths()**: 2D side view (y-z) with:
     - Reservoirs (colored regions)
     - Membrane (gray block)
     - Pores (white with blue border, tapered shape)
     - Trajectories (green lines)
     - Potential annotations
   - **plot_landing_histogram()**: Distribution of exit positions
   - **plot_z_vs_time()**: Time series of z(t)

---

## Key Design Decisions

### **Why 2D (y-z) instead of 3D?**
- Nanopores are typically cylindrical (axisymmetric)
- Reduces computational cost
- Captures essential physics (lateral vs. vertical motion)

### **Why Piecewise-Linear Potential?**
- Approximates full PNPS solution without solving Poisson-Nernst-Planck
- Fast to compute
- Captures key physics: strong field in reservoirs, weak in pore

### **Why Euler-Maruyama?**
- Simple, stable for underdamped systems
- Time step chosen to resolve drag timescale (τ = m/ζ)
- Vectorized for efficiency (all particles at once)

### **Why Track Only Subset?**
- Memory efficiency: full trajectories are large
- Events (translocation) logged immediately
- Can track 1000s of particles, store ~20 full trajectories

---

## Example Workflow

```python
from nanopore_array_sim import run_simulation, load_config

# 1. Load configuration
config = load_config("configs/default_config.yaml")

# 2. Run simulation
result = run_simulation(config)
# Inside: builds geometry, initializes particles, runs time loop

# 3. Analyze results
from nanopore_array_sim.analysis import compute_translocation_stats
stats = compute_translocation_stats(result)
print(f"Translocation rate: {stats['fraction_translocated']:.2%}")

# 4. Visualize
from nanopore_array_sim.visualization import plot_paths
fig = plot_paths(result, show_potential=True)
fig.savefig("trajectories.png")
```

---

## Physical Validation

The simulator should reproduce known physics:

1. **Diffusion**: In no field, mean-squared displacement: ⟨r²⟩ = 2Dt
   - Where D = k_B T / ζ (Einstein relation)

2. **Drift**: In uniform field, drift velocity: v_drift = q_eff E / ζ

3. **Translocation time**: Should match 1D results for single pore

4. **Energy**: In absence of noise, energy approximately conserved

---

## Extensions & Future Work

- **3D geometry**: Full 3D simulation
- **Full PNPS**: Solve Poisson-Nernst-Planck equations
- **Particle-particle interactions**: Excluded volume, electrostatic repulsion
- **Non-spherical particles**: Rods, ellipsoids
- **Time-dependent fields**: AC voltages
- **GPU acceleration**: CUDA/OpenCL for large arrays

---

## Summary

This simulator combines:
- **Classical mechanics** (Langevin dynamics)
- **Electrostatics** (Coulomb forces, Debye screening)
- **Fluid mechanics** (Stokes drag, electro-osmosis)
- **Statistical mechanics** (Brownian motion, thermal fluctuations)

To model **nanoscale transport** through **engineered nanopore arrays**, enabling design optimization for applications in:
- DNA sequencing
- Protein analysis
- Nanoparticle separation
- Biosensing
