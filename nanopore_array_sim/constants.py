"""
Physical constants and material properties.

This module centralizes all physical constants, solvent properties,
and salt definitions used throughout the simulation.
"""

import numpy as np

# Physical constants (SI units)
nm = 1e-9  # nanometer to meter conversion
e_charge = 1.602176634e-19  # elementary charge (C)
kB = 1.380649e-23  # Boltzmann constant (J/K)
eps0 = 8.8541878128e-12  # vacuum permittivity (F/m)
N_A = 6.02214076e23  # Avogadro's number (1/mol)

# Solvent properties
SOLVENTS = {
    "water": {"eta": 0.89e-3, "eps_r": 78.5, "name": "Water"},
    "ethanol": {"eta": 1.074e-3, "eps_r": 24.5, "name": "Ethanol"},
    "glycerol": {"eta": 934e-3, "eps_r": 42.5, "name": "Glycerol"},
    "PBS": {"eta": 0.91e-3, "eps_r": 77.0, "name": "PBS (1X)"},
}

# Salt definitions: list of (stoichiometric_coefficient, charge) pairs
SALTS = {
    "NaCl": [(1, 1), (1, -1)],
    "KCl": [(1, 1), (1, -1)],
    "MgCl2": [(1, 2), (2, -1)],
    "CaCl2": [(1, 2), (2, -1)],
}


def ionic_strength(molarity: float, salt_type: str = "NaCl") -> float:
    """
    Compute ionic strength I from molarity and salt type.
    
    Parameters
    ----------
    molarity : float
        Salt concentration in mol/L
    salt_type : str
        Salt type (must be in SALTS dictionary)
        
    Returns
    -------
    float
        Ionic strength I in mol/L
        
    Raises
    ------
    ValueError
        If salt_type is not defined in SALTS
    """
    if salt_type not in SALTS:
        raise ValueError(f"Salt {salt_type} not defined. Available: {list(SALTS.keys())}")
    
    sum_cz2 = 0.0
    for coeff, z in SALTS[salt_type]:
        c_i = molarity * coeff
        sum_cz2 += c_i * (z ** 2)
    
    return 0.5 * sum_cz2
