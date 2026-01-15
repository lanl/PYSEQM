"""
Public-facing API for seqm.

Import from here (or from `seqm` which re-exports these symbols) to avoid
long dotted paths in user code.
"""

from .dynamics import NACConfig, active_state_tensor, resolve_nac_config
from .dynamics.xlbomd import EnergyXL, ForceXL
from .ElectronicStructure import Electronic_Structure
from .io.xyz import read_xyz, save_xyz
from .MolecularDynamics import XL_BOMD, XL_ESMD, Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin
from .Molecule import Molecule
from .NonadiabaticDynamics import EhrenfestDynamics, NonadiabaticDynamicsBase, SurfaceHoppingDynamics
from .optimization import geomeTRIC_optimization
from .seqm_functions.constants import Constants

# Friendlier alias while keeping legacy name intact
geometric_optimization = geomeTRIC_optimization

__all__ = [
    "XL_BOMD",
    "XL_ESMD",
    "Constants",
    "EhrenfestDynamics",
    "Electronic_Structure",
    "EnergyXL",
    "ForceXL",
    "Molecular_Dynamics_Basic",
    "Molecular_Dynamics_Langevin",
    "Molecule",
    "NACConfig",
    "NonadiabaticDynamicsBase",
    "SurfaceHoppingDynamics",
    "active_state_tensor",
    "geomeTRIC_optimization",
    "geometric_optimization",
    "read_xyz",
    "resolve_nac_config",
    "save_xyz",
]
