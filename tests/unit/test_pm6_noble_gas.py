"""PM6 noble-gas dimers against MOPAC 22.1.1 (PM6 OLDFPC 1SCF), X...X at 3.4 Angstrom.

MOPAC gives the noble gases s and p shells with different principal quantum numbers (npq in parameters_C.F90): s is one
above the period for Ne, Ar, Kr, Xe. The overlap integrals have to follow, or Kr2 and Xe2 are off by about 0.1 eV.
(He has its own tests: test_pm6_helium.py.)
"""

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

MOPAC_ETOT = {10: -544.459527, 18: -528.412745, 36: -517.830125, 54: -1883.632939}


@pytest.mark.parametrize("Z", list(MOPAC_ETOT), ids=["Ne2", "Ar2", "Kr2", "Xe2"])
def test_noble_gas_dimer_total_energy(Z):
    torch.set_default_dtype(torch.float64)
    xyz = torch.tensor(np.array([[[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]]]))
    par = {
        "method": "PM6",
        "scf_eps": 1e-10,
        "scf_converger": [0, 0.3],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
    }
    mol = Molecule(Constants(), par, xyz, torch.tensor([[Z, Z]]), charges=torch.tensor([0.0]))
    Electronic_Structure(par)(mol)
    assert float(mol.Etot) == pytest.approx(MOPAC_ETOT[Z], abs=1e-4)
