"""PM6 core-core repulsion (energy.pair_nuclear_energy) against MOPAC's ccrep (integrals/ccrep.F90).

The expected values come from a line-by-line transcription of ccrep.F90 evaluated with the same parameter files.
"""

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

CASES = {
    # Si-O correction -0.7e-3*exp(-(r - 2.9)**2) uses the distance in Angstrom
    "SiO_1.60": ([14, 8], [[0, 0, 0], [1.6, 0, 0]], 144.067334),
    # C-C triple-bond term 9.278465*exp(-5.983752*r)
    "C2H2": ([6, 6, 1, 1], [[0, 0, 0.6], [0, 0, -0.6], [0, 0, 1.66], [0, 0, -1.66]], 294.194224),
    # pairs without an xfac entry use the generic term 10*exp(-2.18*r)
    "TiNe": ([22, 10], [[0, 0, 0], [2.3, 0, 0]], 131.021997),
    "CrTi_short": ([24, 22], [[0, 0, 0], [0, 1.2, 0]], 251.089694),
}


@pytest.mark.filterwarnings("ignore:SCF for")
@pytest.mark.parametrize("name", list(CASES))
def test_core_core_matches_mopac_ccrep(name):
    torch.set_default_dtype(torch.float64)
    species, xyz, expected = CASES[name]
    par = {
        "method": "PM6",
        "scf_eps": 1e-8,
        "scf_converger": [1],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
    }
    mol = Molecule(
        Constants(),
        par,
        torch.tensor(np.array([xyz]), dtype=torch.float64),
        torch.tensor([species]),
        charges=torch.tensor([0.0]),
    )
    Electronic_Structure(par)(mol)
    assert float(mol.Enuc) == pytest.approx(expected, abs=2e-4)
