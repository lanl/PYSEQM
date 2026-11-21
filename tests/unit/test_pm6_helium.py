"""PM6 helium against MOPAC 22.1.1 (PM6 OLDFPC 1SCF).

In MOPAC He has a 2p shell. Its p orbitals feel no attraction by the other cores (spcore only evaluates the p-type core
attraction for Z >= 3), while its electron-electron integrals use the dipole and quadrupole charge separations that
calpar computes for it.
"""

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

# species, coordinates (Angstrom), MOPAC total energy (eV), MOPAC heat of formation (kcal/mol)
CASES = {
    "He2": ([2, 2], [[0, 0, 0], [3.4, 0, 0]], -108.201454, -0.1885),
    "HeH2": ([2, 1, 1], [[0, 0, 0], [1.4, 0.2, 0], [-0.3, 1.3, 0.4]], -72.162364, 206.3225),
    "HeNe": ([10, 2], [[0, 0, 0], [2.0, 0, 0]], -326.301879, 0.5646),
    "HeCH4": (
        [6, 1, 1, 1, 1, 2],
        [
            [0, 0, 0],
            [1.09, 0, 0],
            [-0.36, 1.03, 0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [2.5, 0.4, 0.2],
        ],
        -230.620558,
        2.6079,
    ),
}


def run(method, species, xyz):
    torch.set_default_dtype(torch.float64)
    order = sorted(range(len(species)), key=lambda i: -species[i])  # species must be non-increasing
    species, xyz = [species[i] for i in order], [xyz[i] for i in order]
    par = {
        "method": method,
        "scf_eps": 1e-10,
        "scf_converger": [0, 0.3],
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
    return mol


@pytest.mark.parametrize("name", list(CASES))
def test_helium_matches_mopac(name):
    species, xyz, etot, hf = CASES[name]
    mol = run("PM6", species, xyz)
    assert float(mol.Etot) == pytest.approx(etot, abs=1e-4)
    assert float(mol.Hf) * 23.061 == pytest.approx(hf, abs=2e-3)


def test_pm6_sp_refuses_helium():
    # its overlap routine has no 2p shell for He and would silently give wrong energies
    with pytest.raises(ValueError, match="He"):
        run("PM6_SP", [2, 1, 1], CASES["HeH2"][1])
