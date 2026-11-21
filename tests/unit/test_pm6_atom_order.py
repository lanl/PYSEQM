"""PM6 packs the Fock matrix as [atoms with a d shell, other heavy atoms, H], so the atoms have to come in that order.

Plain decreasing-Z order puts an sp-only atom heavier than a d-shell atom (Ge, Zn, Hg, Se, Ca ...) before it, which
silently dropped the d orbitals of the lighter atom (Ti-Ge was off by 4.7 eV).
"""

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.read_xyz import read_xyz

# species (d-shell atoms first), coordinates (Angstrom), MOPAC 22.1.1 total energy (PM6 OLDFPC 1SCF) in eV
CASES = {
    "TiGe": ([22, 32], [[0, 0, 0], [2.4, 0, 0]], -141.713512),
    "HgCl2": ([17, 17, 80], [[2.3, 0, 0], [-2.3, 0.2, 0], [0, 0, 0]], -538.802516),
    "ZnCl2H2": (
        [17, 17, 30, 1, 1],
        [[2.1, 0, 0], [-2.1, 0.2, 0], [0, 0, 0], [0, 1.5, 0], [0, -1.5, 0.3]],
        -559.466567,
    ),
}


def molecule(species, xyz):
    torch.set_default_dtype(torch.float64)
    par = {
        "method": "PM6",
        "scf_eps": 1e-10,
        "scf_converger": [0, 0.3],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
    }
    return par, Molecule(Constants(), par, torch.tensor(np.array([xyz]), dtype=torch.float64),
                         torch.tensor([species]), charges=torch.tensor([0.0]))  # fmt: skip


@pytest.mark.parametrize("name", list(CASES))
def test_d_shell_atom_lighter_than_an_sp_atom(name):
    species, xyz, etot = CASES[name]
    par, mol = molecule(species, xyz)
    Electronic_Structure(par)(mol)
    assert float(mol.Etot) == pytest.approx(etot, abs=2e-3)


def test_pm6_rejects_the_wrong_atom_order():
    with pytest.raises(ValueError, match="d shell first"):
        molecule([32, 22], CASES["TiGe"][1])


def test_read_xyz_sorts_for_pm6(tmp_path):
    path = tmp_path / "m.xyz"
    path.write_text("4\n\nGe 0 0 0\nH 0 1.5 0\nTi 2 0 0\nCl 0 0 2\n")
    assert read_xyz([path])[0][0].tolist() == [32, 22, 17, 1]
    assert read_xyz([path], method="PM6")[0][0].tolist() == [22, 17, 32, 1]
