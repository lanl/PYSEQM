"""Electronic energy of the free atom (energy.elec_energy_isolated_atom) against MOPAC's EISOL.

parameters_PM6_MOPAC.csv holds the EISOL that MOPAC prints for every parameterized element (run with OLDFPC, the
constants PYSEQM uses). For partly filled d shells it includes udd*iod and the one-center terms of eiscor.
"""

import os

import numpy as np
import pytest
import torch

import seqm
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.energy import elec_energy_isolated_atom
from seqm.seqm_functions.parameters import params

NAMES = [
    "U_ss",
    "U_pp",
    "U_dd",
    "g_ss",
    "g_pp",
    "g_sp",
    "g_p2",
    "h_sp",
    "s_orb_exp_tail",
    "d_orb_exp_tail",
    "F0SD",
    "G2SD",
    "zeta_s",
    "EISOL",
]
ROOT = os.path.join(os.path.dirname(seqm.__file__), "params") + "/"


@pytest.mark.filterwarnings("ignore:PM6 has no parameters")
def test_isolated_atom_energy_matches_mopac_eisol():
    torch.set_default_dtype(torch.float64)
    const = Constants()
    p = params(method="PM6", elements=list(range(1, 84)), parameters=NAMES, root_dir=ROOT)
    Z = torch.tensor([z for z in range(1, 84) if float(p[z, NAMES.index("zeta_s")]) > 0.1])
    col = {n: p[Z, i] for i, n in enumerate(NAMES)}
    E = elec_energy_isolated_atom(
        const,
        Z,
        col["U_ss"],
        col["U_pp"],
        col["g_ss"],
        col["g_pp"],
        col["g_sp"],
        col["g_p2"],
        col["h_sp"],
        udd=col["U_dd"],
        zsn=col["s_orb_exp_tail"],
        zdn=col["d_orb_exp_tail"],
        f0sd=col["F0SD"],
        g2sd=col["G2SD"],
    )
    diff = (E - col["EISOL"]).abs()
    worst = [(int(z), float(d)) for z, d in zip(Z, diff) if d > 2e-3]
    assert not worst, worst


@pytest.mark.parametrize("Z, hf_mopac", [(40, 91.2124), (72, 482.0117)], ids=["ZrH4", "HfH4"])
def test_d_block_heat_of_formation(Z, hf_mopac):
    # MOPAC 22.1.1, PM6 OLDFPC 1SCF; tetrahedral MH4 with M-H = 1.85 A
    torch.set_default_dtype(torch.float64)
    T = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / np.sqrt(3)
    xyz = torch.tensor(np.vstack([[0, 0, 0], 1.85 * T]))[None]
    par = {
        "method": "PM6",
        "scf_eps": 1e-10,
        "scf_converger": [1],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
    }
    mol = Molecule(Constants(), par, xyz, torch.tensor([[Z, 1, 1, 1, 1]]), charges=torch.tensor([0.0]))
    Electronic_Structure(par)(mol)
    assert float(mol.Hf) * 23.061 == pytest.approx(hf_mopac, abs=0.15)
