"""AM1, RM1 and MNDO against MOPAC 22.1.1 (OLDFPC 1SCF).

Covers what the AM1 tables used to lack: boron's Gaussians and its special B-H/B-C/B-halogen core-core terms, the
Stewart (2004) parameters of Li, Be, Na and Mg together with their pair parameters (alpb, xfac) in AM1 and MNDO,
and RM1.
"""

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

HEXA = [np.array([np.cos(a), np.sin(a), 0.0]) for a in (0, 2.0944, 4.18879)]

# species and coordinates (Angstrom), MOPAC heat of formation (kcal/mol)
CASES = {
    ("AM1", "BeH2"): ([4, 1, 1], [[0, 0, 0], [1.33, 0, 0], [-1.33, 0.1, 0]], 33.100),
    ("AM1", "BH3"): ([5, 1, 1, 1], [[0, 0, 0]] + [1.19 * d for d in HEXA], 26.262),
    ("AM1", "BF3"): ([9, 9, 9, 5], [1.31 * d for d in HEXA] + [[0, 0, 0]], -272.101),
    ("AM1", "CH3BH2"): (
        [6, 5, 1, 1, 1, 1, 1],
        [
            [0, 0, 0],
            [1.57, 0, 0],
            [-0.36, 1.03, 0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [2.1, 1.0, 0],
            [2.1, -1.0, 0.1],
        ],
        11.525,
    ),
    ("AM1", "H2BNH2"): (
        [7, 5, 1, 1, 1, 1],
        [[0, 0, 0], [1.4, 0, 0], [-0.5, 0.85, 0], [-0.5, -0.85, 0.1], [2.0, 1.0, 0], [2.0, -1.0, 0.1]],
        -23.518,
    ),
    ("AM1", "LiH"): ([3, 1], [[0, 0, 0], [1.6, 0, 0]], 37.182),
    ("AM1", "MgH2"): ([12, 1, 1], [[0, 0, 0], [1.7, 0, 0], [-1.7, 0.2, 0]], 40.427),
    ("MNDO", "LiH"): ([3, 1], [[0, 0, 0], [1.6, 0, 0]], 47.913),
    ("MNDO", "MgH2"): ([12, 1, 1], [[0, 0, 0], [1.7, 0, 0], [-1.7, 0.2, 0]], 42.795),
    ("RM1", "H2O"): ([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], -57.811),
}


def run(method, species, xyz, **extra):
    torch.set_default_dtype(torch.float64)
    par = {
        "method": method,
        "scf_eps": 1e-10,
        "scf_converger": [1],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
        **extra,
    }
    mol = Molecule(
        Constants(),
        par,
        torch.tensor(np.array([xyz]), dtype=torch.float64),
        torch.tensor([species]),
        charges=torch.tensor([0.0]),
    )
    if "scf_backward" in extra:
        mol.coordinates.requires_grad_()
    Electronic_Structure(par)(mol)
    return mol


@pytest.mark.parametrize("method, name", list(CASES))
def test_heat_of_formation_matches_mopac(method, name):
    species, xyz, hf = CASES[method, name]
    assert float(run(method, species, xyz).Hf) * 23.061 == pytest.approx(hf, abs=2e-3)


@pytest.mark.parametrize("method, name", [("AM1", "BF3"), ("AM1", "LiH"), ("MNDO", "MgH2")])
def test_analytic_force_matches_autodiff(method, name):
    species, xyz, _ = CASES[method, name]
    xyz = np.array(xyz) + 0.05 * np.sin(np.arange(3 * len(xyz))).reshape(-1, 3)
    autodiff = run(method, species, xyz, scf_backward=1).force
    analytic = run(method, species, xyz, analytical_gradient=[True]).force
    torch.testing.assert_close(autodiff, analytic, rtol=0.0, atol=1e-6)
