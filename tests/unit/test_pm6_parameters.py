"""Checks on the PM6 parameter files (generated from MOPAC's parameters_for_PM6_C.F90)."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

PARAMS = Path(__file__).resolve().parents[2] / "seqm" / "params"
MAIN_FILES = ["parameters_PM6_MOPAC.csv", "parameters_PM6_SP_MOPAC.csv"]
PAIR_FILES = ["PWCCT_PM6_MOPAC.csv", "PWCCT_PM6_SP_MOPAC.csv"]


def read_params(name):
    rows = list(csv.reader(open(PARAMS / name, newline="")))
    header = [h.strip() for h in rows[0]]
    return header, {int(r[0]): dict(zip(header, [x.strip() for x in r])) for r in rows[1:]}


@pytest.mark.parametrize("name", MAIN_FILES)
def test_every_row_has_every_field(name):
    rows = list(csv.reader(open(PARAMS / name, newline="")))
    assert {len(r) for r in rows} == {len(rows[0])}
    assert [int(r[0]) for r in rows[1:]] == list(range(1, 108))
    for r in rows[1:]:
        for field in r[2:]:  # a missing comma (as in the old Re row) leaves two numbers in one field
            float(field)


@pytest.mark.parametrize("name", MAIN_FILES)
def test_values_have_full_precision(name):
    _, p = read_params(name)
    assert float(p[8]["U_ss"]) == -91.678761
    assert float(p[6]["zeta_s"]) == 2.047558
    assert float(p[9]["g_pp"]) == 8.417366


@pytest.mark.parametrize("name", PAIR_FILES)
def test_pair_parameters(name):
    pairs = {(int(r[0]), int(r[1])): (float(r[2]), float(r[3])) for r in csv.reader(open(PARAMS / name))}
    assert pairs[(8, 1)] == (1.260942, 0.192295)
    # pairs are stored for i >= j; d-block pairs used to be missing
    assert pairs[(40, 1)][0] > 0 and pairs[(72, 1)][0] > 0
    assert all(v == (0.0, 0.0) for (i, j), v in pairs.items() if i < j)


def test_d_block_is_parameterized():
    _, p = read_params("parameters_PM6_MOPAC.csv")
    for z in [*range(39, 48), 57, *range(72, 80)]:  # Y-Ag, La, Hf-Au
        for col in ("zeta_s", "zeta_p", "zeta_d", "U_dd", "d_orb_exp_tail"):
            assert float(p[z][col]) != 0.0, (z, col)


def test_sp_file_only_drops_d_orbitals_of_main_group():
    _, full = read_params("parameters_PM6_MOPAC.csv")
    _, sp = read_params("parameters_PM6_SP_MOPAC.csv")
    differing = {(z, c) for z in full for c in full[z] if full[z][c] != sp[z][c]}
    assert differing == {(z, c) for z in range(13, 18) for c in ("U_dd", "zeta_d", "beta_d")}


@pytest.mark.parametrize("Z", [40, 72], ids=["ZrH4", "HfH4"])
def test_d_block_hydride_runs(Z):
    # used to fail with "math domain error" in GetSlaterCondonParameter (zeta_d = 0)
    torch.set_default_dtype(torch.float64)
    T = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]) / np.sqrt(3)
    xyz = torch.tensor(np.vstack([[0, 0, 0], 1.85 * T]))[None]
    par = {
        "method": "PM6",
        "scf_eps": 1e-8,
        "scf_converger": [2, 0.0],
        "sp2": [False],
        "pair_outer_cutoff": 1e10,
        "eig": True,
        "analytical_gradient": [False],
    }
    mol = Molecule(Constants(), par, xyz, torch.tensor([[Z, 1, 1, 1, 1]]), charges=torch.tensor([0.0]))
    Electronic_Structure(par)(mol)
    assert torch.isfinite(mol.Etot).all()
