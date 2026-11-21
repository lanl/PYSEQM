"""The per-element tables in Constants must be indexed by true atomic number."""

import pytest
import torch

from seqm.seqm_functions.constants import Constants

TABLES = [
    "atomic_num",
    "tore",
    "iso",
    "qn",
    "qn_int",
    "qnD_int",
    "ussc",
    "uppc",
    "gssc",
    "gspc",
    "hspc",
    "gp2c",
    "gppc",
    "eheat",
    "mass",
]


@pytest.fixture(scope="module")
def const():
    return Constants()


def test_tables_cover_z_up_to_86(const):
    assert len(const.label) == 87
    for name in TABLES:
        assert len(getattr(const, name)) == 87, name


def test_atomic_num_is_z(const):
    assert torch.equal(const.atomic_num.detach().long(), torch.arange(87))


@pytest.mark.parametrize(
    "z, label, tore, qnd, mass",
    [
        (55, "Cs", 1.0, 0, 132.91),
        (56, "Ba", 2.0, 0, 137.33),
        (57, "La", 3.0, 5, 138.906),
        (71, "Lu", 3.0, 0, 174.967),
        (72, "Hf", 4.0, 5, 178.49),  # reported: Hf values sat at index 58
        (74, "W", 6.0, 5, 183.84),
        (79, "Au", 11.0, 5, 196.97),
        (80, "Hg", 2.0, 5, 200.59),
        (82, "Pb", 4.0, 0, 207.2),
    ],
)
def test_heavy_element_entries(const, z, label, tore, qnd, mass):
    assert const.label[z].strip() == label
    assert float(const.tore[z]) == tore
    assert int(const.qnD_int[z]) == qnd
    assert float(const.mass[z]) == pytest.approx(mass, rel=1e-5)
    assert int(const.qn_int[z]) == 6


def test_valence_electrons_match_occupations(const):
    # core charge = s + p (+ d) occupations, as in MOPAC (tore = ios + iop + iod);
    # for sp elements this must equal ussc + uppc
    for z in list(range(81, 86)):
        assert float(const.tore[z]) == float(const.ussc[z] + const.uppc[z])


def test_zn_cd_core_charge_matches_mopac(const):
    # MOPAC: tore = ios + iop + iod, and iod = 0 for Zn and Cd (filled d shell is not valence)
    assert float(const.tore[30]) == 2.0
    assert float(const.tore[48]) == 2.0


def test_integral_counts_follow_mopac_formulas(const):
    # carbon (s2 p2): <ss|ss>=1, <ss|pp>=4, <pp|pp>+hpp/2=1.5, -hpp/2=-0.5, <sp|sp>=-2
    z = 6
    assert [float(getattr(const, n)[z]) for n in ("gssc", "gspc", "gp2c", "gppc", "hspc")] == [
        1,
        4,
        1.5,
        -0.5,
        -2,
    ]


def test_warns_for_unparameterized_elements():
    import os

    import seqm
    from seqm.seqm_functions.parameters import params

    root_dir = os.path.join(os.path.dirname(seqm.__file__), "params") + "/"
    with pytest.warns(UserWarning, match=r"no parameters for atomic numbers \[58\]"):
        params(method="PM6", elements=[1, 6, 58], root_dir=root_dir)
